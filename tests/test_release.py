"""Release entrypoints must work without checkpoints or private paths."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import infer
from unicache.runtime import build_plan_from_config


class ReleaseTests(unittest.TestCase):
    def args(self, *extra):
        with mock.patch.object(sys, 'argv', ['infer.py', '--prompt', 'test',
                                           '--output-dir', '/tmp/unicache-test', *extra]):
            return infer.parse_args()

    def test_all_backend_task_plans_without_weights(self):
        for backend in ['full', 'torch', 'engine']:
            for task in ['understanding', 'text_to_image', 'editing']:
                with self.subTest(backend=backend, task=task):
                    args = self.args('--backend', backend, '--task', task,
                                     '--image', 'not-loaded.jpg', '--plan-only')
                    bundle = build_plan_from_config(infer.build_runtime_config(args))
                    self.assertTrue(bundle.segment_plan.segments)

    def test_rejects_wrong_backend_config(self):
        path = Path(infer.__file__).parent / 'configs/engine/editing.json'
        args = self.args('--backend', 'torch', '--task', 'editing',
                         '--image', 'not-loaded.jpg', '--plan-only', '--config', str(path))
        with self.assertRaisesRegex(ValueError, 'physical_backend'):
            infer.build_runtime_config(args)

    def test_rejects_wrong_task_config(self):
        path = Path(infer.__file__).parent / 'configs/torch/editing.json'
        args = self.args('--task', 'understanding', '--image', 'not-loaded.jpg',
                         '--plan-only', '--config', str(path))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            infer.build_runtime_config(args)

    def test_full_cannot_silently_use_compression_config(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args('--backend', 'full', '--config', 'anything.json')

    def test_missing_image_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args('--task', 'editing')

    def test_plan_only_never_loads_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args('--task', 'text_to_image', '--backend', 'engine',
                             '--plan-only', '--output-dir', directory)
            with mock.patch.object(infer, 'parse_args', return_value=args), \
                 mock.patch.object(infer, 'build_model') as loader, \
                 contextlib.redirect_stdout(io.StringIO()):
                infer.main()
                loader.assert_not_called()
            self.assertTrue((Path(directory) / 'unicache_plan.json').exists())
            resolved = json.loads((Path(directory) / 'resolved_config.json').read_text())
            self.assertTrue(resolved['runtime']['physical_backend'])

    def test_presets_keep_runtime_modes_distinct(self):
        root = Path(infer.__file__).parent / 'configs'
        for path in (root / 'engine').glob('*.json'):
            config = json.loads(path.read_text())
            self.assertFalse(any('scheduler' in r for r in config['rules']))
            self.assertTrue(config['runtime']['physical_backend'])
        for path in (root / 'torch').glob('*.json'):
            config = json.loads(path.read_text())
            self.assertTrue(any('scheduler' in r for r in config['rules']))
            self.assertFalse(config['runtime'].get('physical_backend', False))


if __name__ == '__main__':
    unittest.main()
