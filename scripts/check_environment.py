"""Check inference dependencies without loading weights or running a model."""
import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['torch', 'engine', 'full'], default='torch')
    parser.add_argument('--task', choices=['understanding', 'text_to_image', 'editing'], default='editing')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    packages = ['torch', 'torchvision', 'transformers', 'accelerate', 'triton', 'flash-attn']
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    print(json.dumps({'python': sys.version.split()[0], 'versions': versions,
                      'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available()}, indent=2))
    for name in ['model_loader', 'inferencer', 'unicache']:
        importlib.import_module(name)
    if args.backend == 'engine':
        if not torch.cuda.is_available():
            raise SystemExit('Engine requires CUDA; dependency check failed (no model was loaded).')
        importlib.import_module('flash_attn')
        importlib.import_module('triton')
        if args.task == 'editing':
            from unicache.storage.kivi_cuda import _load_official_kernel, _load_official_packer
            _load_official_kernel()
            _load_official_packer()
    print('Dependency imports passed; this is not an end-to-end inference test.')


if __name__ == '__main__':
    main()
