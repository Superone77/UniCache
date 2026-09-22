"""Download only the files required for BAGEL inference."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('checkpoints/BAGEL-7B-MoT'))
    parser.add_argument('--revision', default='main', help='Use a commit ID for a reproducible checkpoint download.')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id='ByteDance-Seed/BAGEL-7B-MoT', revision=args.revision,
                      local_dir=str(args.output_dir),
                      allow_patterns=['*.json', '*.txt', '*.model', 'ema.safetensors', 'ae.safetensors'])


if __name__ == '__main__':
    main()
