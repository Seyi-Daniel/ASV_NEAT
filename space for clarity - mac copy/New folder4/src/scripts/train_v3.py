#!/usr/bin/env python3
from training.hparams import HParams
from training.train import train, _cli_type_from_annotation
import argparse
from typing import get_type_hints

if __name__ == '__main__':
    # Build CLI from HParams annotations
    parser = argparse.ArgumentParser()
    hints = get_type_hints(HParams)

    for name, anno in hints.items():
        default = getattr(HParams, name)

        # bool flags: --flag flips the default
        if isinstance(default, bool):
            parser.add_argument(
                f"--{name}",
                action=("store_false" if default else "store_true"),
                help=f"(default: {default})",
            )
        else:
            parser.add_argument(
                f"--{name}",
                type=_cli_type_from_annotation(anno),
                default=default,
                help=f"(default: {default})",
            )

    args = parser.parse_args()
    hp = HParams(**vars(args))
    train(hp)
