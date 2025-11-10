#!/usr/bin/env python3
"""Command line entry point for training."""
from __future__ import annotations

import argparse
import os
import sys
import types
from dataclasses import fields
from typing import Any, Union, get_args, get_origin, get_type_hints


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from asv.hyperparams import GlobalHyperParameters, TrainingHyperParameters
from asv.training import train


def _cli_type_from_annotation(annotation: Any, default: Any) -> Any:
    origin = get_origin(annotation)
    union_type = getattr(types, "UnionType", None)
    if origin in (Union, union_type):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            annotation = args[0]
    if annotation in (int, float, str, bool):
        return annotation
    if default is not None:
        return type(default)
    return str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the multi-boat DQN agent.")
    parser.add_argument(
        "--print-hparams",
        action="store_true",
        help="Print the resolved hyperparameters before training.",
    )

    defaults = TrainingHyperParameters()
    hints = get_type_hints(TrainingHyperParameters)

    for field in fields(TrainingHyperParameters):
        name = field.name
        arg_name = name.replace("_", "-")
        default = getattr(defaults, name)
        annotation = hints.get(name, field.type)

        if annotation is bool:
            if default:
                parser.add_argument(f"--no-{arg_name}", dest=name, action="store_false")
            else:
                parser.add_argument(f"--{arg_name}", dest=name, action="store_true")
            parser.set_defaults(**{name: default})
            continue

        origin = get_origin(annotation)
        union_type = getattr(types, "UnionType", None)
        if origin in (Union, union_type):
            args = [a for a in get_args(annotation) if a is not type(None)]
            if len(args) == 1:
                annotation = args[0]

        arg_type = _cli_type_from_annotation(annotation, default)
        parser.add_argument(f"--{arg_name}", dest=name, type=arg_type, default=default)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print_hparams = args.print_hparams
    arg_dict = vars(args)
    arg_dict.pop("print_hparams")

    hp = TrainingHyperParameters(**arg_dict)
    params = GlobalHyperParameters(training=hp)
    params.synchronize()

    if print_hparams:
        for key, value in params.to_flat_dict().items():
            print(f"{key}: {value}")

    train(params)


if __name__ == "__main__":
    main()
