"""Arguments CLI partagés pour l'entraînement."""

from __future__ import annotations

import argparse


def add_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("checkpointing")
    group.add_argument(
        "--resume",
        type=str,
        default="auto",
        choices=("auto", "none"),
        help=(
            "auto: reprend depuis save_dir/last.pt si présent; "
            "none: toujours repartir de zéro (sauf si --resume-path est fourni)"
        ),
    )
    group.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Chemin explicite vers un checkpoint (.pt). Prioritaire sur --resume.",
    )
    group.add_argument(
        "--fresh",
        action="store_true",
        help="Ignorer tout checkpoint existant et repartir de l'epoch 1.",
    )
