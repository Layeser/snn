import argparse
from pathlib import Path
from typing import Any, Callable

import yaml

Schema = dict[str, tuple[type, str | None]]


def _check_constraint(value: Any, constraint: str | None, key: str) -> None:
    if constraint == "positive" and value <= 0:
        raise ValueError(f"'{key}' doit être > 0 (reçu: {value})")
    if constraint == "non_negative" and value < 0:
        raise ValueError(f"'{key}' doit être >= 0 (reçu: {value})")


def load_and_validate_config(
    config_path: Path,
    schema: Schema,
    extra_validators: list[Callable[[dict[str, Any]], None]] | None = None,
) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Fichier config introuvable: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Le fichier config est vide: {config_path}")
    if not isinstance(raw, dict):
        raise ValueError(f"Le fichier config doit être un mapping YAML: {config_path}")

    missing = [k for k in schema if k not in raw]
    if missing:
        raise ValueError(f"Clés manquantes dans {config_path}: {missing}")

    unknown = [k for k in raw if k not in schema]
    if unknown:
        raise ValueError(f"Clés inconnues dans {config_path}: {unknown}")

    config: dict[str, Any] = {}
    nullable_keys = {"data_dir", "device"}

    for key, (expected_type, constraint) in schema.items():
        value = raw[key]

        if value is None:
            if key in nullable_keys:
                config[key] = None
                continue
            raise ValueError(f"'{key}' ne peut pas être null dans {config_path}")

        if expected_type is str:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"'{key}' doit être une chaîne non vide (reçu: {value!r})")
            value = value.strip()
        elif expected_type in (int, float):
            if isinstance(value, str):
                try:
                    value = expected_type(value)
                except ValueError as exc:
                    raise ValueError(f"'{key}' doit être un nombre (reçu: {value!r})") from exc
            elif not isinstance(value, expected_type):
                if expected_type is int and isinstance(value, float) and value.is_integer():
                    value = int(value)
                else:
                    raise ValueError(
                        f"'{key}' doit être de type {expected_type.__name__} "
                        f"(reçu: {type(value).__name__}, valeur: {value!r})"
                    )
        elif not isinstance(value, expected_type):
            raise ValueError(
                f"'{key}' doit être de type {expected_type.__name__} "
                f"(reçu: {type(value).__name__}, valeur: {value!r})"
            )

        _check_constraint(value, constraint, key)
        config[key] = value

    for validator in extra_validators or []:
        validator(config)

    return config


def parse_train_args(
    default_config_path: Path,
    schema: Schema,
    build_parser: Callable[[dict[str, Any]], argparse.ArgumentParser],
    extra_validators: list[Callable[[dict[str, Any]], None]] | None = None,
) -> tuple[argparse.Namespace, dict[str, Any]]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=str(default_config_path))
    pre_args, remaining = pre_parser.parse_known_args()

    config_path = Path(pre_args.config)
    config = load_and_validate_config(config_path, schema, extra_validators)

    parser = build_parser(config)
    args = parser.parse_args(remaining)
    return args, config
