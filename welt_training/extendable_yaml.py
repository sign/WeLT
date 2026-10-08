"""
YAML configs with `$extends: ./other.yaml` (relative to the file): the extended config is deep-merged under this one.
"""
import re
from pathlib import Path

import yaml

CONFIG_FILE_NAME = "welt.yaml"  # The training config, saved with runs and exports


class Loader(yaml.SafeLoader):
    """Reads exponents without a dot (1e-4) as floats, as YAML 1.2 does (PyYAML's YAML 1.1 reads them as strings)."""


Loader.add_implicit_resolver("tag:yaml.org,2002:float", re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+$"),
                             list("-+0123456789."))


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        both_dicts = isinstance(result.get(key), dict) and isinstance(value, dict)
        result[key] = deep_merge(result[key], value) if both_dicts else value
    return result


def load_yaml_with_extends(path: str | Path, visited: tuple = ()) -> dict:
    path = Path(path).resolve()
    if path in visited:
        raise ValueError(f"Circular $extends: {path}")
    config = yaml.load(path.read_text(), Loader) or {}
    if "$extends" in config:
        parent = load_yaml_with_extends(path.parent / config.pop("$extends"), (*visited, path))
        config = deep_merge(parent, config)
    return config


def load_yaml(path: str, overrides: list[str] = ()) -> dict:
    """Load a YAML config (supporting `$extends`), applying `section.key=value` overrides."""
    config = load_yaml_with_extends(path)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected an override as section.key=value (a YAML value), got {override!r}")
        key, value = override.split("=", 1)
        *sections, name = key.split(".")
        target = config
        for section in sections:
            target = target.setdefault(section, {})
        target[name] = yaml.load(value, Loader)
    return config
