"""
YAML configs with `$extends: ./other.yaml` (relative to the file): the extended config is deep-merged under this one.
"""
from pathlib import Path

from omegaconf import OmegaConf

CONFIG_FILE_NAME = "welt.yaml"  # The training config, saved with runs and exports


def load_yaml_with_extends(path: str | Path):
    path = Path(path).resolve()
    config = OmegaConf.load(path)
    parent = config.pop("$extends", None)
    return config if parent is None else OmegaConf.merge(load_yaml_with_extends(path.parent / parent), config)


def load_yaml(path: str, overrides: list[str] = ()) -> dict:
    """Load a YAML config (supporting `$extends`), applying `section.key=value` overrides (YAML values)."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected an override as section.key=value (a YAML value), got {override!r}")
    config = OmegaConf.merge(load_yaml_with_extends(path), OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_container(config)
