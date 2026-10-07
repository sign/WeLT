"""
YAML file extension utility that supports $extends directive for configuration inheritance.

This module allows YAML configuration files to extend from other YAML files using
the $extends directive, enabling configuration reuse and minimal config specifications.

Example:
    base.yaml:
        model: gpt-3
        temperature: 0.7
        max_tokens: 100

    extended.yaml:
        $extends: ./base.yaml
        temperature: 0.9  # Override temperature
        # model and max_tokens are inherited
"""

import os
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two dictionaries, with override values taking precedence.

    Args:
        base: The base dictionary
        override: The dictionary with override values

    Returns:
        A new merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Override the value
            result[key] = value

    return result


def load_yaml_with_extends(yaml_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file and recursively resolve $extends directives.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        The merged configuration dictionary

    Raises:
        FileNotFoundError: If the YAML file or any extended file is not found
        ValueError: If a circular dependency is detected
    """
    yaml_path = Path(yaml_path).resolve()

    def _load_recursive(path: Path, visited: set[Path]) -> dict[str, Any]:
        """Recursively load and merge YAML files."""
        if path in visited:
            msg = f"Circular dependency detected: {path}"
            raise ValueError(msg)

        if not path.exists():
            msg = f"YAML file not found: {path}"
            raise FileNotFoundError(msg)

        visited.add(path)

        with open(path) as f:
            config = yaml.safe_load(f) or {}

        # Check if this file extends another
        if "$extends" in config:
            extends_path = config.pop("$extends")

            # Resolve the parent path relative to the current file
            if not os.path.isabs(extends_path):
                extends_path = (path.parent / extends_path).resolve()
            else:
                extends_path = Path(extends_path).resolve()

            # Load the parent configuration
            parent_config = _load_recursive(extends_path, visited.copy())

            # Merge: parent as base, current config overrides
            config = deep_merge(parent_config, config)

        return config

    return _load_recursive(yaml_path, set())


CONFIG_FILE_NAME = "welt.yaml"  # The training config, saved with runs and exports


def load_yaml(path: str, overrides: list[str] = ()) -> dict:
    """Load a YAML config (supporting `$extends`), applying `section.key=value` overrides."""
    config = load_yaml_with_extends(path)
    for override in overrides:
        key, value = override.split("=", 1)
        *sections, name = key.split(".")
        target = config
        for section in sections:
            target = target.setdefault(section, {})
        target[name] = yaml.safe_load(value)
    return config
