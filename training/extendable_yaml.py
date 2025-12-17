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
import tempfile
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
            raise ValueError(f"Circular dependency detected: {path}")

        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")

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


def resolve_yaml_file(yaml_path: str | Path) -> str:
    """
    Resolve a YAML file that may contain $extends directives.

    If the YAML file contains $extends, this function creates a temporary
    merged YAML file and returns its path. Otherwise, returns the original path.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        Path to the resolved YAML file (either original or temporary merged file)
    """
    yaml_path = Path(yaml_path)

    # Quick check: does the file contain $extends?
    with open(yaml_path) as f:
        first_line = f.readline().strip()
        if not first_line.startswith("$extends:"):
            # No extension, return original path
            return str(yaml_path.resolve())

    # Load and merge configurations
    merged_config = load_yaml_with_extends(yaml_path)

    # Create a temporary YAML file with the merged configuration
    temp_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yaml',
        prefix='merged_config_',
        delete=False
    )

    try:
        yaml.dump(merged_config, temp_file, default_flow_style=False, sort_keys=False)
        temp_file.flush()
        return temp_file.name
    finally:
        temp_file.close()
