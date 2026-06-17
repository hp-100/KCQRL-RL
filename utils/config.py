"""Small configuration loader with optional PyYAML support."""
from pathlib import Path
from typing import Any, Dict, List


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    try:
        import yaml  # type: ignore
    except ImportError:
        return _load_simple_yaml(path)
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    """Parse the simple subset of YAML used by configs/default.yaml."""
    return _load_simple_yaml_with_lists(path)

def _load_simple_yaml_with_lists(path: Path) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    lines = [ln for ln in path.read_text().splitlines() if ln.split("#", 1)[0].strip()]
    containers: List[tuple[int, Any]] = [(-1, root)]
    for i, raw in enumerate(lines):
        line = raw.split("#", 1)[0].rstrip()
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()
        while containers and indent <= containers[-1][0]:
            containers.pop()
        parent = containers[-1][1]
        if text.startswith("- "):
            parent.append(_scalar(text[2:]))
            continue
        key, _, value = text.partition(":")
        key = key.strip()
        if value.strip():
            parent[key] = _scalar(value)
        else:
            # Peek next meaningful line to choose dict vs list.
            next_text = ""
            for nxt in lines[i + 1:]:
                nxt_line = nxt.split("#", 1)[0].rstrip()
                nxt_indent = len(nxt_line) - len(nxt_line.lstrip(" "))
                if nxt_indent > indent:
                    next_text = nxt_line.strip()
                    break
                if nxt_indent <= indent:
                    break
            parent[key] = [] if next_text.startswith("- ") else {}
        containers.append((indent, parent[key]))
    return root
