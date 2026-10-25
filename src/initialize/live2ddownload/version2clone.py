import json
from collections.abc import Callable
from pathlib import Path


def version2clone(
    base_url: str,
    entry_json: str,
    save_path: Path,
    fetch_fn: Callable[[str, Path], bool],
) -> dict[str, Path]:
    tasks: dict[str, Path] = {}
    base = base_url if base_url.endswith("/") else f"{base_url}/"
    json_path = save_path / entry_json
    fetch_fn(base + entry_json, json_path)

    cfg = json.loads(json_path.read_text(encoding="utf-8"))
    for key in ("model", "physics", "pose"):
        if key in cfg:
            tasks[base + cfg[key]] = save_path / cfg[key]
    for tex in cfg.get("textures", []):
        tasks[base + tex] = save_path / tex
    for exp in cfg.get("expressions", []):
        tasks[base + exp["file"]] = save_path / exp["file"]
    for motion_group in cfg.get("motions", {}).values():
        for m in motion_group:
            tasks[base + m["file"]] = save_path / m["file"]
            if "sound" in m:
                tasks[base + m["sound"]] = save_path / m["sound"]

    return tasks
