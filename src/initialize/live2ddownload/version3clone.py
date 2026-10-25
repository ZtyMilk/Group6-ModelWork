import json
from collections.abc import Callable
from pathlib import Path


def version3clone(
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
    refs = cfg["FileReferences"]

    if "DisplayInfo" in refs:
        cdi_rel = refs["DisplayInfo"]
        cdi_dest = save_path / cdi_rel
        fetch_fn(base + cdi_rel, cdi_dest)
        if not cdi_dest.exists():
            refs.pop("DisplayInfo", None)
            json_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    for key in ("Moc", "Physics", "Pose", "UserData"):
        if key in refs:
            tasks[base + refs[key]] = save_path / refs[key]
    for tex in refs.get("Textures", []):
        tasks[base + tex] = save_path / tex
    for exp in refs.get("Expressions", []):
        tasks[base + exp["File"]] = save_path / exp["File"]
    for motion_group in refs.get("Motions", {}).values():
        for m in motion_group:
            tasks[base + m["File"]] = save_path / m["File"]
            if "Sound" in m and not str(m["Sound"]).startswith(".."):
                tasks[base + m["Sound"]] = save_path / m["Sound"]

    return tasks
