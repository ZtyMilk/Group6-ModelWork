import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


class live2ddownload:
    def __init__(self, direction: str) -> None:
        path = Path(direction)
        self.models_path = path / "models"
        self.vendor_path = path / "vendor"

    def downloader(self) -> None:
        all_tasks = {
            **self._collect_vendor_tasks(),
            **self._collect_model_tasks(),
        }
        missing = {
            url: dest
            for url, dest in all_tasks.items()
            if not (dest.exists() and dest.stat().st_size > 0)
        }

        if not missing:
            print("Live2D assets have existed.")
            return

        print(f"\nDownloading Live2D assets ({len(missing)} files) ...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(self._fetch_file, url, dest)
                for url, dest in missing.items()
            ]
        for fut in as_completed(futures):
            fut.result()

        print(f"Finished: {self.models_path}")
        return

    def _fetch_file(self, url: str, dest: Path, retries: int = 5) -> bool:
        if dest.exists() and dest.stat().st_size > 0:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        last_err: Exception | None = None
        for _ in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = resp.read()
                    dest.write_bytes(data)
                    return True
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return False
                last_err = exc
            except OSError as exc:
                last_err = exc

        raise RuntimeError(f"Failed to download {url}: {last_err}")

    def _collect_vendor_tasks(self) -> dict[str, Path]:
        return {
            "https://cdn.jsdelivr.net/gh/dylanNew/live2d/webgl/Live2D/lib/live2d.min.js": self.vendor_path
            / "live2d.min.js",
            "https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js": self.vendor_path
            / "live2dcubismcore.min.js",
            "https://cdn.jsdelivr.net/npm/pixi.js@6.5.10/dist/browser/pixi.min.js": self.vendor_path
            / "pixi.min.js",
            "https://cdn.jsdelivr.net/npm/pixi-live2d-display@0.4.0/dist/index.min.js": self.vendor_path
            / "pixi-live2d-display.min.js",
        }

    def _collect_model_tasks(self) -> dict[str, Path]:
        tasks: dict[str, Path] = {}

        # Haru (Cubism 4)
        haru_base = (
            "https://cdn.jsdelivr.net/gh/guansss/pixi-live2d-display/test/assets/haru/"
        )
        haru_dir = self.models_path / "haru"
        haru_json_path = haru_dir / "haru_greeter_t03.model3.json"
        self._fetch_file(haru_base + "haru_greeter_t03.model3.json", haru_json_path)

        haru_cfg = json.loads(haru_json_path.read_text(encoding="utf-8"))
        refs = haru_cfg["FileReferences"]
        if "DisplayInfo" in refs:
            refs.pop("DisplayInfo", None)
            haru_json_path.write_text(json.dumps(haru_cfg, indent=2), encoding="utf-8")
        for rel in [refs["Moc"], refs["Physics"], refs["Pose"], *refs["Textures"]]:
            tasks[haru_base + rel] = haru_dir / rel
        for exp in refs.get("Expressions", []):
            tasks[haru_base + exp["File"]] = haru_dir / exp["File"]
        for motion_group in refs.get("Motions", {}).values():
            for m in motion_group:
                tasks[haru_base + m["File"]] = haru_dir / m["File"]

        # Shizuku (Cubism 2)
        shz_base = "https://cdn.jsdelivr.net/gh/guansss/pixi-live2d-display/test/assets/shizuku/"
        shz_dir = self.models_path / "shizuku"
        shz_json_path = shz_dir / "shizuku.model.json"
        self._fetch_file(shz_base + "shizuku.model.json", shz_json_path)
        shz_cfg = json.loads(shz_json_path.read_text(encoding="utf-8"))
        for rel in [
            shz_cfg["model"],
            shz_cfg["physics"],
            shz_cfg["pose"],
            *shz_cfg["textures"],
        ]:
            tasks[shz_base + rel] = shz_dir / rel
        for exp in shz_cfg.get("expressions", []):
            tasks[shz_base + exp["file"]] = shz_dir / exp["file"]
        for motion_group in shz_cfg.get("motions", {}).values():
            for m in motion_group:
                tasks[shz_base + m["file"]] = shz_dir / m["file"]
                if "sound" in m:
                    tasks[shz_base + m["sound"]] = shz_dir / m["sound"]
        return tasks
