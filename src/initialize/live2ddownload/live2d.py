import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .version2clone import version2clone
from .version3clone import version3clone


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
        # Cubism3/4
        v3_models = [
            # Hiyori 桃濑日和：官方超清 JK 校服邻家少女（白月光首选）
            (
                "hiyori",
                "https://cdn.jsdelivr.net/gh/Live2D/CubismWebSamples@master/Samples/Resources/Hiyori/",
                "Hiyori.model3.json",
            ),
            # Haru 春：2K 长发温柔学姐
            (
                "haru",
                "https://cdn.jsdelivr.net/gh/guansss/pixi-live2d-display/test/assets/haru/",
                "haru_greeter_t03.model3.json",
            ),
            # Mao 虹色魔法少女：官方 4.0 旗舰高表现力少女
            (
                "mao",
                "https://cdn.jsdelivr.net/gh/Live2D/CubismWebSamples@master/Samples/Resources/Mao/",
                "Mao.model3.json",
            ),
        ]
        for folder, base_url, entry_json in v3_models:
            tasks.update(
                version3clone(
                    base_url=base_url,
                    entry_json=entry_json,
                    save_path=self.models_path / folder,
                    fetch_fn=self._fetch_file,
                )
            )
        # Cubism2
        v2_models = [
            # Shizuku 雫：经典水手服课桌同桌少女
            (
                "shizuku",
                "https://cdn.jsdelivr.net/gh/guansss/pixi-live2d-display/test/assets/shizuku/",
                "shizuku.model.json",
            ),
            # Hibiki 音无响：日系高校少女
            (
                "hibiki",
                "https://cdn.jsdelivr.net/npm/live2d-widget-model-hibiki@1.0.5/assets/",
                "hibiki.model.json",
            ),
            # Koharu 小春：Q版萌系双马尾桌宠
            (
                "koharu",
                "https://cdn.jsdelivr.net/npm/live2d-widget-model-koharu@1.0.5/assets/",
                "koharu.model.json",
            ),
            # Miku 初音未来：经典双马尾歌姬
            (
                "miku",
                "https://cdn.jsdelivr.net/npm/live2d-widget-model-miku@1.0.5/assets/",
                "miku.model.json",
            ),
        ]
        for folder, base_url, entry_json in v2_models:
            tasks.update(
                version2clone(
                    base_url=base_url,
                    entry_json=entry_json,
                    save_path=self.models_path / folder,
                    fetch_fn=self._fetch_file,
                )
            )

        return tasks
