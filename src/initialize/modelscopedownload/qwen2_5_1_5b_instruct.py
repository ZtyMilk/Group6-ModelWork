from pathlib import Path

from modelscope import snapshot_download


def qwen2_5_1_5b_instruct(direction: str) -> None:
    path = Path(direction)
    target = (
        path / "models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master/model.safetensors"
    )
    if target.exists():
        print("Qwen2.5-1.5B-Instruct has existed.")
        return
    print("\nDownloading Qwen2.5-1.5B-Instruct ...")
    saved_dir = snapshot_download(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        cache_dir=str(path),
    )
    print(f"Finished: {saved_dir}")
