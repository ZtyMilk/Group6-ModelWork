需要有uv，拉取后：

1. cd入项目文件夹。终端运行uv sync
2. 运行uv run python src/initializer.py
3. 运行uv run llamafactory-cli train .config/train.yaml
4. 运行uv run python src/server/pet.py
