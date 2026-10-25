# 项目交接与架构说明手册 (HANDOVER.md)

> **项目名称**：高拟人「白月光」超写实数字人伴侣系统
> **所属课程/背景**：浙江大学 · 创新管理（大五/学期项目）
> **本地硬件基准**：NVIDIA GeForce RTX 4060 Laptop GPU（8GB 显存）/ Windows 11
> **最后更新**：2026-10-24
> **面向对象**：后续接手的 AI 助手 / 团队协作者
>
> 本文件描述磁盘上的真实状态，不以更早的计划为准。

---

## 一、现在到底有什么

主项目还在，数字人子目录已经删掉。

| 部分 | 状态 |
|---|---|
| 主环境 `.venv`（Python 3.11.9，uv 0.12.19） | 在。PyTorch 2.6.0+cu124、LLaMA-Factory |
| 基座 `Qwen2.5-1.5B-Instruct` | 在。`.model/.../snapshots/master/model.safetensors`，约 3.08GB |
| LoRA `saves/persona_lora` | 在。已训练 2 个 epoch，约 28 秒。平均 loss **3.77**，最后几步约 3.0–3.4。人设偏淡，没有学傻 |
| 数据 `.data/persona_toy.json` | 在。ShareGPT，字段 `conversations` |
| 终端聊天 `src/modelchat/chat.py` | 能跑，但和下面的人设规则不一致 |
| 底图 `white_moonlight.jpg` | 在 |
| `validate_data.py` | **不存在** |
| `Linly-Talker/` | **已从磁盘删除**，也没有 `.gitmodules`。不要再 `cd` 进去 |

数字人曾经离线跑通过一次：Edge-TTS 出声，SadTalker 256 出嘴型，再用环境里自带的 ffmpeg 把无声 mp4 和 wav 合成有声视频。网页按钮自己合成失败，因为系统 PATH 里没有 `ffmpeg`。对话大脑没有接上，页面用的是「直接回复」，输入什么就念什么。效果不好，所以整个 `Linly-Talker` 目录被删掉。权重、它的 `.venv`、`results` 都随之删除，不必找。

---

## 二、目录（以当前磁盘为准）

```text
D:\Zhejiang_University\term_5\创新管理\project\
├── .venv/                  # 主环境 Python 3.11，uv 管理
├── pyproject.toml          # torch / torchvision / torchaudio 锁定 pytorch-cu124 源
├── uv.lock
├── .python-version         # 3.11
├── .model/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master/
├── .data/
│   ├── dataset_info.json   # 数据集名 persona_toy
│   └── persona_toy.json
├── .config/
│   ├── train.yaml          # LoRA：rank 16，alpha 32，dropout 0.1，lr 1e-4，epoch 2，bf16
│   └── api.yaml            # 基座 + saves/persona_lora，template qwen。默认端口 8000
├── saves/persona_lora/     # 正在使用的适配器在这一层，不要去读 checkpoint-100 / 125
├── src/modelchat/chat.py
└── white_moonlight.jpg
```

`saves/persona_lora/checkpoint-100` 和 `checkpoint-125` 的适配器约 148MB，是更早一次训练留下的。当前 `adapter_model.safetensors` 约 74MB，和 `checkpoint-50` 一致。`chat.py` 和 `api.yaml` 读的是目录根上的适配器。

主环境用 `uv run`。不要在这个环境里装 Linly-Talker 的依赖。

---

## 三、人设规则（仍然有效）

目标：反讨好、嘴碎、短平快，但必须留互动钩子。不要写成「别烦我」「你快走」。

已经踩过的坑：

1. **客服腔**。没有 system 消息时，Qwen 模板会注入 `You are a helpful assistant`。推理代码必须自己写 `SYSTEM_PROMPT`。`chat.py` 目前没写。
2. **过拟合**。100 条短样本跑到 loss < 0.5 会复读。合适区间大约是 loss 1.2–1.8。当前 3.77 是欠拟合，不是学傻。试聊时会立场翻转，例如先说不去吃火锅，又说「别磨叽」，再说「当然吃了」。
3. **上下文雪球**。历史里骂人的话会把后面的回答带偏。目标参数是 `no_repeat_ngram_size=3`、`repetition_penalty=1.15`、只留最近 4 轮。`chat.py` 只做了 `repetition_penalty=1.15`，历史留的是最近 10 条消息，没有 ngram 限制。
4. **退出方式**。代码认 `q` / `exit` / `quit`。界面上的 `INPUT(Priny C to quit)` 是错的，也没有 `/clear`。

---

## 四、主环境命令

在项目根目录执行：

```powershell
uv run python src/modelchat/chat.py
uv run llamafactory-cli train .config/train.yaml
uv run llamafactory-cli api .config/api.yaml
uv sync
```

`uv sync` 只恢复主环境。它不会恢复已经删除的数字人环境。

8GB 显存不要同时开训练和 API。1.5B 的 bf16 权重约 3GB。

---

## 五、数字人实验结论（不要重蹈）

如果有人把 `Linly-Talker` 重新克隆回来，不要执行 `pip install -r requirements_webui.txt`。那份文件锁的是 Gradio 4 的声明，但同时会把 PyTorch 拉向 CUDA 11.8、NumPy 1.23，把现在这套 cu124 冲掉。`ffmpeg` 也不在那份依赖里。

上次能跑起来的做法是单独建 Python 3.10 环境，保留 PyTorch 2.6+cu124，再把冲突包钉死：

- `gradio==4.44.1`（6.30 对不上旧参数名 `root`、`show_copy_button`）
- `huggingface-hub==0.25.2`（2.x 删了 Gradio 4 需要的 `HfFolder`）
- `transformers==4.44.2`
- `pydantic==2.10.6`
- `fastapi==0.115.6`
- 另装 `g4f`、`scikit-image`、`yacs`、`kornia`、`facexlib`。只为启动和 SadTalker，不是全功能

还改过这些 NumPy 2 不兼容：

- `src/face3d/util/preprocess.py`：`np.VisibleDeprecationWarning` 改为 `DeprecationWarning`
- 同文件 `align_img`：`np.array([w0, h0, s, t[0], t[1]])` 改为对每一项 `float(...)`
- `facexlib` 包内 `alignment/awing_arch.py`：`np.float` 改为 `np.float64`
- `TFG/__init__.py`：注释掉 `Wav2Lipv2`，否则缺 `ultralytics` 会导致整个 `talker` 创建失败

网页在 `http://127.0.0.1:6006`（`configs.py` 的 `port`，不是 7860）。声音用 Edge-TTS `zh-CN-XiaoxiaoNeural`。分辨率用 256，不要开 GFPGAN，不要开 MuseTalk。4060 8GB 跑 SadTalker 256 够用；笔记本报出来的显存常略小于 8，代码里的警告可以忽略。

系统 PATH 没有 `ffmpeg` 时，网页能写出无声的 `temp_*.mp4` 和 `answer.wav`，最后合成会失败。上次用的是环境里的：

`Linly-Talker\.venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe`

这个文件已经随目录删除。

本地大脑也没有接上。Linly 的界面没有「填写 OpenAI 地址」的输入框。`LLM/ChatGPT.py` 用的是旧版 `openai.ChatCompletion`，默认连 OpenAI。下拉里的 Qwen 会再加载一个模型，和 SadTalker 抢显存。

---

## 六、接手后先做什么

1. 先跑 `uv run python src/modelchat/chat.py`，确认主环境还在。退出用 `q`。
2. 若要继续做人设，先改 `chat.py` 的 `SYSTEM_PROMPT` 和上下文窗口，不要先重训。当前 loss 3.77，重训前要加数据，不要把 epoch 堆到 loss < 0.5。
3. 队友数据来了再补质检脚本。仓库里没有 `validate_data.py`。多轮对话保持六成以上。
4. 数字人不要按本文件旧版去填 `http://127.0.0.1:8000/v1`。要重做就先装系统 `ffmpeg`，再按第五节的版本钉死重装，最后改 `ChatGPT.py` 指向本机 API。不建议现在重做。
