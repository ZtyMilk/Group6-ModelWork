import asyncio
import base64
import json
import mimetypes
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import edge_tts
import torch
import transformers
from peft import PeftModel
from PyQt6.QtCore import (
    QBuffer,
    QByteArray,
    QIODevice,
    QObject,
    QPoint,
    Qt,
    QThread,
    QUrl,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QColor, QMouseEvent
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import (
    QWebEngineProfile,
    QWebEngineSettings,
    QWebEngineUrlRequestJob,
    QWebEngineUrlScheme,
    QWebEngineUrlSchemeHandler,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = str(
    PROJECT_ROOT / ".model/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master"
)
ADAPTER_PATH = str(PROJECT_ROOT / "saves/persona_lora")
WEB_DIR = (PROJECT_ROOT / "web").resolve()


def _register_pet_scheme() -> None:
    """注册纯内存自定义协议 petapp://，无需占用任何端口即可加载本地 Live2D 模型"""
    scheme = QWebEngineUrlScheme(b"petapp")
    scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
    scheme.setFlags(
        QWebEngineUrlScheme.Flag.SecureScheme
        | QWebEngineUrlScheme.Flag.LocalScheme
        | QWebEngineUrlScheme.Flag.LocalAccessAllowed
        | QWebEngineUrlScheme.Flag.CorsEnabled
        | QWebEngineUrlScheme.Flag.FetchApiAllowed
    )
    QWebEngineUrlScheme.registerScheme(scheme)


class LocalAssetHandler(QWebEngineUrlSchemeHandler):
    """直接从本地 web/ 目录流式读取前端与 Live2D 模型资源，彻底免除网络依赖"""

    def requestStarted(self, request: QWebEngineUrlRequestJob | None) -> None:
        if request is None:
            return
        rel = request.requestUrl().path().lstrip("/")
        target = (WEB_DIR / rel).resolve()
        if not str(target).startswith(str(WEB_DIR)) or not target.is_file():
            request.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return
        mime, _ = mimetypes.guess_type(str(target))
        if target.suffix == ".js":
            mime = "application/javascript"
        elif target.suffix == ".json":
            mime = "application/json"
        data = QByteArray(target.read_bytes())
        buf = QBuffer(request)
        buf.setData(data)
        buf.open(QIODevice.OpenModeFlag.ReadOnly)
        request.reply((mime or "application/octet-stream").encode("ascii"), buf)


SYSTEM_PROMPT = "你叫KIKI\n"


def parse_emotion(text: str) -> str:
    if any(k in text for k in ["哼", "才不", "笨", "烦", "懒得"]):
        return "傲娇微嗔"
    if any(k in text for k in ["笑", "好呀", "有意思", "听你的"]):
        return "暗喜愉悦"
    if any(k in text for k in ["别", "小心", "怎么了", "吃药", "好好"]):
        return "外冷内热"
    return "托腮注视"


class ModelLoaderWorker(QThread):
    """后台异步加载本地 Qwen2.5 + LoRA 模型，避免阻塞桌宠窗口弹出"""

    loaded = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            tf_mod: Any = transformers
            tokenizer = tf_mod.AutoTokenizer.from_pretrained(
                MODEL_PATH, trust_remote_code=True
            )
            base_model = tf_mod.AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            model.eval()
            self.loaded.emit(tokenizer, model)
        except (OSError, RuntimeError, ValueError) as exc:
            self.failed.emit(str(exc))


class ChatWorker(QThread):
    """后台执行模型推理与 Edge-TTS 语音合成，保持 Live2D 画面满帧流畅"""

    done = pyqtSignal(str, str)

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        messages: list[dict[str, str]],
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.messages = messages

    @staticmethod
    async def _tts_base64(text: str) -> str:
        comm = edge_tts.Communicate(
            text, "zh-CN-XiaoyiNeural", rate="+10%", pitch="+5Hz"
        )
        audio_bytes = bytearray()
        async for chunk in comm.stream():
            if chunk.get("type") == "audio" and "data" in chunk:
                audio_bytes.extend(chunk["data"])
        return base64.b64encode(bytes(audio_bytes)).decode("utf-8")

    def run(self) -> None:
        text_input = self.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=120,
                temperature=0.85,
                top_p=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                do_sample=True,
            )

        response_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        reply: str = self.tokenizer.decode(
            response_ids, skip_special_tokens=True
        ).strip()
        reply = re.sub(r"^[对方晚晚苏晚AI助手]*[：:]\s*", "", reply)

        audio_b64 = ""
        try:
            audio_b64 = asyncio.run(self._tts_base64(reply))
        except (OSError, RuntimeError, ValueError):
            audio_b64 = ""

        self.done.emit(reply, audio_b64)


class PetBridge(QObject):
    replyReady = pyqtSignal(str)
    statusChanged = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer: Any = None
        self.model: Any = None
        self.ready = False
        self.status_text = "正在唤醒KIKI……"
        self.chat_history: list[dict[str, str]] = []
        self.intimacy_score = 60
        self.worker: ChatWorker | None = None

        self.loader = ModelLoaderWorker()
        self.loader.loaded.connect(self._on_model_loaded)
        self.loader.failed.connect(self._on_model_failed)
        self.loader.start()

    def _emit_status(self) -> None:
        payload = json.dumps(
            {"ready": self.ready, "text": self.status_text}, ensure_ascii=False
        )
        self.statusChanged.emit(payload)

    def _on_model_loaded(self, tokenizer: Any, model: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.ready = True
        self.status_text = "嗯..."
        self._emit_status()

    def _on_model_failed(self, err: str) -> None:
        self.ready = False
        self.status_text = f"模型加载失败：{err}"
        self._emit_status()

    @pyqtSlot()
    def notifyReady(self) -> None:
        self._emit_status()

    @pyqtSlot(str)
    def sendMessage(self, user_msg: str) -> None:
        msg = user_msg.strip()
        if not msg or not self.ready:
            return

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.chat_history[-8:])
        messages.append({"role": "user", "content": msg})

        self.worker = ChatWorker(self.tokenizer, self.model, messages)
        self.worker.done.connect(lambda reply, b64: self._on_reply(msg, reply, b64))
        self.worker.start()

    def _on_reply(self, user_msg: str, reply: str, audio_b64: str) -> None:
        self.chat_history.append({"role": "user", "content": user_msg})
        self.chat_history.append({"role": "assistant", "content": reply})
        self.intimacy_score = min(100, self.intimacy_score + 2)
        payload = json.dumps(
            {
                "reply": reply,
                "audio": audio_b64,
                "emotion": parse_emotion(reply),
                "intimacy": self.intimacy_score,
            },
            ensure_ascii=False,
        )
        self.replyReady.emit(payload)


class DragBar(QWidget):
    """顶部迷你半透明拖动与控制栏"""

    def __init__(self, parent_win: QMainWindow, on_switch: Callable[[], None]):
        super().__init__()
        self.win = parent_win
        self.drag_pos = QPoint()
        self.setFixedHeight(28)
        self.setStyleSheet(
            """
            background: rgba(20, 22, 35, 175);
            border: 1px solid rgba(255, 255, 255, 35);
            border-radius: 12px;
            color: #cbd5e1;
            font-size: 12px;
            """
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 8, 0)
        layout.setSpacing(6)

        title = QLabel("⠿ 按住拖动")
        title.setStyleSheet("background: transparent; border: none;")

        switch_btn = QPushButton("换装")
        switch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        switch_btn.setFixedHeight(20)
        switch_btn.setStyleSheet(
            """
            QPushButton {
                background: rgba(255, 133, 162, 60);
                color: #ffd1dc;
                border: none;
                border-radius: 8px;
                padding: 0 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: rgba(255, 133, 162, 120);
            }
            """
        )
        switch_btn.clicked.connect(on_switch)

        close_btn = QPushButton("✕")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet(
            "background: transparent; color: #ff85a2; border: none; font-weight: bold;"
        )
        close_btn.clicked.connect(self.win.close)

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(switch_btn)
        layout.addWidget(close_btn)

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is not None and a0.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = (
                a0.globalPosition().toPoint() - self.win.frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:
        if a0 is not None and (a0.buttons() & Qt.MouseButton.LeftButton):
            self.win.move(a0.globalPosition().toPoint() - self.drag_pos)


class DesktopPet(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("KIKI")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.resize(360, 540)

        primary_screen = QApplication.primaryScreen()
        if primary_screen is not None:
            screen = primary_screen.availableGeometry()
            self.move(screen.width() - 390, screen.height() - 560)

        central = QWidget()
        central.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        central.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.drag_bar = DragBar(self, self.switch_avatar)
        self.webview = QWebEngineView()
        self.webview.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.webview.setStyleSheet("background: transparent;")

        self.bridge = PetBridge()
        self.channel = QWebChannel(self.webview)
        self.channel.registerObject("bridge", self.bridge)

        page = self.webview.page()
        if page is not None:
            page.setBackgroundColor(QColor(0, 0, 0, 0))
            page.setWebChannel(self.channel)

        settings = self.webview.settings()
        if settings is not None:
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False
            )

        self.webview.load(QUrl("petapp://local/pet.html"))

        layout.addWidget(self.drag_bar)
        layout.addWidget(self.webview)
        self.setCentralWidget(central)

    def switch_avatar(self) -> None:
        page = self.webview.page()
        if page is not None:
            page.runJavaScript("window.switchAvatar && window.switchAvatar();")


if __name__ == "__main__":
    _register_pet_scheme()
    app = QApplication(sys.argv)
    asset_handler = LocalAssetHandler(app)
    profile = QWebEngineProfile.defaultProfile()
    if profile is not None:
        profile.installUrlSchemeHandler(b"petapp", asset_handler)
    pet = DesktopPet()
    pet.show()
    sys.exit(app.exec())
