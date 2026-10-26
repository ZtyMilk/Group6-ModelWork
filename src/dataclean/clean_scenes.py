import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / ".data"
RAW_DIR = DATA_DIR / "raw_data"
SCENES_DIR = DATA_DIR / "scenes"
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"

KIKI_PERSONA = (
    "你叫KIKI，是用户的浙大同窗好友兼日常搭子。"
    "说话自然接地气、嘴碎爱损人但关键时刻稳得住、护短靠谱，"
    "绝不用客服腔或说教腔，每次用两三句短话接住情绪并给出实在主意。"
)


def scene_system(scene: str) -> str:
    return f"当前场景：{scene}。{KIKI_PERSONA}只回应这个场景，不要把别的场景里的说法拿过来用。"


SCENE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("小组作业划水", ("小组", "组员", "划水", "分工", "pre", "答辩")),
    ("学习搭子互相监督", ("论文", "ddl", "基图", "文献", "自习")),
    ("考试周刷夜", ("考试", "期末", "背书", "复习", "挂科")),
    ("早八起不来", ("早八", "闹钟", "起床", "赖床")),
    ("选课与绩点焦虑", ("选课", "绩点", "保研", "高数")),
    ("吐槽早操与食堂", ("食堂", "校园跑", "体测", "乐跑")),
    ("宿舍吃瓜与社团吐槽", ("宿舍", "室友", "社团", "八卦")),
    ("Crush不回消息求支招", ("crush", "暗恋", "已读不回", "表白")),
    ("实习兼职与秋招", ("实习", "面试", "简历", "时薪", "老板")),
    ("深夜刷短视频停不下来", ("短视频", "刷手机", "失眠")),
    ("周五火锅局改烧烤", ("火锅", "烧烤", "约饭")),
    ("基图出来买奶茶", ("奶茶",)),
    ("下雨天点外卖与漏餐", ("外卖", "漏餐")),
    ("月底吃土算预算", ("余额", "预算", "没钱")),
    ("快递开箱翻车与退货", ("快递", "退货", "开箱")),
    ("周末宿舍找乐子", ("无聊", "周末", "电玩")),
    ("情绪低落与在场陪伴", ("好累", "累死", "不想动", "破防", "emo")),
)


REPLY_NUDGE = "先接住对方刚才那句话，再补一句具体的，别只回四个字，也别复读对方的话。"


def scene_system_for_text(user_text: str) -> str:
    lowered = user_text.lower()
    for scene, keywords in SCENE_KEYWORDS:
        if any(word in lowered for word in keywords):
            base = scene_system(scene)
        else:
            base = KIKI_PERSONA
    return f"{base}{REPLY_NUDGE}"


def _normalize_kiki(text: str) -> str:
    return text.replace("小满", "KIKI")


def _strip_ua(line: str, prefix: str) -> str:
    if not line.startswith(prefix):
        raise ValueError(f"Expected prefix {prefix!r}, got: {line[:40]!r}")
    body = line[len(prefix) :].strip()
    if (body.startswith("（") and body.endswith("）")) or (
        body.startswith("(") and body.endswith(")")
    ):
        body = body[1:-1].strip()
    body = _normalize_kiki(body)
    if not body:
        raise ValueError(f"Empty message content in line: {line!r}")
    return body


def _pairs_from_lines(lines: list[str]) -> list[tuple[str, str]]:
    ua_lines = [ln for ln in lines if ln.startswith("U=") or ln.startswith("A=")]
    if len(ua_lines) % 2 != 0:
        raise ValueError(f"Odd number of U=/A= lines: {len(ua_lines)}")
    pairs: list[tuple[str, str]] = []
    for idx in range(0, len(ua_lines), 2):
        u = _strip_ua(ua_lines[idx], "U=")
        a = _strip_ua(ua_lines[idx + 1], "A=")
        pairs.append((u, a))
    return pairs


def _build_dialogues(
    pairs: list[tuple[str, str]],
    slices: list[tuple[int, int]],
    scene: str,
) -> list[dict[str, Any]]:
    covered = sum(end - start for start, end in slices)
    if covered != len(pairs):
        raise ValueError(
            f"Slice total {covered} != pair count {len(pairs)} for scene {scene}"
        )
    items: list[dict[str, Any]] = []
    for start, end in slices:
        convs: list[dict[str, str]] = []
        for u, a in pairs[start:end]:
            convs.append({"from": "human", "value": u})
            convs.append({"from": "gpt", "value": a})
        items.append(
            {
                "scene": scene,
                "system": scene_system(scene),
                "conversations": convs,
            }
        )
    return items


def _clean_single_narrative(
    raw_name: str,
    scene_label: str,
    slices: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    lines = [
        ln.strip()
        for ln in (RAW_DIR / raw_name).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    pairs = _pairs_from_lines(lines)
    return _build_dialogues(pairs, slices, scene_label)


def _clean_emotional_partner() -> list[dict[str, Any]]:
    lines = [
        ln.strip()
        for ln in (RAW_DIR / "emotional_partner.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    pairs = _pairs_from_lines(lines)
    part1 = _build_dialogues(
        pairs[:13], [(0, 5), (5, 9), (9, 13)], "情绪低落与在场陪伴"
    )
    part2 = _build_dialogues(
        pairs[13:], [(0, 5), (5, 10)], "情绪恢复与重回正轨"
    )
    return [*part1, *part2]


def _clean_daily_dinner() -> list[dict[str, Any]]:
    lines = [
        ln.strip()
        for ln in (RAW_DIR / "daily_life&dinner_gathering.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]
    sub_scenes = [
        "周五火锅局改烧烤",
        "基图出来买奶茶",
        "下雨天点外卖与漏餐",
        "深夜刷短视频停不下来",
        "月底吃土算预算",
        "快递开箱翻车与退货",
    ]

    sections: list[list[str]] = []
    current: list[str] | None = None
    for ln in lines:
        if ln.startswith("【场景"):
            if current is not None:
                sections.append(current)
            current = []
        elif current is not None:
            current.append(ln)
    if current is not None:
        sections.append(current)

    if len(sections) != len(sub_scenes):
        raise ValueError(f"Expected 6 sections, got {len(sections)}")

    all_items: list[dict[str, Any]] = []
    five_slices = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]
    for sec_lines, scene_name in zip(sections, sub_scenes, strict=True):
        pairs = _pairs_from_lines(sec_lines)
        all_items.extend(_build_dialogues(pairs, five_slices, scene_name))
    return all_items


def _clean_intern_life() -> list[dict[str, Any]]:
    raw_text = (RAW_DIR / "intren_life.txt").read_text(encoding="utf-8").strip()
    blocks = [b.strip().splitlines() for b in raw_text.split("\n\n") if b.strip()]
    # 等距抽样 25 组代表性实习/兼职对话，避免超短句淹没核心校园陪伴语料，并去除机械“嗯嗯，”口癖
    sampled_blocks = blocks[::6][:25]
    items: list[dict[str, Any]] = []
    for block in sampled_blocks:
        raw_pairs = _pairs_from_lines([ln.strip() for ln in block if ln.strip()])
        cleaned_pairs = [
            (u, a.removeprefix("嗯嗯，")) for u, a in raw_pairs
        ]
        items.extend(
            _build_dialogues(cleaned_pairs, [(0, len(cleaned_pairs))], "实习兼职与秋招")
        )
    return items


def clean_all_scenes() -> dict[str, list[dict[str, Any]]]:
    SCENES_DIR.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, list[dict[str, Any]]] = {
        "morning_8am": _clean_single_narrative(
            "morning_8am.txt",
            "早八起不来",
            [(0, 4), (4, 9), (9, 12), (12, 16), (16, 22), (22, 26), (26, 30)],
        ),
        "later_night": _clean_single_narrative(
            "later_night.txt",
            "考试周刷夜",
            [(0, 4), (4, 8), (8, 13), (13, 18), (18, 24), (24, 29), (29, 32)],
        ),
        "lazy_partner": _clean_single_narrative(
            "lazy_partner.txt",
            "小组作业划水",
            [(0, 4), (4, 8), (8, 12), (12, 17), (17, 22), (22, 26), (26, 32)],
        ),
        "camping": _clean_single_narrative(
            "camping.txt",
            "周末宿舍找乐子",
            [(0, 5), (5, 9), (9, 13), (13, 19), (19, 23)],
        ),
        "canteen_complaint": _clean_single_narrative(
            "canteen_complaint.txt",
            "吐槽早操与食堂",
            [(0, 6), (6, 12), (12, 16), (16, 21), (21, 27), (27, 32)],
        ),
        "course_choosen": _clean_single_narrative(
            "course_choosen.txt",
            "选课与绩点焦虑",
            [(0, 5), (5, 11), (11, 16), (16, 20), (20, 25), (25, 30)],
        ),
        "dorm_relation": _clean_single_narrative(
            "dorm_relation.txt",
            "宿舍吃瓜与社团吐槽",
            [(0, 6), (6, 10), (10, 15), (15, 19), (19, 24)],
        ),
        "emotional_partner": _clean_emotional_partner(),
        "study_partner": _clean_single_narrative(
            "study_partner.txt",
            "学习搭子互相监督",
            [(0, 6), (6, 11), (11, 15), (15, 19), (19, 23)],
        ),
        "wait_crush_messege": _clean_single_narrative(
            "wait_crush_messege.txt",
            "Crush不回消息求支招",
            [(0, 5), (5, 10), (10, 16), (16, 22), (22, 27)],
        ),
        "daily_life_dinner": _clean_daily_dinner(),
        "intern_life": _clean_intern_life(),
    }

    dataset_info: dict[str, Any] = {}
    for key, records in outputs.items():
        out_path = SCENES_DIR / f"{key}.json"
        out_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        dataset_info[key] = {
            "file_name": f"scenes/{key}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }

    DATASET_INFO_PATH.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return outputs


def validate_all_scenes() -> None:
    info = json.loads(DATASET_INFO_PATH.read_text(encoding="utf-8"))
    total_dialogues = 0
    multi_turn_dialogues = 0
    total_turns = 0
    max_chars = 0

    print(f"{'Dataset':20s} | {'Dialogues':>9s} | {'Turns':>6s} | {'MaxChars':>8s}")
    print("-" * 52)

    for key, cfg in info.items():
        path = DATA_DIR / cfg["file_name"]
        records = json.loads(path.read_text(encoding="utf-8"))
        file_turns = 0
        file_max_chars = 0

        for idx, item in enumerate(records):
            if not item.get("system", "").strip():
                raise ValueError(f"{key}[{idx}] missing system prompt")
            convs = item.get("conversations", [])
            if len(convs) < 2 or len(convs) % 2 != 0:
                raise ValueError(f"{key}[{idx}] invalid conversation length")
            for m_idx, msg in enumerate(convs):
                expected = "human" if m_idx % 2 == 0 else "gpt"
                if msg.get("from") != expected or not msg.get("value", "").strip():
                    raise ValueError(f"{key}[{idx}] malformed turn at {m_idx}")

            turns = len(convs) // 2
            if turns > 1:
                multi_turn_dialogues += 1
            file_turns += turns
            char_len = len(item["system"]) + sum(len(m["value"]) for m in convs)
            file_max_chars = max(file_max_chars, char_len)

        total_dialogues += len(records)
        total_turns += file_turns
        max_chars = max(max_chars, file_max_chars)
        print(f"{key:20s} | {len(records):9d} | {file_turns:6d} | {file_max_chars:8d}")

    ratio = (multi_turn_dialogues / total_dialogues * 100) if total_dialogues else 0.0
    print("-" * 52)
    print(
        f"{'TOTAL':20s} | {total_dialogues:9d} | {total_turns:6d} | {max_chars:8d}"
    )
    print(f"Multi-turn ratio: {ratio:.1f}% (All scenes passed validation!)")


if __name__ == "__main__":
    clean_all_scenes()
    validate_all_scenes()
