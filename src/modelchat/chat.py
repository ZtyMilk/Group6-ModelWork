from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_PATH = ".model/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master"
ADAPTER_PATH = "saves/persona_lora"
KIKI_SYSTEM_PROMPT = (
    "你叫KIKI，是用户的浙大同窗好友兼日常搭子。"
    "说话自然接地气、嘴碎爱损人但关键时刻稳得住、护短靠谱，"
    "绝不用客服腔、机器腔或大道理说教。每次只说一两句简短实在的口语。"
)


def apply_lora_scale(model_obj: Any, scale_ratio: float = 0.25) -> None:
    try:
        for module in model_obj.modules():
            scaling = getattr(module, "scaling", None)
            if isinstance(scaling, dict):
                for k in list(scaling.keys()):
                    scaling[k] = 2.0 * scale_ratio
    except (AttributeError, TypeError, KeyError):
        return


print("Loading...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
apply_lora_scale(model, 0.25)
model.eval()

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
print("已就绪！输入 q / exit 退出，输入 /clear 清空对话记忆。")

history: list[dict[str, str]] = []

while True:
    try:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["q", "exit", "quit"]:
            print("再见！")
            break
        if user_input.lower() == "/clear":
            history.clear()
            print("已清空对话历史。")
            continue

        messages = [
            {"role": "system", "content": KIKI_SYSTEM_PROMPT},
            *history[-6:],
            {"role": "user", "content": user_input},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

        print("KIKI: ", end="", flush=True)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=80,
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.08,
            streamer=streamer,
            do_sample=True,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ],
        )

        response_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response_text})
        history = history[-6:]

    except KeyboardInterrupt:
        print("\n已退出。")
        break
