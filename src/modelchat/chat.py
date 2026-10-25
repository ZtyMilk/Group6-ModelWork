import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_PATH = ".model/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master"
ADAPTER_PATH = "saves/persona_lora"

print("Loading...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
print("INPUT(Priny C to quit)")

history = []

while True:
    try:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["q", "exit", "quit"]:
            print("再见！")
            break

        # react
        history.append({"role": "user", "content": user_input})

        text = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

        print("对方: ", end="", flush=True)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.85,
            top_p=0.9,
            repetition_penalty=1.15,
            streamer=streamer,
            do_sample=True,
        )

        response_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": response_text})

        # context_length_max
        if len(history) > 10:
            history = history[-10:]

    except KeyboardInterrupt:
        print("\n已退出。")
        break
