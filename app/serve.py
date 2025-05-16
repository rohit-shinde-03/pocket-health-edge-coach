# app/serve.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, textwrap, os

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("⏳ Loading TinyLlama; first run = slow download ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"      # uses CPU on Windows unless you have CUDA
)
print("Model ready")

app = FastAPI(title="Pocket-Health Edge Coach")

class StepsIn(BaseModel):
    steps: int

@app.post("/insight")
def insight(inp: StepsIn):
    SYSTEM_MSG = "You are a concise, friendly health coach."

    USER_TMPL = (
        "Yesterday I walked {steps} steps. "
        # ↓ clearer + stricter wording
        "Respond with exactly ONE actionable suggestion in ONE sentence, "
        "no more than 30 words. Do not add bullet points or extra text."
    )

    chat_prompt = (
        f"<|system|>\n{SYSTEM_MSG}\n"
        f"<|user|>\n{USER_TMPL.format(steps=inp.steps)}\n"
        f"<|assistant|>\n"
    )

    # tighter generation limits
    generated_ids = model.generate(
        tokenizer(chat_prompt, return_tensors="pt").input_ids,
        max_new_tokens=60,   # was 120
        temperature=0.6,
    )

    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    answer = full_text.split("<|assistant|>")[-1].strip()

    # --- post-clamp to ≤ 35 words ----------------------------------------
    words = answer.replace("\n", " ").split()
    if len(words) > 35:
        words = words[:35]
        if words[-1][-1] not in ".!?":  # end with period if needed
            words[-1] += "."
        answer = " ".join(words)

    return {"answer": textwrap.fill(answer, width=70)}