"""端到端冒烟测试：本地 Qwen2.5-1.5B + EasyJailbreak 包装 + GPU 生成。"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from easyjailbreak.models.huggingface_model import HuggingfaceModel

MODEL_PATH = r"D:/models/Qwen2.5-1.5B-Instruct"

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
# 先按 CPU 逐张量加载（low_cpu_mem_usage），再整体搬到 GPU。
# 直接 device_map="cuda" 会走 safetensors 的 GPU 直读路径，在本机 torch 2.8 + safetensors 0.8
# 组合下会报 "invalid python storage"，且会撞 Windows 提交内存上限（os error 1455）。
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True
)
print("cpu 加载完成，正在搬到 GPU ...", flush=True)
model = model.to("cuda")
print("loaded, params: %.2fB" % (sum(p.numel() for p in model.parameters()) / 1e9))
print("dtype:", next(model.parameters()).dtype,
      "| device:", next(model.parameters()).device)

wrapper = HuggingfaceModel(
    model=model,
    tokenizer=tok,
    model_name="qwen-7b-chat",
    generation_config={"max_new_tokens": 64, "do_sample": False},
)

out = wrapper.generate("用一句话解释什么是机器学习。")
print("-" * 50)
print("PROMPT : 用一句话解释什么是机器学习。")
print("RESPONSE:", out.strip()[:300])
print("-" * 50)
print("峰值显存: %.2f GB" % (torch.cuda.max_memory_allocated() / 1024**3))
