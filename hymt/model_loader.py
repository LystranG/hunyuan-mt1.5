import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_path: str, use_8bit_quantization: bool):
    # 加载 tokenizer + 模型（根据是否量化选择不同分支）
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if use_8bit_quantization:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        return tokenizer, model

    # bf16 原生精度加载
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model
