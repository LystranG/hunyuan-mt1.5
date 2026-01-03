import argparse
import os
import sys
from copy import deepcopy

import uvicorn
from dotenv import load_dotenv
from uvicorn.config import LOGGING_CONFIG

from hymt.api import create_app
from hymt.model_loader import load_model


load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--bf", type=int, choices=[8, 16], default=16)
parser.add_argument("--model", type=str, choices=["1.8b", "8b"], default="1.8b")
args = parser.parse_args()

# 读取启动参数与环境变量
MODEL1 = "./hunyuan-model/Tencent-Hunyuan/HY-MT1.5-1.8B"
MODEL2 = "./hunyuan-model/Tencent-Hunyuan/HY-MT1___5-1___8B"
MODEL_PATH = MODEL1 if args.model == "1.8b" else MODEL2

PORT = int(os.getenv("PORT", 8000))
API_KEY = os.getenv("API_KEY", "")
USE_8BIT_QUANTIZATION = args.bf == 8

print(f"正在加载模型: {MODEL_PATH} ...")
print(f"当前模式: {'8-bit 量化' if USE_8BIT_QUANTIZATION else 'bf16 原生精度'}")

# 加载模型与构建应用
tokenizer, model = load_model(MODEL_PATH, use_8bit_quantization=USE_8BIT_QUANTIZATION)
print("模型加载成功")

app = create_app(model=model, tokenizer=tokenizer, api_key=API_KEY)


def _build_uvicorn_log_config() -> dict:
    try:
        config = deepcopy(LOGGING_CONFIG)
        handlers = config.get("handlers", {})
        if "default" in handlers:
            handlers["default"]["stream"] = "ext://sys.stdout"
        if "access" in handlers:
            handlers["access"]["stream"] = "ext://sys.stdout"
        return config
    except Exception:
        return {}


if __name__ == "__main__":
    # 调整 uvicorn 日志输出到 stdout
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        log_config=_build_uvicorn_log_config(),
    )
