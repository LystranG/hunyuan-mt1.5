import torch
import uvicorn
import time
import uuid
import argparse
import os
import json
from dotenv import load_dotenv
from threading import Thread
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

# ================= 参数配置 =================
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bf",
    type=int,
    choices=[8, 16],
    default=16,
    help="Bit format 只能是8或者16"
)

parser.add_argument(
    "--model",
    type=str,
    choices=['1.8b', '8b'],
    default='1.8b',
    help="--model 模型版本只能为1.8b 或 8b"
)

args = parser.parse_args()

# ================= 配置区域 =================
MODEL1 = "./hunyuan-model/Tencent-Hunyuan/HY-MT1.5-1.8B"
MODEL2 = "./hunyuan-model/Tencent-Hunyuan/HY-MT1___5-1___8B"
MODEL_PATH = MODEL1 if args.model == '1.8b' else MODEL2

PORT = int(os.getenv("PORT", 8000))

API_KEY = os.getenv("API_KEY", "")

# 显存控制开关
# True  = 开启 8-bit 量化 (显存占用极低，适合多任务并行)
# False = 使用 bf16 原生精度 (显存占用较高，速度略快)
USE_8BIT_QUANTIZATION = args.bf == 8
# ===========================================

app = FastAPI(title="Hunyuan-MT OpenAI Server")

CORS_ALLOW_ORIGINS = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],  # 允许 Authorization / Content-Type 等
)

print(f"正在加载模型: {MODEL_PATH} ...")
print(f"当前模式: {'8-bit 量化' if USE_8BIT_QUANTIZATION else 'bf16 原生精度'}")

# 1. 加载 Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
except Exception as e:
    print(f"❌ Tokenizer 加载失败，请检查路径: {e}")
    exit(1)

# 2. 加载模型
try:
    if USE_8BIT_QUANTIZATION:
        # === 8-bit 量化模式 ===
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # llm_int8_threshold=6.0
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # === bf16 原生模式 ===
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    print(f"模型加载成功")

except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)


# ================= OpenAI 数据结构定义 =================
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "hunyuan-mt"
    messages: List[Message]
    # 默认采样参数
    top_k: Optional[int] = 20
    top_p: Optional[float] = 0.6
    repetition_penalty: Optional[float] = 1.05
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 60100
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: dict


def _normalize_messages(messages: List[Message]) -> List[dict]:
    return [m.model_dump() for m in messages]


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def verify_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """
    简单的 API Key 鉴权：
    """
    if not API_KEY:
        return

    token: Optional[str] = None
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()

    if not token and x_api_key:
        token = x_api_key.strip()

    if token != API_KEY:
        raise HTTPException(status_code=401, detail="未授权：API key 无效或缺失")


# ================= 核心接口 =================
@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    dependencies=[Depends(verify_api_key)],
)
async def chat_completions(request: ChatCompletionRequest):
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages 不能为空")

        # 1. 处理历史消息
        normalized_messages = _normalize_messages(request.messages)
        input_ids = tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        do_sample = (
            (request.temperature is not None and request.temperature > 1e-5)
            or (request.top_p is not None and request.top_p < 1.0 - 1e-6)
            or (request.top_k is not None and request.top_k > 0)
        )

        # 2. 推理生成
        if request.stream:
            created_ts = int(time.time())
            chat_id = f"chatcmpl-{uuid.uuid4()}"
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_special_tokens=True,
                skip_prompt=True
            )

            generation_kwargs = {
                "inputs": input_ids,
                "max_new_tokens": request.max_tokens,
                "do_sample": do_sample,
                "streamer": streamer,
            }
            if do_sample:
                generation_kwargs["temperature"] = request.temperature
                if request.top_p is not None:
                    generation_kwargs["top_p"] = request.top_p
                if request.top_k is not None:
                    generation_kwargs["top_k"] = request.top_k
            if request.repetition_penalty is not None:
                generation_kwargs["repetition_penalty"] = request.repetition_penalty
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id

            def _run_generate():
                with torch.inference_mode():
                    model.generate(**generation_kwargs)

            Thread(target=_run_generate, daemon=True).start()

            def event_iter():
                # 先发一个带 role 的 chunk，兼容部分 OpenAI 客户端
                yield _sse({
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                })

                for text in streamer:
                    if not text:
                        continue
                    yield _sse({
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                    })

                # 结束 chunk + [DONE]
                yield _sse({
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                })
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_iter(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        with torch.inference_mode():
            generation_kwargs = {
                "inputs": input_ids,
                "max_new_tokens": request.max_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                generation_kwargs["temperature"] = request.temperature
                if request.top_p is not None:
                    generation_kwargs["top_p"] = request.top_p
                if request.top_k is not None:
                    generation_kwargs["top_k"] = request.top_k
            if request.repetition_penalty is not None:
                generation_kwargs["repetition_penalty"] = request.repetition_penalty
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id
            outputs = model.generate(**generation_kwargs)

        # 3. 切片获取新生成的 token
        input_len = input_ids.shape[1]
        generated_ids = outputs[0][input_len:]

        # 4. 解码
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 5. 封装返回
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=output_text),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": input_len,
                "completion_tokens": len(generated_ids),
                "total_tokens": input_len + len(generated_ids)
            }
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 模型列表接口
@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "hunyuan-mt", "object": "model", "owned_by": "local"}]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
