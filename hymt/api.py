import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer

from .generation import build_generation_kwargs, build_input_ids, decode_output, normalize_messages, sse


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "hunyuan-mt"
    messages: List[Message]
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


def verify_api_key_factory(api_key: str):
    def verify_api_key(
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> None:
        if not api_key:
            return

        token: Optional[str] = None
        if authorization:
            parts = authorization.split(" ", 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1].strip()

        if not token and x_api_key:
            token = x_api_key.strip()

        if token != api_key:
            raise HTTPException(status_code=401, detail="未授权：API key 无效或缺失")

    return verify_api_key


class GenerationJob:
    def __init__(self, request: ChatCompletionRequest, streamer: Optional[TextIteratorStreamer]):
        self.request = request
        self.streamer = streamer
        self.done = asyncio.Event()
        self.error: Optional[str] = None
        self.outputs = None
        self.input_len: Optional[int] = None


class GenerationScheduler:
    def __init__(self, model, tokenizer, max_concurrent: int, queue_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_concurrent = max(1, int(max_concurrent))
        self.queue: asyncio.Queue[GenerationJob] = asyncio.Queue(maxsize=max(1, int(queue_size)))
        self._workers: List[asyncio.Task] = []

    async def start(self):
        if self._workers:
            return
        for _ in range(self.max_concurrent):
            self._workers.append(asyncio.create_task(self._worker()))

    async def stop(self):
        for t in self._workers:
            t.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    async def enqueue(self, job: GenerationJob):
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull:
            raise HTTPException(status_code=429, detail="服务繁忙：请稍后重试")
        return job

    async def _worker(self):
        while True:
            job = await self.queue.get()
            try:
                await self._run_job(job)
            finally:
                self.queue.task_done()

    async def _run_job(self, job: GenerationJob):
        request = job.request
        messages = normalize_messages(request.messages)

        def _run_generate():
            input_ids = build_input_ids(self.tokenizer, self.model, messages)
            job.input_len = int(input_ids.shape[1])
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
            generation_kwargs = build_generation_kwargs(
                request=request,
                input_ids=input_ids,
                pad_token_id=pad_token_id,
                streamer=job.streamer,
            )
            with torch.inference_mode():
                return self.model.generate(**generation_kwargs)

        try:
            job.outputs = await asyncio.to_thread(_run_generate)
        except Exception as e:
            job.error = str(e)
        finally:
            if job.streamer is not None:
                try:
                    job.streamer.end()
                except Exception:
                    pass
            job.done.set()


def create_app(model, tokenizer, api_key: str):
    # 读取 CORS 配置并创建 app
    cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
    allow_origins = ["*"] if cors_origins.strip() == "*" else [o.strip() for o in cors_origins.split(",") if o.strip()]
    allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "0").lower() in ("1", "true", "yes", "on")

    # 读取并发与队列配置，初始化后台调度器
    max_concurrent = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "1"))
    queue_size = int(os.getenv("GENERATION_QUEUE_SIZE", "256"))
    scheduler = GenerationScheduler(model=model, tokenizer=tokenizer, max_concurrent=max_concurrent, queue_size=queue_size)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 启动/关闭时启动与停止调度器
        await scheduler.start()
        yield
        await scheduler.stop()

    app = FastAPI(title="Hunyuan-MT OpenAI Server", lifespan=lifespan)
    app.state.scheduler = scheduler
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    verify_api_key = verify_api_key_factory(api_key)

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def chat_completions(request: ChatCompletionRequest):
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages 不能为空")

        created_ts = int(time.time())
        chat_id = f"chatcmpl-{uuid.uuid4()}"

        if request.stream:
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_special_tokens=True,
                skip_prompt=True,
            )
            job = GenerationJob(request=request, streamer=streamer)
            await scheduler.enqueue(job)

            def event_iter():
                yield sse({
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                })

                sent_any = False
                for text in streamer:
                    if not text:
                        continue
                    sent_any = True
                    yield sse({
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                    })

                if job.error and not sent_any:
                    yield sse({
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": job.error}, "finish_reason": None}],
                    })

                yield sse({
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

        job = GenerationJob(request=request, streamer=None)
        await scheduler.enqueue(job)
        await job.done.wait()
        if job.error:
            raise HTTPException(status_code=500, detail=job.error)

        output_text, generated_ids = decode_output(tokenizer, job.outputs, job.input_len or 0)
        input_len = job.input_len or 0
        return ChatCompletionResponse(
            id=chat_id,
            created=created_ts,
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=output_text),
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": input_len,
                "completion_tokens": len(generated_ids),
                "total_tokens": input_len + len(generated_ids),
            },
        )

    @app.get("/v1/models", dependencies=[Depends(verify_api_key)])
    async def list_models():
        return {"object": "list", "data": [{"id": "hunyuan-mt", "object": "model", "owned_by": "local"}]}

    return app
