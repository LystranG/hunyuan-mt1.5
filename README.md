# Hunyuan-MT1.5 本地 OpenAI 兼容服务

## 功能

- OpenAI 兼容接口：`POST /v1/chat/completions`、`GET /v1/models`
- 支持 `stream=true`（SSE）
- API Key 鉴权（`Authorization: Bearer <key>` 或 `X-API-Key: <key>`）
- CORS（浏览器跨域）
- 队列 + 并发调度（队列满返回 429）

## 快速开始

1) 安装依赖（示例）

建议用 conda 装 PyTorch + CUDA，其它依赖用 `uv`/`pip` 装。

2) 配置环境变量

复制一份 `.env.example` 为 `.env` 并按需修改。

3) 启动服务

```bash
python service.py --model 1.8b --bf 16
```

`--bf 8` 表示启用 8-bit 量化，`--bf 16` 表示 bf16 原生精度。

## 调用示例

非流式：

```bash
export API_KEY="your-key"
bash examples/curl_chat.sh
```

流式（SSE）：

```bash
export API_KEY="your-key"
bash examples/curl_chat_stream.sh
```

## 关键环境变量

- `PORT`：端口（默认 8000）
- `API_KEY`：API key（为空则不鉴权）
- `LOG_LEVEL`：uvicorn 日志级别（默认 info）
- `MAX_CONCURRENT_GENERATIONS`：同一时刻最多跑几个生成任务（默认 1）
- `GENERATION_QUEUE_SIZE`：排队队列上限（默认 256，满则 429）
- `CORS_ALLOW_ORIGINS`：允许跨域来源（默认 `*`，多个用逗号分隔）
- `CORS_ALLOW_CREDENTIALS`：是否允许携带凭证（默认 0）

