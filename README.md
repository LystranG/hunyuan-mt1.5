# Hunyuan-MT1.5 本地 OpenAI 兼容服务

## 功能

- OpenAI 兼容接口：`POST /v1/chat/completions`、`GET /v1/models`
- 支持SSE
- API Key 鉴权（`Authorization: Bearer <key>` 或 `X-API-Key: <key>`）
- 流量控制

## 快速开始

1) 安装依赖（示例）

建议用 conda: 
```shell
conda env create -f environment.yml
```

2) 配置环境变量

复制一份 `.env.example` 为 `.env` 并按需修改。

3) 启动服务

```bash
python service.py --model 1.8b --bf 16 --model 8b
```

`--bf 8` 表示启用 8-bit 量化，`--bf 16` 表示 bf16 原生精度，默认为bf16
`--model 7b` 表示使用7b模型，`--model 1.8b` 表示使用1.8b模型 ，默认为1.8b

## 关键环境变量

- `PORT`：端口（默认 8000）
- `API_KEY`：API key（为空则不鉴权）
- `LOG_LEVEL`：uvicorn 日志级别（默认 info）
- `MAX_CONCURRENT_GENERATIONS`：同一时刻最多跑几个生成任务（默认 1）
- `GENERATION_QUEUE_SIZE`：排队队列上限（默认 256，满则 429）
- `CORS_ALLOW_ORIGINS`：允许跨域来源（默认 `*`，多个用逗号分隔）
- `CORS_ALLOW_CREDENTIALS`：是否允许携带凭证（默认 0）

