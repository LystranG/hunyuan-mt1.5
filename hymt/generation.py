import json


def normalize_messages(messages):
    # 兼容 pydantic 模型与 dict，统一转换为 dict 列表
    normalized = []
    for m in messages:
        if hasattr(m, "model_dump"):
            normalized.append(m.model_dump())
        elif isinstance(m, dict):
            normalized.append(m)
        else:
            normalized.append({"role": getattr(m, "role", ""), "content": getattr(m, "content", "")})
    return normalized


def build_input_ids(tokenizer, model, messages):
    # 应用 chat_template 生成输入 token ids
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)


def compute_do_sample(temperature, top_p, top_k):
    if temperature is not None and temperature > 1e-5:
        return True
    if top_p is not None and top_p < 1.0 - 1e-6:
        return True
    if top_k is not None and top_k > 0:
        return True
    return False


def build_generation_kwargs(request, input_ids, pad_token_id, streamer=None):
    # 根据请求参数构造 generate() 需要的 kwargs
    do_sample = compute_do_sample(request.temperature, request.top_p, request.top_k)
    generation_kwargs = {
        "inputs": input_ids,
        "max_new_tokens": request.max_tokens,
        "do_sample": do_sample,
    }
    if streamer is not None:
        generation_kwargs["streamer"] = streamer

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

    return generation_kwargs


def decode_output(tokenizer, outputs, input_len):
    # 切片取出新生成 token 并解码
    generated_ids = outputs[0][input_len:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return output_text, generated_ids


def sse(data: dict) -> str:
    # SSE 行格式：data: <json>\n\n
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
