import os
from openai import OpenAI

# 选一个地域（北京 / 新加坡二选一）
# 北京: https://dashscope.aliyuncs.com/compatible-mode/v1
# 新加坡: 
# DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
# 北京: 
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# 默认模型（可按需改）
QWEN2_MODEL = "qwen2.5-72b-instruct"
QWEN3_MODEL = "qwen3-next-80b-a3b-instruct"
DEEPSEEK_V3_MODEL = "deepseek-v3"
QWEN3_MAX_MODEL = "qwen3-max"


def get_client(api_key: str | None = None) -> OpenAI:
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Missing DASHSCOPE_API_KEY (or pass api_key).")
    return OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL)


def get_qwen3_profile(
    api_key: str | None = None,
    model: str | None = None,
    enable_thinking: bool | None = None,
) -> tuple[OpenAI, str, dict | None]:
    client = get_client(api_key=api_key)
    return client, (model or QWEN3_MODEL), None


def get_qwen3_max_profile(
    api_key: str | None = None,
    model: str | None = None,
    enable_thinking: bool | None = None,
) -> tuple[OpenAI, str, dict | None]:
    client = get_client(api_key=api_key)
    return client, (model or QWEN3_MAX_MODEL), None


def get_qwen2_profile(
    api_key: str | None = None,
    model: str | None = None,
    enable_thinking: bool | None = None,
) -> tuple[OpenAI, str, dict | None]:
    client = get_client(api_key=api_key)
    return client, (model or QWEN2_MODEL), None


def get_deepseek_v3_profile(
    api_key: str | None = None,
    model: str | None = None,
    enable_thinking: bool = False,
) -> tuple[OpenAI, str, dict | None]:
    client = get_client(api_key=api_key)
    extra_body = {"enable_thinking": True} if enable_thinking else None
    return client, (model or DEEPSEEK_V3_MODEL), extra_body


def get_profile(
    name: str, # model name: qwen3, deepseek-v3, qwen3-max
    api_key: str | None = None,
    model: str | None = None,
    enable_thinking: bool = True,
) -> tuple[OpenAI, str, dict | None]:
    profiles = {
        "qwen3": get_qwen3_profile,
        "deepseek-v3": get_deepseek_v3_profile,
        "qwen3-max": get_qwen3_max_profile,
        "qwen2": get_qwen2_profile,
    }
    if name not in profiles:
        raise ValueError(f"Unknown profile '{name}'. Available: {list(profiles.keys())}")
    return profiles[name](api_key=api_key, model=model, enable_thinking=enable_thinking)


if __name__ == "__main__":
    profiles_to_test = ["qwen2", "qwen3", "deepseek-v3", "qwen3-max"]
    
    for profile_name in profiles_to_test:
        print(f"\n[Testing {profile_name}]", flush=True)
        try:
            client, model, extra_body = get_profile(
                profile_name,
                api_key=None,
                model=None,
                enable_thinking=False,
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "你是谁, 你是哪个模型"},
                ],
                temperature=0.3,
                extra_body=extra_body,
            )
            print(f"{profile_name}: {resp.choices[0].message.content}", flush=True)
        except Exception as e:
            print(f"{profile_name} failed: {e}", flush=True)
