"""模型适配层（Model Adapters）

每个模型有专属的 adapter，包含：
- 模型专属 prompt（system prompt 可定制）
- 模型专属 few-shot 示例
- 模型专属参数（max_tokens、temperature 等）

所有 adapter 实现 EvaluationProtocol 接口，确保结果可比。
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

from .models import EvalRequest, EvalResponse
from .protocol import EvaluationProtocol


class BaseAdapter(EvaluationProtocol):
    """适配器基类。

    封装通用评估流程，子类只需覆盖 get_model_config() 和 get_prompt_customization()。
    """

    def evaluate(self, request: EvalRequest, temperature: float = 0.0) -> EvalResponse:
        """通用评估流程"""
        config = self.get_model_config()

        # 构建 prompt（基础 + profile + 模型专属定制）
        system_prompt = self._build_system_prompt(request.evaluation_profile)
        user_prompt = self._build_user_prompt(request)

        # 调用 LLM
        llm = self._create_llm(config, temperature)
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = llm.invoke(messages)
        raw_output = str(response.content) if hasattr(response, "content") else str(response)

        # 解析输出
        from .chain import parse_llm_output, extract_dimensions, extract_flaws, compute_overall_score
        from .chain import apply_veto_rules, determine_verdict, apply_profile_penalties
        from .chain import build_reproducibility_token

        parsed = parse_llm_output(raw_output)
        dimensions = extract_dimensions(parsed)
        flaws = extract_flaws(parsed)
        overall_score = compute_overall_score(dimensions)
        overall_score = apply_veto_rules(overall_score, flaws)
        verdict = determine_verdict(overall_score)
        overall_score, verdict, flaws = apply_profile_penalties(
            request.evaluation_profile, overall_score, verdict, flaws,
            request.before_text, request.after_text,
        )

        return EvalResponse(
            request_id=request.request_id,
            evaluation_profile=request.evaluation_profile,
            dimensions=dimensions,
            overall_score=round(overall_score, 4),
            flaws=flaws,
            verdict=verdict,
            reproducibility_token=build_reproducibility_token(request, temperature),
            model_version=self.get_model_name(),
            prompt_version=self._prompt_version(),
            raw_llm_output=raw_output,
        )

    def get_model_config(self) -> dict[str, Any]:
        """模型配置（子类可覆盖）"""
        return {
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
            "max_tokens": 2048,
        }

    def get_prompt_customization(self) -> dict[str, str]:
        """模型专属 prompt 定制（子类可覆盖）。

        Returns:
            dict: 可包含 system_suffix、user_suffix、few_shot_extra
        """
        return {}

    def _build_system_prompt(self, profile: str) -> str:
        """构建 system prompt（基础 + profile + 模型专属）"""
        from .prompts import build_system_prompt
        base = build_system_prompt(profile)
        custom = self.get_prompt_customization()
        suffix = custom.get("system_suffix", "")
        if suffix:
            base += "\n" + suffix
        return base

    def _build_user_prompt(self, request: EvalRequest) -> str:
        """构建 user prompt"""
        from .prompts import build_user_prompt
        return build_user_prompt(
            before_text=request.before_text,
            after_text=request.after_text,
            segments_before=request.segments_before,
            segments_after=request.segments_after,
        )

    def _create_llm(self, config: dict[str, Any], temperature: float) -> Any:
        """创建 LLM 实例"""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config["model"],
            temperature=temperature,
            max_tokens=config.get("max_tokens", 2048),
            base_url=config["base_url"],
            api_key=config["api_key"],
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def _prompt_version(self) -> str:
        """prompt 版本指纹"""
        from .prompts import SYSTEM_PROMPT
        return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8]


class DeepSeekAdapter(BaseAdapter):
    """DeepSeek 适配器。

    默认适配器，使用标准 prompt，无特殊定制。
    """

    def get_model_name(self) -> str:
        return os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    def get_model_version(self) -> str:
        return self._prompt_version()


class GLMAdapter(BaseAdapter):
    """智谱 GLM 适配器。

    GLM 对指令的响应方式与 DeepSeek 不同，需要定制：
    - 更明确的输出格式指令
    - 更详细的 few-shot 示例
    - API Key 格式：{id}.{secret}，需转换为 JWT
    """

    def _get_glm_api_key(self) -> str:
        """获取 GLM API Key（自动处理 JWT 转换）"""
        key = os.getenv("GLM_API_KEY", "")
        if not key:
            return ""
        # 如果是 {id}.{secret} 格式，转换为 JWT
        if "." in key and not key.startswith("eyJ"):
            try:
                import time
                import jwt
                key_id, secret = key.split(".", 1)
                payload = {
                    "api_key": key_id,
                    "exp": int(time.time()) + 3600,
                    "timestamp": int(time.time()),
                }
                headers = {"alg": "HS256", "sign_type": "SIGN"}
                return jwt.encode(payload, secret, algorithm="HS256", headers=headers)
            except Exception:
                return key
        return key

    def get_model_config(self) -> dict[str, Any]:
        return {
            "model": os.getenv("GLM_MODEL", "glm-4.7-flash"),
            "base_url": os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            "api_key": self._get_glm_api_key(),
            "max_tokens": 2048,
        }

    def get_model_name(self) -> str:
        return os.getenv("GLM_MODEL", "glm-4.7-flash")

    def get_model_version(self) -> str:
        return self._prompt_version() + "-glm"

    def get_prompt_customization(self) -> dict[str, str]:
        return {
            "system_suffix": """
【GLM 特别指令】
请严格按照以下 JSON 格式输出，不要添加任何额外文字或解释：
{
  "dimensions": [...],
  "overall_score": 0.0-1.0,
  "flaws": [...]
}
确保 JSON 可被直接解析，不要包含 markdown 代码块标记。
""",
        }


class GPTAdapter(BaseAdapter):
    """OpenAI GPT 适配器。

    GPT-4o 对"按以下格式输出"的指令响应更好。
    """

    def get_model_config(self) -> dict[str, Any]:
        return {
            "model": os.getenv("GPT_MODEL", "gpt-4o"),
            "base_url": os.getenv("GPT_BASE_URL", "https://api.openai.com/v1"),
            "api_key": os.getenv("GPT_API_KEY", ""),
            "max_tokens": 2048,
        }

    def get_model_name(self) -> str:
        return os.getenv("GPT_MODEL", "gpt-4o")

    def get_model_version(self) -> str:
        return self._prompt_version() + "-gpt"

    def get_prompt_customization(self) -> dict[str, str]:
        return {
            "system_suffix": """
Output Requirements:
1. Output ONLY valid JSON, no markdown fences
2. All scores must be between 0.0 and 1.0
3. Use the exact field names specified in the format
""",
        }
