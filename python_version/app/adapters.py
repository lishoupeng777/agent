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
        """通用评估流程 —— 复用 chain.py 的完整管线，只替换模型。

        这样适配器能获得与主链完全一致的：
        - 锚点预处理 / diff 计算
        - 算法预检调整
        - 置信度评估
        - 判定原因码
        - 线性校准
        """
        from .chain import evaluate as chain_evaluate, create_llm
        from .models import EvalResponse

        config = self.get_model_config()

        # 临时切换 LLM 到适配器指定的模型
        import app.chain as _chain_mod
        original_llm = _chain_mod._llm_instance
        _chain_mod._llm_instance = None  # 强制重建

        try:
            # 设置环境变量让 create_llm() 使用适配器的模型配置
            env_overrides = {}
            if "model" in config:
                env_key = self._env_model_key()
                env_overrides[env_key] = config["model"]
                os.environ[env_key] = config["model"]
            if "base_url" in config:
                env_key = self._env_base_url_key()
                env_overrides[env_key] = config["base_url"]
                os.environ[env_key] = config["base_url"]
            if "api_key" in config:
                env_key = self._env_api_key_key()
                env_overrides[env_key] = config["api_key"]
                os.environ[env_key] = config["api_key"]

            resp = chain_evaluate(request, temperature)

            # 覆盖模型版本信息
            resp.model_version = self.get_model_name()
            resp.prompt_version = self._prompt_version()

            return resp
        finally:
            # 恢复环境变量
            _chain_mod._llm_instance = None  # 强制下次调用时重建
            for key, val in env_overrides.items():
                # 恢复到原始值（从 .env 读取）
                from dotenv import load_dotenv
                load_dotenv(override=True)

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

    def _env_model_key(self) -> str:
        return "DEEPSEEK_MODEL"

    def _env_base_url_key(self) -> str:
        return "DEEPSEEK_BASE_URL"

    def _env_api_key_key(self) -> str:
        return "DEEPSEEK_API_KEY"


class MimoAdapter(BaseAdapter):
    """小米 Mimo 2.5 Pro 适配器。

    Mimo 使用 OpenAI 兼容协议，无需特殊转换。
    API Key 从 .env 读取（MIMO_API_KEY / MIMO_BASE_URL / MIMO_MODEL）。
    """

    def get_model_config(self) -> dict[str, Any]:
        return {
            "model": os.getenv("MIMO_MODEL", "mimo-v2.5-pro"),
            "base_url": os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"),
            "api_key": os.getenv("MIMO_API_KEY", ""),
            "max_tokens": 2048,
        }

    def get_model_name(self) -> str:
        return os.getenv("MIMO_MODEL", "mimo-v2.5-pro")

    def get_model_version(self) -> str:
        return self._prompt_version() + "-mimo"

    def get_prompt_customization(self) -> dict[str, str]:
        return {}

    def _env_model_key(self) -> str:
        return "DEEPSEEK_MODEL"

    def _env_base_url_key(self) -> str:
        return "DEEPSEEK_BASE_URL"

    def _env_api_key_key(self) -> str:
        return "DEEPSEEK_API_KEY"


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

    def _env_model_key(self) -> str:
        return "DEEPSEEK_MODEL"

    def _env_base_url_key(self) -> str:
        return "DEEPSEEK_BASE_URL"

    def _env_api_key_key(self) -> str:
        return "DEEPSEEK_API_KEY"


# ============================================================
# LiteLLM 通用适配器 —— 支持 20+ 模型提供商
# ============================================================

# LiteLLM 支持的模型映射表：短名 → litellm 模型标识
# 用户可通过 .env 中的 API Key 自动启用对应模型
LITELLM_MODEL_CATALOG: dict[str, dict[str, str]] = {
    # OpenAI 系列
    "gpt-4o": {
        "litellm_model": "gpt-4o",
        "provider": "openai",
        "env_key": "OPENAI_API_KEY",
        "display": "GPT-4o (OpenAI)",
    },
    "gpt-4o-mini": {
        "litellm_model": "gpt-4o-mini",
        "provider": "openai",
        "env_key": "OPENAI_API_KEY",
        "display": "GPT-4o Mini (OpenAI)",
    },
    "gpt-4.1": {
        "litellm_model": "gpt-4.1",
        "provider": "openai",
        "env_key": "OPENAI_API_KEY",
        "display": "GPT-4.1 (OpenAI)",
    },
    # Anthropic 系列
    "claude-sonnet-4": {
        "litellm_model": "anthropic/claude-sonnet-4-20250514",
        "provider": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
        "display": "Claude Sonnet 4 (Anthropic)",
    },
    "claude-haiku": {
        "litellm_model": "anthropic/claude-3-5-haiku-20241022",
        "provider": "anthropic",
        "env_key": "ANTHROPIC_API_KEY",
        "display": "Claude 3.5 Haiku (Anthropic)",
    },
    # Google 系列
    "gemini-2.0-flash": {
        "litellm_model": "gemini/gemini-2.0-flash",
        "provider": "google",
        "env_key": "GEMINI_API_KEY",
        "display": "Gemini 2.0 Flash (Google)",
    },
    "gemini-2.5-flash": {
        "litellm_model": "gemini/gemini-2.5-flash",
        "provider": "google",
        "env_key": "GEMINI_API_KEY",
        "display": "Gemini 2.5 Flash (Google)",
    },
    # 阿里 Qwen 系列
    "qwen-plus": {
        "litellm_model": "openai/qwen-plus",
        "provider": "qwen",
        "env_key": "QWEN_API_KEY",
        "display": "Qwen Plus (阿里通义)",
    },
    "qwen-max": {
        "litellm_model": "openai/qwen-max",
        "provider": "qwen",
        "env_key": "QWEN_API_KEY",
        "display": "Qwen Max (阿里通义)",
    },
    # 智谱 GLM 系列
    "glm-4-flash": {
        "litellm_model": "openai/glm-4-flash",
        "provider": "zhipu",
        "env_key": "ZHIPU_API_KEY",
        "display": "GLM-4 Flash (智谱)",
    },
}


class LiteLLMAdapter(BaseAdapter):
    """LiteLLM 通用适配器。

    通过 litellm 统一调用 20+ 模型提供商（OpenAI、Anthropic、Google、阿里、智谱等），
    无需为每个模型写专属 adapter。

    使用方式：
        adapter = LiteLLMAdapter("gpt-4o")
        adapter = LiteLLMAdapter("claude-sonnet-4")
        adapter = LiteLLMAdapter("gemini-2.0-flash")

    环境变量：
        各提供商的 API Key 从 .env 自动读取（OPENAI_API_KEY、ANTHROPIC_API_KEY 等）。
        litellm 会自动识别并使用对应的 Key。
    """

    def __init__(self, model_short_name: str) -> None:
        """
        Args:
            model_short_name: 模型短名（如 "gpt-4o"、"claude-sonnet-4"），
                              必须在 LITELLM_MODEL_CATALOG 中。
        """
        self._short_name = model_short_name
        catalog = LITELLM_MODEL_CATALOG.get(model_short_name)
        if catalog is None:
            raise ValueError(
                f"Unknown model: {model_short_name}. "
                f"Available: {list(LITELLM_MODEL_CATALOG.keys())}"
            )
        self._catalog = catalog

    def evaluate(self, request: EvalRequest, temperature: float = 0.0) -> EvalResponse:
        """使用 litellm 进行评估。

        直接调用 litellm.completion()，然后复用 chain.py 的解析管线。
        """
        import time
        from litellm import completion as litellm_completion
        from .prompts import build_system_prompt, build_user_prompt
        from .chain import (
            parse_llm_output,
            validate_llm_output,
            extract_dimensions,
            extract_flaws,
            resolve_anchor_offsets,
            detect_flaws,
            build_anchored_text,
            compute_overall_score,
            determine_verdict,
            determine_reason_code,
            compute_confidence,
            compute_risk_level,
            _get_rule_hash,
        )
        from .models import AnchorSpan

        t0 = time.time()

        # 构建 prompt
        system_prompt = build_system_prompt(request.evaluation_profile)
        user_prompt = build_user_prompt(
            before_text=request.before_text,
            after_text=request.after_text,
            segments_before=request.segments_before,
            segments_after=request.segments_after,
        )

        # 调用 litellm
        litellm_model = self._catalog["litellm_model"]
        try:
            response = litellm_completion(
                model=litellm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=2048,
            )
            raw_output = response.choices[0].message.content
        except Exception as e:
            return EvalResponse(
                request_id=request.request_id,
                evaluation_profile=request.evaluation_profile,
                dimensions=[],
                overall_score=0.0,
                flaws=[],
                verdict="fail",
                reproducibility_token="error",
                model_version=self.get_model_name(),
                prompt_version=self._prompt_version(),
                raw_llm_output=f"LiteLLM Error: {e}",
                latency_seconds=round(time.time() - t0, 2),
            )

        # 复用 chain.py 的解析管线
        try:
            parsed = parse_llm_output(raw_output)
            validated = validate_llm_output(parsed)
            dimensions = extract_dimensions(validated.model_dump())
            flaws = extract_flaws(validated.model_dump())
        except Exception as e:
            return EvalResponse(
                request_id=request.request_id,
                evaluation_profile=request.evaluation_profile,
                dimensions=[],
                overall_score=0.0,
                flaws=[],
                verdict="fail",
                reproducibility_token="parse-error",
                model_version=self.get_model_name(),
                prompt_version=self._prompt_version(),
                raw_llm_output=f"Parse Error: {e}\n\nRaw: {raw_output[:500]}",
                latency_seconds=round(time.time() - t0, 2),
            )

        # 锚点解析
        anchored_after = build_anchored_text(request.after_text, request.segments_after)
        flaws = resolve_anchor_offsets(flaws, anchored_after, request.after_text)

        # Diff 检测补充位置
        detected_flaws = detect_flaws(request.before_text, request.after_text)
        diff_positions: dict[str, tuple[int, int]] = {}
        for df in detected_flaws:
            if df.get("anchor_after"):
                snippet = df["anchor_after"].split("] ...")[-1].rstrip("...") if "] ..." in df["anchor_after"] else ""
                if snippet and len(snippet) > 3:
                    pos = request.after_text.find(snippet[:20])
                    if pos >= 0:
                        diff_positions[snippet[:20]] = (pos, pos + len(snippet[:20]))

        for flaw in flaws:
            loc = flaw.location
            if loc.start_char == 0 and loc.end_char == 0 and loc.snippet:
                for snip, (s, e) in diff_positions.items():
                    if snip in loc.snippet or loc.snippet[:15] in snip:
                        flaw.location = AnchorSpan(
                            segment_id=loc.segment_id,
                            start_char=s, end_char=e,
                            snippet=loc.snippet,
                        )
                        break
                if flaw.location.start_char == 0 and flaw.location.end_char == 0:
                    search_text = loc.snippet[:15]
                    pos = request.after_text.find(search_text)
                    if pos >= 0:
                        flaw.location = AnchorSpan(
                            segment_id=loc.segment_id,
                            start_char=pos,
                            end_char=pos + len(search_text),
                            snippet=loc.snippet,
                        )

        # 计算总分
        overall_score = compute_overall_score(dimensions)
        verdict = determine_verdict(overall_score)
        reason_code = determine_reason_code(overall_score, flaws)
        confidence = compute_confidence(detected_flaws, flaws, raw_output)
        risk_level = compute_risk_level(overall_score, confidence or 0.5)

        # Token 统计
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None

        latency = round(time.time() - t0, 2)

        return EvalResponse(
            request_id=request.request_id,
            evaluation_profile=request.evaluation_profile,
            dimensions=dimensions,
            overall_score=overall_score,
            flaws=flaws,
            verdict=verdict,
            reason_code=reason_code,
            reproducibility_token=self._prompt_version(),
            model_version=self.get_model_name(),
            prompt_version=self._prompt_version(),
            rule_version=_get_rule_hash(),
            confidence=confidence,
            risk_level=risk_level,
            raw_llm_output=raw_output,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def get_model_name(self) -> str:
        return self._catalog["litellm_model"]

    def get_model_version(self) -> str:
        return self._prompt_version() + f"-litellm-{self._short_name}"

    def get_display_name(self) -> str:
        return self._catalog.get("display", self._short_name)

    def _env_model_key(self) -> str:
        return "DEEPSEEK_MODEL"

    def _env_base_url_key(self) -> str:
        return "DEEPSEEK_BASE_URL"

    def _env_api_key_key(self) -> str:
        return "DEEPSEEK_API_KEY"


def discover_available_litellm_models() -> dict[str, "LiteLLMAdapter"]:
    """根据环境变量自动发现可用的 litellm 模型。

    检查每个模型所需的 API Key 是否已配置，
    只返回 Key 存在的模型适配器。

    Returns:
        dict: 模型短名 → LiteLLMAdapter 实例
    """
    available = {}
    for short_name, catalog in LITELLM_MODEL_CATALOG.items():
        env_key = catalog["env_key"]
        api_key = os.getenv(env_key, "")
        if api_key and len(api_key) > 5:
            try:
                adapter = LiteLLMAdapter(short_name)
                available[short_name] = adapter
            except ValueError:
                continue
    return available
