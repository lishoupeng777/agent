from pydantic import ValidationError

from app.models import EvalRequest, EvalResponse
from app.profiles import (
    PROFILE_GENERAL,
    PROFILE_GOVERNMENT_NOTICE_STRICT,
    PROFILE_LEGAL_STRICT,
    PROFILE_OPTIONS,
    get_profile_config,
    validate_profile_key,
)


def test_eval_request_defaults_to_general_profile():
    req = EvalRequest(request_id="r1", before_text="a", after_text="b")
    assert req.evaluation_profile == PROFILE_GENERAL


def test_eval_response_defaults_to_general_profile():
    resp = EvalResponse(
        request_id="r1",
        dimensions=[],
        overall_score=0.9,
        verdict="pass",
        reproducibility_token="token",
    )
    assert resp.evaluation_profile == PROFILE_GENERAL


def test_validate_profile_key_accepts_builtin_profiles():
    assert validate_profile_key(PROFILE_GENERAL) == PROFILE_GENERAL
    assert validate_profile_key(PROFILE_GOVERNMENT_NOTICE_STRICT) == PROFILE_GOVERNMENT_NOTICE_STRICT
    assert validate_profile_key(PROFILE_LEGAL_STRICT) == PROFILE_LEGAL_STRICT


def test_validate_profile_key_rejects_unknown_profile():
    try:
        validate_profile_key("unknown-profile")
    except ValueError as exc:
        assert "unknown-profile" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown profile")


def test_eval_request_rejects_unknown_profile_with_validation_error():
    try:
        EvalRequest(
            request_id="r1",
            before_text="a",
            after_text="b",
            evaluation_profile="unknown-profile",
        )
    except ValidationError as exc:
        assert "unknown-profile" in str(exc)
        assert "evaluation_profile" in str(exc)
    else:
        raise AssertionError("expected ValidationError for unknown profile")


def test_profile_registry_exposes_labels_for_ui():
    labels = {item["key"]: item["label"] for item in PROFILE_OPTIONS}
    assert labels[PROFILE_GENERAL] == "通用文本"
    assert labels[PROFILE_GOVERNMENT_NOTICE_STRICT] == "政务通告/公告（严格保真）"
    assert labels[PROFILE_LEGAL_STRICT] == "法规合同/条款（严格保真）"


def test_profile_config_contains_prompt_and_penalty_metadata():
    config = get_profile_config(PROFILE_GOVERNMENT_NOTICE_STRICT)
    assert config["key"] == PROFILE_GOVERNMENT_NOTICE_STRICT
    assert "prompt_rules" in config
    assert "critical_fact_types" in config
    assert "penalty_policy" in config
