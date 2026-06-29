# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from `python_version/`:

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env with API key
echo DEEPSEEK_API_KEY=sk-xxx > .env

# Run Streamlit UI (port 8501)
streamlit run app.py

# Run FastAPI backend (port 8081)
python main.py

# Run batch evaluation against eval_dataset.json
python tests/test_evaluate.py

# Run calibration test script
python run_calibration.py
```

There is no automated test suite with pytest. `tests/test_evaluate.py` is a standalone script that hits the live DeepSeek API, so it requires `DEEPSEEK_API_KEY` to be set.

## Architecture

There are **two parallel implementations** that share no code:

**1. FastAPI backend** (`main.py` + `app/`) — a proper REST API with structured Pydantic models, six endpoints under `/api/v1`, and a plain HTML frontend at `static/index.html`. This is the production path.

**2. Streamlit app** (`app.py`) — a self-contained single-file UI that reimplements the LLM call, prompt, and JSON parsing inline. It has its own separate system prompt (two-dimension scoring: `semantic_consistency_score` + `readability_structure_score`) that differs from the FastAPI backend's four-dimension prompt.

When making changes, be deliberate about which implementation you're targeting — a change to `app/prompts.py` does not affect `app.py` and vice versa.

### FastAPI backend internals (`app/`)

The evaluation flow: `routes.py` → `engine.py` → `prompts.py` → DeepSeek API → back through `engine.py` for JSON parsing and scoring.

- `engine.py` — core LLM call; uses a singleton `ChatOpenAI` instance; implements three-tier JSON extraction (direct parse → markdown fence → brace extraction); applies **veto rules** post-parse: critical `structure` or `over_clean` flaws cap `overall_score` to ≤ 0.25, critical `mis_edit` caps to ≤ 0.35; verdict thresholds are pass ≥ 0.8, review ≥ 0.5, fail < 0.5.
- `prompts.py` — four-dimension system prompt (semantic consistency 0.4, over-clean detection 0.3, readability 0.15, structure 0.15) with few-shot examples; calls `debias.py` to append anti-bias instructions at runtime.
- `models.py` — all Pydantic v2 request/response models. `EvalRequest` has a `stabilize` flag that triggers multi-sample averaging in `routes.py`.
- `calibration.py` — computes Pearson/Spearman correlation, MAE, RMSE, consistency rate against `human_label.overall_score` on each request.
- `stability.py` — runs `evaluate()` N times (temperature=0.0 throughout) and reports variance; stable threshold is variance < 0.005.
- `reporter.py` — orchestrates the full acceptance report: evaluation + calibration + stability + flaw metrics + anchor accuracy + bias detection.
- `metrics.py` — flaw detection Precision/Recall/F1 and anchor location accuracy (with configurable char tolerance, default 10).
- `debias.py` — generates anti-length-bias and anti-position-bias prompt supplements; also computes bias statistics from flaw lists.

### LLM configuration

The backend reads three env vars: `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` (default `https://api.deepseek.com/v1`), `DEEPSEEK_MODEL` (default `deepseek-chat`). These are loaded via `python-dotenv` from `.env` at startup.

The Streamlit app hardcodes `deepseek-chat` and `https://api.deepseek.com/v1` and takes the key from the sidebar input (falling back to `DEEPSEEK_API_KEY` env var).
