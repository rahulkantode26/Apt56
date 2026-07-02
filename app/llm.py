"""LLM explanation layer.

Turns the classifier's structured output into a plain-language explanation for
the user. The LLM never makes the classification itself — it only explains the
number the CNN already produced, and always appends a not-a-diagnosis
disclaimer.

Three backends, selected by the LLM_PROVIDER env var:
  - "local"     : a local model via an OpenAI-compatible endpoint (Ollama by
                  default; also works with LM Studio / llama.cpp).
  - "anthropic" : the Anthropic API (needs ANTHROPIC_API_KEY).
  - "none"      : always use the templated fallback.
  - "auto"      : (default) local if it's reachable, else Anthropic if a key is
                  set, else the template.

Any failure falls back to a templated explanation so the app still works — the
classifier is the core of the product; the LLM is an enhancement.

Local-model env vars:
  LOCAL_LLM_BASE_URL   default http://localhost:11434/v1   (Ollama)
  LOCAL_LLM_MODEL      default llama3.2
  LOCAL_LLM_API_KEY    default "ollama" (dummy; the endpoint ignores it)
"""

from __future__ import annotations

import os

import httpx

PROVIDER = os.environ.get("LLM_PROVIDER", "auto").lower()

ANTHROPIC_MODEL = os.environ.get("EXPLAINER_MODEL", "claude-opus-4-8")

LOCAL_BASE_URL = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
LOCAL_MODEL = os.environ.get("LOCAL_LLM_MODEL", "llama3.2")
LOCAL_API_KEY = os.environ.get("LOCAL_LLM_API_KEY", "ollama")

DISCLAIMER = (
    "This is an automated screening aid, not a medical diagnosis. Please see a "
    "qualified eye care professional for any concerns about your vision."
)

SYSTEM_PROMPT = (
    "You are a careful medical-screening assistant for an eye-photo app. You are "
    "given the output of an image classifier that has ALREADY analyzed a photo of "
    "a person's eye. Your job is ONLY to explain that result in plain, calm, "
    "non-alarming language for a layperson — you do NOT diagnose, and you must "
    "never contradict or second-guess the classifier's numbers. Keep it to 2–3 "
    "short sentences. Do not invent findings beyond what the scores say. Always be "
    "clear that this is a screening aid, not a diagnosis, and encourage seeing an "
    "eye professional if the result suggests a condition or the user is concerned."
)


def _templated_explanation(prediction: str, confidence: float, probabilities: dict) -> str:
    pct = f"{confidence:.0%}"
    if prediction.lower() == "cataract":
        body = (
            f"The image classifier flagged possible signs of a cataract with {pct} "
            "confidence. A cataract is a clouding of the eye's lens that can blur "
            "vision, and it is very treatable when caught by a professional."
        )
    else:
        body = (
            f"The image classifier did not find signs of a cataract in this photo "
            f"({pct} confidence it looks typical)."
        )
    return f"{body} {DISCLAIMER}"


def _user_message(prediction: str, confidence: float, probabilities: dict) -> str:
    scores = ", ".join(f"{k}: {v:.1%}" for k, v in probabilities.items())
    return (
        "Classifier result for the uploaded eye photo:\n"
        f"- Predicted class: {prediction}\n"
        f"- Confidence: {confidence:.1%}\n"
        f"- Full probabilities: {scores}\n\n"
        "Explain this to the user."
    )


def _ensure_disclaimer(text: str) -> str:
    if "not a" not in text.lower() and "screening" not in text.lower():
        return f"{text}\n\n{DISCLAIMER}"
    return text


def _explain_local(prediction: str, confidence: float, probabilities: dict) -> str:
    """Call a local model through an OpenAI-compatible chat endpoint."""
    resp = httpx.post(
        f"{LOCAL_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LOCAL_API_KEY}"},
        json={
            "model": LOCAL_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_message(prediction, confidence, probabilities)},
            ],
            "temperature": 0.3,
            "stream": False,
        },
        timeout=120.0,  # local models can be slow to load on the first request
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    if not text:
        raise ValueError("empty local LLM response")
    return _ensure_disclaimer(text)


def _explain_anthropic(prediction: str, confidence: float, probabilities: dict) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_message(prediction, confidence, probabilities)}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "").strip()
    if not text:
        raise ValueError("empty Anthropic response")
    return _ensure_disclaimer(text)


def _local_reachable() -> bool:
    try:
        httpx.get(f"{LOCAL_BASE_URL}/models", timeout=1.5)
        return True
    except Exception:  # noqa: BLE001
        return False


def _resolve_provider() -> str:
    if PROVIDER in ("local", "anthropic", "none"):
        return PROVIDER
    # auto: prefer a running local model, then Anthropic, then template.
    if _local_reachable():
        return "local"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    return "none"


def explain(prediction: str, confidence: float, probabilities: dict) -> dict:
    """Return {'explanation': str, 'source': 'local'|'llm'|'fallback'}."""
    provider = _resolve_provider()

    if provider == "local":
        try:
            return {"explanation": _explain_local(prediction, confidence, probabilities),
                    "source": "local"}
        except Exception as exc:  # noqa: BLE001
            print(f"[llm] local model failed, using template: {exc}")

    elif provider == "anthropic":
        try:
            return {"explanation": _explain_anthropic(prediction, confidence, probabilities),
                    "source": "llm"}
        except Exception as exc:  # noqa: BLE001
            print(f"[llm] Anthropic failed, using template: {exc}")

    return {"explanation": _templated_explanation(prediction, confidence, probabilities),
            "source": "fallback"}
