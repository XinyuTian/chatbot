import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hello API")

STATIC_DIR = Path(__file__).resolve().parent / "static"

AI_BUILDERS_BASE_URL = "https://space.ai-builders.com/backend"


@app.get("/", include_in_schema=False)
async def chat_page() -> FileResponse:
    """Serve the chat web UI (ChatGPT-style)."""
    return FileResponse(STATIC_DIR / "index.html")


class HelloRequest(BaseModel):
    """Request body for POST hello endpoint."""

    name: str = Field(..., examples=["Alice"], description="Your name")


class HelloResponse(BaseModel):
    """Response body for hello endpoints."""

    message: str = Field(..., examples=["Hello Alice"], description="Greeting message")


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    prompt: str = Field(..., examples=["Write a short haiku about coding."])
    system_prompt: str | None = Field(
        default=None,
        examples=["You are a concise assistant."],
        description="Optional system instruction.",
    )


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""

    message: str = Field(..., description="Assistant response text")
    model: str = Field(..., examples=["gpt-5"])
    used_search: bool = Field(
        False,
        description="True if a web search was run and the reply used those results.",
    )
    search_query: str | None = Field(
        None,
        description="Search phrase sent to the search API when used_search is true.",
    )


PLANNER_SYSTEM = """You route user requests. Decide if answering needs fresh web search
(facts after your knowledge cutoff, current events, prices, "latest", specific URLs, etc.).

Reply with ONLY valid JSON (no markdown fences, no other text) using exactly these keys:
- "need_search": boolean
- "search_query": string or null — short Tavily keyword phrase if need_search is true
- "direct_answer": string or null — if need_search is false, your complete answer here;
  if need_search is true, use null"""


class SearchRequest(BaseModel):
    """Request body for search endpoint."""

    prompt: str = Field(
        ...,
        examples=["latest Python 3.13 release notes"],
        description="Search query; sent as a keyword to AI Builders search API.",
    )
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Max results per keyword (1–20). Defaults to provider default if omitted.",
    )


@app.get(
    "/hello",
    response_model=HelloResponse,
    summary="Say hello with GET",
    response_description="JSON with a greeting for the given name",
)
def hello_get(
    name: str = Query(..., description="Your name", examples=["Alice"]),
) -> HelloResponse:
    return HelloResponse(message=f"Hello {name}")


@app.post(
    "/hello",
    response_model=HelloResponse,
    summary="Say hello with POST",
    response_description="JSON with a greeting for the given name",
)
def hello_post(payload: HelloRequest) -> HelloResponse:
    return HelloResponse(message=f"Hello {payload.name}")


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    return ""


def _truncate(text: str, max_len: int = 800) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n… ({len(text) - max_len} more chars)"


def _summarize_search_payload(data: dict[str, Any], preview_len: int = 1200) -> str:
    lines: list[str] = []
    if "combined_answer" in data and data["combined_answer"]:
        ca = str(data["combined_answer"])
        lines.append(f"combined_answer ({len(ca)} chars): {_truncate(ca, 400)}")
    queries = data.get("queries")
    if isinstance(queries, list):
        lines.append(f"queries: {len(queries)} keyword(s)")
        for i, q in enumerate(queries[:3]):
            if isinstance(q, dict):
                kw = q.get("keyword", "?")
                resp = q.get("response")
                if isinstance(resp, dict):
                    lines.append(
                        f"  [{i}] keyword={kw!r} response_keys={list(resp.keys())}"
                    )
    errors = data.get("errors")
    if errors:
        lines.append(f"errors: {errors!r}")
    raw = json.dumps(data, ensure_ascii=False, indent=2)
    lines.append(f"raw_json_chars={len(raw)}")
    lines.append(f"preview:\n{_truncate(raw, preview_len)}")
    return "\n".join(lines)


def _parse_planner_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


async def _chat_complete(
    client: httpx.AsyncClient,
    api_key: str,
    messages: list[dict[str, Any]],
    *,
    log_label: str = "chat_completions",
) -> dict[str, Any]:
    payload = {
        "model": "gpt-5",
        "messages": messages,
        "temperature": 1.0,
    }
    logger.info(
        "[%s] LLM POST /v1/chat/completions model=%s messages=%s",
        log_label,
        payload["model"],
        len(messages),
    )
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, str):
            logger.info(
                "[%s]   msg[%s] role=%s chars=%s preview=%r",
                log_label,
                i,
                role,
                len(content),
                _truncate(content, 300),
            )
        else:
            logger.info("[%s]   msg[%s] role=%s content=<non-string>", log_label, i, role)

    response = await client.post(
        f"{AI_BUILDERS_BASE_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
    )
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage")
    logger.info(
        "[%s] LLM response model=%s usage=%s",
        log_label,
        data.get("model"),
        usage,
    )
    return data


async def _ai_builders_search(
    client: httpx.AsyncClient,
    api_key: str,
    query: str,
    max_results: int = 6,
    *,
    log_label: str = "search_tool",
) -> dict[str, Any]:
    body = {"keywords": [query.strip()], "max_results": max_results}
    logger.info(
        "[%s] TOOL search POST /v1/search/ params=%s",
        log_label,
        json.dumps(body, ensure_ascii=False),
    )
    response = await client.post(
        f"{AI_BUILDERS_BASE_URL}/v1/search/",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
    )
    response.raise_for_status()
    out = response.json()
    logger.info(
        "[%s] TOOL search result summary:\n%s",
        log_label,
        _summarize_search_payload(out),
    )
    return out


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Agentic chat (one round): plan → optional search → answer",
    response_description="GPT-5 decides whether to search; at most one search, then final reply.",
)
async def chat(payload: ChatRequest) -> ChatResponse:
    api_key = os.getenv("AI_BUILDERS_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Missing AI_BUILDERS_API_KEY environment variable.",
        )

    logger.info("========== POST /chat ==========")
    logger.info("user prompt (%s chars): %s", len(payload.prompt), _truncate(payload.prompt, 500))
    if payload.system_prompt:
        logger.info(
            "system_prompt (%s chars): %s",
            len(payload.system_prompt),
            _truncate(payload.system_prompt, 300),
        )

    planner_system = PLANNER_SYSTEM
    if payload.system_prompt:
        planner_system = f"{payload.system_prompt}\n\n{PLANNER_SYSTEM}"

    planner_messages: list[dict[str, Any]] = [
        {"role": "system", "content": planner_system},
        {"role": "user", "content": payload.prompt},
    ]

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            data = await _chat_complete(
                client,
                api_key,
                planner_messages,
                log_label="planner",
            )
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"AI Builders API error: {exc.response.text}",
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Network error: {exc}") from exc

    model = data.get("model", "gpt-5")
    raw_planner = _message_text(data.get("choices", [{}])[0].get("message", {}))
    if not raw_planner:
        raise HTTPException(status_code=502, detail="Planner returned empty content.")

    logger.info("planner raw assistant text (%s chars):\n%s", len(raw_planner), _truncate(raw_planner, 1200))

    try:
        plan = _parse_planner_json(raw_planner)
    except (json.JSONDecodeError, TypeError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="Planner did not return valid JSON.",
        ) from None

    need_search = bool(plan.get("need_search"))
    search_query = plan.get("search_query")
    direct_answer = plan.get("direct_answer")

    logger.info(
        "planner parsed: need_search=%s search_query=%r direct_answer_len=%s",
        need_search,
        search_query,
        len(direct_answer) if isinstance(direct_answer, str) else None,
    )

    if not need_search:
        if isinstance(direct_answer, str) and direct_answer.strip():
            logger.info("path: no search — returning direct_answer from planner")
            return ChatResponse(
                message=direct_answer.strip(),
                model=model,
                used_search=False,
                search_query=None,
            )
        logger.warning("path: no search but no direct_answer — returning raw planner text")
        return ChatResponse(
            message=raw_planner,
            model=model,
            used_search=False,
            search_query=None,
        )

    if not isinstance(search_query, str) or not search_query.strip():
        raise HTTPException(
            status_code=502,
            detail="Planner requested search but did not provide search_query.",
        )
    q = search_query.strip()

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            search_results = await _ai_builders_search(
                client, api_key, q, log_label="planner→search"
            )
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Search API error: {exc.response.text}",
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Search network error: {exc}") from exc

        results_blob = json.dumps(search_results, ensure_ascii=False, indent=2)
        if len(results_blob) > 24_000:
            results_blob = results_blob[:24_000] + "\n…[truncated]"

        logger.info(
            "synthesis input: results_blob_chars=%s (sent to LLM)",
            len(results_blob),
        )

        synth_system = (
            "You already have web search results below. Answer the user's question "
            "directly and accurately. Use the results; do not ask to search again or "
            "claim you will use tools. If results are insufficient, say so briefly."
        )
        if payload.system_prompt:
            synth_system = f"{payload.system_prompt}\n\n{synth_system}"

        synth_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": synth_system,
            },
            {
                "role": "user",
                "content": (
                    f"User question:\n{payload.prompt}\n\n"
                    f"Search results (JSON):\n{results_blob}"
                ),
            },
        ]

        try:
            synth_data = await _chat_complete(
                client,
                api_key,
                synth_messages,
                log_label="synthesis",
            )
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"AI Builders API error: {exc.response.text}",
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Network error: {exc}") from exc

    final_text = _message_text(synth_data.get("choices", [{}])[0].get("message", {}))
    if not final_text:
        raise HTTPException(status_code=502, detail="Synthesis returned empty content.")

    logger.info(
        "final assistant message (%s chars): %s",
        len(final_text),
        _truncate(final_text, 600),
    )
    logger.info("========== /chat done (used_search=True) ==========")

    return ChatResponse(
        message=final_text.strip(),
        model=synth_data.get("model", model),
        used_search=True,
        search_query=q,
    )


@app.post(
    "/search",
    summary="Web search via AI Builders",
    response_description="Raw JSON from AI Builders POST /v1/search/",
)
async def search(payload: SearchRequest) -> dict[str, Any]:
    """Proxy to AI Builders Tavily search; maps `prompt` to a single keyword."""
    api_key = os.getenv("AI_BUILDERS_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Missing AI_BUILDERS_API_KEY environment variable.",
        )

    q = payload.prompt.strip()
    if not q:
        raise HTTPException(status_code=422, detail="prompt must not be empty.")

    body: dict[str, Any] = {"keywords": [q]}
    if payload.max_results is not None:
        body["max_results"] = payload.max_results

    logger.info("========== POST /search ==========")
    logger.info("POST /search request body: %s", json.dumps(body, ensure_ascii=False))

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            response = await client.post(
                f"{AI_BUILDERS_BASE_URL}/v1/search/",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"AI Builders API error: {exc.response.text}",
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Network error: {exc}") from exc

    out = response.json()
    logger.info("POST /search response summary:\n%s", _summarize_search_payload(out))
    logger.info("========== /search done ==========")
    return out
