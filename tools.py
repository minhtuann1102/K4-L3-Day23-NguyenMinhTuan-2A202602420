"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import random
import re
import time  # noqa: F401
import xml.etree.ElementTree as ET

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

_LAST_ARXIV_CALL = 0.0


def _redact(text: str) -> str:
    key = os.getenv("EXA_API_KEY", "").strip()
    if key and key in text:
        text = text.replace(key, "[REDACTED]")
    return text


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            is_retryable = False
            retry_after = None

            if isinstance(e, RetryableError):
                is_retryable = True
                retry_after = e.retry_after
            elif isinstance(e, httpx.HTTPStatusError):
                if e.response.status_code in {429, 500, 502, 503, 504}:
                    is_retryable = True
                    ra = e.response.headers.get("retry-after")
                    if ra:
                        try:
                            retry_after = float(ra)
                        except (ValueError, TypeError):
                            pass
            elif isinstance(e, httpx.TransportError):
                is_retryable = True

            if not is_retryable or attempt == attempts - 1:
                raise

            if retry_after is not None:
                delay = min(cap, max(0.0, retry_after))
            else:
                backoff = base * (2 ** attempt)
                jitter = random.uniform(0.0, 0.5 * base)
                delay = min(cap, backoff + jitter)

            time.sleep(delay)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _LAST_ARXIV_CALL
    try:
        terms = re.findall(r"[A-Za-z0-9\-]+", query)
        if not terms:
            return "NO RESULTS"
        search_query = " AND ".join(f"all:{t}" for t in terms)

        elapsed = time.monotonic() - _LAST_ARXIV_CALL
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        _LAST_ARXIV_CALL = time.monotonic()

        clamped_max = max(1, min(int(max_results), 30))
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": clamped_max,
        }

        def _call():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(ARXIV_URL, params=params)
                resp.raise_for_status()
                return resp.text

        xml_text = with_retry(_call, attempts=5, base=2.0, cap=60.0)

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "NO RESULTS"

        records = []
        for entry in entries:
            raw_id = entry.findtext("atom:id", "", ns).strip()
            arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
            clean_id = re.sub(r"v\d+$", "", arxiv_id)
            raw_title = entry.findtext("atom:title", "", ns)
            title = " ".join(raw_title.split())
            if not clean_id or title.lower() == "error":
                continue
            url = f"https://arxiv.org/abs/{clean_id}"
            published = entry.findtext("atom:published", "", ns).strip()[:10]
            raw_summary = entry.findtext("atom:summary", "", ns)
            summary = " ".join(raw_summary.split())[:600]

            records.append({
                "id": clean_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _redact(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        clamped_limit = max(1, min(int(limit), 100))
        params = {"limit": clamped_limit}
        if date and date.strip():
            params["date"] = date.strip()

        def _call():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(HF_DAILY_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        data = with_retry(_call, attempts=5, base=1.0, cap=30.0)
        if not isinstance(data, list) or not data:
            return "NO RESULTS"

        records = []
        for item in data:
            p = item.get("paper") if isinstance(item, dict) and isinstance(item.get("paper"), dict) else item
            paper_id = p.get("id") or item.get("id")
            if not paper_id:
                continue
            title = " ".join(str(p.get("title") or item.get("title") or "").split())
            summary = " ".join(str(p.get("summary") or item.get("summary") or "").split())[:600]
            published = str(p.get("publishedAt") or item.get("publishedAt") or "")[:10]
            upvotes = int(p.get("upvotes") or item.get("upvotes") or 0)
            github = str(p.get("githubRepo") or item.get("githubRepo") or "")
            stars = int(p.get("githubStars") or item.get("githubStars") or 0)

            records.append({
                "id": str(paper_id),
                "url": f"https://huggingface.co/papers/{paper_id}",
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if keyword and keyword.strip():
            kw = keyword.strip().lower()
            records = [r for r in records if kw in r["title"].lower() or kw in r["summary"].lower()]

        records.sort(key=lambda r: r.get("upvotes", 0), reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _redact(f"ERROR: {type(exc).__name__}: {exc}")


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        clean_q = query.strip()
        if not clean_q:
            return "NO RESULTS"
        clamped_limit = max(1, min(int(limit), 50))
        params = {"q": clean_q, "limit": clamped_limit}

        def _call():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(HF_SEARCH_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        data = with_retry(_call, attempts=5, base=1.0, cap=30.0)
        if not isinstance(data, list) or not data:
            return "NO RESULTS"

        records = []
        for item in data:
            p = item.get("paper") if isinstance(item, dict) and isinstance(item.get("paper"), dict) else item
            paper_id = p.get("id") or item.get("id")
            if not paper_id:
                continue
            title = " ".join(str(p.get("title") or item.get("title") or "").split())
            raw_summary = p.get("ai_summary") or p.get("summary") or item.get("summary") or ""
            summary = " ".join(str(raw_summary).split())[:600]
            published = str(p.get("publishedAt") or item.get("publishedAt") or "")[:10]
            upvotes = int(p.get("upvotes") or item.get("upvotes") or 0)
            github = str(p.get("githubRepo") or item.get("githubRepo") or "")
            stars = int(p.get("githubStars") or item.get("githubStars") or 0)

            records.append({
                "id": str(paper_id),
                "url": f"https://huggingface.co/papers/{paper_id}",
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _redact(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _call_exa_mcp(tool_name: str, arguments: dict) -> str:
    api_key = os.getenv("EXA_API_KEY", "").strip()
    url = f"{EXA_URL}?exaApiKey={api_key}" if api_key else EXA_URL
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }

    def _request():
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code == 429:
                ra = resp.headers.get("retry-after")
                try:
                    retry_sec = float(ra) if ra else 20.0
                except (ValueError, TypeError):
                    retry_sec = 20.0
                raise RetryableError("Exa rate limited (HTTP 429)", retry_after=min(30.0, retry_sec))
            resp.raise_for_status()
            text = resp.text

            data = None
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    raw = line[5:].strip()
                    if raw:
                        try:
                            data = json.loads(raw)
                            break
                        except json.JSONDecodeError:
                            pass
            if data is None:
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as err:
                    raise RuntimeError(f"Invalid JSON from Exa MCP: {err}")

            if "error" in data:
                err_msg = str(data["error"].get("message") if isinstance(data["error"], dict) else data["error"])
                if "rate limit" in err_msg.lower():
                    raise RetryableError("Exa rate limited", retry_after=20.0)
                raise RuntimeError(err_msg)

            result = data.get("result", {})
            meta = result.get("_meta", {})
            if meta and any("rate" in str(k).lower() or "limit" in str(k).lower() for k in meta.keys()):
                raise RetryableError("Exa rate limited in meta", retry_after=20.0)

            content = result.get("content", [])
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    t = item.get("text", "")
                    if "rate limit" in t.lower() and "dashboard.exa.ai" in t.lower():
                        raise RetryableError("Exa rate limited in content text", retry_after=20.0)
                    texts.append(t)

            return "\n\n".join(texts).strip()

    return with_retry(_request, attempts=5, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        clean_q = query.strip()
        if not clean_q:
            return "NO RESULTS"
        obj = objective.strip() or f"Find research and survey papers about {clean_q}"
        num = max(1, min(int(num_results), 10))
        text = _call_exa_mcp("web_search_exa", {"query": clean_q, "objective": obj, "numResults": num})
        return text if text else "NO RESULTS"
    except Exception as exc:
        return _redact(f"ERROR: {type(exc).__name__}: {exc}")


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        clean_url = url.strip()
        if not clean_url:
            return "NO RESULTS"
        text = _call_exa_mcp("web_fetch_exa", {"urls": [clean_url]})
        if not text:
            return "NO RESULTS"
        return text[:12000]
    except Exception as exc:
        return _redact(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
