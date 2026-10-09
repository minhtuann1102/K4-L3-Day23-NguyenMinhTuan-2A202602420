"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


class ResearchProgressCallback(BaseCallbackHandler):
    """Prints live progress of tool calls and subagent activations to stdout."""

    def on_tool_start(self, serialized, input_str, **kwargs):
        try:
            name = serialized.get("name", "tool") if serialized else "tool"
            inp = (input_str or "").strip().replace("\n", " ")
            if len(inp) > 100:
                inp = inp[:97] + "..."
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] [TOOL START] {name}: {inp}", flush=True)
        except Exception:
            pass

    def on_tool_end(self, output, **kwargs):
        try:
            out_str = str(output or "").strip().replace("\n", " ")
            if len(out_str) > 80:
                out_str = out_str[:77] + "..."
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] [TOOL END] -> {out_str}", flush=True)
        except Exception:
            pass


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    if not topic or not topic.strip():
        return "topic"
    clean = re.sub(r"[^\w]+", "-", topic.strip().lower()).strip("-")
    clean = clean[:60].rstrip("-")
    return clean or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Conduct deep research and produce a comprehensive, verified literature survey report on the topic: '{topic}'.\n\n"
        "Requirements:\n"
        "1. Plan with write_todos and split the topic into at least 3 sub-questions.\n"
        "2. Delegate research for each sub-question in parallel to the researcher subagents using the task tool.\n"
        "3. Ensure the gathered research covers at least 3 distinct source families (arxiv, hf-daily, hf-search, web).\n"
        "4. Merge and write all sources into /tmp/work/research/sources.json.\n"
        "5. Draft the survey report in /tmp/work/report/report.md following REPORT_TEMPLATE.md with inline citations [n]. Do not write ## References manually.\n"
        "6. Execute finalize_citations.py and check_citations.py using the execute tool until validation passes.\n"
        "7. Spot-check 2-3 key claims with citation-checker."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}."""
    tool_calls = Counter()
    subagent_calls = 0
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        calls = getattr(msg, "tool_calls", None) or []
        for call in calls:
            cname = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if cname:
                tool_calls[cname] += 1
                if cname == "task":
                    subagent_calls += 1

        usage = getattr(msg, "usage_metadata", None)
        if isinstance(usage, dict):
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)

    return {
        "model": str(model_name),
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_calls),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes:
        raise RuntimeError("Report file is missing or empty in sandbox")

    report_text = report_bytes.decode("utf-8", errors="replace")
    if not report_text.strip():
        raise RuntimeError("Report file is empty")

    if not sources_bytes:
        raise RuntimeError("sources.json is missing in sandbox")

    try:
        sources = json.loads(sources_bytes.decode("utf-8"))
        if not isinstance(sources, list):
            raise ValueError("sources.json must be a list")
    except Exception as exc:
        raise RuntimeError(f"sources.json is invalid JSON: {exc}") from exc

    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)

    source_families = sorted(list({s.get("source") for s in sources if isinstance(s, dict) and s.get("source")}))
    meta = summarize(messages, elapsed, model_name)
    meta["topic"] = topic
    meta["n_sources"] = len(sources)
    meta["source_families"] = source_families

    sources_out = reports_dir / f"{slug}.sources.json"
    meta_out = reports_dir / f"{slug}.meta.json"
    report_out = reports_dir / f"{slug}.md"

    sources_out.write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
    meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    report_out.write_text(report_text, encoding="utf-8")

    return report_out


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = topic.strip()
    if not topic:
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2

    model = make_model()
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL", "model")
    start = time.monotonic()
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] === Starting research for: '{topic}' (model: {model_name}) ===", flush=True)

    try:
        print(f"[{time.strftime('%H:%M:%S')}] Provisioning Daytona sandbox...", flush=True)
        with open_sandbox() as backend:
            print(f"[{time.strftime('%H:%M:%S')}] Sandbox ready ({backend.id}). Setting up workspace...", flush=True)
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })
            print(f"[{time.strftime('%H:%M:%S')}] Launching Lead Research Agent...", flush=True)
            agent = build_lead_agent(backend, model)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000, "callbacks": [ResearchProgressCallback()]},
            )
            elapsed = time.monotonic() - start
            print(f"[{time.strftime('%H:%M:%S')}] Lead agent finished in {elapsed:.1f}s. Saving outputs...", flush=True)
            messages = result.get("messages", []) if isinstance(result, dict) else []
            report_path = save_outputs(backend, topic, messages, elapsed, model_name)
            print(f"[{time.strftime('%H:%M:%S')}] Report successfully saved to {report_path}", flush=True)
            return 0
    except RuntimeError as exc:
        print(f"[{time.strftime('%H:%M:%S')}] FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        print(f"[{time.strftime('%H:%M:%S')}] FAILED with unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
