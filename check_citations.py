"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3]  [1, 2]  [1-3]  [2-3]; not [3](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            if part.isdigit():
                numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]

    seen_urls = {}
    source_by_n = {}
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"source entry {entry!r} is not an object")
            continue
        n = entry.get("n")
        if not isinstance(n, int):
            problems.append(f"source n must be an int, got {n!r}")
        elif n in source_by_n:
            problems.append(f"duplicate source number: {n}")
        else:
            source_by_n[n] = entry

        url = entry.get("url")
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source n={n!r} url must start with http:// or https://, got {url!r}")
        elif url in seen_urls:
            problems.append(f"the same url appears twice: {url}")
        else:
            seen_urls[url] = n

    ref_matches = list(_REF_HEADING.finditer(report_text))
    if not ref_matches:
        problems.append("missing '## References' heading")
        body = report_text
        ref_text = ""
    else:
        body = report_text[:ref_matches[-1].start()]
        ref_text = report_text[ref_matches[-1].end():]

    # Find citations in the body only (excluding code blocks and markdown links)
    cited = set()
    segments = _CODE.split(body)
    for i, segment in enumerate(segments):
        if i % 2:  # inside code block
            continue
        for match in _GROUP.finditer(segment):
            for num in _group_numbers(match.group(1)):
                cited.add(num)

    # Check cited against sources
    for n in sorted(cited):
        if n not in source_by_n:
            problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(source_by_n.keys()):
        if n not in cited:
            problems.append(f"source [{n}] never cited")

    # If ## References exists, check reference lines
    if ref_matches:
        ref_lines = []
        for line in ref_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            m = re.match(r"^\[(\d+)\]", stripped)
            if m:
                ref_lines.append((int(m.group(1)), stripped))

        ref_by_n = {}
        ref_counts = {}
        for num, line in ref_lines:
            ref_counts[num] = ref_counts.get(num, 0) + 1
            ref_by_n.setdefault(num, []).append(line)

        for num, count in ref_counts.items():
            if count > 1:
                problems.append(f"reference [{num}] appears {count} times in References")
            if num not in source_by_n:
                problems.append(f"reference [{num}] is not a valid source")

        for num in sorted(source_by_n.keys()):
            if num not in ref_counts:
                problems.append(f"source [{num}] has no reference line in References")

        for num, lines in ref_by_n.items():
            if num in source_by_n:
                expected_url = source_by_n[num].get("url")
                for line in lines:
                    urls = [u.rstrip(".,;)") for u in re.findall(r"https?://[^\s)\]>]+", line)]
                    if len(urls) != 1:
                        problems.append(f"reference line [{num}] must contain exactly one URL, found {len(urls)}")
                    elif expected_url and urls[0] != expected_url:
                        problems.append(f"reference line [{num}] URL does not match source url: {urls[0]!r} != {expected_url!r}")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
