from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .m3_harness import aggregate_by_buckets, build_default_m3_cases, run_teacher_benchmark


def main() -> None:
    cases = build_default_m3_cases()
    rows = run_teacher_benchmark(cases)
    bucket_summary = aggregate_by_buckets(rows)

    # Resolve to repo root regardless of working directory
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "artifacts" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "case_count": len(cases),
        "row_count": len(rows),
        "bucket_summary": [asdict(item) for item in bucket_summary],
    }
    (out_dir / "m3_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
