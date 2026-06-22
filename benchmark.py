

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from config import OUTPUT_DIR, REPORT_DIR, ensure_project_dirs


FULL_BENCHMARK_COMMANDS = [
    ["recommender_item_cf.py"],
    ["recommender_lsh.py", "--skip_brute_force"],
    ["recommender_mf.py"],
    ["recommender_graph.py"],
    ["recommender_diversity_coldstart.py"],
    ["recommender_api.py", "--eval-only"],
]


def run_command(args: list[str]) -> tuple[int, float]:
    
    t0 = time.perf_counter()
    completed = subprocess.run([sys.executable, *args], check=False)
    return completed.returncode, time.perf_counter() - t0


def write_final_report(rows: list[dict[str, object]], quick: bool) -> Path:
    
    ensure_project_dirs()
    report_path = REPORT_DIR / "final_evaluation_report.md"
    status = "PASS" if all(row["exit_code"] == 0 for row in rows) else "FAIL"
    mode = "quick API validation" if quick else "full task benchmark"

    lines = [
        "# Week 4 Movie Recommendation System Remediation Report",
        "",
        f"Mode: {mode}",
        f"Final status: {status}",
        "",
        "| Check | Exit Code | Seconds |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['name']} | {row['exit_code']} | {row['seconds']:.2f} |")
    lines.extend(
        [
            "",
            "Validation scope:",
            "- Leakage-safe train/validation/test tuning for Item-CF, MF, and MMR.",
            "- Cross-platform LSH output and dependency coverage.",
            "- Portable outputs/ and artifacts/ directories.",
            "- BFS and Personalized PageRank graph metrics.",
            "- FastAPI schema validation, cache benchmark, and endpoint tests.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    
    parser = argparse.ArgumentParser(description="Run project remediation benchmarks")
    parser.add_argument("--full", action="store_true", help="Run all task scripts; otherwise run API eval only")
    args = parser.parse_args()

    commands = FULL_BENCHMARK_COMMANDS if args.full else [["recommender_api.py", "--eval-only"]]
    rows: list[dict[str, object]] = []
    for command in commands:
        exit_code, seconds = run_command(command)
        rows.append({"name": " ".join(command), "exit_code": exit_code, "seconds": seconds})
        if exit_code != 0:
            break

    report_path = write_final_report(rows, quick=not args.full)
    print(f"Benchmark report written to {report_path}")
    sys.exit(0 if all(row["exit_code"] == 0 for row in rows) else 1)


if __name__ == "__main__":
    main()

