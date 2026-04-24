"""
Full RPA Pipeline Runner
========================
Orchestrates the complete two-phase RPA loan processing pipeline:

  Phase 1 — UI Automation (rpa_web_bot.py)
      • Launches a headless Chrome browser
      • Opens the First National Bank online loan portal (loan_form.html)
      • Fills in each loan application form field with human-like typing
      • Takes "before" and "after" screenshots for every application
      • Demonstrates real Robotic Process Automation on a live web UI

  Phase 2 — Back-end Processing (rpa_loan_bot.py)
      • Ingests the CSV application file
      • Validates applicant data and calls the credit bureau API (simulated)
      • Applies underwriting business rules (DTI, LTV, credit score, employment)
      • Generates approval / rejection decisions with narrative reasons
      • Creates email notification files for each applicant
      • Writes disbursement records for approved loans
      • Outputs processed_applications.csv with full audit trail

Usage:
    python run_full_rpa.py

Author: RPA Automation Team
Version: 2.0.0
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent

PHASE1_SCRIPT = BASE_DIR / "rpa_web_bot.py"
PHASE2_SCRIPT = BASE_DIR / "rpa_loan_bot.py"


def banner(title: str, char: str = "=", width: int = 70) -> None:
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def run_phase(label: str, script: Path) -> dict:
    """Run a phase script and return timing + exit-code info."""
    banner(f"PHASE: {label}", "─")
    print(f"  Script : {script}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print()

    t_start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(BASE_DIR),
    )
    elapsed = round(time.perf_counter() - t_start, 2)

    status = "SUCCESS" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
    print()
    print(f"  Status : {status}")
    print(f"  Elapsed: {elapsed} s")
    return {"label": label, "elapsed": elapsed, "status": status, "returncode": result.returncode}


def main() -> None:
    pipeline_start = time.perf_counter()

    banner("FULL RPA PIPELINE — First National Bank Loan Processing")
    print("  This runner executes both phases of the RPA solution:")
    print("    Phase 1 → Browser UI automation (Selenium + headless Chrome)")
    print("    Phase 2 → Back-end decisioning (credit rules + email generation)")
    print()
    print(f"  Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    phases = []

    # ── Phase 1: UI Automation ────────────────────────────────────────────────
    p1 = run_phase("UI Automation — Web Form Bot (Selenium)", PHASE1_SCRIPT)
    phases.append(p1)

    if p1["returncode"] != 0:
        print("  WARNING: Phase 1 encountered errors. Continuing to Phase 2 …")

    # ── Phase 2: Back-end Processing ──────────────────────────────────────────
    p2 = run_phase("Back-end Processing — Loan Decision Engine", PHASE2_SCRIPT)
    phases.append(p2)

    # ── Pipeline summary ──────────────────────────────────────────────────────
    total_elapsed = round(time.perf_counter() - pipeline_start, 2)

    banner("COMBINED PIPELINE SUMMARY")
    print(f"  {'Phase':<50} {'Status':<20} {'Time':>8}")
    print("  " + "-" * 80)
    for ph in phases:
        print(f"  {ph['label']:<50} {ph['status']:<20} {ph['elapsed']:>6.2f}s")
    print("  " + "-" * 80)
    print(f"  {'TOTAL PIPELINE':<50} {'':20} {total_elapsed:>6.2f}s")

    # Screenshot count
    ss_dir = BASE_DIR / "screenshots"
    if ss_dir.exists():
        shots = list(ss_dir.glob("*.png"))
        print()
        print(f"  Screenshots captured : {len(shots)}")
        for s in sorted(shots):
            print(f"    • {s.name}")

    # Outputs
    print()
    print("  Key output files:")
    outputs = [
        BASE_DIR / "processed_applications.csv",
        BASE_DIR / "disbursement_records.csv",
        BASE_DIR / "logs" / "audit_trail.log",
        BASE_DIR / "web_bot_output.txt",
    ]
    for out in outputs:
        exists = "✓" if out.exists() else "✗ (missing)"
        print(f"    {exists}  {out.relative_to(BASE_DIR)}")

    all_ok = all(ph["returncode"] == 0 for ph in phases)
    print()
    if all_ok:
        print("  ✓  All pipeline phases completed successfully.")
    else:
        print("  ⚠  One or more phases exited with errors — review output above.")

    banner("Pipeline complete.", "=")


if __name__ == "__main__":
    main()
