"""
run_rpa.py — RPA Pipeline Runner
=================================
Entry point script to execute the full RPA Loan Processing Bot pipeline.

Usage:
    python run_rpa.py
    python run_rpa.py --input path/to/custom_applications.csv

The runner:
    1. Parses command-line arguments
    2. Validates the environment (dependencies, directories)
    3. Delegates to the core rpa_loan_bot module
    4. Prints a final status banner
    5. Returns exit code 0 on success, 1 on failure
"""

import sys
import time
import argparse
import traceback
import importlib
import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES = {"pandas": "pandas"}

def check_dependencies() -> None:
    """Verify that all required third-party packages are importable."""
    missing = []
    for pkg, import_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n[ERROR] Missing required packages: {missing}")
        print(f"        Run: pip install {' '.join(missing)}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Banner helpers
# ---------------------------------------------------------------------------

WIDTH = 72

def banner(text: str, char: str = "=") -> None:
    print(char * WIDTH)
    padding = (WIDTH - len(text) - 2) // 2
    print(f"{char}{' ' * padding}{text}{' ' * (WIDTH - padding - len(text) - 2)}{char}")
    print(char * WIDTH)


def header() -> None:
    print()
    banner("FIRST NATIONAL BANK — RPA LOAN PROCESSING SYSTEM")
    print(f"  Version  : 1.0.0")
    print(f"  Module   : rpa_loan_bot")
    print(f"  Started  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*WIDTH}\n")


def footer(summary: dict, elapsed: float, success: bool) -> None:
    status = "SUCCESS" if success else "FAILURE"
    print(f"\n{'='*WIDTH}")
    print(f"  PIPELINE STATUS : {status}")
    print(f"  Elapsed Time    : {elapsed:.2f} seconds")
    if success:
        print(f"\n  KEY METRICS")
        print(f"  {'─'*40}")
        print(f"  Applications submitted   : {summary.get('total_applications', 'N/A')}")
        print(f"  Approved                 : {summary.get('approved', 'N/A')}")
        print(f"  Rejected                 : {summary.get('rejected', 'N/A')}")
        print(f"  Errors / Invalid data    : {summary.get('errors', 'N/A')}")
        print(f"  Approval rate            : {summary.get('approval_rate_pct', 'N/A')}%")
        total_val = summary.get('total_loan_value_approved', 0)
        print(f"  Total approved loan value: ${total_val:,.2f}")
    print(f"{'='*WIDTH}\n")


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RPA Loan Processing Bot — Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_rpa.py
  python run_rpa.py --input /data/applications.csv
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to the input CSV file (default: sample_applications.csv in same directory)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Environment Validation
# ---------------------------------------------------------------------------

def validate_environment(input_path: Path) -> None:
    """Check that required input files and directories are accessible."""
    print(f"  [ENV] Checking environment...")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"       Ensure 'sample_applications.csv' is in the same directory as this script,\n"
            f"       or pass --input <path> with a valid file."
        )

    # Ensure output directories exist
    base = Path(__file__).parent
    (base / "emails").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)

    print(f"  [ENV] Input file   : {input_path.name}  ✓")
    print(f"  [ENV] Output dirs  : emails/, logs/  ✓")
    print(f"  [ENV] Dependencies : pandas ✓")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """
    Orchestrate the RPA pipeline run.

    Returns:
        int: Exit code (0 = success, 1 = failure)
    """
    args = parse_args()

    # Check packages first, before any other import that might fail
    check_dependencies()

    # Now safe to import the bot module
    import rpa_loan_bot  # noqa: E402 — intentional late import

    # Resolve input path
    base_dir   = Path(__file__).parent
    input_file = Path(args.input) if args.input else base_dir / "sample_applications.csv"

    header()

    # Validate environment
    try:
        validate_environment(input_file)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        return 1

    # Execute pipeline
    t0      = time.time()
    success = False
    summary = {}

    try:
        print(f"  [RUN] Launching RPA bot pipeline...\n")
        summary = rpa_loan_bot.run(input_file=input_file)
        success = True

    except FileNotFoundError as exc:
        print(f"\n[ERROR] Input file error: {exc}")
        traceback.print_exc()

    except ValueError as exc:
        print(f"\n[ERROR] Data validation error: {exc}")
        traceback.print_exc()

    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n[ERROR] Unexpected pipeline failure: {exc}")
        traceback.print_exc()

    finally:
        elapsed = time.time() - t0
        footer(summary, elapsed, success)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
