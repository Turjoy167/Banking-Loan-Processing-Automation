"""
RPA Loan Processing Bot
=======================
Automates the end-to-end processing of banking loan applications.

Workflow:
    1. Ingest loan applications from CSV
    2. Validate required fields and data integrity
    3. Simulate credit bureau API call
    4. Apply underwriting business rules
    5. Generate approval/rejection decisions with reasons
    6. Create email notification files
    7. Write disbursement records for approved loans
    8. Output processed results to CSV
    9. Maintain a full audit log throughout

Author: RPA Automation Team
Version: 1.0.0
"""

import os
import csv
import json
import time
import random
import logging
import hashlib
import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "sample_applications.csv"
OUTPUT_FILE = BASE_DIR / "processed_applications.csv"
DISBURSEMENT_FILE = BASE_DIR / "disbursement_records.csv"
AUDIT_LOG_FILE = BASE_DIR / "logs" / "audit_trail.log"
EMAIL_DIR = BASE_DIR / "emails"

# Underwriting thresholds
MIN_CREDIT_SCORE = 650
MAX_DTI_RATIO = 0.43          # Debt-to-Income ratio ceiling
MIN_ANNUAL_INCOME = 30_000    # Absolute income floor
MAX_LOAN_TO_INCOME_RATIO = 5  # Loan amount cannot exceed 5× annual income
EMPLOYED_STATUSES = {"Employed", "Self-Employed"}

# Simulated interest rates by credit tier (annual percentage rate)
RATE_TABLE = {
    "Excellent": 0.0499,   # 760+
    "Good":      0.0699,   # 700–759
    "Fair":      0.0899,   # 650–699
}

# Required CSV columns
REQUIRED_COLUMNS = [
    "application_id", "applicant_name", "email",
    "annual_income", "loan_amount", "loan_purpose",
    "employment_status", "years_employed", "existing_debt",
    "credit_score",
]


# ---------------------------------------------------------------------------
# Logging / Audit Setup
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure dual-output logging: structured audit file + console."""
    AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    EMAIL_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rpa_loan_bot")
    logger.setLevel(logging.DEBUG)

    # File handler — full debug detail for audit trail
    fh = logging.FileHandler(AUDIT_LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Console handler — INFO level for operator visibility
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("  %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logging()


# ---------------------------------------------------------------------------
# Step 1 — Data Ingestion
# ---------------------------------------------------------------------------

def ingest_applications(filepath: Path) -> pd.DataFrame:
    """
    Read loan applications from a CSV file and return a DataFrame.

    Raises:
        FileNotFoundError: If the input CSV does not exist.
        ValueError: If required columns are missing.
    """
    logger.info(f"[INGEST] Reading applications from: {filepath.name}")
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.debug(f"[INGEST] Raw DataFrame shape: {df.shape}")

    # Verify schema
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Normalise string columns
    for col in ["applicant_name", "email", "employment_status", "loan_purpose"]:
        df[col] = df[col].astype(str).str.strip()

    logger.info(f"[INGEST] Loaded {len(df)} application(s) successfully.")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Field Validation
# ---------------------------------------------------------------------------

def validate_application(row: pd.Series) -> tuple[bool, list[str]]:
    """
    Validate a single loan application row for completeness and basic sanity.

    Returns:
        (is_valid: bool, errors: list[str])
    """
    errors: list[str] = []

    # Null / empty checks
    for col in REQUIRED_COLUMNS:
        val = row.get(col)
        if pd.isna(val) or str(val).strip() == "":
            errors.append(f"Missing value for '{col}'")

    if errors:
        return False, errors

    # Numeric range checks
    try:
        income = float(row["annual_income"])
        loan   = float(row["loan_amount"])
        debt   = float(row["existing_debt"])
        score  = int(row["credit_score"])
        yrs    = float(row["years_employed"])

        if income <= 0:
            errors.append("annual_income must be positive")
        if loan <= 0:
            errors.append("loan_amount must be positive")
        if debt < 0:
            errors.append("existing_debt cannot be negative")
        if not (300 <= score <= 850):
            errors.append(f"credit_score {score} out of valid range 300–850")
        if yrs < 0:
            errors.append("years_employed cannot be negative")

    except (ValueError, TypeError) as exc:
        errors.append(f"Non-numeric value in numeric field: {exc}")

    # E-mail basic format check
    email = str(row["email"])
    if "@" not in email or "." not in email.split("@")[-1]:
        errors.append(f"Invalid email format: {email}")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Step 3 — Credit Bureau API Simulation
# ---------------------------------------------------------------------------

def call_credit_bureau_api(application_id: str, applicant_name: str,
                            reported_score: int) -> dict:
    """
    Simulate a credit bureau API call.

    In production this would be an HTTP request to Experian / Equifax / TransUnion.
    Here we introduce a small deterministic jitter around the reported score to
    mimic real-world bureau variation, then return a structured response.

    Args:
        application_id: Unique application identifier.
        applicant_name:  Full name for the request payload.
        reported_score:  Credit score provided on the application form.

    Returns:
        dict with 'bureau_score', 'bureau_reference', 'timestamp', 'status'
    """
    logger.debug(f"[CREDIT API] Calling bureau for {application_id} ({applicant_name})")

    # Simulate network latency (0.05–0.15 s)
    time.sleep(random.uniform(0.05, 0.15))

    # Deterministic jitter: ±10 points seeded by application_id
    seed = int(hashlib.md5(application_id.encode()).hexdigest(), 16)
    rng  = random.Random(seed)
    jitter = rng.randint(-10, 10)
    bureau_score = max(300, min(850, reported_score + jitter))

    reference = hashlib.sha256(
        f"{application_id}{applicant_name}{bureau_score}".encode()
    ).hexdigest()[:16].upper()

    response = {
        "status":          "SUCCESS",
        "bureau_score":    bureau_score,
        "bureau_reference": f"CBR-{reference}",
        "timestamp":       datetime.datetime.utcnow().isoformat() + "Z",
        "agency":          "Simulated Credit Bureau",
    }
    logger.debug(f"[CREDIT API] Response for {application_id}: {response}")
    return response


# ---------------------------------------------------------------------------
# Step 4 — Underwriting / Business Rules Engine
# ---------------------------------------------------------------------------

def apply_business_rules(row: pd.Series, bureau_score: int) -> tuple[str, list[str], Optional[float]]:
    """
    Apply underwriting business rules to determine loan decision.

    Rules checked (in priority order):
        1. Employment status must be in approved set
        2. Annual income must meet minimum threshold
        3. Credit score must meet minimum floor
        4. Debt-to-Income ratio must be below ceiling
        5. Loan-to-Income ratio must be within bounds

    Args:
        row:          Application data row.
        bureau_score: Verified credit score from bureau API.

    Returns:
        (decision: str, reasons: list[str], interest_rate: float | None)
            decision is 'APPROVED' or 'REJECTED'
    """
    income      = float(row["annual_income"])
    loan_amount = float(row["loan_amount"])
    debt        = float(row["existing_debt"])
    emp_status  = str(row["employment_status"]).strip()

    rejection_reasons: list[str] = []

    # Rule 1 — Employment
    if emp_status not in EMPLOYED_STATUSES:
        rejection_reasons.append(
            f"Employment status '{emp_status}' does not meet requirement "
            f"(must be one of {sorted(EMPLOYED_STATUSES)})"
        )

    # Rule 2 — Minimum income
    if income < MIN_ANNUAL_INCOME:
        rejection_reasons.append(
            f"Annual income ${income:,.0f} below minimum threshold ${MIN_ANNUAL_INCOME:,.0f}"
        )

    # Rule 3 — Credit score floor
    if bureau_score < MIN_CREDIT_SCORE:
        rejection_reasons.append(
            f"Bureau credit score {bureau_score} below minimum required score {MIN_CREDIT_SCORE}"
        )

    # Rule 4 — Debt-to-Income ratio
    # DTI = (annual debt payments + new loan payment) / gross income
    # Approximate annual debt payment = existing_debt × 5% servicing rate
    annual_debt_service  = debt * 0.05
    annual_loan_payment  = loan_amount * 0.07   # approx annualised payment
    dti = (annual_debt_service + annual_loan_payment) / income if income > 0 else 1.0
    if dti > MAX_DTI_RATIO:
        rejection_reasons.append(
            f"Debt-to-Income ratio {dti:.2%} exceeds maximum allowed {MAX_DTI_RATIO:.0%}"
        )

    # Rule 5 — Loan-to-Income ratio
    lti = loan_amount / income if income > 0 else 999
    if lti > MAX_LOAN_TO_INCOME_RATIO:
        rejection_reasons.append(
            f"Loan-to-Income ratio {lti:.2f}× exceeds maximum allowed {MAX_LOAN_TO_INCOME_RATIO}×"
        )

    if rejection_reasons:
        return "REJECTED", rejection_reasons, None

    # Determine interest rate tier for approved applications
    if bureau_score >= 760:
        tier = "Excellent"
    elif bureau_score >= 700:
        tier = "Good"
    else:
        tier = "Fair"

    rate = RATE_TABLE[tier]
    approval_reasons = [
        f"Credit score {bureau_score} meets minimum requirement",
        f"DTI ratio {dti:.2%} within acceptable range",
        f"Employment status '{emp_status}' verified",
        f"Income ${income:,.0f} meets threshold",
        f"Assigned interest rate {rate:.2%} ({tier} tier)",
    ]
    return "APPROVED", approval_reasons, rate


# ---------------------------------------------------------------------------
# Step 5 — Email Generation
# ---------------------------------------------------------------------------

def _format_currency(amount: float) -> str:
    return f"${amount:,.2f}"


def generate_approval_email(row: pd.Series, bureau_score: int,
                             interest_rate: float, reasons: list[str]) -> str:
    """Generate a professional approval notification email body."""
    loan_amount  = float(row["loan_amount"])
    monthly_rate = interest_rate / 12
    n_payments   = 60  # 5-year term
    # Monthly payment formula: M = P * [r(1+r)^n] / [(1+r)^n - 1]
    monthly_payment = (
        loan_amount * monthly_rate * (1 + monthly_rate) ** n_payments
        / ((1 + monthly_rate) ** n_payments - 1)
    )
    total_repayable = monthly_payment * n_payments

    return f"""
================================================================================
  FIRST NATIONAL BANK — LOAN APPROVAL NOTIFICATION
================================================================================
Date:         {datetime.date.today().strftime('%B %d, %Y')}
Reference:    {row['application_id']}
To:           {row['applicant_name']}
Email:        {row['email']}
--------------------------------------------------------------------------------

Dear {row['applicant_name']},

We are pleased to inform you that your loan application has been APPROVED.

LOAN DETAILS
------------
  Loan Purpose      : {row['loan_purpose']}
  Loan Amount       : {_format_currency(loan_amount)}
  Annual Rate (APR) : {interest_rate:.2%}
  Loan Term         : 60 months (5 years)
  Monthly Payment   : {_format_currency(monthly_payment)}
  Total Repayable   : {_format_currency(total_repayable)}

APPROVAL BASIS
--------------
{chr(10).join('  • ' + r for r in reasons)}

NEXT STEPS
----------
  1. A loan agreement will be sent to your registered email within 2 business days.
  2. Please review, sign, and return the agreement.
  3. Funds will be disbursed within 3–5 business days of agreement receipt.

If you have any questions, please contact your loan officer or call 1-800-FNB-LOAN.

Sincerely,
Automated Underwriting Division
First National Bank

================================================================================
  This is an automated notification. Please do not reply to this message.
================================================================================
""".strip()


def generate_rejection_email(row: pd.Series, bureau_score: int,
                              reasons: list[str]) -> str:
    """Generate a professional rejection notification email body."""
    return f"""
================================================================================
  FIRST NATIONAL BANK — LOAN APPLICATION DECISION
================================================================================
Date:         {datetime.date.today().strftime('%B %d, %Y')}
Reference:    {row['application_id']}
To:           {row['applicant_name']}
Email:        {row['email']}
--------------------------------------------------------------------------------

Dear {row['applicant_name']},

Thank you for your interest in a loan from First National Bank. After careful
review of your application (Ref: {row['application_id']}), we regret to inform
you that we are unable to approve your request at this time.

REASON(S) FOR DECISION
-----------------------
{chr(10).join('  • ' + r for r in reasons)}

WHAT YOU CAN DO
---------------
  • Address the factors listed above and reapply after 90 days.
  • Contact a financial advisor for guidance on improving your credit profile.
  • Consider applying for a smaller loan amount if income/DTI was the concern.
  • Call 1-800-FNB-LOAN to speak with a lending specialist.

Under the Equal Credit Opportunity Act, you have the right to request the
specific reasons for this decision within 60 days of receiving this notice.

Sincerely,
Automated Underwriting Division
First National Bank

================================================================================
  This is an automated notification. Please do not reply to this message.
================================================================================
""".strip()


def save_email(application_id: str, decision: str, email_body: str) -> Path:
    """Persist email notification to disk."""
    filename = EMAIL_DIR / f"{application_id}_{decision.lower()}_email.txt"
    filename.write_text(email_body, encoding="utf-8")
    logger.debug(f"[EMAIL] Saved notification to: {filename.name}")
    return filename


# ---------------------------------------------------------------------------
# Step 6 — Disbursement Record
# ---------------------------------------------------------------------------

def write_disbursement_record(row: pd.Series, interest_rate: float,
                               bureau_score: int) -> dict:
    """
    Build a disbursement record for an approved loan.

    Returns the record dict (to be collected for batch CSV write).
    """
    loan_amount  = float(row["loan_amount"])
    monthly_rate = interest_rate / 12
    n_payments   = 60
    monthly_payment = (
        loan_amount * monthly_rate * (1 + monthly_rate) ** n_payments
        / ((1 + monthly_rate) ** n_payments - 1)
    )
    disbursement_date = datetime.date.today() + datetime.timedelta(days=5)

    record = {
        "application_id":       row["application_id"],
        "applicant_name":       row["applicant_name"],
        "loan_amount":          loan_amount,
        "interest_rate":        f"{interest_rate:.4f}",
        "term_months":          n_payments,
        "monthly_payment":      round(monthly_payment, 2),
        "total_repayable":      round(monthly_payment * n_payments, 2),
        "disbursement_date":    disbursement_date.isoformat(),
        "account_status":       "PENDING_SIGNATURE",
        "bureau_score_at_approval": bureau_score,
        "created_at":           datetime.datetime.utcnow().isoformat() + "Z",
    }
    return record


# ---------------------------------------------------------------------------
# Step 7 — Audit Trail Entry
# ---------------------------------------------------------------------------

def log_audit_event(application_id: str, event: str,
                    detail: Optional[str] = None) -> None:
    """
    Write a structured audit event to the logger (captured by file handler).

    Args:
        application_id: The application being processed.
        event:          Short event label (e.g. 'VALIDATION_PASS').
        detail:         Optional additional detail.
    """
    msg = f"[AUDIT] AppID={application_id} | Event={event}"
    if detail:
        msg += f" | Detail={detail}"
    logger.debug(msg)


# ---------------------------------------------------------------------------
# Main Processing Pipeline
# ---------------------------------------------------------------------------

def process_applications(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Run the full RPA pipeline over all applications.

    Args:
        df: Raw applications DataFrame.

    Returns:
        (results_df, disbursement_records)
    """
    results      : list[dict] = []
    disbursements: list[dict] = []

    total = len(df)
    print(f"\n{'='*72}")
    print(f"  RPA LOAN PROCESSING BOT  |  Processing {total} application(s)")
    print(f"{'='*72}\n")

    for idx, row in df.iterrows():
        app_id = str(row["application_id"])
        name   = str(row["applicant_name"])

        print(f"  ┌─ [{idx + 1}/{total}] {app_id} — {name}")
        log_audit_event(app_id, "PROCESSING_START")

        result = {
            "application_id":   app_id,
            "applicant_name":   name,
            "email":            row["email"],
            "loan_amount":      row["loan_amount"],
            "loan_purpose":     row["loan_purpose"],
            "employment_status": row["employment_status"],
            "annual_income":    row["annual_income"],
            "existing_debt":    row["existing_debt"],
            "reported_credit_score": row["credit_score"],
            "bureau_credit_score":   None,
            "bureau_reference":      None,
            "decision":         None,
            "interest_rate":    None,
            "decision_reasons": None,
            "email_file":       None,
            "validation_errors": None,
            "processed_at":     datetime.datetime.utcnow().isoformat() + "Z",
        }

        # --- Step 2: Validation ---
        is_valid, val_errors = validate_application(row)
        if not is_valid:
            result["decision"]          = "ERROR"
            result["validation_errors"] = "; ".join(val_errors)
            result["decision_reasons"]  = "; ".join(val_errors)
            log_audit_event(app_id, "VALIDATION_FAIL", "; ".join(val_errors))
            print(f"  │  [VALIDATION]  FAILED — {val_errors}")
            print(f"  └─ Decision: ERROR\n")
            results.append(result)
            continue

        log_audit_event(app_id, "VALIDATION_PASS")
        print(f"  │  [VALIDATION]  Passed all field checks")

        # --- Step 3: Credit Bureau API ---
        try:
            bureau_resp = call_credit_bureau_api(
                app_id, name, int(row["credit_score"])
            )
            bureau_score = bureau_resp["bureau_score"]
            bureau_ref   = bureau_resp["bureau_reference"]
            result["bureau_credit_score"] = bureau_score
            result["bureau_reference"]    = bureau_ref
            log_audit_event(app_id, "CREDIT_BUREAU_CALL",
                            f"score={bureau_score} ref={bureau_ref}")
            print(f"  │  [CREDIT API]  Bureau score: {bureau_score} (ref: {bureau_ref})")
        except Exception as exc:
            result["decision"]         = "ERROR"
            result["decision_reasons"] = f"Credit bureau API failure: {exc}"
            log_audit_event(app_id, "CREDIT_BUREAU_ERROR", str(exc))
            print(f"  │  [CREDIT API]  ERROR — {exc}")
            print(f"  └─ Decision: ERROR\n")
            results.append(result)
            continue

        # --- Step 4: Business Rules ---
        decision, reasons, interest_rate = apply_business_rules(row, bureau_score)
        result["decision"]        = decision
        result["interest_rate"]   = interest_rate
        result["decision_reasons"] = "; ".join(reasons)
        log_audit_event(app_id, f"DECISION_{decision}",
                        f"reasons={reasons}")

        decision_icon = "✓ APPROVED" if decision == "APPROVED" else "✗ REJECTED"
        print(f"  │  [RULES]       Decision: {decision_icon}")
        for r in reasons:
            print(f"  │               → {r}")

        # --- Step 5: Email Generation ---
        if decision == "APPROVED":
            email_body = generate_approval_email(row, bureau_score, interest_rate, reasons)
            disburse   = write_disbursement_record(row, interest_rate, bureau_score)
            disbursements.append(disburse)
            log_audit_event(app_id, "DISBURSEMENT_RECORD_CREATED",
                            f"monthly_payment={disburse['monthly_payment']}")
            print(f"  │  [DISBURSE]    Monthly payment: ${disburse['monthly_payment']:,.2f} "
                  f"@ {interest_rate:.2%} APR")
        else:
            email_body = generate_rejection_email(row, bureau_score, reasons)

        email_path = save_email(app_id, decision, email_body)
        result["email_file"] = email_path.name
        log_audit_event(app_id, "EMAIL_GENERATED", email_path.name)
        print(f"  │  [EMAIL]       Notification saved: {email_path.name}")
        print(f"  └─ Complete\n")

        results.append(result)

    results_df = pd.DataFrame(results)
    return results_df, disbursements


# ---------------------------------------------------------------------------
# Output Writers
# ---------------------------------------------------------------------------

def write_processed_csv(results_df: pd.DataFrame) -> None:
    """Persist processed applications to output CSV."""
    results_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"[OUTPUT] Processed results written to: {OUTPUT_FILE.name}")
    logger.debug(f"[OUTPUT] Columns: {list(results_df.columns)}")


def write_disbursement_csv(disbursements: list[dict]) -> None:
    """Persist disbursement records to CSV."""
    if not disbursements:
        logger.info("[DISBURSE] No disbursement records to write.")
        return
    df = pd.DataFrame(disbursements)
    df.to_csv(DISBURSEMENT_FILE, index=False)
    logger.info(f"[DISBURSE] {len(disbursements)} record(s) written to: "
                f"{DISBURSEMENT_FILE.name}")


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------

def print_summary(results_df: pd.DataFrame) -> dict:
    """
    Print a formatted summary of the RPA run and return metrics dict.

    Args:
        results_df: DataFrame of processed results.

    Returns:
        Summary metrics dictionary.
    """
    total     = len(results_df)
    approved  = (results_df["decision"] == "APPROVED").sum()
    rejected  = (results_df["decision"] == "REJECTED").sum()
    errors    = (results_df["decision"] == "ERROR").sum()
    app_rate  = approved / total * 100 if total else 0

    approved_df = results_df[results_df["decision"] == "APPROVED"]
    total_disbursed = approved_df["loan_amount"].sum() if len(approved_df) else 0

    summary = {
        "total_applications": int(total),
        "approved":           int(approved),
        "rejected":           int(rejected),
        "errors":             int(errors),
        "approval_rate_pct":  round(app_rate, 1),
        "total_loan_value_approved": float(total_disbursed),
    }

    print(f"\n{'='*72}")
    print(f"  RPA RUN SUMMARY")
    print(f"{'='*72}")
    print(f"  Total Applications Processed : {total}")
    print(f"  Approved                     : {approved}")
    print(f"  Rejected                     : {rejected}")
    print(f"  Errors / Invalid             : {errors}")
    print(f"  Approval Rate                : {app_rate:.1f}%")
    print(f"  Total Approved Loan Value    : ${total_disbursed:,.2f}")
    print(f"{'='*72}")
    print(f"\n  Output Files:")
    print(f"    • {OUTPUT_FILE.name}")
    print(f"    • {DISBURSEMENT_FILE.name}")
    print(f"    • {AUDIT_LOG_FILE.relative_to(BASE_DIR)}")
    print(f"    • emails/  ({approved + rejected} notification file(s))")
    print(f"\n  Run completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*72}\n")

    return summary


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def run(input_file: Path = INPUT_FILE) -> dict:
    """
    Execute the full RPA loan processing pipeline.

    Args:
        input_file: Path to the input CSV (default: sample_applications.csv).

    Returns:
        Summary metrics dictionary.
    """
    run_start = time.time()
    logger.info("=" * 60)
    logger.info("RPA LOAN PROCESSING BOT — RUN STARTED")
    logger.info(f"Input file : {input_file}")
    logger.info(f"Start time : {datetime.datetime.now().isoformat()}")
    logger.info("=" * 60)

    try:
        df = ingest_applications(input_file)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"Fatal ingestion error: {exc}")
        raise

    results_df, disbursements = process_applications(df)
    write_processed_csv(results_df)
    write_disbursement_csv(disbursements)
    summary = print_summary(results_df)

    elapsed = time.time() - run_start
    logger.info(f"RPA RUN COMPLETE — elapsed {elapsed:.2f}s")
    logger.info(f"Summary: {json.dumps(summary)}")

    return summary


if __name__ == "__main__":
    run()
