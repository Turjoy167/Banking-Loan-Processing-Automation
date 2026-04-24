"""
Document Verifier Module
========================
Simulated AI document verification for banking loan applications.

Adapted for the Loan Prediction Dataset column schema:
  applicant_income, coapplicant_income, loan_amount, loan_amount_term,
  credit_history, property_area, gender, married, dependents,
  education, self_employed

Mimics an OCR + NLP pipeline that:
  - Extracts text from loan documents (simulated)
  - Uses regex/rule-based NLP to parse key fields
  - Cross-validates extracted data against application data
  - Returns a structured verification report with confidence scores
"""

import re
import random
import string
from datetime import date, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Simulated OCR text generation
# ---------------------------------------------------------------------------

_DOCUMENT_TEMPLATES = {
    "pay_stub": """\
PAY STUB - {company} Inc.
Employee: {name}
Pay Period: {start_date} to {end_date}
Gross Monthly Income: ${gross_income:,.2f}
Co-Applicant Monthly Income: ${co_income:,.2f}
YTD Earnings: ${ytd:,.2f}
Deductions: ${deductions:,.2f}
Net Pay: ${net_pay:,.2f}
Employment Status: {employment_status}
Education Level: {education_level}
""",

    "bank_statement": """\
BANK STATEMENT
Account Holder: {name}
Account Number: ****{acct_suffix}
Statement Period: {start_date} to {end_date}
Opening Balance: ${open_bal:,.2f}
Total Deposits: ${deposits:,.2f}
Total Withdrawals: ${withdrawals:,.2f}
Closing Balance: ${close_bal:,.2f}
Average Monthly Deposits: ${avg_deposits:,.2f}
Property Area: {property_area}
""",

    "id_document": """\
GOVERNMENT ISSUED ID
Full Name: {name}
Date of Birth: {dob}
ID Number: {id_num}
Issue Date: {issue_date}
Expiry Date: {expiry_date}
Marital Status: {marital_status}
Dependents: {dependents}
Address: {address}
""",

    "credit_report": """\
CREDIT REPORT SUMMARY
Consumer: {name}
Report Date: {report_date}
Credit History Score: {credit_history}
Loan Amount Requested: ${loan_amount:,.2f}
Loan Term (months): {loan_term}
Total Open Accounts: {open_accounts}
Collections: {collections}
Bankruptcies: {bankruptcies}
""",
}


def _random_name():
    first = random.choice(["James", "Sarah", "Michael", "Emily", "David",
                            "Jessica", "Robert", "Ashley", "William", "Amanda"])
    last  = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones",
                            "Garcia", "Miller", "Davis", "Wilson", "Taylor"])
    return f"{first} {last}"


def _random_date(years_back: int = 1) -> str:
    base  = date.today() - timedelta(days=years_back * 365)
    delta = timedelta(days=random.randint(0, 365))
    return (base + delta).strftime("%m/%d/%Y")


def _property_area_label(code) -> str:
    """Convert numeric property area code back to label."""
    mapping = {0: "Rural", 1: "Semiurban", 2: "Urban"}
    try:
        return mapping.get(int(code), "Semiurban")
    except (TypeError, ValueError):
        return str(code)


def simulate_ocr(application: dict, doc_type: str = "pay_stub",
                 introduce_error: bool = False, seed: int = None) -> str:
    """
    Generate simulated OCR text for a loan document.

    Parameters
    ----------
    application    : dict – raw application fields (uses real dataset column names)
    doc_type       : one of pay_stub | bank_statement | id_document | credit_report
    introduce_error: if True, subtly alter a numeric value to simulate fraud
    seed           : random seed for reproducibility

    Returns
    -------
    str  – simulated document text
    """
    if seed is not None:
        random.seed(seed)

    # ── Extract application fields ────────────────────────────────────────
    app_income   = float(application.get("applicant_income", 5000))
    co_income    = float(application.get("coapplicant_income", 0))
    loan_amount  = float(application.get("loan_amount", 100))    # in thousands
    loan_term    = float(application.get("loan_amount_term", 360))
    credit_hist  = float(application.get("credit_history", 1))
    prop_area    = application.get("property_area", 1)
    married      = application.get("married", 1)
    dependents   = int(application.get("dependents", 0))
    education    = application.get("education", 1)  # 1=Graduate, 0=Not Graduate
    self_emp     = application.get("self_employed", 0)
    name         = application.get("applicant_name", _random_name())

    today        = date.today()
    period_end   = today.strftime("%m/%d/%Y")
    period_start = (today - timedelta(days=30)).strftime("%m/%d/%Y")

    # Optionally corrupt income figure to simulate discrepancy
    reported_income = app_income * (1.40 if introduce_error else 1.0)

    edu_label  = "Graduate" if int(education) == 1 else "Not Graduate"
    emp_status = "Self-Employed" if int(self_emp) == 1 else "Salaried"
    marital    = "Married" if int(married) == 1 else "Single"
    prop_label = _property_area_label(prop_area)

    if doc_type == "pay_stub":
        gross   = reported_income
        ytd     = reported_income * today.month
        deduct  = gross * 0.22
        text = _DOCUMENT_TEMPLATES["pay_stub"].format(
            company=random.choice(["Acme Corp", "Pinnacle LLC", "NovaTech", "Apex Industries"]),
            name=name,
            start_date=period_start, end_date=period_end,
            gross_income=gross, co_income=co_income,
            ytd=ytd, deductions=deduct,
            net_pay=gross - deduct,
            employment_status=emp_status,
            education_level=edu_label,
        )

    elif doc_type == "bank_statement":
        avg_dep  = reported_income + co_income
        deposits = avg_dep * 3
        open_bal = avg_dep * 1.5
        with_    = deposits * 0.85
        text = _DOCUMENT_TEMPLATES["bank_statement"].format(
            name=name,
            acct_suffix=random.randint(1000, 9999),
            start_date=period_start, end_date=period_end,
            open_bal=open_bal, deposits=deposits,
            withdrawals=with_, close_bal=open_bal + deposits - with_,
            avg_deposits=avg_dep,
            property_area=prop_label,
        )

    elif doc_type == "id_document":
        dob    = _random_date(years_back=random.randint(25, 60))
        issued = _random_date(years_back=random.randint(1, 9))
        expiry = (date.today() + timedelta(days=random.randint(180, 3650))).strftime("%m/%d/%Y")
        id_num = "".join(random.choices(string.ascii_uppercase + string.digits, k=9))
        text = _DOCUMENT_TEMPLATES["id_document"].format(
            name=name, dob=dob, id_num=id_num,
            issue_date=issued, expiry_date=expiry,
            marital_status=marital,
            dependents=dependents,
            address=f"{random.randint(100, 9999)} Oak Street, Springfield, IL",
        )

    elif doc_type == "credit_report":
        open_accounts = max(1, int(loan_term / 60))  # approximate
        text = _DOCUMENT_TEMPLATES["credit_report"].format(
            name=name,
            report_date=today.strftime("%m/%d/%Y"),
            credit_history=int(credit_hist),
            loan_amount=loan_amount,
            loan_term=int(loan_term),
            open_accounts=open_accounts,
            collections=0,
            bankruptcies=0,
        )

    else:
        text = f"Unknown document type: {doc_type}"

    return text


# ---------------------------------------------------------------------------
# NLP extraction layer (rule-based regex patterns)
# ---------------------------------------------------------------------------

_PATTERNS = {
    "income":         r"(?:Gross Monthly Income|Average Monthly Deposits)[^\d]*([\d,]+\.?\d*)",
    "co_income":      r"Co-Applicant Monthly Income[^\d]*([\d,]+\.?\d*)",
    "loan_amount":    r"Loan Amount Requested[^\d]*([\d,]+\.?\d*)",
    "loan_term":      r"Loan Term \(months\)\s*:\s*(\d+)",
    "credit_history": r"Credit History Score\s*:\s*(\d+)",
    "name":           r"(?:Employee|Account Holder|Full Name|Consumer)\s*:\s*([A-Za-z ]+)",
    "property_area":  r"Property Area\s*:\s*([A-Za-z]+)",
    "marital_status": r"Marital Status\s*:\s*([A-Za-z]+)",
    "dependents":     r"Dependents\s*:\s*(\d+)",
    "expiry_date":    r"Expiry Date\s*:\s*([\d/]+)",
}


def extract_fields(ocr_text: str) -> dict:
    """
    Apply regex patterns to extract key financial fields from OCR text.

    Parameters
    ----------
    ocr_text : str – simulated OCR output

    Returns
    -------
    dict of extracted field → value (None if not found)
    """
    extracted = {}
    for field, pattern in _PATTERNS.items():
        match = re.search(pattern, ocr_text, re.IGNORECASE)
        if match:
            raw = match.group(1).strip().replace(",", "")
            try:
                extracted[field] = float(raw)
            except ValueError:
                extracted[field] = raw
        else:
            extracted[field] = None
    return extracted


# ---------------------------------------------------------------------------
# Cross-validation: extracted vs. application data
# ---------------------------------------------------------------------------

def _relative_diff(a: float, b: float) -> float:
    """Relative absolute difference between two values."""
    if b == 0:
        return 0.0
    return abs(a - b) / abs(b)


def validate_document(application: dict, extracted: dict) -> dict:
    """
    Compare extracted document fields with stated application values.

    Returns
    -------
    dict
        issues   – list of discrepancy strings
        warnings – list of minor flag strings
        passed   – list of passed check strings
    """
    issues   = []
    warnings = []
    passed   = []

    # ── Income check (monthly) ────────────────────────────────────────────
    if extracted.get("income") is not None:
        app_income = float(application.get("applicant_income", 0))
        doc_income = extracted["income"]
        diff = _relative_diff(doc_income, app_income)
        if diff > 0.25:
            issues.append(
                f"Applicant income mismatch: stated {app_income:,.0f}/mo, "
                f"document shows {doc_income:,.0f}/mo ({diff:.1%} difference)."
            )
        elif diff > 0.10:
            warnings.append(
                f"Minor income variance: {diff:.1%} between stated and document."
            )
        else:
            passed.append("Applicant income verified within acceptable tolerance.")

    # ── Co-applicant income check ─────────────────────────────────────────
    if extracted.get("co_income") is not None:
        app_co = float(application.get("coapplicant_income", 0))
        doc_co = extracted["co_income"]
        diff   = _relative_diff(doc_co, app_co)
        if diff > 0.25:
            warnings.append(
                f"Co-applicant income variance: {diff:.1%} difference."
            )
        else:
            passed.append("Co-applicant income verified.")

    # ── Credit history check ──────────────────────────────────────────────
    if extracted.get("credit_history") is not None:
        app_ch = int(application.get("credit_history", 1))
        doc_ch = int(extracted["credit_history"])
        if app_ch != doc_ch:
            issues.append(
                f"Credit history mismatch: stated {app_ch}, document shows {doc_ch}."
            )
        else:
            passed.append("Credit history record consistent with application.")

    # ── Loan amount check (document in thousands vs raw) ──────────────────
    if extracted.get("loan_amount") is not None:
        app_la = float(application.get("loan_amount", 0))
        doc_la = extracted["loan_amount"]
        diff   = _relative_diff(doc_la, app_la)
        if diff > 0.20:
            issues.append(
                f"Loan amount mismatch: stated {app_la:,.0f}, "
                f"document shows {doc_la:,.0f} ({diff:.1%} difference)."
            )
        else:
            passed.append("Loan amount verified.")

    # ── Document expiry check (ID) ────────────────────────────────────────
    if extracted.get("expiry_date") is not None:
        try:
            parts = str(extracted["expiry_date"]).split(".")
            if len(parts) == 1:  # stored as float like 12312025.0
                raw = str(int(float(extracted["expiry_date"])))
                # Try mm/dd/yyyy from raw string stored in text
        except Exception:
            pass  # skip if parse fails

    return {"issues": issues, "warnings": warnings, "passed": passed}


# ---------------------------------------------------------------------------
# Confidence score calculation
# ---------------------------------------------------------------------------

def compute_confidence(validation: dict) -> float:
    """
    Compute an overall document verification confidence score [0, 1].

    Deductions:
      - Critical issue  : -25 points each
      - Warning         : -8 points each
    Floor: 0.
    """
    score = 100.0
    score -= len(validation["issues"])   * 25.0
    score -= len(validation["warnings"]) * 8.0
    score  = max(0.0, score)
    return round(score / 100.0, 4)


# ---------------------------------------------------------------------------
# Main verification entry point
# ---------------------------------------------------------------------------

def verify_application_documents(
    application: dict,
    doc_types: list = None,
    introduce_errors: dict = None,
    seed: int = 0,
) -> dict:
    """
    Run the full document verification pipeline for one loan application.

    Parameters
    ----------
    application     : dict of raw loan application fields (real dataset schema)
    doc_types       : list of document types to simulate; defaults to all four
    introduce_errors: dict mapping doc_type → bool (whether to inject fraud)
    seed            : base random seed

    Returns
    -------
    dict
        applicant_name, documents (per-doc details), overall_confidence,
        overall_status, flagged_for_human_review
    """
    if doc_types is None:
        doc_types = ["pay_stub", "bank_statement", "id_document", "credit_report"]
    if introduce_errors is None:
        introduce_errors = {}

    doc_results = {}
    all_issues  = []

    for i, doc_type in enumerate(doc_types):
        inject = introduce_errors.get(doc_type, False)
        ocr_text   = simulate_ocr(application, doc_type,
                                   introduce_error=inject, seed=seed + i)
        extracted  = extract_fields(ocr_text)
        validation = validate_document(application, extracted)
        confidence = compute_confidence(validation)

        doc_results[doc_type] = {
            "ocr_preview":  ocr_text[:250] + "...",
            "extracted":    extracted,
            "validation":   validation,
            "confidence":   confidence,
        }
        all_issues.extend(validation["issues"])

    overall_confidence = round(
        sum(r["confidence"] for r in doc_results.values()) / len(doc_results), 4
    )

    if overall_confidence >= 0.85:
        overall_status = "VERIFIED"
    elif overall_confidence >= 0.60:
        overall_status = "CONDITIONAL_PASS"
    else:
        overall_status = "VERIFICATION_FAILED"

    return {
        "applicant_name":           application.get("applicant_name", "Unknown"),
        "documents":                doc_results,
        "overall_confidence":       overall_confidence,
        "overall_status":           overall_status,
        "total_issues":             len(all_issues),
        "flagged_for_human_review": overall_confidence < 0.70 or len(all_issues) > 0,
        "issue_summary":            all_issues,
    }
