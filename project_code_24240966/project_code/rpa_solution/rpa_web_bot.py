"""
RPA Web UI Automation Bot
=========================
Demonstrates true Robotic Process Automation by automating a real browser:
 - Opens the First National Bank loan application portal (loan_form.html)
 - Reads each application record from sample_applications.csv
 - Fills every form field character-by-character (human-like typing)
 - Takes a screenshot of the filled-in form BEFORE submission
 - Clicks "Submit Application"
 - Takes a screenshot of the confirmation page AFTER submission
 - Logs every action with timestamps
 - Prints a full summary at the end

Requirements:
    pip install selenium chromedriver-binary-auto

Author: RPA Automation Team
Version: 2.0.0
"""

import csv
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# ── Selenium imports ──────────────────────────────────────────────────────────
import chromedriver_binary  # noqa: F401  (adds chromedriver to PATH)
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
FORM_HTML       = BASE_DIR / "loan_form.html"
CSV_INPUT       = BASE_DIR / "sample_applications.csv"
SCREENSHOT_DIR  = BASE_DIR / "screenshots"
LOG_FILE        = BASE_DIR / "web_bot_output.txt"
CHROME_BINARY   = Path.home() / ".cache/ms-playwright/chromium-1217/chrome-linux64/chrome"

# ── How many applications to process (keep demo fast) ────────────────────────
MAX_APPLICATIONS = 3

# ── Typing simulation delay (seconds per character) ──────────────────────────
TYPING_DELAY = 0.05

# ── Map CSV loan_purpose values → dropdown option values in the HTML form ─────
PURPOSE_MAP = {
    "Home Improvement": "Home",
    "Home Purchase":    "Home",
    "Business Expansion": "Business",
    "Auto Purchase":    "Car",
    "Education":        "Education",
    "Personal":         "Personal",
    "Medical Expenses": "Personal",
    "Vacation":         "Personal",
    "Debt Consolidation": "Personal",
}

# ── Logging setup ─────────────────────────────────────────────────────────────
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("rpa_web_bot")

# ── Action log (for final summary) ───────────────────────────────────────────
action_log: list[dict] = []


def action(app_id: str, step: str, detail: str = "") -> None:
    """Record and print one RPA action."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    msg = f"[{app_id}]  {step}"
    if detail:
        msg += f"  →  {detail}"
    log.info(msg)
    action_log.append({"app_id": app_id, "timestamp": ts, "step": step, "detail": detail})


# ── Driver factory ────────────────────────────────────────────────────────────
def build_driver() -> webdriver.Chrome:
    """Create a headless Chrome WebDriver with a large viewport."""
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument("--force-device-scale-factor=1")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-web-security")   # allow file:// local form
    opts.add_argument("--allow-file-access-from-files")
    if CHROME_BINARY.exists():
        opts.binary_location = str(CHROME_BINARY)

    svc = Service(chromedriver_binary.chromedriver_filename)
    driver = webdriver.Chrome(service=svc, options=opts)
    driver.set_window_size(1280, 900)
    return driver


# ── Typing helper ─────────────────────────────────────────────────────────────
def human_type(element, text: str) -> None:
    """
    Clear the element then type each character with a small delay
    to simulate genuine human interaction — a key RPA characteristic.
    """
    element.clear()
    for char in str(text):
        element.send_keys(char)
        time.sleep(TYPING_DELAY)


# ── Screenshot helper ─────────────────────────────────────────────────────────
def take_screenshot(driver: webdriver.Chrome, filename: str) -> Path:
    """Save a full-page screenshot and return its path."""
    path = SCREENSHOT_DIR / filename
    driver.save_screenshot(str(path))
    return path


# ── Main form-filling function ────────────────────────────────────────────────
def fill_application(driver: webdriver.Chrome, app: dict) -> dict:
    """
    Navigate to the loan form and fill it in for one application.
    Returns timing metrics.
    """
    app_id   = app["application_id"]
    app_num  = app_id.split("-")[-1]        # e.g.  "001"
    wait     = WebDriverWait(driver, 10)

    t_start = time.perf_counter()

    # ── Step 1: Open form ─────────────────────────────────────────────────────
    form_url = f"file://{FORM_HTML}"
    action(app_id, "NAVIGATE", f"Loading {FORM_HTML.name}")
    driver.get(form_url)

    # Wait for the form to render
    wait.until(EC.presence_of_element_located((By.ID, "loanForm")))
    time.sleep(0.3)   # short pause so page is fully painted

    # ── Step 2: Scroll to top ─────────────────────────────────────────────────
    driver.execute_script("window.scrollTo(0, 0);")

    # ── Step 3: Fill Personal Information ────────────────────────────────────
    action(app_id, "TYPE", f"Full Name  →  {app['applicant_name']}")
    name_field = driver.find_element(By.ID, "fullName")
    human_type(name_field, app["applicant_name"])

    action(app_id, "TYPE", f"Email  →  {app['email']}")
    email_field = driver.find_element(By.ID, "email")
    human_type(email_field, app["email"])

    # ── Step 4: Fill Financial Information ───────────────────────────────────
    action(app_id, "TYPE", f"Annual Income  →  ${app['annual_income']}")
    income_field = driver.find_element(By.ID, "annualIncome")
    human_type(income_field, app["annual_income"])

    action(app_id, "TYPE", f"Loan Amount  →  ${app['loan_amount']}")
    loan_field = driver.find_element(By.ID, "loanAmount")
    human_type(loan_field, app["loan_amount"])

    # Map CSV purpose to dropdown option
    raw_purpose = app["loan_purpose"]
    mapped_purpose = PURPOSE_MAP.get(raw_purpose, "Personal")
    action(app_id, "SELECT", f"Loan Purpose  →  {mapped_purpose} (from '{raw_purpose}')")
    purpose_sel = Select(driver.find_element(By.ID, "loanPurpose"))
    purpose_sel.select_by_value(mapped_purpose)
    time.sleep(0.15)

    action(app_id, "TYPE", f"Existing Monthly Debt  →  ${app['existing_debt']}")
    debt_field = driver.find_element(By.ID, "existingDebt")
    human_type(debt_field, app["existing_debt"])

    # ── Step 5: Fill Employment Information ──────────────────────────────────
    raw_status = app["employment_status"]
    action(app_id, "SELECT", f"Employment Status  →  {raw_status}")
    status_sel = Select(driver.find_element(By.ID, "employmentStatus"))
    status_sel.select_by_value(raw_status)
    time.sleep(0.15)

    action(app_id, "TYPE", f"Years Employed  →  {app['years_employed']}")
    years_field = driver.find_element(By.ID, "yearsEmployed")
    human_type(years_field, app["years_employed"])

    # ── Step 6: Fill Credit Information ──────────────────────────────────────
    action(app_id, "TYPE", f"Credit Score  →  {app['credit_score']}")
    credit_field = driver.find_element(By.ID, "creditScore")
    human_type(credit_field, app["credit_score"])

    t_filled = time.perf_counter()

    # ── Step 7: Screenshot BEFORE submit ─────────────────────────────────────
    shot_filled = f"app_{app_num}_filled.png"
    path_filled = take_screenshot(driver, shot_filled)
    action(app_id, "SCREENSHOT", f"Form filled  →  {path_filled}")

    # ── Step 8: Click Submit ──────────────────────────────────────────────────
    action(app_id, "CLICK", "Submit Application button")
    submit_btn = driver.find_element(By.ID, "submitBtn")
    driver.execute_script("arguments[0].scrollIntoView(true);", submit_btn)
    time.sleep(0.2)
    submit_btn.click()

    # Wait for confirmation panel to appear
    wait.until(EC.visibility_of_element_located((By.ID, "confirmation")))
    time.sleep(0.4)

    # ── Step 9: Screenshot AFTER submit ──────────────────────────────────────
    shot_submitted = f"app_{app_num}_submitted.png"
    path_submitted = take_screenshot(driver, shot_submitted)
    action(app_id, "SCREENSHOT", f"Confirmation  →  {path_submitted}")

    t_end = time.perf_counter()

    return {
        "app_id":         app_id,
        "fill_time_s":    round(t_filled - t_start, 2),
        "total_time_s":   round(t_end - t_start, 2),
        "screenshot_filled":    str(path_filled),
        "screenshot_submitted": str(path_submitted),
    }


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    overall_start = time.perf_counter()

    log.info("=" * 68)
    log.info("  RPA WEB UI AUTOMATION BOT  —  First National Bank Portal")
    log.info("=" * 68)
    log.info(f"  Form:       {FORM_HTML}")
    log.info(f"  Input CSV:  {CSV_INPUT}")
    log.info(f"  Screenshots:{SCREENSHOT_DIR}")
    log.info(f"  Max apps:   {MAX_APPLICATIONS}")
    log.info(f"  Typing delay: {TYPING_DELAY * 1000:.0f} ms / character")
    log.info("=" * 68)
    print()

    # ── Read CSV ──────────────────────────────────────────────────────────────
    with open(CSV_INPUT, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        applications = [
            {k.strip(): v.strip() for k, v in row.items()}
            for row in reader
        ]

    # Trim to first MAX_APPLICATIONS
    applications = applications[:MAX_APPLICATIONS]
    log.info(f"Loaded {len(applications)} application(s) from {CSV_INPUT.name}")
    print()

    # ── Launch browser ────────────────────────────────────────────────────────
    log.info("Launching headless Chrome browser …")
    driver = build_driver()
    log.info(f"Browser started  (Chrome {driver.capabilities.get('browserVersion', '?')})")
    print()

    results: list[dict] = []
    total_screenshots = 0

    for idx, app in enumerate(applications, 1):
        app_id = app.get("application_id", f"APP-{idx:03d}")
        log.info("-" * 60)
        log.info(f"  Processing application {idx}/{len(applications)}:  {app_id}  ({app.get('applicant_name', '?')})")
        log.info("-" * 60)

        try:
            metrics = fill_application(driver, app)
            results.append(metrics)
            total_screenshots += 2
            action(app_id, "DONE",
                   f"Completed in {metrics['total_time_s']} s  "
                   f"(fill: {metrics['fill_time_s']} s)")
        except (NoSuchElementException, TimeoutException, WebDriverException) as exc:
            action(app_id, "ERROR", str(exc))
            log.error(f"  ⚠ Failed to process {app_id}: {exc}")

        print()

    # ── Quit browser ──────────────────────────────────────────────────────────
    driver.quit()
    log.info("Browser closed.")
    print()

    # ── Final summary ─────────────────────────────────────────────────────────
    overall_elapsed = round(time.perf_counter() - overall_start, 2)

    log.info("=" * 68)
    log.info("  RPA WEB BOT — EXECUTION SUMMARY")
    log.info("=" * 68)
    log.info(f"  Applications processed : {len(results)}")
    log.info(f"  Total screenshots saved: {total_screenshots}")
    log.info(f"  Total elapsed time     : {overall_elapsed} seconds")
    log.info(f"  Screenshot directory   : {SCREENSHOT_DIR}")
    print()
    log.info("  Per-Application Timing:")
    log.info("  " + "-" * 50)
    for r in results:
        log.info(
            f"  {r['app_id']}  |  fill: {r['fill_time_s']:5.2f}s  |  total: {r['total_time_s']:5.2f}s"
        )
    print()
    log.info("  Screenshots saved:")
    log.info("  " + "-" * 50)
    for r in results:
        log.info(f"  {Path(r['screenshot_filled']).name}")
        log.info(f"  {Path(r['screenshot_submitted']).name}")
    print()
    log.info("  Full action log:")
    log.info("  " + "-" * 50)
    for entry in action_log:
        detail = f"  →  {entry['detail']}" if entry["detail"] else ""
        log.info(f"  {entry['timestamp']}  [{entry['app_id']}]  {entry['step']}{detail}")
    log.info("=" * 68)
    log.info("  RPA UI Automation complete.")
    log.info("=" * 68)


if __name__ == "__main__":
    main()
