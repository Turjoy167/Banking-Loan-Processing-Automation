# Banking Loan Processing Automation

**Student:** Plabon Banik (24240966) | MSc Artificial Intelligence | National College of Ireland  
**Module:** H9IAPA — Intelligent Agents and Process Automation

---

## Project Overview

This project automates the end-to-end banking loan processing pipeline using Robotic Process Automation (RPA) and Artificial Intelligence (AI) techniques. It combines rule-based workflow automation with machine learning-driven credit risk assessment to reduce manual processing time and improve decision consistency.

---

## Project Structure

```
banking-loan-processing/
├── ai_solution/
│   ├── ai_loan_assessor.py       # ML-based loan eligibility classifier
│   ├── document_verifier.py      # Automated document validation module
│   └── risk_dashboard.py         # Interactive risk analytics dashboard
├── rpa_solution/
│   ├── rpa_loan_bot.py           # RPA backend workflow automation
│   └── rpa_web_bot.py            # Selenium-driven web form bot
├── bpmn_diagrams/
│   ├── loan_process_as_is.bpmn   # Current-state process diagram
│   └── loan_process_to_be.bpmn   # Automated future-state diagram
├── dataset/
│   └── loan_data.csv             # Analytics Vidhya Loan Prediction Dataset
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10 or higher
- Google Chrome (for Selenium web automation)
- ChromeDriver (auto-managed via `chromedriver-binary-auto` in `requirements.txt`)

---

## Installation

Install all required Python dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### RPA Backend
Runs the core rule-based loan processing automation:
```bash
python rpa_solution/rpa_loan_bot.py
```

### RPA Web Bot
Automates web form submission via Selenium (requires Flask server to be running first):
```bash
python rpa_solution/rpa_web_bot.py
```

### AI Loan Assessor
Trains and evaluates the machine learning loan eligibility model:
```bash
python ai_solution/ai_loan_assessor.py
```

### Document Verifier
Runs automated validation checks on submitted loan documents:
```bash
python ai_solution/document_verifier.py
```

### Risk Dashboard
Launches the loan risk analytics and visualisation dashboard:
```bash
python ai_solution/risk_dashboard.py
```

---

## Dataset

**Analytics Vidhya — Loan Prediction Dataset**  
614 loan applications with applicant demographics, income details, credit history, and loan approval outcomes. Used to train and evaluate the AI credit risk classifier.

---

## BPMN Diagrams

Process models are located in the `bpmn_diagrams/` directory. These include both the as-is manual loan process and the to-be automated process, modelled in compliance with BPMN 2.0 notation.
