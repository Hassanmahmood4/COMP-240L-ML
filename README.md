# COMP-240L — Machine Learning Lab

Coursework for **Machine Learning Lab** (Semester 5) at Pak-Austria Fachhochschule Institute of Applied Sciences and Technology (PAF-IAST).

| | |
|---|---|
| **Student** | Hassan Mahmood |
| **Registration** | B24F0089CE001 |
| **Instructor** | Mr Rafiullah |

## Labs

| Lab | Topic | Folder |
|-----|--------|--------|
| 1 | PIP, packages, and external libraries | `Lab 1/` |
| 2 | Environments, timing, visualisation, model persistence | `Lab 2/` |
| 3 | Data wrangling with NumPy and Pandas (housing) | `Lab 3/` |
| 4 | Data preprocessing for ML (loan pipeline + Section 13 tasks) | `Lab 4/` |

Word lab reports live in `Lab Reports/` (`ML LAB 1.docx` … `ML LAB 4.docx`).

## Lab 4 layout

- `preprocessing.py` — main Parts A–H pipeline (dataset → split → impute/scale/encode → models → leakage demo)
- `common.py` — shared load/split and preprocess helpers for the exercises
- `task1_*.py` … `task8_*.py` — Section 13 exercises (robust scaling, imputation, indicators, encoding, trees, thresholds, unseen categories, income clipping)
- `data/` — raw and prepared CSVs
- `outputs/` — figures only (text run logs are gitignored)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib scikit-learn python-docx pillow
```

Run a task from its lab folder, for example:

```bash
cd "Lab 4"
python task1_robust_scaling.py
```

## Notes

- Virtual environments (`.venv`, `mlenv`) and `__pycache__` are ignored.
- Console `*_output.txt` files under `outputs/` are ignored; keep PNG figures for reports.
