# ml_project_lab02

Breast Cancer classification project using `sklearn.datasets.load_breast_cancer`.

A scikit-learn `Pipeline` applies `StandardScaler` then `LogisticRegression`. After a stratified 70/30 split, the model is trained, evaluated, and saved with joblib. `predict.py` reloads that model and scores the first 10 samples.

## Recreate from a clean machine

### 1. Prerequisites

- Python 3.10+ installed (`python3 --version`)
- Terminal access (macOS / Linux / Windows with Python)

### 2. Get the project

Copy or clone this folder so you have:

```text
ml_project_lab02/
  train.py
  predict.py
  check_requirements.py
  requirements.txt
  README.md
```

### 3. Create and activate a virtual environment

```bash
cd ml_project_lab02
python3 -m venv .venv
```

Activate it:

- macOS / Linux: `source .venv/bin/activate`
- Windows (cmd): `.venv\Scripts\activate.bat`
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`

### 4. Install pinned dependencies only

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip list
```

Optional check:

```bash
python check_requirements.py
```

You should see: `2 of 2 requirements satisfied`.

### 5. Train

```bash
python train.py
```

This prints test accuracy and writes `breast_cancer_pipeline.joblib`.

### 6. Predict

```bash
python predict.py
```

This prints true vs predicted labels for the first 10 dataset rows.

## Notes

- `requirements.txt` lists **direct** dependencies only (`scikit-learn`, `joblib`), pinned with `==`.
- Other packages in `pip list` (numpy, scipy, …) are installed automatically as transitive dependencies.
- Task 6 peer-review notes go in `PEER_REVIEW.md`.
