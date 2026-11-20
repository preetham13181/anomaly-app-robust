# Transaction Monitor: protected copy

A copy of the Anomaly Detection App with two protections added. **The trained model (`model.joblib`) is unchanged.**

| File | What it is |
|---|---|
| `app.py` | Protected app (deploy this one) |
| `app_baseline.py` | Original app, no protections. Only a file selector was added so the same test files can be fed to it ("before" screenshots) |
| `checks.py` | The two checks: row validation and flag-rate watch |
| `model.joblib`, `history.csv`, `new_transactions.csv`, `requirements.txt` | Unchanged from the original app |
| `test_data/` | Demonstration files that trigger each failure (built by `make_test_data.py` from `new_transactions.csv`) |

## Protections
1. **Invalid or missing data**: rows with blank, non-numeric, zero or negative values are marked NOT CHECKED (never OK, never REVIEW) and listed with the reason and next step. A file with a missing column, an empty file or a file that is more than 10% unusable stops with a clear message.
2. **Model no longer fits the data**: while the stream runs, the app compares the share flagged with the review budget. At 2x the budget (or at 0.25x or lower) it shows a warning with what to do, and suggests a review order that fits the planned workload.

## Deploy
Create a public GitHub repo, upload everything here, then on share.streamlit.io create one app with main file `app.py` (protected) and a second app with main file `app_baseline.py` (baseline).
