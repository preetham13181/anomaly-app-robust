Transaction Monitor: a safer Anomaly Detection App

A Streamlit app that scores transactions with an Isolation Forest and sends the most unusual ones to human review. This is a protected copy of the original Anomaly Detection App. It adds two safeguards so the app does not mislead the people who rely on it. The trained model (model.joblib) is unchanged: nothing was retrained or replaced.



Link

Protected app

https://anomaly-app-robust-74jzz7bymff946neqmslqp.streamlit.app/

Baseline app (original behavior, no protections)

https://anomaly-app-robust-dvhvt2adwrik9app4sjkj8b.streamlit.app/

Why this exists

An app can run successfully and still give a wrong answer. Testing the original app showed two problems:

Bad data passes as "OK". Blank or negative values are scored anyway and come out as normal. A typical amount of 0 crashes the stream, and a missing column crashes it at the first row.

Changed behavior goes unnoticed. If customers behave differently from the history the model learned from, the app flags far too many (or too few) transactions and still says "Done".

What was added

1. Protection against invalid or missing data

Detects: missing columns, empty or unreadable files, and rows with blank, non-numeric, zero or negative values.

Responds:

A missing column, empty file or unreadable file stops the run with an error naming the problem and what to ask the data owner for. Nothing is approved.

Bad rows are marked NOT CHECKED, never OK and never REVIEW. They are listed with the reason and the next step: send to manual review or back to the data owner.

If more than 10% of the rows are unusable, the app says not to rely on the run, because the data feed is probably broken.

The green "Done" appears only when every transaction was checked.

2. Protection against a model that no longer fits the data

Detects: after 100 scored transactions, the app compares the share flagged with the review budget. At 2 times the budget or more (or 0.25 times or less), it raises a warning.

Responds: the warning explains the likely cause (changed behavior, units or data source) and the next steps: do not trust the list or add reviewers because of it, ask the data owner, check a sample of the flags, and ask the model owner to retrain.
It also shows a suggested review order: the top-scoring transactions, sized to the planned workload.

Test results (review budget 4%)

In the Live stream tab, choose a file in the sidebar box Transaction file for the live stream and press Start the stream.

Demo file

Original (baseline app)

Protected app

Blank / negative values

Green "Done. 20 transactions were sent to review". The 6 bad rows passed as OK.

6 rows NOT CHECKED with reasons, yellow "Run finished with problems"

Zero values

ZeroDivisionError crash after 9 of 600 transactions

No crash. 3 rows NOT CHECKED, the other 597 scored

A column is missing

KeyError crash

Red error naming customer_typical_amount and the next step

Customer behavior has changed

Green "Done. 156 transactions were sent to review". Only 20 were real.

Flag-rate warning (26.0% against a 4% budget), suggested review order. 16 of its top 24 were real.

Normal transactions (default)

20 flagged, 16 correct

Identical, with "Not checked: 0" and a green "Done"

Repository contents

File

Purpose

app.py

The protected app (deploy this one)

app_baseline.py

The original app without protections. Only a file selector was added so the same test files can be fed to it.

checks.py

The two checks: row validation and flag-rate monitoring

model.joblib

Trained Isolation Forest (unchanged)

history.csv

Past transactions the model learned from (unchanged)

new_transactions.csv

Demonstration stream (unchanged)

test_data/

Demonstration files that trigger each failure

make_test_data.py

Builds the files in test_data/ from new_transactions.csv

requirements.txt

Python packages (scikit-learn is pinned to the version that trained the model)

All data is synthetic demonstration data.

Run it yourself

pip install -r requirements.txt
streamlit run app.py            # protected app
streamlit run app_baseline.py   # baseline for comparison

Known limits

The drift check watches the flag rate only. A shift that keeps the flag rate near the budget would not be caught.

The 2-times threshold is a judgement call that a business could tune.

Validation catches impossible values, not plausible but wrong ones (for example the wrong currency with realistic numbers).

Other risks are not handled in code, such as reading "OK" as "safe", review capacity limits, and fraud that stays within normal ranges.

The real fix for drift is retraining the model on recent data, which is outside the scope of this project.
