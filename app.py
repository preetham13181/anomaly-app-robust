import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from checks import (REQUIRED_COLUMNS, MAX_UNUSABLE_SHARE,
                    missing_columns, find_problems, flag_rate_status)

st.set_page_config(page_title='Transaction Monitor', layout='wide')

# ---- load the trained model (unchanged, made in the Colab notebook) ----
bundle = joblib.load('model.joblib')
model = bundle['model']
history_scores = bundle['history_scores']
features = bundle['features']


def score(amount, typical, last_hour, gap, age):
    """Turn one transaction into the five features and return its anomaly score."""
    row = pd.DataFrame([{
        'log_amount': np.log1p(amount),
        'amount_ratio': amount / typical,
        'transactions_last_hour': last_hour,
        'log_gap': np.log1p(gap),
        'log_age': np.log1p(age),
    }])[features]
    return -model.score_samples(row)[0]


# ---- demo files (the first one is the supplied demonstration data) ----
SOURCES = {
    'Demo: normal transactions (default)': 'new_transactions.csv',
    'Demo: blank / negative values': 'test_data/invalid_values.csv',
    'Demo: zero values (crash in original)': 'test_data/crash_values.csv',
    'Demo: a column is missing': 'test_data/missing_column.csv',
    'Demo: customer behavior has changed': 'test_data/behavior_changed.csv',
    'Upload my own CSV file': None,
}


def load_stream(choice, uploaded):
    """Return (table, error text). Never raises."""
    try:
        if SOURCES[choice] is None:
            if uploaded is None:
                return None, 'Upload a CSV file in the sidebar first, then press the button again.'
            return pd.read_csv(uploaded), None
        return pd.read_csv(SOURCES[choice]), None
    except Exception:
        return None, ('The file could not be read as a CSV table (it is empty or damaged). '
                      'Export it again from the source system and upload the new file.')


# ---- sidebar: the review budget ----
st.title('Transaction Monitor')
st.write('Each new transaction is scored by Isolation Forest. '
         'If the score is above the cutoff, it goes to review.')

pct = st.sidebar.slider('Send this share of transactions to review (%)', 1, 10, 4)
cutoff = np.percentile(history_scores, 100 - pct)
st.sidebar.write(f'Score cutoff: **{cutoff:.3f}**')
st.sidebar.caption('The cutoff is set from past transactions: '
                   'a transaction scoring above it is more unusual than the top '
                   f'{pct}% of history.')

st.sidebar.divider()
source = st.sidebar.selectbox('Transaction file for the live stream', list(SOURCES))
uploaded = None
if SOURCES[source] is None:
    uploaded = st.sidebar.file_uploader('CSV file', type='csv')

tab1, tab2 = st.tabs(['Check one transaction', 'Live stream'])

# ---- tab 1: type in one transaction (unchanged) ----
with tab1:
    c1, c2 = st.columns(2)
    amount = c1.number_input('Amount ($)', min_value=1.0, value=60.0)
    typical = c1.number_input("Customer's typical amount ($)", min_value=1.0, value=50.0)
    last_hour = c1.number_input('Transactions in the last hour', min_value=0, value=1)
    gap = c2.number_input('Minutes since previous transaction', min_value=0.0, value=120.0)
    age = c2.number_input('Account age (days)', min_value=0, value=700)

    if st.button('Score this transaction'):
        s = score(amount, typical, last_hour, gap, age)
        if s > cutoff:
            st.error(f'REVIEW  (score {s:.3f}, cutoff {cutoff:.3f})')
        else:
            st.success(f'OK  (score {s:.3f}, cutoff {cutoff:.3f})')
        share = (history_scores < s).mean() * 100
        st.write(f'This is more unusual than {share:.0f}% of past transactions.')


def drift_message(status, pct):
    kind, rate, ratio = status
    if kind == 'high':
        return (f'**Unusual flag rate: {rate:.1f}% of transactions are being flagged, but the review '
                f'budget is {pct}% ({ratio:.1f} times higher).** With a model that still fits the data, '
                f'about {pct}% would be flagged. Customer behavior or the data feed has probably '
                f'changed (for example a new currency or unit, a seasonal peak or a new product), so '
                f'the model\'s idea of "normal" is out of date and many flags may be false alarms.\n\n'
                f'**What to do:** (1) Do not treat this list as reliable and do not add reviewers '
                f'because of it. (2) Ask the data owner whether the source or its units changed. '
                f'(3) Check a sample of the flagged transactions to see how many are real. '
                f'(4) If behavior really changed, ask the model owner to retrain on recent history. '
                f'Until then, review only the highest-scoring transactions that fit your planned '
                f'workload (suggested list at the end of the run).')
    return (f'**Unusually few flags: only {rate:.1f}% of transactions are being flagged, but the '
            f'review budget is {pct}%.** The model may be missing unusual transactions, or the data '
            f'may be incomplete (for example many default or empty values).\n\n'
            f'**What to do:** ask the data owner to confirm the file is complete and that the data '
            f'source did not change. Do not read the short list as "everything is fine".')


# ---- tab 2: transactions arriving one by one ----
with tab2:
    st.write('Press the button. The app receives the new transactions one at a time '
             'and scores each one as it arrives.')
    speed = st.select_slider('Speed', options=['Slow', 'Medium', 'Fast'], value='Medium')
    pause = {'Slow': 0.5, 'Medium': 0.15, 'Fast': 0.02}[speed]

    if st.button('Start the stream'):
        stream, load_error = load_stream(source, uploaded)
        missing = missing_columns(stream) if stream is not None else []

        # PROTECTION 1a: file cannot be used at all -> stop this run and say what to do
        if load_error:
            st.error(load_error)
        elif missing:
            st.error('**This file cannot be checked: required column(s) missing: '
                     + ', '.join(missing) + '.** No transaction was scored, so none of them is '
                     'approved. **What to do:** ask the data owner for a file with all of these '
                     'columns: ' + ', '.join(REQUIRED_COLUMNS) + '.')
        elif len(stream) == 0:
            st.error('The file has no transactions. Nothing was checked. '
                     'Ask the data owner to send the file again.')
        else:
            numbers, problems = find_problems(stream)
            unusable = problems != ''
            n_unusable = int(unusable.sum())

            # PROTECTION 1b: warn before the run if some rows cannot be scored
            if n_unusable:
                st.warning(f'{n_unusable} of {len(stream)} rows have missing or impossible values. '
                           'They will be marked NOT CHECKED and will not be scored.')

            box = st.empty()
            counts = st.empty()
            drift_box = st.empty()
            st.subheader('Sent to review')
            table = st.empty()
            flagged = []
            not_checked = []
            n_scored = 0

            for k, (i, r) in enumerate(stream.iterrows()):
                s = None
                if unusable[i]:
                    reason = problems[i]
                else:
                    try:
                        v = numbers.loc[i]
                        s = score(v['amount'], v['customer_typical_amount'],
                                  v['transactions_last_hour'], v['minutes_since_previous'],
                                  v['account_age_days'])
                    except Exception:
                        reason = 'the score could not be calculated'

                if s is None:
                    # never counted as OK and never as REVIEW
                    not_checked.append({'id': str(r['transaction_id']), 'problem': reason,
                                        'amount': str(r['amount']),
                                        'typical': str(r['customer_typical_amount'])})
                    box.warning(f"{r['transaction_id']}: NOT CHECKED  ->  {reason}")
                else:
                    n_scored += 1
                    msg = f"{r['transaction_id']}: ${numbers.loc[i, 'amount']:.2f}  score {s:.3f}"
                    if s > cutoff:
                        flagged.append({'id': r['transaction_id'], 'amount': r['amount'],
                                        'typical': r['customer_typical_amount'],
                                        'last_hour': r['transactions_last_hour'],
                                        'score': round(s, 3)})
                        box.error(msg + '  ->  REVIEW')
                    else:
                        box.info(msg + '  ->  ok')

                counts.write(f'Received **{k + 1}** of {len(stream)}.  '
                             f'Sent to review: **{len(flagged)}**.  '
                             f'Not checked: **{len(not_checked)}**')
                if flagged:
                    table.dataframe(pd.DataFrame(flagged), hide_index=True)

                # PROTECTION 2: watch the flag rate while the stream runs
                status = flag_rate_status(n_scored, len(flagged), pct)
                if status:
                    drift_box.warning(drift_message(status, pct))
                else:
                    drift_box.empty()
                time.sleep(pause)

            # ---- end of run: say honestly how it went ----
            share_unusable = n_unusable / len(stream)
            status = flag_rate_status(n_scored, len(flagged), pct)

            if share_unusable > MAX_UNUSABLE_SHARE:
                st.error(f'**Do not rely on this run: {n_unusable} of {len(stream)} transactions '
                         f'({share_unusable:.0%}) could not be checked.** When this many rows are '
                         'unusable, the data feed is probably broken (a changed export, a wrong file '
                         'or a broken column mapping). **What to do:** ask the data owner to fix the '
                         'source, then run the file again. Until then, handle these transactions '
                         'manually.')
            elif not_checked or status:
                st.warning(f'Run finished with problems. {len(flagged)} transactions were sent to '
                           f'review, {len(not_checked)} could not be checked, and '
                           f'{n_scored} were scored.')
            else:
                st.success(f'Done. {len(flagged)} transactions were sent to review.')

            if not_checked:
                st.subheader('NOT CHECKED: neither approved nor flagged')
                st.write(f'These {len(not_checked)} transactions have missing or impossible values, so '
                         'the model could not judge them. **What to do:** send them to manual review '
                         'or back to the data owner for correction, then run them again. Do not treat '
                         'them as OK.')
                st.dataframe(pd.DataFrame(not_checked), hide_index=True)

            if status and status[0] == 'high' and flagged:
                workload = max(1, round(pct / 100 * n_scored))
                st.subheader(f'Suggested review order (top {workload} by score)')
                st.write('The flag rate is much higher than planned, so work from the top of this list '
                         f'and stop at your planned workload of {workload} transactions.')
                top = pd.DataFrame(flagged).sort_values('score', ascending=False).head(workload)
                st.dataframe(top, hide_index=True)

            # if the file has an answer key, check how we did
            if 'needs_review' in stream.columns and flagged:
                answer = stream.set_index('transaction_id')['needs_review']
                hits = sum(answer[f['id']] for f in flagged)
                st.write(f'{hits} of the {len(flagged)} flagged transactions really needed '
                         f'review. In total, {answer.sum()} transactions needed review.')
