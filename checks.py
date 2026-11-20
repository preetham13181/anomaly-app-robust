"""Checks used by the protected Transaction Monitor. The trained model is not touched here."""
from functools import reduce

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ['transaction_id', 'amount', 'customer_typical_amount',
                    'transactions_last_hour', 'minutes_since_previous', 'account_age_days']
NUMBER_COLUMNS = REQUIRED_COLUMNS[1:]

# value rules: column -> (text for the message, test that returns True when the value is valid)
RULES = {
    'amount':                  ('greater than 0', lambda s: s > 0),
    'customer_typical_amount': ('greater than 0', lambda s: s > 0),
    'transactions_last_hour':  ('0 or more',      lambda s: s >= 0),
    'minutes_since_previous':  ('0 or more',      lambda s: s >= 0),
    'account_age_days':        ('0 or more',      lambda s: s >= 0),
}

MAX_UNUSABLE_SHARE = 0.10   # above this, the whole file is treated as a broken feed


def missing_columns(df):
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def find_problems(df):
    """Return (numbers, problems).
    numbers  : the five numeric columns as floats (bad values become NaN)
    problems : one text per row, empty when the row is usable."""
    numbers = df[NUMBER_COLUMNS].apply(pd.to_numeric, errors='coerce')
    numbers = numbers.replace([np.inf, -np.inf], np.nan)
    parts = []
    for col, (rule, is_valid) in RULES.items():
        parts.append(pd.Series(
            np.where(numbers[col].isna(), f'{col} is missing or not a number; ',
                     np.where(~is_valid(numbers[col]), f'{col} must be {rule}; ', '')),
            index=df.index))
    problems = reduce(lambda a, b: a + b, parts).str.strip().str.rstrip(';')
    return numbers.astype(float), problems


def flag_rate_status(n_scored, n_flagged, budget_pct, min_scored=100, high=2.0, low=0.25):
    """Compare the share flagged so far with the review budget.
    Returns None (normal) or ('high' / 'low', flagged_percent, ratio_to_budget)."""
    if n_scored < min_scored:
        return None
    rate = 100 * n_flagged / n_scored
    ratio = rate / budget_pct
    if ratio >= high:
        return ('high', rate, ratio)
    if ratio <= low:
        return ('low', rate, ratio)
    return None
