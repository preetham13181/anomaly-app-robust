"""Builds the demonstration files in test_data/ from the supplied new_transactions.csv."""
import numpy as np, pandas as pd
base = pd.read_csv('new_transactions.csv')

# 1) invalid values the ORIGINAL app silently scores (no crash): blank, negative, impossible
a = base.copy()
a['minutes_since_previous'] = a['minutes_since_previous'].astype(float)
a.loc[2,  'amount'] = np.nan                    # blank amount
a.loc[6,  'amount'] = -25.0                     # negative amount
a.loc[11, 'customer_typical_amount'] = -40.0    # negative typical amount
a.loc[15, 'minutes_since_previous'] = np.nan    # blank gap
a.loc[21, 'account_age_days'] = -3              # negative account age
a.loc[30, 'transactions_last_hour'] = np.nan    # blank count
a.to_csv('test_data/invalid_values.csv', index=False)

# 2) zero values that CRASH the original app (division by a typical amount of 0)
b = base.copy()
b.loc[9,  'customer_typical_amount'] = 0.0
b.loc[40, 'amount'] = 0.0
b.loc[55, 'customer_typical_amount'] = 0.0
b.to_csv('test_data/crash_values.csv', index=False)

# 3) a column is missing (an export changed)
base.drop(columns=['customer_typical_amount']).to_csv('test_data/missing_column.csv', index=False)

# 4) customer behavior changed: amounts 2.5x higher and 2 extra transactions per hour
d = base.copy()
d['amount'] = (d['amount'] * 2.5).round(2)
d['transactions_last_hour'] = d['transactions_last_hour'] + 2
d.to_csv('test_data/behavior_changed.csv', index=False)
