import pandas as pd
import numpy as np

df_w = pd.read_csv('data/wallenbeck_event_table_v2.csv')


# 1, gold-plating

def classify_gold_plating(row):
    if row['Left-Censored (pre-1991)'] == 'YES':
        return 'LEFT_CENSORED'
    if (row['Country'] == 'AUT' and
        'confinement limit' in str(row['Dimension'])):
        return 'BEYOND_NO_EU_EQUIVALENT'
    pre = row['Pre-Directive']
    if pre == 'YES':
        return 'PRE_DIRECTIVE'
    elif pre == 'SAME YEAR':
        return 'SAME_YEAR_EXCEEDS_MINIMUM'
    elif pre == 'NO':
        return 'POST_DIRECTIVE_EXCEEDS_MINIMUM'
    else:
        return 'UNKNOWN'

# Flag unreliable lag measures separately

def lag_reliability(row):
    if row['Left-Censored (pre-1991)'] == 'YES':
        return 'SWEDEN_REFERENCE'
    if pd.isna(row['years_after_sweden']):
        return 'NO_SWEDEN_EQUIVALENT'
    if row['gold_plating_category'] == 'SAME_YEAR_EXCEEDS_MINIMUM':
        return 'UNRELIABLE_LAG_DIRECTIVE_CONFOUNDED'
    if row['gold_plating_category'] == 'POST_DIRECTIVE_EXCEEDS_MINIMUM':
        return 'UNRELIABLE_LAG_POST_DIRECTIVE'
    return 'RELIABLE'

df_w['gold_plating_category'] = df_w.apply(classify_gold_plating, axis=1)
df_w['lag_reliability'] = df_w.apply(lag_reliability, axis=1)

# 2. Years ahead of Sweden

sweden_years = (df_w[df_w['Country'] == 'SWE']
                .set_index('Dimension')['Year Adopted']
                .to_dict())

def years_after_sweden(row):
    if row['Country'] == 'SWE':
        return 0
    dim = row['Dimension']
    if dim not in sweden_years:
        return np.nan
    swe_year = sweden_years[dim]
    return row['Year Adopted'] - swe_year

df_w['years_after_sweden'] = df_w.apply(years_after_sweden, axis=1)

# ── 3. In corpus window ───────────────────────────────────────────────────────

CORPUS_WINDOWS = {
    'AUT': (1996, 2019),
    'DEU': (2009, 2021),
    'DNK': (2007, 2022),
}

def in_corpus_window(row):
    window = CORPUS_WINDOWS.get(row['Country'])
    if window is None:
        return False
    return window[0] <= row['Year Adopted'] <= window[1]

df_w['in_corpus_window'] = df_w.apply(in_corpus_window, axis=1)

df_w.to_csv('data/wallenbeck_event_table_v3.csv', index=False)