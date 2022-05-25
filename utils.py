from operator import itemgetter

import unicodedata
import arff
import pandas as pd


def load_arff_to_df(filepath):
    data = arff.load(open(filepath, 'r'))
    getitem_0 = itemgetter(0)
    columns = list(map(getitem_0, data['attributes']))
    return pd.DataFrame(data['data'], columns=columns)




def normalize_text_cols(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).apply(
                lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode("ascii").lower()).apply(
                lambda s: ''.join(filter(str.isalnum, s)))
    return df


def merge_levels(df, threshold=0.05, label="other"):
    count_threshold = len(df) * threshold
    for col in df.columns:
        if df[col].dtype == object:
            count_df = df[col].value_counts()
            for cat, cnt in count_df.iteritems():
                if cnt < count_threshold:
                    df[col][df[col] == cat] = label


def print_value_counts(df, dtype=object):
    for col in df.columns:
        if df[col].dtype == dtype:
            print(col)
            print(df[col].value_counts(dropna=False))


def find_value_counts_below(df, threshold, dtype=object):
    count_threshold = len(df) * threshold
    for col in df.columns:
        if df[col].dtype == dtype:
            count_df = df[col].value_counts()
            for cat, cnt in count_df.iteritems():
                if cnt < count_threshold:
                    print(col, cat)


def booleans_to_int(df: pd.DataFrame):
    mappings = {}
    for col in df.columns[df.dtypes == object]:
        categories = df[col].astype("category").cat.categories
        if len(categories) == 2:
            mappings[col] = {i: val for i, val in enumerate(categories)}
            df[col] = df[col].astype("category").cat.codes
    return mappings

def text_cols_to_int(df: pd.DataFrame):
    mappings = {}
    for col in df.columns[df.dtypes == object]:
        categories = df[col].astype("category").cat.categories
        mappings[col] = {i: val for i, val in enumerate(categories)}
        df[col] = df[col].astype("category").cat.codes
        # df[col] = df[col].astype("category")
    return mappings
