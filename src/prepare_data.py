import pandas as pd
from sklearn.model_selection import train_test_split


def rename_cols(df):
    df.rename(columns={'A': 'age', 'B': 'sex', 'C': 'text', 'D': 'isced', 'E': 'isced2', 'F': '#anwps', 'G': 'clean_text', 'H': 'count_char', 'I': 'count_punct', 'J': 'count_word', 'K': 'avg_wordlength', 'L': 'count_misspelled', 'M': 'word_uniqueness'}, inplace=True)
    df.dropna(subset=['isced', 'clean_text'], inplace=True)
    df['isced'].mask(df['isced'].isin([3.0, 5.0, 1.0]), 0, inplace=True)
    df['isced'].mask(df['isced'].isin([6.0, 7.0, 8.0]), 1, inplace=True)
    df['clean_text'] = df['clean_text'].str.replace('\d+', ' ')


def select_testset(df):
    df, df_test = train_test_split(df, stratify=df['isced'], test_size=0.25, random_state=0)
    df_test.to_csv(r"../data/raw/testset.csv")
    return df


def separate_data(df):
    clean_text = df['clean_text']
    target = df.isced
    meta = df.iloc[:, 5:13]
    meta = meta.loc[:, meta.columns != 'clean_text']
    liwc = df.iloc[:, 13:]
    liwc.replace(',', '.', inplace=True, regex=True)
    liwc = liwc.astype(float)
    liwc_text = pd.concat([liwc, clean_text], axis=1)

    # Write sprated data to "data/processed/separate" folder
    clean_text.to_csv(r"../data/processed/separate/clean_text.csv", index=False)
    target.to_csv(r"../data/processed/separate/target.csv", index=False)
    meta.to_csv(r"../data/processed/separate/meta.csv", index=False)
    liwc.to_csv(r"../data/processed/separate/liwc.csv", index=False)
    liwc_text.to_csv(r"../data/processed/separate/liwc_text.csv", index=False)
