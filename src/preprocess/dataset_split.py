from os.path import join
import pandas as pd


def custom_train_val_test_split(dataframe, output_dirpath):
    train_df = dataframe.groupby("claim", group_keys=False).apply(lambda x: x.sample(frac=.8))
    rem_df = dataframe.iloc[~dataframe.index.isin(train_df.index)]
    val_df = rem_df.groupby("claim", group_keys=False).apply(lambda x: x.sample(frac=.5))
    test_df = rem_df.iloc[~rem_df.index.isin(val_df.index)]

    for df, df_name in zip([train_df, val_df, test_df], ["train.csv", "val.csv", "test.csv"]):
        df.to_csv(join(output_dirpath, df_name), index=False)
    return train_df, val_df, test_df


if __name__ == '__main__':
    claim_df = pd.read_csv("/home/agusriscos/verifiable-phrase-detection/data/ml_test_data.csv")
    # custom_train_val_test_split(claim_df, output_dirpath="/home/agusriscos/verifiable-phrase-detection/data")
