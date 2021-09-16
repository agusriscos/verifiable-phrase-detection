from os.path import join
from sklearn.model_selection import train_test_split
from src.preprocess.preprocess_spacy import filter_significant_words, load_processing_pipeline
import pandas as pd


def train_val_test_split(dataframe, output_dirpath):
    train_df, rem_df = train_test_split(dataframe, train_size=0.8, stratify=dataframe["claim"])
    val_df, test_df = train_test_split(rem_df, train_size=0.5, stratify=rem_df["claim"])

    for df, df_name in zip([train_df, val_df, test_df], ["raw_train.csv", "raw_val.csv", "raw_test.csv"]):
        out_df = df.copy()
        if "train" in df_name:
            filter_cols = ["text", "claim", "language_detection_confidence"]
        else:
            filter_cols = ["text", "claim"]
        out_df.drop(columns=[col for col in out_df.columns if col not in filter_cols], inplace=True)
        out_df.to_csv(join(output_dirpath, df_name), index=False)

    return train_df, val_df, test_df


def prepare_training_data(pipeline, df, bt_df, output_dirpath, output_filename="prep_train.csv"):
    """
    Prepare data for training the model.
        * Remove negative sentences with few significant words.
        * Remove negative sentences with worst language detection confidence (most of them  have grammar and spell errors).
        * Add data augmentation rows.
        * Apply class balancing reducing the negative class of the remaining data
    """

    df["num_significant_words"] = df["text"].apply(
        lambda x: len(set(filter_significant_words(pipeline, x))))
    aux_df = df.loc[~((df["claim"] == 0) & (df["num_significant_words"] < 2))]
    aux_df = aux_df.loc[~((aux_df["claim"] == 0) & (aux_df["language_detection_confidence"] < 0.8))]
    aux_df = pd.concat([aux_df[["text", "claim"]], bt_df[["text", "claim"]]]).drop_duplicates(subset=["text"])

    claim_count = aux_df["claim"].value_counts().values
    not_claim = aux_df.loc[aux_df["claim"] == 0].sample(claim_count[1])
    out_df = pd.concat([aux_df.loc[aux_df["claim"] == 1], not_claim]).sample(frac=1)
    out_df.to_csv(join(output_dirpath, output_filename), index=False)
    return out_df


def main():
    data_dirpath = "/home/agusriscos/verifiable-phrase-detection/data"
    claim_pipeline = load_processing_pipeline()
    train = pd.read_csv("/home/agusriscos/verifiable-phrase-detection/data/raw_train.csv")
    bt_train = pd.read_csv("/home/agusriscos/verifiable-phrase-detection/data/bt_train.csv")
    out_df = prepare_training_data(claim_pipeline, train, bt_train, data_dirpath)
    print(out_df)


if __name__ == '__main__':
    main()
