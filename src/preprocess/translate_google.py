import os
from os.path import join, exists
import pandas as pd
from pycountry import languages
from google.cloud import translate_v2 as translator

from src.preprocess import dataset_split


def get_translation_client():
    return translator.Client()


def detect_language(client, text):
    result = client.detect_language(text)
    return result["language"], result["confidence"]


def get_language_names(language_iso_list):
    language_name_list = []
    for language_iso in language_iso_list:
        if len(language_iso) == 2:
            language_name = languages.get(alpha_2=language_iso).name
        else:
            language_name = languages.get(alpha_3=language_iso).name
        language_name_list.append(language_name)
    return language_name_list


def translate(client, text, src_language, dst_language):
    response = client.translate(text, source_language=src_language, target_language=dst_language)
    return response["translatedText"]


def main():
    # This code was executed to make Data Augmentation with back-translation and the Google Translation API
    translate_client = get_translation_client()
    claim_df = pd.read_csv("/home/agusriscos/verifiable-phrase-detection/data/ml_test_data.csv")

    # Detect language of all texts
    claim_df["language"], claim_df["language_detection_confidence"] = zip(*claim_df["text"].map(
        lambda x: detect_language(translate_client, x)))

    # Replace text translating into spanish (only claims in other languages)
    text_to_replace = claim_df.loc[(claim_df["language"] != "es") & (claim_df["claim"] == 1)][["text", "language"]]
    translated_text = list(text_to_replace.apply(
        lambda x: translate(translate_client, x["text"], x["language"], "es"), axis=1).values)
    claim_df.replace(list(text_to_replace["text"]), translated_text, inplace=True)

    # Translate all claims into english for paraphrasing and back-translation purposes
    claim_df["en_text"] = claim_df.apply(
        lambda x: translate(translate_client, x["text"], "es", "en") if x["claim"] == 1 else None, axis=1)
    claim_df.to_csv("/home/agusriscos/verifiable-phrase-detection/data/translated-raw-data.csv", index=False)

    # Train and test split (with significant representations of both labels) before transformations
    # to avoid data leakage
    output_dirpath = "/home/agusriscos/verifiable-phrase-detection/data"
    train_df, val_df, test_df = dataset_split.custom_train_val_test_split(
        claim_df,
        output_dirpath=output_dirpath
    )

    # Back translate es-en-es for data augmentation (only claim sentences)
    bt_output_dirpath = "/home/agusriscos/verifiable-phrase-detection/data/back-translation"
    if not exists(bt_output_dirpath):
        os.mkdir(bt_output_dirpath)
    for df, df_name in zip([train_df, val_df, test_df],
                           ["backtransl-train.csv", "backtransl-val.csv", "backtransl-test.csv"]):
        bt_claim_df = df.loc[df["claim"] == 1]
        bt_claim_df["text"] = bt_claim_df["en_text"].apply(lambda x: translate(translate_client, x, "en", "es"))
        bt_claim_df.to_csv(join(bt_output_dirpath, df_name), index=False)


if __name__ == "__main__":
    main()
