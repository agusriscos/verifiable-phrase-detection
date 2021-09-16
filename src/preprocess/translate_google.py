import sys
import os
from os.path import join, exists
import subprocess

import pandas as pd
from pycountry import languages
from google.cloud import translate_v2 as translator
from src.preprocess import prepare_model_data


def get_translation_client():
    """
    Get Google Translator API
    """
    return translator.Client()


def detect_language(client, text):
    """
    Detect language of a text with Google API
    """
    result = client.detect_language(text)
    return result["language"], result["confidence"]


def get_language_names(language_iso_list):
    """
    Get countries of a list of ISO names
    """
    language_name_list = []
    for language_iso in language_iso_list:
        if len(language_iso) == 2:
            language_name = languages.get(alpha_2=language_iso).name
        else:
            language_name = languages.get(alpha_3=language_iso).name
        language_name_list.append(language_name)
    return language_name_list


def translate(client, text, src_language, dst_language):
    """
    Translate a text with Google API
    """
    response = client.translate(text, source_language=src_language, target_language=dst_language)
    return response["translatedText"]


def prepare_for_data_augmentation(translate_client, dataframe, output_dirpath, output_filename="in_da_train.csv"):
    """
    Prepare train claims into english for data augmentation purposes.
    We use the output of this function as input for EDA (https://github.com/jasonwei20/eda_nlp).
    Clone their repo and follow their instructions.
    """
    if not exists(output_dirpath):
        os.makedirs(output_dirpath)

    en_df = dataframe.loc[dataframe["claim"] == 1][["claim", "text"]]
    en_df["en_text"] = en_df["text"].apply(
        lambda x: translate(translate_client, x, "es", "en"))
    en_df.drop(columns=["text"], inplace=True)
    en_df.to_csv(join(output_dirpath, output_filename), header=False, index=False, sep="\t")


def back_translate_claims(translate_client, dataframe, output_dirpath, output_filename="bt_train.csv"):
    """
    Back translate EDA texts into spanish
    """
    if not exists(output_dirpath):
        os.makedirs(output_dirpath)

    bt_df = dataframe.copy()
    bt_df["text"] = bt_df["en_text"].apply(lambda x: translate(translate_client, x, "en", "es"))
    bt_df.drop(columns=["en_text"], inplace=True)
    bt_df.to_csv(join(output_dirpath, output_filename), index=False)
    return bt_df


def main():
    """
    This code was executed to make Data Augmentation with back-translation and the Google Translation API.
    """
    # Path variables
    data_dirpath = "/home/agusriscos/verifiable-phrase-detection/data"
    raw_input_filepath = join(data_dirpath, "ml_test_data.csv")
    da_input_dirpath = join(data_dirpath, "data-augmentation/input")
    da_output_filepath = join(data_dirpath, "data-augmentation/output/out_da_train.csv")

    translate_client = get_translation_client()
    claim_df = pd.read_csv(raw_input_filepath)
    sample = claim_df.groupby("claim", group_keys=False).apply(lambda x: x.sample(frac=0.01))

    # Detect language of all texts
    sample["language"], sample["language_detection_confidence"] = zip(*sample["text"].map(
        lambda x: detect_language(translate_client, x)))

    # Replace text translating into spanish (only claims in other languages)
    text_to_replace = sample.loc[(sample["language"] != "es") & (sample["claim"] == 1)][["text", "language"]]
    translated_text = list(text_to_replace.apply(
        lambda x: translate(translate_client, x["text"], x["language"], "es"), axis=1).values)
    sample.replace(list(text_to_replace["text"]), translated_text, inplace=True)

    # Train and test split (with significant representations of both labels) before transformations
    # to avoid data leakage
    train_df, val_df, test_df = prepare_model_data.train_val_test_split(sample, output_dirpath=data_dirpath)

    # Translate train claims into english for data augmentation purposes
    prepare_for_data_augmentation(translate_client, train_df, output_dirpath=da_input_dirpath)

    # Back translate es-en-es for data augmentation (only claim sentences)
    # Execute EDA augment.py script before this.
    if exists(da_output_filepath):
        output_da_df = pd.read_csv(da_output_filepath, names=["claim", "en_text"], sep="\t")
    else:
        print("The back-translation cannot be performed because the file is not found.")
        sys.exit()

    bt_train_df = back_translate_claims(translate_client, output_da_df, output_dirpath=data_dirpath)
    print(bt_train_df.head())


if __name__ == "__main__":
    main()
