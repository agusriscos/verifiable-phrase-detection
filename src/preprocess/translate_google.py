import pandas as pd
from pycountry import languages
from google.cloud import translate_v2 as translator


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


def back_translate(client, text, src_language, dst_language):
    translated_text = translate(client, text, src_language, dst_language)
    back_translated_text = translate(client, translated_text, dst_language, src_language)
    return back_translated_text


if __name__ == "__main__":
    # This code was executed to detect and translated claim texts with Google API
    client = translate.Client()
    claim_df = pd.read_csv("/home/agusriscos/verifiable-phrase-detection/data/ml_test_data.csv").sample(10)
    claim_df["language"], claim_df["language_detection_confidence"] = zip(*claim_df["text"].map(
        lambda x: detect_language(client, x)
    ))

    claim_text = "El paro en España ha bajado un 20% en los últimos 10 años."
    print(claim_text)
    print(back_translate(translate_client, claim_text, "es", "en"))
