import pandas as pd
from pycountry import languages
from google.cloud import translate_v2 as translate


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


if __name__ == "__main__":
    claim_text_list = [
        {
            "text": "El paro en España ha bajado un 20% en los últimos 10 años."
        },
        {
            "text": "O paro en España caeu un 20% nos últimos 10 anos."
        }
    ]
    df = pd.DataFrame(claim_text_list)
    translate_client = translate.Client()
    df["language"], df["language_detection_confidence"] = zip(*df["text"].map(
        lambda x: detect_language(translate_client, x)
    ))
    print(df.head())
