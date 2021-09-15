import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.preprocess.preprocess_spacy import tokenize, load_processing_pipeline


def load_paraphraser(model_name="Vamsi/T5_Paraphrase_Paws"):
    t5_tokenizer = AutoTokenizer.from_pretrained(model_name)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return t5_tokenizer, t5_model


def paraphrase(tokenizer, paraphraser, sentence):
    text = "paraphrase: " + sentence + " </s>"
    inputs = tokenizer(text, padding=True, return_tensors="pt")
    tokens = paraphraser.generate(**inputs)
    return tokenizer.decode(tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


if __name__ == '__main__':

    paraphraser_tokenizer, paraphrase_model = load_paraphraser()
    print(paraphrase(paraphraser_tokenizer, paraphrase_model,
                     "Unemployment in Spain has dropped 20% in the last 10 years."))
    claim_df = pd.read_csv("/home/agusriscos/verifiable-phrase-detection/data/translated-raw-data.csv")

    claim = claim_df.loc[(claim_df["claim"] == 1)]
    claim_pipeline = load_processing_pipeline("en_core_web_sm")
    claim["num_words"] = claim["en_text"].apply(
        lambda x: len(tokenize(claim_pipeline, x, with_punctuation=False)))
    short_claim = claim.loc[claim["num_words"] < 20][["en_text", "claim"]]
    short_claim["text"] = short_claim["en_text"].apply(
        lambda x: paraphrase(paraphraser_tokenizer, paraphrase_model, x)
    )