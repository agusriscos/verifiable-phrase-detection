import pandas as pd
import spacy


def read_claim_data(src_filepath):
    out_df = pd.read_csv(src_filepath)
    return out_df


def load_processing_pipeline(processing_pipeline_name="es_core_news_sm", disable_list=None, exclude_list=None):
    if exclude_list is None:
        exclude_list = []
    if disable_list is None:
        disable_list = []
    pipeline = spacy.load(processing_pipeline_name, disable=disable_list, exclude=exclude_list)
    return pipeline


def tokenize(pipeline, text, with_punctuation=True):
    # Only tokenization with/without punctuation removal
    claim_doc = pipeline(text)
    if with_punctuation:
        token_list = [token.text.lower() for token in claim_doc]
    else:
        token_list = [token.text.lower() for token in claim_doc if not token.is_punct]
    return token_list


def normalize(pipeline, text):
    # Tokenization and lemmatization removing stop_words and punctuation (for descriptive analysis)
    claim_doc = pipeline(text)
    # A token is significative if it is not a punctuation neither a stopword and it is not a number
    return [token.lemma_ for token in claim_doc if (not token.is_punct | token.is_stop)]


if __name__ == '__main__':
    claim_df = read_claim_data("../../data/ml_test_data.csv").sample(10)
    claim_pipeline = load_processing_pipeline()

    # Testing the NLP pipeline
    claim_text = "Las escuelas gratuitas están superando a las estatales en cuanto a nivel de alumnos."
    tokenized_claim = tokenize(claim_pipeline, claim_text)
    # print(tokenized_claim)
    normalized_claim = normalize(claim_pipeline, claim_text)
    # print(normalized_claim)

    # claim_df["num_words"] = claim_df["text"].apply(lambda x: len(tokenize(claim_pipeline, x, with_punctuation=False)))
    # claim_df["num_unique_words"] = claim_df["text"].apply(
    #     lambda x: len(set(normalize(claim_pipeline, x)))
    # )
    # print(claim_df.head())
