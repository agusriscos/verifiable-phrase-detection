import pandas as pd
import spacy


def read_claim_data(src_filepath):
    out_df = pd.read_csv(src_filepath)
    return out_df


def load_processing_pipeline(processing_pipeline_name="es_core_news_sm", disable_list=None, exclude_list=None):
    """
    Load Spacy NLP pipeline
    """
    if exclude_list is None:
        exclude_list = []
    if disable_list is None:
        disable_list = []
    pipeline = spacy.load(processing_pipeline_name, disable=disable_list, exclude=exclude_list)
    return pipeline


def tokenize(pipeline, text, with_punctuation=False):
    """
    Simple tokenization for word count analysis
    """
    claim_doc = pipeline(text)
    if with_punctuation:
        token_list = [token.text for token in claim_doc]
    else:
        token_list = [token.text for token in claim_doc if not token.is_punct]
    return token_list


def filter_significant_words(pipeline, text):
    """
    Function to filter significant words with custom rules:
        - With numerical words and pronouns in order to detect quantities and claims based on personal experience.
        - Without stop words (including CCONJ words and punctuation).
    """
    claim_doc = pipeline(text)
    token_list = []
    for token in claim_doc:
        if str(token.pos_) in ["NUM", "PRON"]:
            token_list.append(token.lemma_.lower())
        elif (not token.is_punct | token.is_stop) and (str(token.pos_) != "CCONJ"):
            token_list.append(token.lemma_.lower())
        else:
            pass
    return set(token_list)


def normalize(pipeline, text, with_stopwords=False):
    """
    Normalize text (tokenization -> punctuation (always) and stop words removal (optional))
    """
    claim_doc = pipeline(text)
    if with_stopwords:
        return [token.lower() for token in claim_doc if not token.is_punct | token.is_stop]
    else:
        return "".join([token.lower() for token in claim_doc if not token.is_punct])


if __name__ == '__main__':
    # claim_df = read_claim_data("../../data/ml_test_data.csv").sample(10)
    claim_pipeline = load_processing_pipeline("es_core_news_lg")

    # Testing the NLP pipeline
    claim_text = "El paro en España subirá un 20% en los próximos cuatro años y el PIB bajará un 5%."
    tokenized_claim = tokenize(claim_pipeline, claim_text)
    print(tokenized_claim)
    normalized_claim = filter_significant_words(claim_pipeline, claim_text)
    print(normalized_claim)
