from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if __name__ == '__main__':
    sentence = "unemployment has fallen by 10% in spain in the last four years."
    text = "paraphrase: " + sentence + " </s>"

    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

    inputs = tokenizer(text, pad_to_max_length=True, return_tensors="pt")
    tokens = model.generate(**inputs)

    print(tokenizer.decode(tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
