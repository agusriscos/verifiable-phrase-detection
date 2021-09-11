import pandas as pd
import transformers
from src.detector.bert_finetune_imdb import CustomDataset, compute_metrics

if __name__ == '__main__':
    test_df = pd.read_csv('../../data/aclImdb/imdb_test.csv')
    test_texts = test_df["text"].values.tolist()
    test_labels = test_df["label"].values.tolist()

    tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-cased")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = CustomDataset(test_encodings, test_labels)

    model_weights_path = "../../data/imdb-bert-test-results/checkpoint-4800"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_weights_path)
    args = transformers.TrainingArguments(
        output_dir='../../data/imdb-bert-logs',
        logging_dir='../../data/imdb-bert-logs/test'
    )

    trainer = transformers.Trainer(
        model=model,
        args=args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    print(trainer.evaluate())
