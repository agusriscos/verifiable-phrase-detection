from os.path import join
import pandas as pd
import transformers
from src.model.finetune_bert import CustomDataset, compute_metrics

if __name__ == '__main__':
    data_dirpath = "/home/agusriscos/verifiable-phrase-detection/data"
    logs_dirpath = "../../training/bert"
    model_weights_path = join(logs_dirpath, "checkpoint-1400")

    test_df = pd.read_csv(join(data_dirpath, 'test.csv'))
    test_texts = test_df["text"].values.tolist()
    test_labels = test_df["claim"].values.tolist()

    tokenizer = transformers.AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = CustomDataset(test_encodings, test_labels)

    args = transformers.TrainingArguments(
        output_dir='../../training/dummy',
        logging_dir=join(logs_dirpath, 'test-logs')
    )

    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_weights_path)
    trainer = transformers.Trainer(
        model=model,
        args=args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    result = trainer.predict(test_dataset)
    test_df["predict_claim"] = result.predictions.argmax(-1)
    test_df.to_csv(join(data_dirpath, "predictions.csv"))
    print(result.metrics)

