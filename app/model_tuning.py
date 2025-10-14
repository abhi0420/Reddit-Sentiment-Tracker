import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)

from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


class AbsaDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        encoding = self.tokenizer(item["text"],
                                  text_pair = item["text_pair"],
                                  truncation=True,
                                  padding = "max_length",
                                  max_length = self.max_len,
                                  return_tensors="pt" 
                                  )
        
        return {
            'input_ids' : encoding['input_ids'].flatten(),
            'attention_mask' : encoding['attention_mask'].flatten(),
            'labels' : torch.tensor(item['labels'], dtype=torch.long)
        }
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'classification_report': classification_report(labels, predictions, output_dict=True)
    }

def fine_tune_model():

    train_df = pd.read_csv("")

    train_data = train_df[:int(0.8*len(train_df))]
    val_data = train_df[int(0.8*len(train_df)):]

    print("Training size :", len(train_data))
    print("Validation size :", len(val_data))

    model_name = "yangheng/deberta-v3-base-absa-v1.1"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3,
        problem_type="single_label_classification"
    )

    train_dataset = AbsaDataset(train_data, tokenizer)
    val_dataset = AbsaDataset(val_data, tokenizer)

    training_args = TrainingArguments(
        output_dir="./fine_tuned_absa_model_v1",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to = None,
        save_total_limit=2
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Starting fine-tuning...")

    trainer.train()

    tokenizer.save_pretrained("./fine_tuned_absa_model_v1")

    print("Fine-tuning complete. Model saved to ./fine_tuned_absa_model_v1")

    eval_results = trainer.evaluate()

    print("Evaluation results:", eval_results)

    return trainer.model, tokenizer



if __name__ == "__main__":
    fine_tune_model()
    