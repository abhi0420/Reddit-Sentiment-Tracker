import pandas as pd
import torch.nn.functional as F

def read_test_data(file_path: str) -> pd.DataFrame:
    """Reads test data from a CSV file and returns it as a DataFrame."""
    df = pd.read_csv(file_path)
    df.fillna("", inplace=True)

    return df


def evaluate_models(absa_model, sarcasm_detection_pipeline, tokenizer):
    
    test_data = read_test_data("Data/test_data.csv")
    total_samples = len(test_data)
    accurate_sentiment = 0
    accurate_sarcasm = 0

    for i, row in test_data.iterrows():
        text = row.get('Text', "")
        title = row.get('Title', "")
        aspect = row.get('Aspect', "")
        true_sentiment = row.get('Sentiment', "").lower()
        true_sarcasm = str(row.get('Sarcasm Flag', "")).lower()

        full_text = (title + "\n" + text).strip()

        if absa_model:
            print("Analyzing text:", full_text)
            inputs = tokenizer(full_text, text_pair=aspect, return_tensors="pt", truncation=True)
            outputs = absa_model(**inputs)
            probs =  F.softmax(outputs.logits, dim=-1)[0].cpu().tolist()
            id2label = {int(k): v for k, v in absa_model.config.id2label.items()}
            label_probs = {id2label[i].lower(): probs[i] for i in range(len(probs))}
            predicted_sentiment = max(label_probs, key=label_probs.get)
            
            if predicted_sentiment == true_sentiment:
                accurate_sentiment += 1



        if sarcasm_detection_pipeline:
            sarcasm = sarcasm_detection_pipeline(text)[0]
            if sarcasm['label'] == 'irony':
                if sarcasm['score'] > 0.8:
                    predicted_sarcasm = 'True'
                else:
                    predicted_sarcasm = 'False'
            else:
                predicted_sarcasm = 'False'
            if predicted_sarcasm.lower() == true_sarcasm:
                accurate_sarcasm += 1
        
    sentiment_accuracy = (accurate_sentiment / total_samples) if total_samples > 0 else 0
    sarcasm_accuracy = (accurate_sarcasm / total_samples) if total_samples > 0 else 0

    return sentiment_accuracy, sarcasm_accuracy

        




def main():
    absa_model = None 
    sarcasm_model = None  
    tokenizer = None  
    evaluate_models(absa_model, sarcasm_model, tokenizer)
    

if __name__ == "__main__":
    main()