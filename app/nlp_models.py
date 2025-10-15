from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import warnings

warnings.filterwarnings("ignore")





def load_models(use_latest = True):

    DEVICE = "cuda" if torch.cuda.is_available() else -1

    fine_tuned_model_path = "./fine_tuned_absa_model_v1"

    if use_latest and os.path.exists(fine_tuned_model_path):
        print("Loading fine tuned model for ABSA....")
        ABSA_MODEL_NAME = fine_tuned_model_path

    else:
        print("Loading base model for ABSA....")
        ABSA_MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"


    absa_model = AutoModelForSequenceClassification.from_pretrained(ABSA_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(ABSA_MODEL_NAME)
    sentiment_analysis_pipeline = pipeline("sentiment-analysis",model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    sarcasm_detection_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony")

    print("Models Loaded Successfully")



    return {
        "absa_model": absa_model,
        "sentiment_analysis_pipeline": sentiment_analysis_pipeline,
        "sarcasm_detection_pipeline": sarcasm_detection_pipeline,
        "tokenizer": tokenizer
    }
     

