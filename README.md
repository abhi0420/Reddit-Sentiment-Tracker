# 🎯 Reddit Sentiment Tracker

> An advanced sentiment analysis system for Reddit posts using fine-tuned transformers and multi-model ensemble approach.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Features

- **Aspect-Based Sentiment Analysis (ABSA)** - Analyze sentiment toward specific entities (Tesla, Bitcoin, etc.)
- **Sarcasm Detection** - Automatically detects and handles sarcastic comments
- **Custom Fine-tuning** - Fine-tune models on domain-specific Reddit data
- **Multi-Model Ensemble** - Combines DeBERTa ABSA, RoBERTa sentiment, and irony detection
- **FastAPI Backend** - Production-ready REST API
- **Comprehensive Analytics** - Sentiment trends, top posts, and detailed summaries

## Quick Start

### Prerequisites

```bash
Python 3.8+
pip or conda
Reddit API credentials (optional, for live data)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Reddit-Sentiment-Tracker.git
cd Reddit-Sentiment-Tracker

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
```

### Run the API

```bash
cd app
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Usage

### Basic Sentiment Analysis

```python
import requests

response = requests.post("http://localhost:8000/analyze", json={
    "text": "Tesla",
    "subreddit": "technology",
    "limit": 10
})

results = response.json()
print(f"Average Sentiment: {results['summary']['average_sentiment']}")
```

### Example Response

```json
{
  "posts": [
    {
      "title": "Tesla Model Y Review",
      "sentiment": "positive",
      "sentiment_score": 0.85,
      "sarcasm_flag": false,
      "method": "absa"
    }
  ],
  "summary": {
    "average_sentiment": 0.72,
    "most_positive_post": {...},
    "most_negative_post": {...}
  }
}
```

## Architecture

```
        ┌─────────────────┐
        │  Reddit API     │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  Data Fetcher   │
        └────────┬────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│      Multi-Model Ensemble           │
│  ┌───────────┐  ┌────────────┐      │
│  │ DeBERTa   │  │  RoBERTa   │      │
│  │   ABSA    │  │ Sentiment  │      │
│  └─────┬─────┘  └──────┬─────┘      │
│        │               │            │
│        └────────┬──────┘            │
│                 ▼                   │
│        ┌─────────────────┐          │
│        │ Sarcasm Detector│          │
│        └─────────┬───────┘          │
└──────────────────┼──────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │   Sentiment     │
         │   Aggregation   │
         └─────────────────┘
```

## Fine-tuning

Fine-tune the ABSA model on your own data:

```python
from model_tuning import fine_tune_model

# Prepare your data in CSV format:
# text, aspect, label
# "Tesla is amazing", "Tesla", "positive"

model, tokenizer = fine_tune_model()
```

## Model Performance

| Model | Accuracy | Use Case |
|-------|----------|----------|

| DeBERTa ABSA (Base) | 79% | Aspect-specific sentiment |

| RoBERTa Irony | 87% | Sarcasm detection |

## Tech Stack

- **Backend**: FastAPI, Python 3.8+
- **ML Framework**: PyTorch, Transformers (Hugging Face)
- **Models**: 
  - DeBERTa v3 (ABSA)
  - RoBERTa (Sentiment & Irony)
- **Data**: PRAW (Reddit API)
- **Deployment**: Docker-ready

## Project Structure

```
Reddit-Sentiment-Tracker/
├── app/
│   ├── main.py              # FastAPI application
│   ├── services/
│   │   ├── fetch_data.py    # Reddit data fetching
│   │   └── analyze.py       # Sentiment analysis
│   ├── nlp_models.py        # Model loading
│   ├── model_tuning.py      # Fine-tuning pipeline
│   ├── evaluate.py          # Model evaluation
│   └── insights.py          # Analytics & visualization
├── Data/
│   ├── test_data.csv        # Evaluation dataset
│   └── extended_training_data.csv
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add Feature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for pre-trained models
- [DeBERTa ABSA](https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1)
- [Cardiff NLP](https://huggingface.co/cardiffnlp) for RoBERTa models

## Contact

Abhinand H - [https://www.linkedin.com/in/abhinand-h-74616a1b8/]

Project Link: [https://github.com/abhi0420/Reddit-Sentiment-Tracker]

---

⭐ Star this repo if you find it helpful!