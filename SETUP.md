## Setup Guide

## System Requirements

- **OS**: Windows 10+, macOS 10.15+, or Linux
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space for models

## Step-by-Step Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Reddit-Sentiment-Tracker.git
cd Reddit-Sentiment-Tracker
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**If you encounter issues:**
```bash
# For PyTorch with CUDA (GPU support):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only:
pip install torch torchvision torchaudio
```

### 4. Set Up Reddit API

1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Fill in:
   - **name**: Your app name
   - **App type**: script
   - **redirect uri**: http://localhost:8000

4. Copy your `client_id` and `client_secret`

5. Create `.env` file:
```env
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=MyRedditSentimentBot/1.0
```

### 5. Download Models (First Run)

Models will auto-download on first run :
cd app
python nlp_models.py
```

### 6. Run the Application

```bash
uvicorn main:app --reload --port 8000
```

Visit: http://localhost:8000/docs



### Issue: Reddit API authentication fails
**Solution**: Double-check `.env` credentials and ensure no extra spaces

### Issue: Import errors
**Solution**: Ensure virtual environment is activated:
```bash
# Check Python path
which python  # Should point to .venv
```

