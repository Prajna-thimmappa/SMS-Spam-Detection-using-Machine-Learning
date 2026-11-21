# SMS-Spam-Detection-using-Machine-Learning
# SpamShield: SMS Spam Detection using NLP and Machine Learning

A machine learning project that classifies SMS messages as **spam** or **ham** (not spam) using text preprocessing, TF-IDF, and classical ML models.

[Live Demo](#) · [Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## Overview

Spam messages waste time and can be harmful (phishing, scams). This project builds a classifier that detects spam from the text of a message alone, and compares several models to find the best performer.

## Dataset

- **Source:** SMS Spam Collection (UCI / Kaggle)
- **Size:** ~5,500 messages
- **Classes:** ham (~87%), spam (~13%), so the data is imbalanced

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- NLTK
- Matplotlib, Seaborn
- Streamlit (for the web app)

## Project Workflow

1. **Data exploration:** class balance, message length, common words
2. **Text cleaning:** lowercasing, removing punctuation and stopwords, lemmatization
3. **Feature extraction:** TF-IDF vectorization
4. **Train/test split:** stratified to preserve the spam ratio
5. **Model training:** Multinomial Naive Bayes, Logistic Regression, Linear SVM
6. **Evaluation:** precision, recall, F1-score, confusion matrix
7. **Pipeline:** scikit-learn `Pipeline` to prevent data leakage
8. **Explainability:** word-level spam indicators per prediction (`POST /explain`, Streamlit)
9. **Deployment:** Streamlit app and FastAPI API, plus Docker

## Results

| Model                | Accuracy | Precision | Recall | F1-score |
|----------------------|----------|-----------|--------|----------|
| Multinomial NB       | 97%      | 100%      | 81%    | 90%      |
| Logistic Regression  | 97%      | 100%      | 77%    | 87%      |
| Linear SVM           | 98%      | 97%       | 89%    | 93%      |

**Best model:** Multinomial NB, chosen because it supports `predict_proba` for confidence scores in the Streamlit app and FastAPI API.

Linear SVM has the highest spam F1 (93%) but lacks `predict_proba`, so it can't serve confidence scores.

Accuracy alone is misleading on imbalanced data, so precision, recall, and F1 for the spam class were the main metrics.

## Project Structure

```
spamshield-sms-classifier/
├── spam.csv
├── spam_pipeline.joblib
├── sms_spam.ipynb
├── app.py
├── backend.py
├── preprocess.py
├── train_pipeline.py
├── explain.py
├── test_spam.py
├── test_explain.py
├── test_preprocess.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .github/workflows/ci.yml
├── classification_report.json
├── confusion_matrices/
│   ├── confusion_matrix_MultinomialNB.png
│   ├── confusion_matrix_LogisticRegression.png
│   └── confusion_matrix_LinearSVC.png
├── step1_explore.py
├── step2_clean.py
├── step3_tfidf.py
├── step4_split.py
├── step5_train.py
├── step6_eval.py
├── step7_pipeline.py
├── step8_custom.py
└── README.md
```

## Installation

```bash
git clone https://github.com/[your-username]/spamshield-sms-classifier.git
cd spamshield-sms-classifier
pip install -r requirements.txt
```

Or build and run with Docker:

```bash
docker compose up
# Streamlit: http://localhost:8501
# FastAPI:   http://localhost:8000
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then paste a message into the text box and click **Predict**.

**Example:**

| Message | Prediction |
|---------|------------|
| "Congratulations! You've won a free iPhone. Click here to claim." | Spam |
| "Hey, are we still meeting at 6 today?" | Ham |

Run the FastAPI API:

```bash
uvicorn backend:app --reload
# GET  http://localhost:8000/health
# POST http://localhost:8000/predict   {"text": "..."}
# POST http://localhost:8000/explain   {"text": "..."}
```

Or with Docker:

```bash
docker compose up
```

**Example (API):**

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text":"Congratulations! You won a free iPhone. Click here"}'
```

Prediction includes a confidence score; `/explain` returns the top words driving the spam label.

## Key Learnings

- Converting text to numbers with TF-IDF
- Why accuracy fails on imbalanced datasets
- Using pipelines to avoid data leakage
- Comparing multiple models fairly

## Limitations and Future Improvements

- Trained on English SMS only; may not generalize to other languages or email
- Hinglish/code-mixed support added (`is_hinglish()`, `clean_text(remove_hinglish=True)`)
- Could improve with a transformer model such as DistilBERT
- Add model explainability to show which words trigger the spam label (done — `/explain`)

## Author

**[Your Name]**
[LinkedIn](#) · [GitHub](#)

## License

This project is licensed under the MIT License.