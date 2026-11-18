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
8. **Deployment:** Streamlit app

## Results

| Model                | Accuracy | Precision | Recall | F1-score |
|----------------------|----------|-----------|--------|----------|
| Multinomial NB       | [xx%]    | [xx%]     | [xx%]  | [xx%]    |
| Logistic Regression  | [xx%]    | [xx%]     | [xx%]  | [xx%]    |
| Linear SVM           | [xx%]    | [xx%]     | [xx%]  | [xx%]    |

**Best model:** [model name], chosen because [reason, e.g. highest F1 on spam class].

Accuracy alone is misleading on imbalanced data, so precision, recall, and F1 for the spam class were the main metrics.

## Project Structure

```
spamshield-sms-classifier/
├── data/
│   └── spam.csv
├── notebooks/
│   └── spam_classifier.ipynb
├── app.py
├── model.pkl
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/[your-username]/spamshield-sms-classifier.git
cd spamshield-sms-classifier
pip install -r requirements.txt
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

## Key Learnings

- Converting text to numbers with TF-IDF
- Why accuracy fails on imbalanced datasets
- Using pipelines to avoid data leakage
- Comparing multiple models fairly

## Limitations and Future Improvements

- Trained on English SMS only; may not generalize to other languages or email
- Could improve with a transformer model such as DistilBERT
- Add support for Indian languages and code-mixed text (Hinglish)
- Add model explainability to show which words trigger the spam label

## Author

**[Your Name]**
[LinkedIn](#) · [GitHub](#)

## License

This project is licensed under the MIT License.