# 📰 Fake News Detector

A machine learning project that classifies news articles as **real** or **fake** using **TF-IDF vectorization** and a **Logistic Regression** classifier, built in Python.
## 📌 Overview

The spread of misinformation online makes it increasingly important to have tools that can quickly assess the credibility of a news article. This project builds a supervised text-classification pipeline that:

1. Cleans and preprocesses raw news text
2. Converts text into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**
3. Trains a **Logistic Regression** model to classify articles as `REAL` or `FAKE`
4. Evaluates the model's performance on a held-out test set
## ✨ Features

- Text preprocessing (cleaning, tokenization, stopword removal)
- TF-IDF feature extraction
- Logistic Regression classification
- Model evaluation with accuracy, precision, recall, and confusion matrix
- Simple script/notebook to test the model on custom input text
## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML / Data | scikit-learn, pandas, NumPy |
| Text Processing | TF-IDF (`sklearn.feature_extraction.text`) |
| Model Persistence | joblib |
## 📂 Project Structure

```
fake-news-detector/
│
├── data/
│   └── fake_or_real_news.csv   # Dataset (title, text, label)
├── src/
│   ├── preprocess.py           # Text cleaning (lowercase, remove URLs/punctuation/stopwords)
│   ├── train.py                # Full training pipeline (TF-IDF + Logistic Regression)
│   └── predict.py              # Load saved model and classify new text
├── model/
│   ├── logistic_regression.pkl # Trained classifier
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
│   └── metrics.txt             # Saved evaluation metrics
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```
## 📊 Dataset

- **Source:** [fake_or_real_news.csv](https://github.com/lutzhamel/fake-news) — originally compiled by George McIntire
- **Size:** 6,335 articles — 3,171 REAL / 3,164 FAKE (well balanced)
- **Features used:** Article title + text (combined)
- **Labels:** `REAL`, `FAKE`
## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/data96-pixel/fake-news-detector.git
   cd fake-news-detector
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## ▶️ Usage

1. **Train the model:**
   ```bash
   cd src
   python train.py
   ```
   This loads `data/fake_or_real_news.csv`, cleans the text, fits a TF-IDF vectorizer,
   trains the Logistic Regression model, prints evaluation metrics, and saves the
   trained model + vectorizer into `model/`.
2. **Classify new text:**
   ```bash
   python predict.py --text "Your news article text here"
   ```
   Output includes the predicted label (`FAKE`/`REAL`) and the model's confidence for each class.
## 📈 Results

Evaluated on a held-out 20% test set (1,267 articles):

| Metric | Score |
|---|---|
| Accuracy | 0.9266 |
| Precision (FAKE) | 0.9030 |
| Recall (FAKE) | 0.9558 |
| F1-score (FAKE) | 0.9286 |

**Confusion matrix** (rows = actual, cols = predicted):

|            | Pred. FAKE | Pred. REAL |
|---|---|---|
| **Actual FAKE** | 605 | 28 |
| **Actual REAL** | 65  | 569 |

> ⚠️ **Note:** The model is trained on ~6,300 news articles from 2016-era political
> coverage, so it performs best on similarly styled, longer-form political news text.
> Short, generic, or out-of-domain sentences (e.g., a single made-up test sentence)
> may be misclassified — this is expected behavior for a TF-IDF + Logistic Regression
> baseline, not a bug.
### 🧪 Real-world sanity check

Testing the model on an article from its own 2016-era training data correctly
predicted **REAL** with 84.5% confidence. However, testing it on a genuine, current
(2026) news article about an unrelated topic (international energy/business news)
was **misclassified as FAKE** (61% confidence).

This is a textbook example of **data drift**: the model doesn't learn "what makes
news true," it learns the statistical word patterns of its training set. A decade
later, different vocabulary, entities, and topics (companies, geopolitical events,
years) shift the input distribution away from what the model was trained on, so
accuracy drops on out-of-domain or newer text — even though the article is 100%
real. This is a known limitation of simple bag-of-words models and a good argument
for periodically retraining on fresh data, or using more context-aware models
(e.g., transformer-based ones) for production use.
## 🚀 Future Improvements
- Experiment with additional models (e.g., Naive Bayes, SVM, or transformer-based models like BERT)
- Add a web interface for real-time article checking
- Expand the dataset for better generalization
- Deploy as an API or browser extension
## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.
## 📄 License
This project is licensed under the [MIT License](LICENSE) — feel free to use and modify it.
## 👤 Author
Khayala Aliyeva - I am a data analytics/science student at Baku Engineering University & INHA University. I am also studying at datacras on DevOps & MLOps & Data Engineering & Platform Engineering|Data Reporting Analyst intern| 🔗 GitHub Repository
Data Reporting Analyst Intern | ICT Student, Baku Engineering University
🔗 [GitHub Repository](https://github.com/data96-pixel/fake-news-detector)
