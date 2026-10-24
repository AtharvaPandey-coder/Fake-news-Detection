# 📰 Fake News Detector

A machine learning web app that classifies a news headline or article as **Real** or **Fake**, explains *why* using SHAP, and also reports the text's **sentiment** and **category**.

---

## Features

- **Fake / Real prediction** with a confidence score (TF-IDF + Logistic Regression)
- **Explainability (SHAP):** shows the top words that pushed the prediction toward Fake or Real
- **Sentiment analysis:** Positive / Negative / Neutral using VADER
- **Category tagging:** politics, sports, business, entertainment, health, world, or general (keyword-based)
- **Streamlit web app:** paste any text and get results instantly

---

## Dataset

- **Source:** TODO (add Kaggle dataset name and link)
- **Size:** 45,757 labelled news articles
- **Columns:** `text`, `label` (0 = Real, 1 = Fake)
- **Class balance:** roughly 50/50 (test set: 4,572 Real vs 4,580 Fake)
- **Split:** 80% train / 20% test (9,152 test samples)

---

## Approach

1. **Text cleaning:** lowercasing, removing punctuation and stopwords (`clean_text` function)
2. **Feature extraction:** TF-IDF vectorizer (fit on training data only, to avoid data leakage)
3. **Model training and comparison:** Decision Tree, AdaBoost, Logistic Regression
4. **Overfitting check:** compared train vs test accuracy
5. **Explainability:** SHAP `LinearExplainer` on the Logistic Regression model
6. **Enrichment:** VADER sentiment and keyword-based category tagging
7. **Deployment:** Streamlit app using pickled model, vectorizer and explainer

---

## Results

| Model | Test Accuracy | F1-score | Notes |
|---|---|---|---|
| Decision Tree (default) | 0.9773 | 0.98 | Train accuracy was 0.9999, so it overfits (about 2.3% train-test gap) |
| **Logistic Regression** | **0.9717** | 0.97 | ROC-AUC: 0.9717, precision and recall 0.97 for both classes |

### Why Logistic Regression was chosen

The unconstrained Decision Tree memorised the training data (about 2.3% train–test gap). Logistic Regression scored almost the same on test data (0.972 vs 0.977), but it is a much simpler model and works directly with SHAP's `LinearExplainer`, so every prediction can be explained quickly. For this project I valued **explainability and simplicity** over a tiny difference in accuracy.

---

## Explainability (SHAP)

For every prediction, the app shows the words that contributed most:

- 🔴 Red bars push the prediction toward **Fake**
- 🟢 Green bars push the prediction toward **Real**

---

## Project Structure

```
fake-news-detector/
│
├── app.py                  # Streamlit app
├── logistic.pkl            # Trained Logistic Regression model
├── tfidf.pkl               # Fitted TF-IDF vectorizer
├── shap_explainer.pkl      # SHAP LinearExplainer
├── fake_news_notebook.ipynb  # Training and evaluation notebook
├── requirements.txt
└── README.md
```

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

**requirements.txt**
```
streamlit
scikit-learn
pandas
numpy
shap
matplotlib
vaderSentiment
```

---

## Tech Stack

Python · scikit-learn · SHAP · VADER · Pandas · NumPy · Matplotlib · Streamlit

---

## Limitations

- **Category tagging is keyword-based, not a trained model.** One matching keyword can decide the category (for example, a single mention of "war" tags a text as "world"). A trained classifier or zero-shot model would be more accurate; I chose keywords to keep the app lightweight.
- **The model learns patterns from this dataset only.** It may be learning the writing style of the news sources in the data, not whether a claim is true. It does **not** fact-check claims against the real world.
- **Source bias has not been tested.** I have not yet evaluated the model on articles from sources outside the training dataset.
- **Predictions are probabilistic.** Treat the output as a signal, not a verdict.

---

## Future Improvements

- Error analysis on misclassified articles
- Test generalisation by holding out an entire news source
- Replace keyword category tagging with a trained or zero-shot classifier
- Compare TF-IDF against sentence embeddings or a BiLSTM
- Robustness testing with reworded headlines

---

## Author

**Atharva Pandey**
B.Tech Computer Science | Aspiring Data Science / ML Intern
