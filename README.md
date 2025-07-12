# Uncovering Player Sentiment Through Gaming Community Conversations on Twitter
[Kaggle Notebook](https://www.kaggle.com/code/rothindrohait/sentiment-analysis-bilstm) | [Web App](https://da-sentiment-analysis.streamlit.app/)
---
## 📌Overview:

Sentiment analysis (or opinion mining) is an NLP technique that detects emotions (positive, negative, neutral) in text. It helps businesses understand public perception by analyzing social media, reviews, and customer feedback.
In this rise of AI in business, AI/ML-based sentiment analysis tools can help in many ways, like:

- Tracking customer sentiment about products/services in real-time
- Detect PR crises early (e.g., sudden spike in negative tweets)
- Automatically categorize support tickets by urgency (anger = high priority)
- Improve products based on recurring complaints in reviews &
- Compare sentiment toward competitors to identify market gaps, etc
  
Employing this system in a gaming environment will help to track toxic behaviour among players, improve community engagement, etc.

Used entity-level sentiment analysis dataset of twitter from [Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/data)

- __Twitter Sentiment Analysis Dataset__


In this project, we have utilized the above-mentioned dataset, which contains **69,491 records** in the training set and **1000 records** in the test set of twitter conversations between gamers of different games with 4 attributes. Key features include:

| Feature              | Relevance to Churn                                  |
|----------------------|-----------------------------------------------------|
| **Tweet ID**         | Individual twitter IDs                              |
| **entity**           | Company/Game names                                          |
| **sentiment**        | Projected impression of the tweets/conversation     |
| **Tweet content**    | Tweets/converstations                               |


## ⚙️Methods
The project employs time series analysis techniques, including:

- __Data Preprocessing__
- __Lemmitization__
- __Tokenization & Padding__
- __Model Training & Evaluation__
- __Hyperparameter Tuning__
- __Final Model Selection & Evaluation__
- __Pipeline__

Click here to access the Kaggle Notebook

## 🛠️ Tools & Technologies
- __Programming Language:__ Python
- __Libraries:__ NumPy, Pandas, Seaborn, Matplotlib, Scikit-learn, Spacy, Tensorflow/Keras (BiLSTM)
- __Environment:__ Kaggle Notebook & VS Code
- __Deployment:__ Streamlit
