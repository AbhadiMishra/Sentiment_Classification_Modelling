# Sentiment Classification Models with TF-IDF Vectorization

This repository hosts implementations of text classification models trained using TF-IDF vectorization. The models are evaluated based on their performance metrics, and each model is saved for future use.<br>
` 1. Training File: Model_Check.ipynb`<br>
` 2. Testing File: Testing_on_Random_Chat.ipynb` 

## Models and Performance Metrics

Below are the details and performance metrics of each model:

- **Multinomial Naive Bayes (MNB)**
  - Accuracy: 0.63
  - F1-score: 0.61
  - Precision: 0.71
  - Model file: pipeline_mnb.joblib

- **Logistic Regression (LR)**
  - Accuracy: 0.72
  - F1-score: 0.72
  - Precision: 0.72
  - Model file: logistic_reg.joblib

- **Support Vector Classifier (SVM)**
  - Accuracy: 0.73
  - F1-score: 0.72
  - Precision: 0.73
  - Model file: pipeline_svm.joblib

- **Random Forest Classifier (RF)**
  - Accuracy: 0.69
  - F1-score: 0.69
  - Precision: 0.70
  - Model file: pipeline_rf_692121.joblib

- **K-Nearest Neighbors Classifier (KNN)**
  - Accuracy: 0.43
  - F1-score: 0.36
  - Precision: 0.66
  - Model file: pipeline_knn.joblib

- **Decision Tree Classifier (DT)**
  - Accuracy: 0.64
  - F1-score: 0.64
  - Precision: 0.64
  - Model file: pipeline_dt.joblib

- **Neural Network MLPClassifier (NN)**
  - Accuracy: 0.69
  - F1-score: 0.69
  - Precision: 0.69
  - Model file: pipeline_mlp.joblib

- **Gradient Boosting Classifier (GBC)**
  - Accuracy: 0.71
  - F1-score: 0.69
  - Precision: 0.72
  - Model file: pipeline_gbc.joblib

- **AdaBoost Classifier (ABC)**
  - Accuracy: 0.69
  - F1-score: 0.69
  - Precision: 0.71
  - Model file: pipeline_abc.joblib

## Usage

1. Ensure Python 3.x is installed on your system.
2. Clone this repository to your local machine.
3. Place your dataset in a CSV file named `Chat_for_train.csv`.
4. Run the provided code to train and evaluate the sentiment classification models.

## WhatsApp Chat Sentiment Analysis

This repository also includes code for sentiment analysis on WhatsApp chat data. The code extracts message data from the chat file and predicts the sentiment of each message using pre-trained machine learning models.

Feel free to explore and experiment with these models for your text classification tasks!
