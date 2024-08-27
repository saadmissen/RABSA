import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import pandas as pd


def load_data(file_path):
    sentences = []
    labels = []
    
    ## This data collection is modified from the one as prepared. 
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split("\t")
                sentence = parts[0]
                label = parts[1]  
                sentences.append(sentence)
                labels.append(label)
    
    return sentences, labels

# file path positive
file_path = 'D:/PhD_Work/Data/positive_reviews.txt'

#file_path = 'D:/PhD_Work/Data/negative_reviews.txt'   # for negative
#file_path = 'D:/PhD_Work/Data/neutral_reviews.txt'    # for neutral



# Load data
sentences, labels = load_data(file_path)

# Convert labels to numeric values (assuming binary or multiclass labels)
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
labels = [label_mapping[label] for label in labels]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# Vector Space Model (VSM) using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)

# VSM with SVM Classifier
vsm_model = make_pipeline(vectorizer, SVC(kernel='linear'))
vsm_model.fit(X_train, y_train)
y_pred_vsm = vsm_model.predict(X_test)
print("VSM + SVM")
print("Accuracy:", accuracy_score(y_test, y_pred_vsm))
print(classification_report(y_test, y_pred_vsm))

# Random Forest Baseline - @ASMA update this from decision tree to RF

rf_model = make_pipeline(vectorizer, RandomForestClassifier(n_estimators=100, random_state=42))
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Naive Bayes Baseline
nb_model = make_pipeline(vectorizer, MultinomialNB())
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print("Naive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))
