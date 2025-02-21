import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_model(input_path):

    df = pd.read_csv(input_path)
    texts = df['cleaned_text']
    labels = [0, 1, 0]  
    
   
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
   
    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

    import joblib
    joblib.dump(model, "results/models/text_classifier.pkl")
    print("Model saved to results/models/text_classifier.pkl")

if __name__ == "__main__":
    train_model("data/processed/cleaned_data.csv")
