import pandas as pd
import re
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """Cleans the text by removing unwanted characters and stopwords."""
    text = re.sub(r"http\S+", "", text)  
    text = re.sub(r"[^\w\s]", "", text) 
    text = text.lower()  # Convert to lowercase
    words = text.split() 
    words = [word for word in words if word not in STOPWORDS]  
    return " ".join(words)

def preprocess_csv(input_path, output_path):
    """Reads a CSV file, cleans the text column, and saves the processed data."""
    df = pd.read_csv(input_path)
    df['cleaned_text'] = df['text'].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_csv("data/raw/sample_data.csv", "data/processed/cleaned_data.csv")
