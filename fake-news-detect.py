import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class FakeNewsDetector:
    def __init__(self):
        self.classifier = None
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def preprocess_text(self, text):
        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        return text

    def load_and_analyze_csv(self, csv_path):
        # Load dataset
        df = pd.read_csv(csv_path)

        # Preprocess texts
        df['processed_text'] = df['text'].apply(self.preprocess_text)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['label'], test_size=0.2, random_state=42
        )

        # Vectorize texts
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)

        # Train Naive Bayes classifier
        self.classifier = MultinomialNB()
        self.classifier.fit(X_train_vectorized, y_train)

        # Evaluate the model
        y_pred = self.classifier.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:\n", report)

        # Analyze each text in the dataset
        results = []
        for _, row in df.iterrows():
            text_vectorized = self.vectorizer.transform([self.preprocess_text(row['text'])])
            prediction = self.classifier.predict(text_vectorized)[0]
            confidence = max(self.classifier.predict_proba(text_vectorized)[0]) * 100

            results.append({
                'text': row['text'],
                'actual_label': row['label'],
                'predicted_label': prediction,
                'confidence': f"{confidence:.2f}%"
            })

        # Convert results to DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv('fake_news_predictions.csv', index=False)

        # Print results
        for result in results:
            print(f"Text: {result['text']}")
            print(f"Actual Label: {result['actual_label']}")
            print(f"Predicted Label: {result['predicted_label']}")
            print(f"Confidence: {result['confidence']}\n")

        return results_df

def main():
    detector = FakeNewsDetector()
    results = detector.load_and_analyze_csv('fake-news-dataset.csv')  # Replace with your actual CSV file path
    print("Analysis saved to 'fake_news_predictions.csv'.")

if __name__ == "__main__":
    main()
