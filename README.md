# Fake News Detector

This project is a simple Fake News Detector that uses Natural Language Processing (NLP) techniques to classify news articles as real or fake. The model is trained using the Naive Bayes algorithm with TF-IDF vectorization.

## Features
- **Text Preprocessing:** Cleans and prepares text data by removing special characters and converting text to lowercase.
- **TF-IDF Vectorization:** Converts text data into numerical representations for machine learning.
- **Naive Bayes Classification:** A probabilistic classifier used to distinguish between real and fake news.
- **Model Evaluation:** Provides accuracy score and classification report.
- **Dataset Analysis:** Generates predictions with confidence scores and saves the results.

## Prerequisites
Ensure you have the following installed:
- Python 3.x
- Required Python libraries:
  - `pandas`
  - `sklearn`
  - `re`

Install dependencies using:
```bash
pip install pandas scikit-learn
```

## Project Structure
```
.
├── fake_news_detector.py  # Main Python script
├── fake-news-dataset.csv  # Sample dataset (replace with actual file)
├── fake_news_predictions.csv  # Output results
├── README.md  # Project documentation
```

## Usage

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

2. **Run the script:**
```bash
python fake_news_detector.py
```

3. **Provide the dataset:**
   - Place your news dataset as `fake-news-dataset.csv` in the project directory.
   - Ensure the dataset has columns: `text` (news content) and `label` (0 for real, 1 for fake).

4. **Output:**
   - Model accuracy and evaluation metrics will be displayed in the console.
   - The predictions will be saved in `fake_news_predictions.csv`.

## Code Overview

### 1. `FakeNewsDetector` Class
- **`__init__()`**: Initializes the TF-IDF vectorizer and Naive Bayes classifier.
- **`preprocess_text(text)`**: Prepares input text by cleaning and normalizing.
- **`load_and_analyze_csv(csv_path)`**: Loads, preprocesses, and analyzes data.

### 2. `main()` Function
- Instantiates the `FakeNewsDetector` class.
- Loads the dataset and processes it.
- Displays the analysis results.

## Example Output
```
Model Accuracy: 0.87

Classification Report:
              precision    recall  f1-score   support

         0       0.85      0.89      0.87       500
         1       0.90      0.85      0.87       500

    accuracy                           0.87      1000
   macro avg       0.87      0.87      0.87      1000
weighted avg       0.87      0.87      0.87      1000

Analysis saved to 'fake_news_predictions.csv'.
```
