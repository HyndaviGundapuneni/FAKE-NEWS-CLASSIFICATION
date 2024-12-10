# Fake News Classification

## Project Overview

The goal of this project is to classify news articles as either **real** or **fake** using machine learning techniques. With the rise of misinformation, fake news classification has become an essential task in combating the spread of misleading information. This project utilizes **Logistic Regression** for classifying news articles based on their textual content.

The dataset consists of news articles that are labeled as real or fake, and a series of preprocessing steps and machine learning models are applied to create an effective classification system.

## Dataset

The dataset consists of two CSV files:

1. `Fake.csv` - Contains fake news articles.
2. `True.csv` - Contains real news articles.

These datasets are preprocessed and cleaned before being used to train a machine learning model.

## Libraries Used

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computing.
- **Matplotlib & Seaborn**: For data visualization.
- **Plotly**: For interactive data visualization.
- **Scikit-learn**: For machine learning models, feature extraction, and evaluation metrics.
- **NLTK & SpaCy**: For text preprocessing, including lemmatization and tokenization.
- **WordCloud**: For visualizing the most frequent words in the dataset.

## Data Preprocessing

1. **Cleaning Data**:
   - Removed duplicates from both the fake and real news datasets.
   - Added a new column `category` (0 for real news, 1 for fake news).
   - Dropped irrelevant columns such as `title`, `subject`, and `date`.

2. **Text Preprocessing**:
   - Removed punctuation and converted all text to lowercase.
   - Used **SpaCy** for lemmatization, which reduces words to their base form.
   
3. **Feature Extraction**:
   - Applied **CountVectorizer** to convert text data into numerical vectors for model training.

## Model Development

The model is built using **Logistic Regression**, a widely used algorithm for binary classification tasks.

### Steps:
1. **Data Splitting**: The dataset was split into training and testing sets (80% training, 20% testing).
2. **Text Vectorization**: The text data was transformed using **CountVectorizer** into a format suitable for machine learning.
3. **Model Training**: **Logistic Regression** was used to train the model on the preprocessed text data.
4. **Model Evaluation**: The model's performance was evaluated using metrics such as **accuracy**, **precision**, **recall**, and **F1-score**. A **confusion matrix** was used to visualize the model’s predictions.

## Results

- The model achieved **99% accuracy** in classifying both real and fake news articles, demonstrating its high performance in identifying fake news.
- **Confusion Matrix** and **Classification Report** metrics (precision, recall, and F1-score) were used to further evaluate the model’s performance.

## Conclusion

The project successfully builds a machine learning model that can accurately classify news articles as real or fake. Although the model performs well, the challenge of detecting fake news remains complex due to the constantly evolving nature of misinformation.

By investing in improving these models and incorporating more advanced techniques, we can contribute to reducing the spread of fake news and promoting a more informed public.

