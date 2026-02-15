# Sentiment Analysis on Amazon Appliance Reviews

A natural language processing (NLP) project analyzing sentiment in Amazon appliance reviews using both lexicon-based and machine learning approaches.

## Project Overview

This project analyzes customer sentiment from Amazon appliance reviews across two phases:

- **Phase 1**: Lexicon-based sentiment analysis using established NLP libraries (VADER, TextBlob)
- **Phase 2**: Machine learning-based approach (planned) for more sophisticated sentiment prediction

## Project Structure

```
sentiment_analysis_amazon_appliance_reviews/
├── phase_one/
│   ├── main.ipynb           # Primary analysis notebook
│   ├── data/                # Raw appliance review data
│   └── outputs/             # Generated results and visualizations
├── phase_two/               # Machine learning approach (in development)
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Phase One: Lexicon-Based Sentiment Analysis

### Approach

Phase 1 uses lexicon-based methods to classify sentiment. The process includes:

1. **Data Loading & Exploration**
   - Load gzipped JSON appliance reviews
   - Analyze review distribution, ratings, and verification status

2. **Text Preprocessing**
   - Remove duplicates and empty reviews
   - Tokenization using NLTK
   - Stopword removal
   - Lemmatization with WordNetLemmatizer

3. **Sentiment Classification**
   - **VADER**: Rule-based sentiment analyzer optimized for social media
   - **TextBlob**: Simple polarity-based sentiment scoring
   - Both analyzers compared for validation

4. **Evaluation**
   - Confusion matrices
   - Performance metrics (accuracy, precision, recall, F1-score)
   - Cross-method validation

### Key Libraries

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **nltk**: Natural language toolkit for text processing
- **vaderSentiment**: Lexicon-based sentiment analysis
- **textblob**: Simple text processing and sentiment analysis
- **scikit-learn**: Machine learning metrics and evaluation
- **matplotlib & seaborn**: Data visualization

## Phase Two: Machine Learning Approach (Planned)

Phase 2 will implement machine learning models for sentiment classification, including:

- Feature extraction (TF-IDF, Word embeddings)
- Model training (Logistic Regression, SVM, Neural Networks, etc.)
- Cross-validation and hyperparameter tuning
- Performance comparison with Phase 1 results

## Installation

### Prerequisites

- Python 3.13+

### Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the analysis notebook:
   ```bash
   jupyter notebook phase_one/main.ipynb
   ```

## Dataset

The project uses Amazon appliance reviews in gzipped JSON format (`Appliances_5.json.gz`). Each review contains:
- Rating (1-5 stars)
- Review text
- Verification status
- Other metadata

## Results & Outputs

Phase 1 generates:
- Sentiment predictions from VADER and TextBlob
- Comparative analysis and visualizations
- Confusion matrices and performance metrics
- Results stored in `phase_one/outputs/`

## Dependencies

See `requirements.txt` for the complete list of dependencies.

Main packages:
- ipykernel ≥7.2.0
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- vaderSentiment
- textblob
- scikit-learn

## Usage

To run the sentiment analysis:

1. Open `phase_one/main.ipynb` in Jupyter Notebook
2. Execute cells sequentially to:
   - Load and explore the data
   - Preprocess reviews
   - Run sentiment analysis
   - Generate evaluation metrics and visualizations

## Notes

- NLTK models are automatically downloaded on first run (punkt, stopwords, wordnet)
- The analysis focuses on appliance reviews but can be adapted for other product categories
- Results are compared across multiple lexicon-based approaches for robustness

## Future Enhancements

- Complete Phase 2 machine learning implementation
- Explore deep learning approaches (BERT, transformers)
- Multi-class sentiment analysis (fine-grained emotions)
- Aspect-based sentiment analysis
- Real-time prediction API

## Author Notes

This project demonstrates the effectiveness of NLP techniques in understanding customer sentiment, providing insights for product improvement and customer satisfaction analysis.
