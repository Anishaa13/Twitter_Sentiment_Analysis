# Twitter Sentiment Analysis

This project demonstrates the use of Natural Language Processing (NLP) and machine learning to predict tweet sentiments (positive, negative, or neutral). The analysis pipeline includes data preprocessing, exploratory data analysis (EDA), model building using deep learning, and result visualization.

---

## Project Workflow

### 1. **Data Collection**
- **Source**: Kaggle’s tweet sentiment dataset.
- The dataset includes a variety of tweets labeled with sentiment categories, providing the foundation for training and evaluation.

### 2. **Data Preprocessing**
- **Objective**: Clean and normalize raw tweet text for analysis.
- **Steps**:
  - **Remove special characters**: Strip hashtags, mentions, links, and emojis.
  - **Convert text to lowercase**: Ensures uniformity.
  - **Tokenization**: Breaks text into words for easier processing.
  - **Stop-word removal**: Excludes common words like "the" and "is" to focus on meaningful terms.
  - **Padding sequences**: Ensures input size consistency for the model.

### 3. **Exploratory Data Analysis (EDA)**
- Visualizes sentiment distribution using libraries such as Matplotlib and Seaborn.
- **Insights Gathered**:
  - Proportion of positive, negative, and neutral sentiments.
  - Commonly used keywords in tweets with each sentiment.

### 4. **Feature Engineering**
- Converts preprocessed text into numeric data using:
  - **Word embeddings**: Translates words into vectors for capturing semantic meaning.
  - **Sequence padding**: Maintains uniformity for input into the model.

### 5. **Model Building**
- Implements a Long Short-Term Memory (LSTM) network using TensorFlow and Keras.
- **Steps**:
  - **Input layer**: Accepts processed sequences.
  - **LSTM layers**: Captures temporal dependencies in text.
  - **Dense layer**: Maps outputs to sentiment categories.
  - **Compilation**: Optimized using `adam` optimizer and `categorical_crossentropy` loss.

### 6. **Model Training**
- Splits the dataset into training and validation sets.
- Trains the model over multiple epochs, monitoring loss and accuracy to prevent overfitting.

### 7. **Model Evaluation**
- **Metrics**:
  - Accuracy: Achieved 90% on validation data.
  - Precision and Recall: Evaluated using a classification report.
- Plots training/validation loss and accuracy for performance visualization.

### 8. **Sentiment Prediction**
- Takes new tweets as input.
- Predicts sentiment category based on trained model outputs.

### 9. **Result Visualization**
- **Visual Tools**: Matplotlib, Seaborn.
- Generates bar plots and word clouds to illustrate:
  - Distribution of sentiments.
  - Frequently used terms by sentiment type.

---

## Installation

### Prerequisites
- Python 3.8 or above
- Jupyter Notebook or Google Colab
- Required Libraries: TensorFlow, Keras, Pandas, NumPy, Matplotlib, Seaborn

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Anishaa13/TwitterSentiment.git
   ```
2. Navigate to the project folder:
   ```bash
   cd TwitterSentiment
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Project

1. **Load the notebook**: Open `TwitterSentiment.ipynb` in Jupyter Notebook or Colab.
2. **Run Preprocessing**: Execute cells for cleaning and preparing the data.
3. **Train the Model**: Run the training steps and save the best model.
4. **Test Predictions**: Input sample tweets and observe the sentiment classification.

---

## Results

- Achieved 90% accuracy on sentiment classification.
- Provided insights into public sentiment trends, useful for marketing, customer feedback, and social media strategies.

--
