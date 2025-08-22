# Amazon-alexa-review-sentiment-analysis

1. Data Preparation
This is the starting point. The code first loads your raw data, which contains customer reviews and star ratings. The most important part of this step is transforming the numerical star ratings (1 to 5) into the sentiment categories you want to predict: positive, neutral, and negative. This is a form of feature engineering, where you're creating a new label for your machine learning model to learn from.

2. Text Preprocessing
This is the "cleaning station" for your text. Raw, messy language isn't directly usable by a machine learning model, which only understands numbers. This is where NLTK comes in. The code uses it to:

Tokenize: It breaks down each review into individual words.

Remove Stop Words: It gets rid of common words like "the," "is," and "a" that don't carry much sentiment. This reduces the noise and helps the model focus on important words.

Stemming: It reduces words to their base or root form (e.g., "running" becomes "run"). This ensures that different variations of the same word are treated as a single feature, which improves model performance.

3. Text Vectorization
Now that the text is clean, it needs to be converted into a numerical format. This is the job of the TF-IDF Vectorizer. TF-IDF stands for Term Frequency-Inverse Document Frequency. It's an algorithm that:

Counts how often a word appears in a specific review (Term Frequency).

Considers how rare or common a word is across all reviews (Inverse Document Frequency).

The result is a large matrix of numbers, where each number represents the importance of a specific word in a specific review. This numerical matrix is the final input for your machine learning model.

4. Model Training & Evaluation
With the data in a numerical format, the Logistic Regression model can now be trained. The model looks for patterns in the TF-IDF matrix that correspond to each of the three sentiment labels. It learns which words or combinations of words are most strongly associated with positive, neutral, or negative reviews.

After training, the model is tested on a separate set of data it has never seen before. The code uses metrics like accuracy and the classification report to evaluate how well the model performed. This tells you if the model's predictions are reliable.

5. Prediction
This is the final step where the model is put to work. When you provide a new, unseen review, the code puts it through the exact same pipeline: it's preprocessed by NLTK, vectorized by TF-IDF, and then fed to the trained Logistic Regression model. The model then outputs its prediction of the sentiment for that review.