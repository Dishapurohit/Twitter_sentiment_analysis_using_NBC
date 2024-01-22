import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import string

import warnings
warnings.filterwarnings('ignore')

import nltk  # nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer

from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.metrics import *
from sklearn.metrics import roc_curve

# DATA EXPLORATION
tweets = pd.read_csv('Training.csv', encoding='latin', names=['sentiment', 'id', 'date', 'query', 'user', 'tweet'])
print(tweets)

tweets = tweets.sample(frac=1)
tweets = tweets[:200000]
print("Dataset shape:", tweets.shape)
tweets['sentiment'].unique()
tweets['sentiment'] = tweets['sentiment'].replace(4, 1)
print(tweets)

# Removing the unnecessary columns
tweets.drop(['date', 'query', 'user'], axis=1, inplace=True)
tweets.drop('id', axis=1, inplace=True)
print(tweets.head(10))

# Checking if any null values present
(tweets.isnull().sum() / len(tweets)) * 100
sns.heatmap(tweets.isnull(), yticklabels='False', cbar='False', cmap='Blues')
tweets.hist(bins=30, figsize=(13, 5), color='r')
plt.show()

# converting pandas object to a string type
tweets['tweet'] = tweets['tweet'].astype('str')

# Check the number of positive vs. negative tagged sentences
positives = tweets['sentiment'][tweets.sentiment == 1]
negatives = tweets['sentiment'][tweets.sentiment == 0]

print('Total length of the data is:         {}'.format(tweets.shape[0]))
print('No. of positve tagged sentences is:  {}'.format(len(positives)))
print('No. of negative tagged sentences is: {}'.format(len(negatives)))

tweets['length'] = tweets['tweet'].apply(len)
print(tweets)

tweets['length'].plot(bins=80, figsize=(13, 5), kind='hist')
plt.show()

tweets.describe()
print(tweets.describe())

var1 = tweets[tweets['length'] == 15]['tweet'].iloc[0]  # shortest comment
var2 = tweets[tweets['length'] == 115]['tweet'].iloc[0]  # longest comment
var3 = tweets[tweets['length'] == 57]['tweet'].iloc[0]  # average comment
print("Shortest Comment is = ", var1)
print("Longest Comment is = ", var2)
print("Average comment is = ", var3)

positive_comments = tweets[tweets['sentiment'] == 1]
print(positive_comments)
print("Positive Comments")
negative_comments = tweets[tweets['sentiment'] == 0]
print(negative_comments)
print("Negative Comments")

# Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”)
# that a search engine has been programmed to ignore,
# both when indexing entries for searching and when retrieving them as the result of a search query.
nltk.download('stopwords')
stopword = set(stopwords.words('english'))
print(stopword)

# Data Cleaning
urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern = '@[^\s]+'
some = 'amp,today,tomorrow,going,girl'


def process_tweets(tweet):
    tweet = tweet.lower()
    tweet = tweet[1:]
    # Removing all URls
    tweet = re.sub(urlPattern, '', tweet)
    # Removing all @username.
    tweet = re.sub(userPattern, '', tweet)
    # remove some words
    tweet = re.sub(some, '', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    # tokenizing words
    tokens = word_tokenize(tweet)
    # tokens = [w for w in tokens if len(w)>2]
    # Removing Stop Words
    final_tokens = [w for w in tokens if w not in stopword]
    # reducing a word to its word stem
    wordLemm = WordNetLemmatizer()
    finalwords = []
    for w in final_tokens:
        if len(w) > 1:
            word = wordLemm.lemmatize(w)
            finalwords.append(word)
    return ' '.join(finalwords)


# Text Processing Completed
tweets['processed_tweets'] = tweets['tweet'].apply(lambda x: process_tweets(x))
print('Text Preprocessing complete.')
print(tweets)

# removing short words
tweets['processed_tweets'] = tweets['processed_tweets'].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))
print(tweets.head(5))

tweets = shuffle(tweets).reset_index(drop=True)

# Tokenization
tokenized_tweet = tweets['processed_tweets'].apply(lambda x: x.split())
print(tokenized_tweet.head(5))

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
text_counts = cv.fit_transform(tweets['processed_tweets'].values.astype('U'))

# Vectorization
vectorizer = CountVectorizer(analyzer=process_tweets)
tokenized_tweets = vectorizer.fit_transform(tweets['processed_tweets'])
print(tokenized_tweets.shape)
X = tokenized_tweets
y = tweets['sentiment']

# Train and Test Split
X = text_counts
y = tweets['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=19)

# Train the model using a Naive Bayes Classifier
cnb = ComplementNB()
cnb.fit(X_train, y_train)
cross_cnb = cross_val_score(cnb, X, y, n_jobs=-1)
print("Cross Validation score = ", cross_cnb)
print("Train accuracy ={:0.2f}%".format(cnb.score(X_train, y_train)*100))
print("Test accuracy ={:0.2f}%".format(cnb.score(X_test, y_test)*100))
train_acc_cnb = cnb.score(X_train, y_train)
test_acc_cnb = cnb.score(X_test, y_test)

# Plotting the best parameters
data_cnb = [train_acc_cnb, test_acc_cnb]
labels = ['Train Accuracy', 'Test Accuracy']
plt.xticks(range(len(data_cnb)), labels)
plt.ylabel('Accuracy')
plt.title('Accuracy plot with best parameters')
plt.bar(range(len(data_cnb)), data_cnb, color=['blue', 'darkorange'])
Train_acc = mpatches.Patch(color='blue', label='Train_acc')
Test_acc = mpatches.Patch(color='darkorange', label='Test_acc')
plt.legend(handles=[Train_acc, Test_acc], loc='best')
plt.gcf().set_size_inches(8, 8)
plt.show()

# Evaluating the Naive-Bayes Classifier Performance
# Confusion Matrix
# Predict test data set
y_pred_cnb = cnb.predict(X_test)

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)
# This is the confusion matrix :
print(confusion_matrix(y_test, y_pred_cnb))
cm = confusion_matrix(y_test, y_pred_cnb)

group_names = ['True Positive', 'False Positive', 'False Negative', 'True Negative']
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:0.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
plt.show()

# Checking performance our model with classification report
print(classification_report(y_test, y_pred_cnb))

# Calculating F1, prescision and recall scores
print("F1 score ={:0.2f}%".format(f1_score(y_test, y_pred_cnb, average="macro")*100))
f1_cnb = f1_score(y_test, y_pred_cnb, average="macro")
print("Precision score ={:0.2f}%".format(precision_score(y_test, y_pred_cnb, average="macro")*100))
precision_cnb = precision_score(y_test, y_pred_cnb, average="macro")
print("Recall score ={:0.2f}%".format(recall_score(y_test, y_pred_cnb, average="macro")*100))
recall_cnb = recall_score(y_test, y_pred_cnb, average="macro")

# Checking performance our model with ROC Score
roc_score_cnb = roc_auc_score(y_test, y_pred_cnb)
print("Area Under the Curve = ", roc_score_cnb)

# Drawing the ROC curve
fpr_dt_1, tpr_dt_1, _ = roc_curve(y_test, cnb.predict_proba(X_test)[:, 1])
plt.plot(fpr_dt_1, tpr_dt_1, label="ROC curve")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.gcf().set_size_inches(8, 8)
plt.show()
