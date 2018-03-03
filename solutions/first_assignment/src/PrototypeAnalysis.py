import pandas as pd
import os
from math import log
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk.downloader
nltk.download('punkt')


def extract_reviews(filepath: str):
    review_list = []
    pos_or_neg = []

    for file in os.listdir(filepath):
        score = int(file.split('.')[0].split('_')[1])
        review_list.append(open(filepath + file, encoding='utf-8').read())
        if score >= 7:
            pos_or_neg.append(1)
        else:
            pos_or_neg.append(0)

    return pd.DataFrame({'reviews': review_list, 'is_positive': pos_or_neg})


stop_words = stopwords.words('english')


def clean_text(text: str):
    cleaned = []
    tokens = word_tokenize(text)
    for word in tokens:
        word = word.lower()
        if word in stop_words:
            continue
        cleaned.append(word)
    return cleaned


def construct_dictionary(reviews):
    dictionary = {}
    for text in reviews:
        # Alternative: for word in clean_text(text):
        for word in text.split(' '):
            word = word.lower()
            if word not in dictionary:
                dictionary[word] = 1.0
            else:
                dictionary[word] += 1.0
    return dictionary


print("Extracting train reviews..")
train_df_pos = extract_reviews('../../../datasets/first_assignment/train/pos/')
train_df_neg = extract_reviews('../../../datasets/first_assignment/train/neg/')
train_total = pd.concat([train_df_neg, train_df_pos])

# prior probability of negative or positive review given training data
p_of_positive = log(train_df_pos.size / (train_df_neg.size + train_df_pos.size))
p_of_negative = log(train_df_neg.size / (train_df_neg.size + train_df_pos.size))

# vocab with frequency for the negative reviews and the positive
print("Making dictionary for all reviews..\n")
pos_dictionary = construct_dictionary(train_df_pos['reviews'])
neg_dictionary = construct_dictionary(train_df_neg['reviews'])
total_dictionary = construct_dictionary(train_total['reviews'])

# nr of words in total vocab
vocab_size = float(len(total_dictionary))

# positive and negative words total (including recounted words)
pos_word_count = float(sum(pos_dictionary.values()))
neg_word_count = float(sum(neg_dictionary.values()))


def condprob_word(word: str, positive: bool, alpha: float = 1.0):
    """
    :param word: word to calculate conditional probability for
    :param positive: calculate for positive dictionary or not
    :param alpha: laplace smoothing (default 1)
    :return: log-likelihood of word given class c (positive or negative)
    """
    if positive and word in pos_dictionary:
        return log((pos_dictionary[word] + alpha) / (pos_word_count + vocab_size))
    elif positive:
        return log(alpha / (pos_word_count + vocab_size))
    elif word in neg_dictionary:
        return log((neg_dictionary[word] + alpha) / (neg_word_count + vocab_size))
    else:
        return log(alpha / (neg_word_count + vocab_size))


def classify_reviews(reviews):
    positive_count = 0
    negative_count = 0
    for text in reviews:
        is_positive = p_of_positive
        is_negative = p_of_negative

        # Alternative: for word in clean_text(text):
        for word in text.split(' '):
            word = word.lower()
            is_positive += condprob_word(word, True)
            is_negative += condprob_word(word, False)

        if is_negative > is_positive:
            negative_count += 1
        else:
            positive_count += 1

        # print("Review: ", text, "\n Results -> pos or neg:\n ", is_positive, " vs ", is_negative, "\n")
    return positive_count, negative_count


# reviews to be classified
print("Extracting test reviews..")
test_df_pos = extract_reviews('../../../datasets/first_assignment/test/pos/')
test_df_neg = extract_reviews('../../../datasets/first_assignment/test/neg/')

print("Classifying test reviews..\n")
true_pos, false_pos = classify_reviews(test_df_pos['reviews'])
false_neg, true_neg = classify_reviews(test_df_neg['reviews'])

print("TP   FP")
print(true_pos, false_pos, "\n")
print("FN   TN")
print(false_neg, true_neg, "\n")
print("Precision: ", (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos))

# precision on fitted data
print("Classifying train reviews..\n")
true_pos, false_pos = classify_reviews(train_df_pos['reviews'])
false_neg, true_neg = classify_reviews(train_df_neg['reviews'])

print("TP   FP")
print(true_pos, false_pos, "\n")
print("FN   TN")
print(false_neg, true_neg, "\n")
print("Precision: ", (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos))