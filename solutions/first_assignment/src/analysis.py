import pandas as pd
import numpy as np
import os
from math import log
from functools import reduce
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from stop_words import get_stop_words


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


def construct_dictionary(reviews):
    dictionary = {}
    for text in reviews:
        for word in text.split(' '):
            word = word.lower()
            if word not in dictionary:
                dictionary[word] = 1.0
            else:
                dictionary[word] += 1.0
    return dictionary


print("Extracting positive reviews..")
train_df_pos = extract_reviews('../../../datasets/first_assignment/train/pos/')
print("Extracting negative reviews..")
train_df_neg = extract_reviews('../../../datasets/first_assignment/train/neg/')
print("Concatenating reviews..\n")
train_total = pd.concat([train_df_neg, train_df_pos])

# prior probability of negative or positive review given training data
p_of_positive = log(train_df_pos.size / (train_df_neg.size + train_df_pos.size))
p_of_negative = log(train_df_neg.size / (train_df_neg.size + train_df_pos.size))

# vocab with frequency for the negative reviews and the positive
print("Making dictionary for positive reviews..")
pos_dictionary = construct_dictionary(train_df_pos['reviews'])
print("Making dictionary for negative reviews..")
neg_dictionary = construct_dictionary(train_df_neg['reviews'])
print("Making dictionary for all reviews..\n")
total_dictionary = construct_dictionary(train_total['reviews'])

# nr of words in total vocab
total_cardinality = float(len(total_dictionary))

# positive and negative words total (including recounted words)
nr_positive_words = float(sum(pos_dictionary.values()))
nr_negative_words = float(sum(neg_dictionary.values()))


def condprob_word(word: str, positive: bool, alpha: float):
    if positive and word in pos_dictionary:
        return (pos_dictionary[word] + alpha) / (nr_positive_words + total_cardinality)
    elif positive:
        return alpha / (nr_positive_words + total_cardinality)
    elif word in neg_dictionary:
        return (neg_dictionary[word] + alpha) / (nr_negative_words + total_cardinality)
    else:
        return alpha / (nr_negative_words + total_cardinality)


def classify_reviews(reviews):
    positive_count = 0
    negative_count = 0
    for text in reviews:
        is_positive = p_of_positive
        is_negative = p_of_negative

        for word in text.split(' '):
            word = word.lower()
            is_positive += log(condprob_word(word, True, 1.0))
            is_negative += log(condprob_word(word, False, 1.0))

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
test_total = pd.concat([test_df_neg, test_df_pos])

print("Classifying test reviews..\n")
true_pos, false_pos = classify_reviews(test_df_pos['reviews'])
false_neg, true_neg = classify_reviews(test_df_neg['reviews'])

print("TP   FP")
print(true_pos, false_pos, "\n")
print("FN   TN")
print(false_neg, true_neg, "\n")
print("Precision: ", (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos))
