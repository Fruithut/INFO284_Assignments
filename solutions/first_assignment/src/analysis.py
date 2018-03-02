import pandas as pd
import numpy as np
import os
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
    """
    :param reviews: takes a pandas.Series object
    :return: a dictionary of every word that exist pandas.Series object together
    with the total number of times the word has been encountered
    """
    dictionary = {}
    for text in reviews:
        for word in text.split(' '):
            word = word.lower()
            if word not in dictionary:
                dictionary[word] = 1
            else:
                dictionary[word] += 1
    return dictionary


print("Extracting positive reviews..")
train_df_pos = extract_reviews('../../../datasets/first_assignment/train/pos/')
print("Extracting negative reviews..")
train_df_neg = extract_reviews('../../../datasets/first_assignment/train/neg/')
print("Concatenating reviews..\n")
train_total = pd.concat([train_df_neg, train_df_pos])

# prior probability of negative or positive review given training data
p_of_positive = train_df_pos.size / (train_df_neg.size + train_df_pos.size)
p_of_negative = 1 - p_of_positive

# vocab with frequency for the negative reviews and the positive
print("Making dictionary for positive reviews..")
pos_dictionary = construct_dictionary(train_df_pos['reviews'])
print("Making dictionary for negative reviews..")
neg_dictionary = construct_dictionary(train_df_neg['reviews'])
print("Making dictionary for all reviews..\n")
total_dictionary = construct_dictionary(train_total['reviews'])

# nr of words in total vocab
total_cardinality = len(total_dictionary)

# positive and negative words total (including recounted words)
nr_positive_words = sum(pos_dictionary.values())
nr_negative_words = sum(neg_dictionary.values())


def classify_reviews(reviews):
    positive_count = 0
    negative_count = 0
    for text in reviews:
        p_word_positive = []
        p_word_negative = []

        # p(frequency of word in training set | given positive or negative)
        for word in text.split(' '):
            word = word.lower()

            denominator = 1
            if word in total_dictionary:
                denominator += total_dictionary[word]

            # calculate for p(word| given positive class)
            if word in pos_dictionary:
                p_word_positive.append(
                    (pos_dictionary[word] + 1 / (nr_positive_words + total_cardinality)) / denominator)
            elif word in neg_dictionary:
                # laplace smoothing
                p_word_positive.append((1 / (nr_positive_words + total_cardinality)))

            # calculate for p(word| given negative class)
            if word in neg_dictionary:
                p_word_negative.append(
                    (neg_dictionary[word] + 1 / (nr_negative_words + total_cardinality)) / denominator)
            elif word in pos_dictionary:
                # laplace smoothing
                p_word_negative.append(1 / (nr_negative_words + total_cardinality))

        # probability of review being positive
        result_pos = reduce(lambda x, y: x * y, p_word_positive)
        result_pos *= p_of_positive

        # probability of review being negative
        result_neg = reduce(lambda x, y: x * y, p_word_negative)
        result_neg *= p_of_negative

        if result_neg > result_pos:
            negative_count += 1
        else:
            positive_count += 1

        # print("Review: ", text, "\n Results -> pos or neg:\n ", result_neg,
        #  " ", result_pos, "\n", "Result: ", is_positive)

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
