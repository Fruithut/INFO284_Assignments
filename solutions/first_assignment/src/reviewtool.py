import pandas as pd
import os
from math import log

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk.downloader


def extract_reviews(path: str, encoding: str = 'utf-8'):
    """
    :param path: takes the path to a folder of .txt files and reads them
    :param encoding: encoding of .txt files
    :return: pandas.DataFrame with reviews and target values (pos or neg)
    """
    review_list = []
    target_list = []

    for file in os.listdir(path):
        score = int(file.split('.')[0].split('_')[1])
        review_list.append(open(path + file, encoding=encoding).read())
        if score >= 7:
            target_list.append(1)
        else:
            target_list.append(0)

    return pd.DataFrame({'reviews': review_list, 'target': target_list})


class NBReviewClassifier(object):

    def __init__(self, to_clean: bool = False, stopwords_lang: str = 'english'):
        """
        Initializes the classifier with instance variables and the required resources for data fitting and analysis
        :param to_clean: to clean and remove stopwords from data which will be fitted and classified
        :param stopwords_lang: define language for stopwords
        """
        self.to_clean = to_clean
        if to_clean:
            nltk.download('punkt')
            nltk.download('stopwords')
            self.stop_words = stopwords.words(stopwords_lang)

        # these variables will hold frequencies, probabilities and
        # dictionaries after a call to the fit() method
        self.positive_dict = None
        self.negative_dict = None
        self.prob_negative = None
        self.prob_positive = None
        self.vocab_size = None
        self.positive_word_count = None
        self.negative_word_count = None

        # values are assigned after call to classify() method
        self.true_positive = self.false_positive = self.true_negative = self.false_negative = 0

        # boolean flag for if the model has been fitted
        self.fitted = False

    def clean_text(self, string: str):
        """
        :param string: to be cleaned and filtered
        :return: a list of words that av been cleaned and filtered by the stop_words-list
        """
        cleaned = []
        tokens = word_tokenize(string)
        for word in tokens:
            word = word.lower()
            if word in self.stop_words:
                continue
            cleaned.append(word)
        return cleaned

    def conditional_word(self, word: str, positive: bool, alpha: float = 1.0):
        """
        :param word: word to calculate conditional probability for
        :param positive: calculate for positive dictionary or not
        :param alpha: laplace smoothing (default 1)
        :return: log-likelihood of word given class c (positive or negative)
        """
        if positive and word in self.positive_dict:
            return log((self.positive_dict[word] + alpha) / (self.positive_word_count + self.vocab_size))
        elif positive:
            return log(alpha / (self.positive_word_count + self.vocab_size))
        elif word in self.negative_dict:
            return log((self.negative_dict[word] + alpha) / (self.negative_word_count + self.vocab_size))
        else:
            return log(alpha / (self.negative_word_count + self.vocab_size))

    def construct_dictionary(self, strings):
        """
        Creates a dictionary for all the strings in the format: {string: times_encountered}
        :param strings: pandas.Series object with strings
        :return: dict() object with frequency for all unique strings
        """
        dictionary = {}
        for text in strings:
            if self.to_clean:
                for word in self.clean_text(text):
                    dictionary[word] = dictionary.get(word, 0.0) + 1.0
            else:
                for word in text.split(' '):
                    word = word.lower()
                    dictionary[word] = dictionary.get(word, 0.0) + 1.0
        return dictionary

    def fit(self, positive_path: str, negative_path: str, encoding: str = 'utf-8'):
        """
        Fits the training data to the classification model
        :param positive_path: path to folder containing positive reviews
        :param negative_path: path to folder containing negative reviews
        :param encoding: text encoding of the reviews
        """

        # reset results from last call to classify() method
        self.true_positive = self.false_positive = self.true_negative = self.false_negative = 0

        print("FITTING TRAINING DATA")
        print("\tExtracting texts..")
        train_pos = extract_reviews(positive_path, encoding)
        train_neg = extract_reviews(negative_path, encoding)

        print("\tConstructing dictionaries..")
        self.positive_dict = self.construct_dictionary(train_pos['reviews'])
        self.negative_dict = self.construct_dictionary(train_neg['reviews'])
        # concatenate the two dictionaries
        total_dict = self.positive_dict.copy()
        total_dict.update(self.negative_dict)

        self.prob_negative = log(train_neg.size / (train_pos.size + train_neg.size))
        self.prob_positive = log(train_pos.size / (train_neg.size + train_pos.size))
        self.vocab_size = float(len(total_dict))
        self.positive_word_count = float(sum(self.positive_dict.values()))
        self.negative_word_count = float(sum(self.negative_dict.values()))

        self.fitted = True
        print("\tReady.\n")

    def classify(self, positive_path: str, negative_path: str, encoding: str = 'utf-8'):
        """
        :param positive_path: path to positive test reviews
        :param negative_path: path to negative test reviews
        :param encoding: encoding of the text documents
        :return: a list of predictions for each review, None if model hasn't been fit
        """
        if not self.fitted:
            return None

        # reset results from last call to classify() method
        self.true_positive = self.false_positive = self.true_negative = self.false_negative = 0

        print("ESTIMATING TEST DATA")
        print("\tExtracting texts to be classified..")
        test_positive = extract_reviews(positive_path, encoding)
        test_negative = extract_reviews(negative_path, encoding)
        all_reviews = pd.concat([test_negative, test_positive])

        classified_targets = []
        print("\tEstimating..")
        for index, text in enumerate(all_reviews['reviews']):
            is_positive = self.prob_positive
            is_negative = self.prob_negative
            result = 0

            if self.to_clean:
                for word in self.clean_text(text):
                    is_positive += self.conditional_word(word, True)
                    is_negative += self.conditional_word(word, False)
            else:
                for word in text.split(' '):
                    word = word.lower()
                    is_positive += self.conditional_word(word, True)
                    is_negative += self.conditional_word(word, False)

            # count true/false positive and negatives
            if all_reviews['target'].iloc[index] == 1:
                if is_positive > is_negative:
                    self.true_positive += 1
                else:
                    self.false_positive += 1
            else:
                if is_negative > is_positive:
                    self.true_negative += 1
                else:
                    self.false_negative += 1

            # result of classifying a given text
            if is_positive > is_negative:
                result = 1
            classified_targets.append(result)

        print("\tDone.\n")
        return classified_targets

    def predict(self, test_review: str):
        """
        :param test_review: text to be evaluated
        :return: 0 if negative, 1 if positive, None if model hasn't been fit
        """
        if not self.fitted:
            return None

        is_positive = self.prob_positive
        is_negative = self.prob_negative
        result = 0

        if self.to_clean:
            for word in self.clean_text(test_review):
                is_positive += self.conditional_word(word, True)
                is_negative += self.conditional_word(word, False)
        else:
            for word in test_review.split(' '):
                word = word.lower()
                is_positive += self.conditional_word(word, True)
                is_negative += self.conditional_word(word, False)

        if is_positive > is_negative:
            result = 1

        return result

    def confusion_matrix(self):
        """
        :return: a dictionary containing the true/false positives and negatives,
        None if model hasn't been fit
        """
        if not self.fitted:
            return None

        return {'tp': self.true_positive, 'fp': self.false_positive,
                'fn': self.false_negative, 'tn': self.true_negative}

    def print_scores(self):
        """
        Prints a summary of the algorithm's performance after classification
        has taken place.
        :return: None if model hasn't been fit
        """
        if not self.fitted:
            return None

        print("Confusion Matrix:")
        print("TP   FP")
        print(self.true_positive, self.false_positive)
        print("FN   TN")
        print(self.false_negative, self.true_negative)

        print("\nPrecision: ", (self.true_positive + self.true_negative) /
              (self.true_positive + self.true_negative + self.false_negative + self.false_positive))

        print("Positive-class precision: ", self.true_positive / (self.true_positive + self.false_positive))
        print("Negative-class precision: ", self.true_negative / (self.true_negative + self.false_negative))

        print("\nError rate:", 1 - (self.true_positive + self.true_negative) /
              (self.true_positive + self.true_negative + self.false_negative + self.false_positive))
