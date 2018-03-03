import pandas as pd
import os
from math import log

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk.downloader


def extract_reviews(path: str, encoding: str = 'utf-8'):
    """
    :param path: takes a folder of .txt files and reads them
    :param encoding: encoding of .txt file
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


class NBTextClassifier:

    def __init__(self, positive_path: str, negative_path: str,
                 to_clean: bool = False, encoding: str = 'utf-8', stopwords_lang: str = 'english'):
        """
        //todo: write desc
        :param positive_path: path to positive train text
        :param negative_path: path to negative train text
        :param to_clean: to clean text and remove stopwords
        :param encoding: encoding of text
        :param stopwords_lang: define language for stopwords
        """
        self.to_clean = to_clean
        if to_clean:
            nltk.download('punkt')
            self.stop_words = stopwords.words(stopwords_lang)

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
        print("\tReady.\n")

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

    def classify(self, positive_path: str, negative_path: str, encoding: str = 'utf-8'):
        """
        :param positive_path: path to positive test reviews
        :param negative_path: path to negative test reviews
        :param encoding: encoding of the text documents
        :return: //todo
        """
        print("ESTIMATING TEST DATA")

        print("\tExtracting texts to be classified..")
        test_positive = extract_reviews(positive_path, encoding)
        test_negative = extract_reviews(negative_path, encoding)
        all_reviews = pd.concat([test_negative, test_positive])

        classified_targets = []
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

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
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if is_negative > is_positive:
                    true_negative += 1
                else:
                    false_negative += 1

            # result of classifying a given text
            if is_positive > is_negative:
                result = 1
            classified_targets.append(result)
        print("\tDone.\n")

        # todo: remove this and return a confusion matrix from method
        print("Confusion Matrix:\n"
              "----------------")
        print("TP   FP")
        print(true_positive, false_positive)
        print("FN   TN")
        print(false_negative, true_negative)
        print("\nPrecision: ", (true_positive + true_negative) /
              (true_positive + true_negative + false_negative + false_positive))
