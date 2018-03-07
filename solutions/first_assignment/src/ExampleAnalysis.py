from reviewtool import NBReviewClassifier

# Example use of the classifier

# initialize classifier
review_classifier = NBReviewClassifier(to_clean=False)
# alternatively:
# review_classifier = NBReviewClassifier(to_clean=True, stopwords_lang='english')

# performance on test set
review_classifier.fit(positive_path='../../../datasets/first_assignment/train/pos/',
                      negative_path='../../../datasets/first_assignment/train/neg/', encoding='utf-8')

results = review_classifier.classify(positive_path='../../../datasets/first_assignment/test/pos/',
                                     negative_path='../../../datasets/first_assignment/test/neg/', encoding='utf-8')
review_classifier.print_scores()

# performance on training set
review_classifier.classify(positive_path='../../../datasets/first_assignment/train/pos/',
                           negative_path='../../../datasets/first_assignment/train/neg/', encoding='utf-8')
review_classifier.print_scores()
