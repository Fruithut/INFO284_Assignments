from reviewtool import NBReviewClassifier

# Example use of the classifier

# initialize the classifier
review_classifier = NBReviewClassifier()

# alternatively -> review_classifier = NBReviewClassifier(to_clean=True, stopwords_lang='english')

# fit training data
review_classifier.fit(positive_path='../../../datasets/first_assignment/train/pos/',
                      negative_path='../../../datasets/first_assignment/train/neg/', encoding='utf-8')
# classify on test data
results = review_classifier.classify(positive_path='../../../datasets/first_assignment/test/pos/',
                                     negative_path='../../../datasets/first_assignment/test/neg/', encoding='utf-8')

result = review_classifier.predict("This move was very good, the actors were fantastic")
print("Single review prediction: ", result)

c_matrix = review_classifier.confusion_matrix()
print("Confusion matrix dictionary: ", c_matrix)

review_classifier.print_scores()

# performance on training data
review_classifier.classify(positive_path='../../../datasets/first_assignment/train/pos/',
                           negative_path='../../../datasets/first_assignment/train/neg/', encoding='utf-8')

review_classifier.print_scores()
