from reviewtool import NBReviewClassifier

# Example use of the classifier

review_classifier = NBReviewClassifier()

review_classifier.fit(positive_path='../../../datasets/first_assignment/train/pos/',
                      negative_path='../../../datasets/first_assignment/train/neg/')

review_classifier.classify(positive_path='../../../datasets/first_assignment/test/pos/',
                           negative_path='../../../datasets/first_assignment/test/neg/')

# single review classification
# review_classifier.confusion_matrix()
