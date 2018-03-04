from Classifier import NBTextClassifier

# Example use of the classifier

review_classifier = NBTextClassifier()

review_classifier.fit(positive_path='../../../datasets/first_assignment/train/pos/',
                      negative_path='../../../datasets/first_assignment/train/neg/')

review_classifier.classify(positive_path='../../../datasets/first_assignment/test/pos/',
                           negative_path='../../../datasets/first_assignment/test/neg/')

review_classifier.confusion_matrix()
