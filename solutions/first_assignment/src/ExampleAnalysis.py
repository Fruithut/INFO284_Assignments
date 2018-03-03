from Classifier import NBTextClassifier

# Example use of the classifier

classifier = NBTextClassifier(positive_path='../../../datasets/first_assignment/train/pos/',
                              negative_path='../../../datasets/first_assignment/train/neg/')
classifier.classify(positive_path='../../../datasets/first_assignment/test/pos/',
                    negative_path='../../../datasets/first_assignment/test/neg/')
