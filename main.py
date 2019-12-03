# @author Micheal Dunne
import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Step 1 - Import dataset iris
def importLoadIris():
    return datasets.load_iris()


# Step 2 - Split information from the dataset into Train, Test, Validation subset
def splitDataset(datasets):
    a = datasets.data
    b = datasets.target

    # Splits Train and Test Data. Train 50%. Test 50%
    aDataTrain, aDataTest, bDataTrain, bDataTest = train_test_split(a, b, test_size=0.5, random_state=5000)

    # print(aDataTrain)
    # print(bDataTrain)
    # Splits Test and Validate. Test %25. Validate 25%.
    aDataTest, aDataValid, bDataTest, bDataValid = train_test_split(aDataTest, bDataTest, test_size=0.5,
                                                                    random_state=7000)

    # print(aDataTest)
    # print("-------------")
    # print(aDataValid)
    # print("-------------")
    # print(bDataTest)
    # print("-------------")
    # print(bDataValid)

    # Return Array of each value
    return aDataTrain, aDataValid, aDataTest, bDataTrain, bDataValid, bDataTest


# Step 3 - Ensure the subsets are Independent and Representative of the original dataset.
def subsetIndependancy(splitdata, datasets):
    uniqueVal, uniqueValcount = np.unique(datasets.target, return_counts=True)
    # Model Selection
    print("Model Selection")
    print(uniqueVal)
    print(uniqueValcount)
    #  Train and count of unique values
    print("bDataTrain")
    uniqueValBTrain, uniqueValcountBTrain = np.unique(splitdata[0], return_counts=True)
    print(uniqueValBTrain)
    print(uniqueValcountBTrain)
    # Test and count of unique values
    print("bDataTest")
    uniqueValBTest, uniqueValcountBTest = np.unique(splitdata[2], return_counts=True)
    print(uniqueValBTest)
    print(uniqueValcountBTest)
    # Validate and count of unique values
    print("bDataValid")
    uniqueValBValid, uniqueValcountBValid = np.unique(splitdata[1], return_counts=True)
    print(uniqueValBValid)
    print(uniqueValcountBValid)
    return uniqueValBValid, uniqueValcountBValid, uniqueValBTest, uniqueValcountBTest, uniqueValBTrain, uniqueValcountBTrain, uniqueVal, uniqueValcount


#     Model Selection
#     [0 1 2]
#     [50 50 50]
#     bDataTrain
#     [0.1 0.2 0.3 0.4 0.5 0.6 1.  1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.  2.1
#     2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3.  3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9
#     4.  4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5.  5.1 5.2 5.3 5.4 5.5 5.6 5.7
#     5.8 5.9 6.  6.1 6.2 6.3 6.4 6.5 6.7 6.8 6.9 7.  7.1 7.2 7.3 7.4 7.7 7.9]
#     [ 2  9  4  5  1  1  6  1  6 11  9  7  6  4  4  3  6  3  2  8  4  4  3  5
#     8  4 10  3 10  5  6  3  2  2  5  4  5  3  2  1  3  5  1  6  3  6  7  6
#     4  2  4  4  6  7  6  4  3  5  2  4  3  2  6  2  2  1  1  2  1  1  3  1]
#     bDataTest
#     [0.1 0.2 0.3 0.4 1.  1.1 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.  2.1 2.2 2.3 2.4
#     2.5 2.6 2.7 2.8 2.9 3.  3.1 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.  4.1 4.3 4.4
#     4.5 4.6 4.8 4.9 5.  5.1 5.2 5.4 5.5 5.6 5.7 5.8 6.  6.1 6.2 6.3 6.4 6.5
#     6.6 6.7 6.9 7.6 7.7]
#     [1 8 2 2 1 1 6 3 9 3 2 5 1 1 2 2 3 1 4 1 1 3 3 5 7 1 2 3 2 2 1 1 1 1 1 3 1
#     2 3 2 4 4 1 3 3 5 3 1 1 1 2 3 3 1 1 4 3 1 1]
#     bDataValid
#     [0.1 0.2 0.3 1.  1.1 1.2 1.3 1.4 1.5 1.6 1.8 1.9 2.1 2.2 2.3 2.4 2.5 2.6
#     2.7 2.8 2.9 3.  3.1 3.2 3.3 3.4 3.5 3.6 3.8 4.2 4.3 4.4 4.5 4.6 4.7 4.8
#     4.9 5.  5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6.  6.1 6.3 6.4 6.5 6.6 6.8
#     7.2]
#     [ 2 12  1  1  2  1  3  9  9  2  3  3  1  2  1  1  3  1  3  3  3 12  1  3
#     2  4  2  1  1  3  1  2  3  4  1  3  3  3  7  1  1  1  3  1  1  3  1  4
#     3  3  2  2  2  1  1]

# Step 4 - Build the first classifier for the problem.
def firstClassifier(splitdata):
    # Model
    classifier = svm.LinearSVC(random_state=0, tol=1e-5)
    # Support Vector Machine
    classifier = classifier.fit(splitdata[0], splitdata[3])
    return classifier


# Step 5 - Build the second classifier for the problem.
def secondClassifier(splitdata):
    # Decision Tree Classifier
    classifier = DecisionTreeClassifier()
    # Fit
    classifier = classifier.fit(splitdata[0], splitdata[3])
    return classifier


# Step 6 - Build the third and final classifier.
def thirdClassifier(splitdata):
    # Logistic Regression
    classifer = LogisticRegression(solver='lbfgs', multi_class='auto')
    # Fit
    classifer = classifer.fit(splitdata[0], splitdata[3])
    return classifer


def bestClassifier(linearSVC, decisionTree, logisticRegression, splitTestA, splitTestB, splitValidA, splitValidB):

    predictTestA = linearSVC.predict(splitTestA)
    print("Linear SVC Predict A")
    print("Classification")
    print(classification_report(splitTestB,predictTestA))
    print("Accuracy Score")
    print(accuracy_score(splitTestB, predictTestA))

    predictTestB = logisticRegression.predict(splitTestA)
    print("Logistic Regression Predict B")
    print("Classification")
    print(classification_report(splitTestB,predictTestB))
    print("Accuracy Score")
    print(accuracy_score(splitTestB, predictTestB))

    predictTestC = decisionTree.predict(splitTestA)
    print("Decision Tree Predict C")
    print("Classification")
    print(classification_report(splitTestB,predictTestC))
    print("Accuracy Score")
    print(accuracy_score(splitTestB, predictTestC))

    validate = linearSVC.predict(splitValidA)
    print("Linear SVC Vaidation")
    print("Classification")
    print(classification_report(splitValidB,validate))
    print("Accuracy Score")
    print(accuracy_score(splitValidB,validate))



#     Step 7 - Select the best out of the three classifiers
#     Linear SVC Predict A
#     0.972972972972973
#     Logistic Regression Predict B
#     0.9459459459459459
#     Decision Tree Predict C
#     0.9459459459459459
#     From these results I can see that Linear SVC is the most accurate out of the three with a result of 0.972972972972973.
#
#     Linear SVC Predict A
#     Classification
#               precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00        13
#            1       1.00      0.92      0.96        12
#            2       0.92      1.00      0.96        12
#
#     accuracy                           0.97        37
#    macro avg       0.97      0.97      0.97        37
# weighted avg       0.98      0.97      0.97        37




# Step 8 - Report on the future performance of the selected classifier.
# Linear SVC Vaidation
# Classification
#               precision    recall  f1-score   support
#
#            0       1.00      1.00      1.00        15
#            1       0.77      1.00      0.87        10
#            2       1.00      0.77      0.87        13
#
#     accuracy                           0.92        38
#    macro avg       0.92      0.92      0.91        38
# weighted avg       0.94      0.92      0.92        38
#
# Accuracy Score
# 0.9210526315789473

# The future performance of the Linear SVC classifier is good. It does become less accurate over time but still remains about 0.9
# which is still very accurate. The accuracy drops by 0.05192034139.
# This accuracy says that it will remain accurate on other data sets.

# Main
def main():
    datasets = importLoadIris()
    splitdata = splitDataset(datasets)
    # subsetIndependancy(splitdata, datasets)
    linearSVC = firstClassifier(splitdata)
    decisionTree = secondClassifier(splitdata)
    logisticRegression = thirdClassifier(splitdata)
    bestClassifier(linearSVC, decisionTree, logisticRegression, splitdata[2], splitdata[5], splitdata[1], splitdata[4])


main()
