from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    from a.preprocessing import *            
except ImportError:
    try:
        from preprocessing import *                   
    except ImportError as error:
        print("ERROR: There was an error loading required modules. Please make sure that all modules are in the correct folders.")



def plot_tree_depths(x_train, x_val, y_train, y_val):
    """
    plot accuracy-depth graph for validation and train data
    
    :param [x_train, x_val, y_train, y_val]: The train and test data and their respective labels.
    """
    train_acc = []
    val_acc = []
    min_depth = 5
    max_depth = 64

    for i in tqdm(range(min_depth, max_depth)):
        clf = tree.DecisionTreeClassifier(max_depth=i)
        clf = clf.fit(x_train, y_train)

        train_acc.append(clf.score(x_train, y_train))
        val_acc.append(clf.score(x_val, y_val))

    train_line, = plt.plot(range(min_depth, max_depth), train_acc)
    val_line, = plt.plot(range(min_depth, max_depth), val_acc)
    plt.xlabel('tree depth')
    plt.ylabel('accuracy')
    plt.legend((train_line, val_line), ('train accuracy', 'val accuracy'))
    plt.savefig('depth_vs_accuracy.png')
    plt.show()


def predict(feature_matrix, type, sentence):
    """
    Predicts the category of a given sentence using either the Logistic Regression or
    Decision Tree model.
    
    :param feature_matrix: the bag-of-words feature matrix
    :param type: defines which model to use, "log_reg" for Logistic Regression and "dec_tree" for the Decision Tree
    :param sentence: the sentence that will be classified
    :return: the prediction of the selected model
    """
    if (type == "log_reg"):
        pkl_filename = "b/data/log_reg_model.pkl"
    elif (type == "dec_tree"):
        pkl_filename = "b/data/dec_tree_model.pkl"

    # Load from file
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)

    prediction = model.predict(feature_matrix.transform([sentence]))[0]

    return (prediction)


def prepare_sklearn_models(x_train, y_train, x_test=[], y_test=[], print_accuracies=False):
    """
    Trains the Logistic Regression and Decision Tree models and saves them.
    :param x_train: Bag of words representation of the train utterances
    :param y_train: Train labels
    :param x_test: Bag of words representation of the train utterances
    :param y_test: Test labels
    :param print_accuracies: Whether you want to print the model quantitative results or not
    :return:
    """

    dec_tree = tree.DecisionTreeClassifier(max_depth=30)
    dec_tree = dec_tree.fit(x_train, y_train)

    if not os.path.exists('b/data'):
        os.makedirs('b/data')
    with open("b/data/dec_tree_model.pkl", 'wb') as file:
        pickle.dump(dec_tree, file)

    #   logistic regression, no validation set cause we don't optimize hyperparameters
    log_reg = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=250)
    log_reg.fit(x_train, y_train)

    with open("b/data/log_reg_model.pkl", 'wb') as file:
        pickle.dump(log_reg, file)

    if print_accuracies:
        # decision tree quantitave evaluation:
        x_train_small, x_val, y_train_small, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=23)
        print("Decision tree accuracies:\n"
              "train {:.2%} validation {:.2%} test {:.2%}".format(*map(dec_tree.score,
                                                                       [x_train_small, x_val, x_test],
                                                                       [y_train_small, y_val, y_test])))

        print('Decision tree metrics on train data:\n'
              "precision {:.2%} recall {:.2%} fscore {:.2%}".format(
            *precision_recall_fscore_support(y_train, dec_tree.predict(x_train), average='weighted')))

        print('Decision tree metrics on test data:\n'
              "precision {:.2%} recall {:.2%} fscore {:.2%}".format(
            *precision_recall_fscore_support(y_test, dec_tree.predict(x_test), average='weighted')))

        # logistic regression quantitative evaluation:
        print("Logistic regression accuracies:\n"
              "train {:.2%} test {:.2%}".format(*map(log_reg.score,
                                                     [x_train, x_test],
                                                     [y_train, y_test])))

        print('Logistic regression metrics on train data:\n'
              "precision {:.2%} recall {:.2%} fscore {:.2%}".format(
            *precision_recall_fscore_support(y_train, log_reg.predict(x_train), average='weighted')))

        print('Logistic regression metrics on test data:\n'
              "precision {:.2%} recall {:.2%} fscore {:.2%}".format(
            *precision_recall_fscore_support(y_test, log_reg.predict(x_test), average='weighted')))
