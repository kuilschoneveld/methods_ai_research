from collections import Counter
from sklearn.model_selection import train_test_split
import sys
import copy
# nltk.download('stopwords')
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import *

try:
    import nn
except ImportError:
    nn = None
    print("Could not import the nn dependencies")

import decisionTree_and_logisticRegression as dtlr
import baselines
baselines.init()


def baseline_stats(function, data):
    """
    Determines the accuracy of both baseline models.
    
    :param function: determines which baseline function should be evaluated
    :param data: the test set with which the function should be evaluated
    :return: the determined accuracy of the "model"
    
    """
    correct = 0
    total = 0
    
    result_dict = {}
    
    for cat in baselines.categories:
        for stat in ["TP", "TN", "FP", "FN"]:
            result_dict[cat + "_" + stat] = 0

    for item in data:
        if item[0] == function(item[1]):
            correct += 1
            result_dict[item[0] + "_TP"] += 1
            for c in baselines.categories:
                if c != item[0] and c != function(item[1]):
                    result_dict[c + "_TN"] += 1
        else:
            result_dict[function(item[1]) + "_FP"] += 1
            result_dict[item[0] + "_FN"] += 1
            for c in baselines.categories:
                if c != item[0] and c != function(item[1]):
                    result_dict[c + "_TN"] += 1

        # For Debugging:
        #             print("RIGHT: " + item[1] + " - " + item[0])
        #         else:
        #             print("WRONG: " + item[1] + " - " + item[0] + ", but " + baseline_2(item[1]))

        total += 1

    return correct / total, result_dict

def nn_stats(keras_model, feature_matrix, x, y, y_cats):
    
    cats = ['inform', 'confirm', 'reqalts', 'request', 'null', 'affirm', 'thankyou', 'negate', 'hello', 'bye', 'repeat',
        'deny', 'reqmore', 'restart', 'ack']
    
    correct = 0
    total = 0
      
    result_dict = {}
    
    for cat in baselines.categories:
        for stat in ["TP", "TN", "FP", "FN"]:
            result_dict[cat + "_" + stat] = 0

    for i in range(len(y)):
        pred = nn.make_prediction(keras_model, x[i], feature_matrix)
        if y_cats[i] == pred:
            correct += 1
            result_dict[y_cats[i] + "_TP"] += 1
            for c in baselines.categories:
                if c != y_cats[i] and c != pred:
                    result_dict[c + "_TN"] += 1
        else:
            result_dict[pred + "_FP"] += 1
            result_dict[y_cats[i] + "_FN"] += 1
            for c in baselines.categories:
                if c != y_cats[i] and c != pred:
                    result_dict[c + "_TN"] += 1
        total += 1
        
    return correct / total, result_dict

if __name__ == '__main__':
    """
    The main function of the program. This function should be called when testing it's functionality.
    It first prepares/trains the models and then starts a user-input-dialog where new sentences
    can be entered. It then shows the predicted category for all models.
    
    The current models are:
    - Baseline 1
    - Baseline 2
    - Logistic Regression
    - Decision Tree
    - Simple Dense Neural Network
    
    For optimisation and debugging purposes, all models are currently trained each time the program is started.
    
    """

    print("Please wait, preparing the models...")

    data = read_data("dialog_acts.dat")

    sentences, labels = split_x_y(read_data("dialog_acts.dat"))

    #   fixed random state to keep the test data unbiased
    x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.15, random_state=42)
    
    baseline_train, baseline_test = train_test_split(data, test_size=0.15, random_state=42)

    #   fit the bag of words representation on train set
    feature_matrix = CountVectorizer(max_features=700)
    feature_matrix.fit(x_train)

    #   apply the bag of words representation to the data
    x_train_bow, x_test_bow = map(feature_matrix.transform, [x_train, x_test])

    baselines.most_common_category = baselines.most_common(np.array(baseline_train))
    baselines.category_ratios = baselines.category_ranking(np.array(baseline_train))

    dtlr.prepare_sklearn_models(x_train_bow, y_train, x_test_bow, y_test, True)

    cats = ['inform', 'confirm', 'reqalts', 'request', 'null', 'affirm', 'thankyou', 'negate', 'hello', 'bye', 'repeat',
            'deny', 'reqmore', 'restart', 'ack']

    Y = np.zeros((len(y_train), 15), dtype=np.int32)
    for i in range(len(y_train)):
        Y_i = np.zeros((1, 15), dtype=np.int32)
        Y_i[0, cats.index(y_train[i])] = 1.0
        Y[i, :] = Y_i
    y_train_vec = Y

    Y = np.zeros((len(y_test), 15), dtype=np.int32)
    for i in range(len(y_test)):
        Y_i = np.zeros((1, 15), dtype=np.int32)
        Y_i[0, cats.index(y_test[i])] = 1.0
        Y[i, :] = Y_i
    y_test_vec = Y
    

    if nn is not None:
        keras_model = nn.create_model(x_train_bow, y_train_vec, x_test_bow, y_test_vec)

    print("Finished preparing the models!")
    print("***")
    
    b1_acc, b1_result_dict = baseline_stats(baselines.baseline_1, baseline_test)
    print(b1_acc)
    print(b1_result_dict)
    b2_acc, b2_result_dict = baseline_stats(baselines.baseline_2, baseline_test)
    print(b2_acc)
    print(b2_result_dict)
    if nn is not None:
        nn_acc, nn_result_dict = nn_stats(keras_model, feature_matrix, x_test, y_test_vec, y_test)
        print(nn_acc)
        print(nn_result_dict)

#     next_sentence = True
#     round = 1
# 
#     sentence = ""
#     print("You can now type in your sentences for prediction.")
#     print('To quit: type q or quit followed by an enter')
#     while True:
#         if sentence == "quit" or sentence == "q":
#             sys.exit("Thank you for using our tool!")
#         else:
#             if (round == 1):
#                 print("-- Round 1 --")
#                 sentence = input("Please type in a sentence: ")
#             else:
#                 print("-- Round " + str(round) + " --")
#                 sentence = input("Please type in another sentence: ")
# 
#             print("Prediction results:")
#             print("  Baseline 1: " + baselines.baseline_1(sentence))
#             print("  Baseline 2: " + baselines.baseline_2(sentence))
#             print("  Logistic Regression: " + dtlr.predict(feature_matrix, "log_reg", sentence))
#             print("  Decision Tree: " + dtlr.predict(feature_matrix, "dec_tree", sentence))
#             if nn is not None:
#                 print("  Neural Net: " + nn.make_prediction(keras_model, sentence, feature_matrix))
#             round += 1
