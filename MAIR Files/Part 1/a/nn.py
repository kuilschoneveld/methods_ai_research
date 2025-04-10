#!/usr/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 1 means "Info" messages will not be printed.
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
# import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np



def create_model(x_train, y_train, x_val, y_val):
    """
    
    A sentence classification script using Bag-of-Words and Keras.
    
    There are currently one trainable NN: A simple NN consisting only of dense layers

    :param [x_train, y_train, x_val, y_val]: The train and test data and their respective labels.
    :return: the trained model
    
    """     
  
    # The second neural net that can be used.  
        # In future versions, the training might be skipped if a finished model is available.
        # For easier optimisation we currently did not include this feature.
    #         if os.path.isfile('data/keras_model_nn.h5'):
    #             print("model found, skipping training of neural net!")
    #         else:
    #             print("no model found, beginning training of neural net!")
    
    # The input dimensions of the neural net
    input_dim = x_train.shape[1]
    input_dim_val = x_val.shape[1]
    
    # The model definition
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation='relu'))
    model.add(Dense(15, activation='softmax'))
     
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
     
#         print(model.summary())
     
    beginning_time = time.time()
    callback = [EarlyStopping(monitor='val_loss',
                        min_delta=0,
                        patience=10,
                        verbose=0, mode='auto')]

    history = model.fit(x_train, y_train,
                    epochs=33,
                    # Set "verbose=True" if you would like to see a progress bar of the training!
                    verbose=False,
                    validation_data=(x_val, y_val),
                    batch_size=50,
                    callbacks=callback)
    
    time_passed = time.time() - beginning_time
    #     print(" --- Time passed: " + str(time_passed) + " --- ")
            
    # OPTIONALLY: Saves the model and shows graph of the development of accuracy and loss during training.
#     model.save('data/keras_model_nn.h5') 
#     print("***Model saved successfully***") 
#
#     print(history.history.keys())  
    
    # OPTIONALLY: Creates and saves a plot, requires the matplotlib import to be enabled
#     fig = plt.figure(1)  
#       
#     plt.subplot(211)  
#     plt.plot(history.history['accuracy'])  
#     plt.plot(history.history['val_accuracy'])  
#     plt.title('model accuracy')  
#     plt.ylabel('accuracy')  
#     plt.xlabel('epoch')  
#     plt.legend(['train', 'test'], loc='upper left')  
#       
#     plt.subplot(212)  
#     plt.plot(history.history['loss'])  
#     plt.plot(history.history['val_loss'])  
#     plt.title('model loss')  
#     plt.ylabel('loss')  
#     plt.xlabel('epoch')  
#     plt.legend(['train', 'test'], loc='upper left')  
# 
#     save_file = os.path.join('data/', 'nn_plot.png')  
#     plt.savefig(save_file)
#
#     plt.close(fig)
    
    return model

def make_prediction(model, sentence, feature_matrix):
    """
    
    A class that returns a category as a prediction
    
    :param sentence: the sentence that will be predicted
    :param model: the trained model of the NN
    :param feature_matrix: the bag-of-words feature matrix for the dense NN
    :return: the predicted category (currently only possible for the dense NN)
    
    """      
    X = feature_matrix.transform([sentence])
         
    result = model.predict(X)
    
    cats = ['inform', 'confirm', 'reqalts', 'request', 'null', 'affirm', 'thankyou', 'negate', 'hello', 'bye', 'repeat', 'deny', 'reqmore', 'restart', 'ack']
    
    return cats[np.argmax(result)]
    
    
    
    
    
    
