import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer
from numpy.core.fromnumeric import sort
from tflearn.layers.core import activation
stemmer = LancasterStemmer()

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf 
import tensorflow.keras
import tflearn
import random
import json
import pickle

with open("AI_Chatbot/intents.json") as file:
    data = json.load(file)

try:    # Try to open data that is already preprocessed
    with open ("data.pickle", "rb") as f:   # Read pickle data
        words, labels, training, output = pickle.load(f)    # Load data from pickle file

except: # Else re-train model on new data

    # print(data) # Print all the text in the json file
    # print(data["intents"])  # Only print contents inside intents

    words = []
    labels = []
    docs_x = [] # Words from intent
    docs_y = [] # Which intent the words are apart of

    # Use stemming to get root of word/ sentence:
    # Need to tokenize first:
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            words_t = nltk.word_tokenize(pattern) # Get list of words from pattern in json file
            # print(words_t, "\n")
            words.extend(words_t)   # Add words_t to list, words
            docs_x.append(words_t )    
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels: 
            labels.append(intent["tag"])    # list labels contains all tags

    print("\nWords:\n", words)
    print("\ndocs_x:\n", docs_x)
    print("\ndocs_y:\n", docs_y)
    print("\nLabels:\n", labels)

    # Stem words that we have in words list
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]    # Convert words to lowercase and get stem words of scentance
    print("\nstemmed words:\n", words)
    words = sorted(list(set(words)))    # Take words and make set to remove duplicates en convert back to list

    labels = sorted(labels) # Sort list, labels

    # Need to convert sting to bag-of-words/ one-hot encoded:   (Bag of words is like one-hot but how many times the word appears)
    # We use one-hot because we only care about 1 key word

    # Input is one-hot list that says if list exist
    # Output is tags that need to be one-hot

    training = []   # Input to NN
    output = []

    out_empty = [0 for _ in range(len(labels))]    #  Create empty output list with 0's that is the length of the number of classes we have

    for x, doc in enumerate(docs_x):    # enumerate will create a tuple with the xth value and the value in list docs_x
        bag = []    # one hot input

        words_t = [stemmer.stem(w) for w in doc]    # Stem letters in words

        for w in words:     # Go through stemmed words and check if in main word list
            if w in words_t:    
                bag.append(1)   # Word exists
            else:
                bag.append(0)   # Word does not exist

        output_row = out_empty[:]   # Copy of out_empty
        output_row[labels.index(docs_y[x])] = 1     # Look through labels list and find tag, then set that value to 1 in output

        training.append(bag)
        output.append(output_row)

    print("\nwords_t:\n", words_t)  # Stem the words 
    print("\ntraining:\n", training) 
    print("\noutput:\n", output)  

    # Convert input and output lists to numpy arrays for tensorflow
    training = np.array(training)   
    output = np.array(output)

    print("\ntraining:\n", training)  
    print("\noutput:\n", output)  

    with open("data.pickle", "wb") as f:    # After all preprocessing then save pickle model
        pickle.dump((words, labels, training, output), f)

# tf.reset_default_graph()    # Get rid of prev settings    (don't word here)

net = tflearn.input_data(shape = [None, len(training[0])])  # Define input shape of length training
net = tflearn.fully_connected(net, 8)   # Fully connected hidden layer
net = tflearn.fully_connected(net, 8)   # Fully connected hidden layer
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")  # Output layer of length of the output and use softmax to get propability of each outcome
net = tflearn.regression(net)

model = tflearn.DNN(net)    # Train model   (DNN = deep neural network)(DNN is normal neural network)(DNN is an ANN)

try:    # try to load in already trained model
    model.load("model.tflearn")
except: # Else train model and save model
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)   # pass training data to model
    model.save("model.tflearn") # Save model as model.tflearn

# Make predictions:
 
def baf_of_words(s, words): # Convert sentence to back of words # s is sentence and words is words list to create bag of words of

    bag = []
    
    s_words = nltk.words_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i].append(1)

    return np.array(bag)

    