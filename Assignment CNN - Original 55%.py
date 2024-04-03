# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 10:20:20 2023

@author: Moritz
"""
# %% General setup loading libraries etc. and setting working directory
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Dropout
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.chdir(r"C:\Users\User\OneDrive\Dokumente\Uni\12. Big Data and Deep Learning\Data")
#%% Check working directory
print(os.getcwd())
#%% Loading provided datasets 
amzn = pd.read_csv("AMZN.csv")
fb = pd.read_csv("FB.csv")
intc = pd.read_csv("INTC.csv")

# Computing daily returns
amzn["Return"] = amzn["Close"].pct_change()
fb["Return"] = fb["Close"].pct_change()
intc["Return"] = intc["Close"].pct_change()

# Dropping NA values 
amzn = amzn.dropna()
fb = fb.dropna()
intc = intc.dropna()
#%% Check if data is loaded correctly because there was a compatibility issue with the dates
print(amzn.head(50))

#%% define the provided training, validation and test dates
train_start_date="2015-04-28"
train_end_date="2017-12-31"
val_start_date="2018-01-03"
val_end_date="2018-12-31"
test_start_date="2019-01-02"
test_end_date="2020-01-31"

# Splitting the datasets into training, validation,test sets based on dates
train_amzn = amzn[(amzn["Date"] >= train_start_date) & (amzn["Date"] <= train_end_date)]
val_amzn = amzn[(amzn["Date"] >= val_start_date) & (amzn["Date"] <= val_end_date)]
test_amzn = amzn[(amzn["Date"] >= test_start_date) & (amzn["Date"] <= test_end_date)]

train_fb = fb[(fb["Date"] >= train_start_date) & (fb["Date"] <= train_end_date)]
val_fb = fb[(fb["Date"] >= val_start_date) & (fb["Date"] <= val_end_date)]
test_fb = fb[(fb["Date"] >= test_start_date) & (fb["Date"] <= test_end_date)]

train_intc = intc[(intc["Date"] >= train_start_date) & (intc["Date"] <= train_end_date)]
val_intc = intc[(intc["Date"] >= val_start_date) & (intc["Date"] <= val_end_date)]
test_intc = intc[(intc["Date"] >= test_start_date) & (intc["Date"] <= test_end_date)]
#%%Create the forecasting variable
train_amzn["Label"] = (train_amzn["Return"] > 0).astype(int)
val_amzn["Label"] = (val_amzn["Return"] > 0).astype(int)
test_amzn["Label"] = (test_amzn["Return"] > 0).astype(int)

# Check if everything is correctly implemented
train_amzn.head(), val_amzn.head(), test_amzn.head()
#%%
def create_sequences(data, sequence_length=14):
    """
    Creating sequences from the data.
    
   Arguments:
    - data (DataFrame): dataframe containing the data.
    - sequence_length (int):  length of  sequences.
    
    Returns:
    -list of sequences and list of corresponding labels.
    """
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data.iloc[i:i+sequence_length][["Return", "Volume"]].values)
        labels.append(data.iloc[i+sequence_length]["Label"])
    return np.array(sequences), np.array(labels)

train_sequences, train_labels = create_sequences(train_amzn)
val_sequences, val_labels = create_sequences(val_amzn)
test_sequences, test_labels = create_sequences(test_amzn)
#%% Step 7
def model_fn(params):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(14, 2)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))  # Binary classification

    model.compile(optimizer=tf.keras.optimizers.Adam(params["learning_rate"]), loss="binary_crossentropy", metrics=["accuracy"])
# Training the model
    history = model.fit(train_sequences, train_labels, validation_data=(val_sequences, val_labels), epochs=100, batch_size=params["batch_size"], verbose=0)
    
    return history, model
#%% Step 8
search_space = {
    "filters": [32, 64, 128],
    "kernel_size": [2,3, 4],
    "lstm_units": [30, 50, 70],
    "dropout": [0.1, 0.3, 0.5],
    "learning_rate": [0.001, 0.01, 0.1],
    "batch_size": [32, 64, 128]
}
#%%
def random_search(model_fn, search_space, n_iter, search_dir): 
    results = []  # initialise an empty set 

    if not os.path.exists(search_dir):
        os.mkdir(search_dir)  # use os and create a directory if it doesn"t exist

    best_model_path = os.path.join(search_dir, "best_model.h5")
    results_path = os.path.join(search_dir, "results.csv")

    best_val_acc = 0  # Initialize best validation accuracy to 0

    for i in range(n_iter):
        params = {k: v[np.random.randint(len(v))] for k, v in search_space.items()}

        history, model = model_fn(params)
        epochs = np.argmax(history.history["val_accuracy"]) + 1 
        result = {k: v[epochs - 1] for k, v in history.history.items()}
        params["epochs"] = epochs 

        if result["val_accuracy"] > best_val_acc:
            best_val_acc = result["val_accuracy"]
            model.save(best_model_path)

        result = {**params, **result}
        results.append(result)
        tf.keras.backend.clear_session()
        print(f"iteration {i + 1} – {', '.join(f'{k}:{v:.4g}' for k, v in result.items())}")

    best_model = tf.keras.models.load_model(best_model_path)
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path)

    return results_df, best_model
#%%Step 9
results, best_model = random_search(model_fn, search_space, n_iter=30, search_dir="model_search")

#%% 
model = load_model("C:/Users/User/OneDrive/Dokumente/Uni/12. Big Data and Deep Learning/Data/model_search/best_model.h5")


#%% Evaluate the training data
train_loss, train_accuracy = model.evaluate(train_sequences, train_labels)
print(f"Training Accuracy: {train_accuracy*100:.2f}%")
#%% Evaluate on the validation data
val_loss, val_accuracy = model.evaluate(val_sequences, val_labels)
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
#%% Make predictions
predictions = model.predict(test_sequences)
classified_predictions = [1 if p >= 0.5 else 0 for p in predictions]

#%%
print(classified_predictions)
#%%Extract true lables for accuracy evaluation

true_labels = test_labels.tolist()

# Count correct predictions
correct_predictions = sum([1 if true_labels[i] == classified_predictions[i] else 0 for i in range(len(true_labels))])

# Compute  accuracy
accuracy = correct_predictions / len(true_labels)

print(f"Accuracy: {accuracy*100:.2f}%")
#%% Check metrics
accuracy = accuracy_score(true_labels, classified_predictions)
precision = precision_score(true_labels, classified_predictions)
recall = recall_score(true_labels, classified_predictions)
f1 = f1_score(true_labels, classified_predictions)

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1-Score: {f1*100:.2f}%")
