import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

def pre_processing(x, sc=None):
    if sc is None:
        sc=StandardScaler()
        x=sc.fit_transform(x)
        return x,sc
    else:
        x=sc.transform(x)
        return x,sc

def create_timeseq(x,y=None,seq_length=10):
    x_seq=[]
    y_seq=[] if y is not None else None

    for i in range((len(x)-seq_length)+1):
        x_seq.append(x[i:i+seq_length])
        if y is not None:
            y_seq.append(y[i+seq_length-1])

    x_seq=np.array(x_seq)
    if len(x_seq) == 0:
        print("Not enough data points to create sequences")
        exit(1)
    
    x_seq = x_seq.reshape(x_seq.shape[0], x_seq.shape[1], -1)
    
    if y is not None:
        y_seq=np.array(y_seq).reshape(-1, 1)
        return x_seq, y_seq

    return x_seq

def split_data(x_seq,y_seq):
    x_train,x_test,y_train,y_test=train_test_split(x_seq,y_seq,test_size=0.2,random_state=42,stratify=y_seq)
    return x_train,x_test,y_train,y_test

def train_model(x_train,y_train,x_test,y_test):
    model=Sequential([
        #First Layer
        LSTM(64,return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])),
        Dropout(0.3), #Prevents Overfitting by disabling 30% neurons- better generalization

        #Second Layer
        LSTM(32,return_sequences=False),
        Dropout(0.3),

        #Connected Layer
        Dense(16, activation='relu'), # rectified linear unit (value; if x>=0, 0;otherwise)

        #Output Layer
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
    model.save("eeg_lstm_model.h5") #HDF5 format
    print("Model successfully saved")
    return model

def predict_output(model,x_test):    
    y_pred_prob = model.predict(x_test) #probability of confusion
    y_pred = (y_pred_prob > 0.5).astype(int) # output 0 or 1
    return y_pred

def eval_model(y_pred,y_test):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n📊 LSTM Model Evaluation:")
    print(f"✅ Accuracy:  {accuracy:.2f}")
    print(f"🎯 Precision: {precision:.2f}")
    print(f"🔄 Recall:    {recall:.2f}")
    print(f"⭐ F1 Score:  {f1:.2f}")
#
data=pd.read_csv("Processed_Data.csv")
features = ['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2', 'Attention', 'Mediation']
x=data[features]
y=data['user-definedlabel']

x,sc=pre_processing(x)
joblib.dump(sc,"scaler.pkl")

x_seq,y_seq=create_timeseq(x,y)
x_train,x_test,y_train,y_test=split_data(x_seq,y_seq)
model=train_model(x_train,y_train,x_test,y_test)
y_pred=predict_output(model,x_test)
eval_model(y_pred,y_test)

##For new Data
'''data=pd.read_csv("Example_data.csv")
features = ['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2', 'Attention', 'Mediation']
x=data[features]
scaler=joblib.load("scaler.pkl")

x, sc = pre_processing(x)  # Normalize features
x_seq= create_timeseq(x)
model=load_model("eeg_lstm_model.h5")
y_pred=predict_output(model,x_seq)
if y_pred==1:
    print("Confused")
else:
    print("Not Confused")'''
