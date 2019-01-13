import pandas as pd
import matplotlib.pyplot as plt
import matplotlib #only needed to determine Matplotlib version number
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers.recurrent import SimpleRNN
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Sequential

class MyRNNModelV1:
    """
    とりあえず自然データの学習ができた最初のモデル
    """
    def __init__(self,maxlen=25,n_input=1,n_output=1,n_hidden=20):
        self.maxlen = maxlen
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
    def load_data(self,df_list):
        # 初期化
        self.data = []
        self.target = []

        for i in range(0,len(df_list) - self.maxlen):
            self.data.append(df_list[i:i+self.maxlen])
            self.target.append(df_list[i+self.maxlen])
        # モデルに入力できる形にreshape
        self.X = np.array(self.data).reshape(len(self.data),self.maxlen,1)
        self.Y = np.array(self.target).reshape(len(self.target),1)

        # データの分割
        N_train = int(len(self.data) * 0.9)
        N_validation = len(self.data) - N_train
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(self.X, self.Y, test_size=N_validation)

    def build(self):
        self.model = Sequential()
        self.model.add(SimpleRNN(self.n_hidden,input_shape=(self.maxlen,self.n_input)))
        self.model.add(Dense(self.n_output))
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.model.compile(loss='mean_squared_error',optimizer=optimizer)
    def train(self,epochs=500,batch=10):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        self.model.fit(self.X_train, self.Y_train,
          batch_size=batch,
          epochs=epochs,
          validation_data=(self.X_validation, self.Y_validation),
         callbacks=[early_stopping])
    def predict(self,df_list,stop=100):
        """
        df_listで与えられたデータから未来を予測しかえす。
        """
        self.predicted_data = []
        data = df_list[len(df_list)-self.maxlen-1:-1]
        for i in range(stop):
            # 入力を初期化
            input_arrray = np.array(data).reshape(1,self.maxlen,self.n_input)
            v = self.model.predict(input_arrray)
            self.predicted_data.append(v[0][0])
            # input_arrayの初期化
            data.append(v[0][0])
            data.pop(0)
        return self.predicted_data

class MyDeepRNNModelV1(MyRNNModelV1):
    def build(self):
        self.model = Sequential()
        self.model.add(SimpleRNN(self.n_hidden,input_shape=(self.maxlen,self.n_input),return_sequences=True))
        self.model.add(SimpleRNN(self.n_hidden,return_sequences=True))
        self.model.add(SimpleRNN(self.n_hidden,return_sequences=True))
        self.model.add(SimpleRNN(self.n_hidden))
        self.model.add(Dense(self.n_output))
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self.model.compile(loss='mean_squared_error',optimizer=optimizer)
        self.model.summary()


        

