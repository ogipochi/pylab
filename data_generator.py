import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split

class SimpleSin:
    def __init__(self,T=100,maxlen=25):
        np.random.seed(0)
        self.T = T
        self.maxlen = maxlen
    def create_toy(self,ample=0.05):
        x = np.arange(0,2*self.T + 1)
        noise = ample * np.random.uniform(low=-1.0,high=1.0,size=len(x))
        self.wave = np.sin(2.0 * np.pi * x/self.T) + noise
        return self.wave
    def construct_data(self):
        length_of_sequences = 2 * self.T
        self.data = []
        self.target = []
        for i in range(0,length_of_sequences-self.maxlen+1):
            self.data.append(self.wave[i:i+self.maxlen])
            self.target.append(self.wave[i+self.maxlen])
        X = np.array(self.data).reshape(len(self.data), self.maxlen, 1)
        Y = np.array(self.target).reshape(len(self.target), 1)
        # データ設定
        N_train = int(len(self.data) * 0.9)
        N_validation = len(self.data) - N_train
        
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=N_validation)
        return (X_train,X_validation,Y_train,Y_validation)


class WaveData():
    def __init__(self,T=100,ampl=0.05,A=1):
        self.x = np.arange(0,2*T+1)
        self.T = T
        self.ampl = ampl
        self.A = A
        self.wave = None
    def generate(self):
        self.wave = self.A * (np.sin(2.0*np.pi*self.x/self.T)*np.cos(4*np.pi*self.x/self.T)*np.sin(7*np.pi*self.x/self.T))
        return self.wave
    def add_noise(self):
        
        noise = self.ampl * np.random.uniform(
            low = -1*(self.A),
            high= self.A,
            size=len(self.x)
        )
        self.wave += noise
        return self.wave
    def plot(self):
        plt.plot(self.x,self.wave)
        plt.show()
    def get_data(self,maxlen=25):
        length_of_sequences = 2*self.T
        self.data=[]
        self.target = []
        for i in range(0,length_of_sequences - maxlen + 1):
            self.data.append(self.wave[i:i+maxlen]) # i番目からi+maxlen-1番目までのデータ
            self.target.append(self.wave[i+maxlen]) # その次のmaxlen番目のデータ
        self.X = np.array(self.data).reshape(len(self.data),maxlen,1)
        self.Y = np.array(self.target).reshape(len(self.data),1)

        # データを訓練用とテストように分割する
        num_train = int(len(self.data) * 0.9)
        num_validation = len(self.data) - num_train
        # 指定した数でデータを分割
        self.X_train,self.X_validation,self.Y_train,self.Y_validation = train_test_split(self.X,self.Y,test_size=num_validation)
        
        return self.X_train,self.X_validation,self.Y_train,self.Y_validation

class MultiRandomWave():
    def __init__(self,num_wave=2,T=1000,ampl=0.1,A=1):
        self.num_wave = num_wave
        self.T = T
        random_b= random.randint(0,self.T)
        self.x = np.arange(random_b,random_b + 2*T+1)
        print(self.x)
        
        self.ampl = ampl
        self.A = A
        self.wave_list = []
    def generate(self):
        for i in range(self.num_wave):
            num_sin = random.randint(2,10)
            num_cos = random.randint(2,10)
            wave = 1
            
            for i in range(num_sin):
                deg_sin = random.random()*(10/self.T)
                wave = wave * np.sin(deg_sin * np.pi * self.x)
            for i in range(num_cos):
                deg_cos = random.random()*(10/self.T)
                wave = wave * np.sin(deg_cos * np.pi * self.x)
            wave = self.A * wave
            wave = wave/max(np.abs(wave))
            self.wave_list.append(wave)
        return self.wave_list
    def add_noise(self):
        for i,wave in enumerate(self.wave_list):
            noise = self.ampl * np.random.uniform(
            low = -1*(self.A),
            high= self.A,
            size=len(self.x))
            self.wave_list[i]+=noise
        return self.wave_list
    def plot(self):
        for i,wave in enumerate(self.wave_list):
            plt.plot(self.x,wave)
        plt.show()
    
        
    
