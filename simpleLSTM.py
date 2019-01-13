import os
import nltk
import collections
import numpy as np
import codecs
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM

from keras.callbacks import TensorBoard

class SampleData:
    """
    サンプルのデータをnumpy型で作成する
    データは
    https://www.kaggle.com/c/si650winter11/data
    からダウンロードして./dataに解凍しておく
    """
    def __init__(self,data_dir="./data",file_train = "umich-sentiment-train.txt"):
        self.data_dir = data_dir
        self.file_train = file_train
        # 各カウンターの初期化
        self.word_freq = collections.Counter()   # 単語の出現数のカウンタ
        self.maxlen = 0                          # １文章の最大長
        self.num_recs = 0                        # データファイルに含まれる文章数
        self.vocab_size = 0                      # 学習に使用する単語数
        self.word2index =dict()                  # 単語からIDへの辞書
        self.index2word = dict()                 # IDから単語への辞書

    def read_data(self,deliminator = "\t",max_features=2000):
        # １文章に含まれる最大単語数のカウンタを初期化
        with codecs.open(os.path.join(self.data_dir,self.file_train),"r","utf-8") as ftrain:
            for line in ftrain:
                # ラベルと文章はタブで区切られている
                label , sentence = line.strip().split(deliminator)
                try:
                    # 単語ごとに正規化してリスト化
                    words = nltk.word_tokenize(sentence.lower())
                except LookupError:
                    print("Englisth tokenize does not downloaded. So download it.")
                    nltk.download("punkt")
                    words = nltk.word_tokenize(sentence.lower())
                # 最大単語数のレコードを更新
                self.maxlen = max(self.maxlen,len(words))
                # 各単語のカウンタを更新
                for word in words:
                    self.word_freq[word] += 1
                # 文章数のカウンタを更新
                self.num_recs += 1
        # 学習に使用する単語数の更新
        # 読み込んだデータに出現する全単語数が
        # max_featuresより大きい場合max_featuresの値までで切り捨てる
        self.vocab_size = min(max_features,len(self.word_freq)) + 2
        # 出現頻度順にIDをふっていく
        for i,x in enumerate(self.word_freq.most_common(max_features)):
            # idの0と1はPADとUNKで予約しておくので+2したidにする
            self.word2index[x[0]] = i+2
        # 系列長より短い文章の場合、文章の残った部分はPADとする
        # 学習時に出現しなかった単語に関してはUNKとする
        self.word2index["PAD"] = 0
        self.word2index["UNK"] = 1

        for k,v in self.word2index.items():
            self.index2word[v] = k
    def create_data(self,deliminator="\t",max_sentence_length=40):
        self.X = np.empty((self.num_recs,),dtype=list)
        self.y = np.zeros((self.num_recs, ))
        i = 0
        with codecs.open(os.path.join(self.data_dir,self.file_train),"r","utf-8") as ftrain:
            for line in ftrain:
                label, sentence = line.strip().split(deliminator)
                # 単語ごとに正規化してリスト化
                words = nltk.word_tokenize(sentence.lower())
                seqs = []
                for word in words:
                    if word in self.word2index:
                        seqs.append(self.word2index[word])
                    else:
                        seqs.append(self.word2index["UNK"])
                self.X[i] = seqs
                self.y[i] = int(label)
                i +=1
        # 文章の最大長までパディングする
        self.X = sequence.pad_sequences(self.X , maxlen=max_sentence_length)
        # データをランダムに並び替えて訓練データとテストデータにする
        self.Xtrain,self.Xtest,self.ytrain,self.ytest = train_test_split(self.X,self.y,test_size=0.2,random_state=42)




class KerasEmbeddingLSTM:
    def build(self,vocab_size=50,embedding_size=128,max_sentence_length=40,hidden_size=64):
        self.model = Sequential()
        # 単語のIDの並びのみで表されている入力をembedding_sizeの列サイズの行列に
        # 変換する学習を行う層
        self.model.add(Embedding(input_dim=vocab_size,output_dim=embedding_size,input_length=max_sentence_length))
        # 入力の一部を捨てる層
        self.model.add(Dropout(0.5))
        # シーケンスの学習を行う層
        self.model.add(LSTM(units=hidden_size, dropout=0.5, recurrent_dropout=0.5))
        # 最終的な出力を行う層入力はEmbeddingで１xembedding_sizeになっている
        # そこからLSTMのユニットに拡張され最終的に１の出力に全結合される
        self.model.add(Dense(1))
        self.model.add(Activation("sigmoid"))

        self.model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"])
    def train_model(self,x_train,y_train,x_test,y_test,batch_size=32,num_epochs=10,log_dir="./logs"):
        self.history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            callbacks=[TensorBoard(log_dir)],
            validation_data=(x_test,y_test))
        self.score,self.acc = self.model.evaluate(x_test,y_test,batch_size=batch_size)
        print("Test Score:{:.3f}, accuracy: {:.3f}".format(self.score, self.acc))
    
    