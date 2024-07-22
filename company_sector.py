import keras
from keras import layers
import docx
from company import OpenTextHandler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical,pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import argparse,statistics
import numpy as np,os
from database import Database
import re
import pandas as pd
import warnings
import requests,zipfile
from tqdm import tqdm
warnings.filterwarnings('ignore')


MAX_NB_WORDS=20000
REPLACE_BY_SPACE_RE=re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE=re.compile('[^0-9a-z #+_]')
STOPWORDS=set(stopwords.words('english'))
EMBD_DIM=100

# 2. Load Glove Embeddings in order to use pretrained Embeddings Instead of train an embedding layer from scratch
def get_glove_embedding_layer(folder_for_GloVe:str,word_index,embedding_dim):
    def load_GloVe_embeddings(filepath,word_index,embedding_dim): # Check how can i load the glove embeddings and use a pretrained model for that
        vocab_size=len(word_index)+1
        embedding_matrix_vocab=np.zeros((vocab_size,embedding_dim))

        with open(filepath,encoding='utf-8') as file:
            for line in file:
                word,*vector=line.split()
                if word in word_index:
                    idx=word_index[word]
                    embedding_matrix_vocab[idx]=np.array(vector,dtype=np.float32)
        return embedding_matrix_vocab
    
    def download_GloVe_embeddings(folder_to_be_saved:str):
        url="http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
        filepath=os.path.join(folder_to_be_saved,"globe.6B.zip")
        print(f"Download GloVe Embeddings from {url}...") 
        response=requests.get(url,stream=True)
        
        if response.status_code==200:
            file_size=int(response.headers.get('Content-Length',0))
            progress=tqdm(total=file_size,unit='B',unit_scale=True,unit_divisor=1024)
            with open(filepath,"wb") as file:
                for chunk in response.iter_content(1024):
                    if chunk:  # filter out keep-alive new chunks
                        file.write(chunk)
                        progress.update(len(chunk))
            progress.close()
            print("Download complete!")
        else:
            print(f'Download failed with status code: {response.status_code}')
        
        print("Unzipping the archive...")
        with zipfile.ZipFile(filepath,'r') as zip_ref:
            zip_ref.extractall(path=folder_to_be_saved)
        print("Unzipping complete")
    
    if not os.path.exists(folder_for_GloVe):
        os.makedirs(folder_for_GloVe,exist_ok=True)
        download_GloVe_embeddings(folder_to_be_saved=folder_for_GloVe)
    
    filepath=os.path.join(folder_for_GloVe,f'glove.6B.{EMBD_DIM}d.txt')
    return load_GloVe_embeddings(filepath=filepath,word_index=word_index,embedding_dim=embedding_dim)
    
def prepare_data(descriptions:list,sectors:list):
    company_sectors=OpenTextHandler.get_instance().company_sector_categories
    sector_to_integer_idx={company_sectors[i]:i for i in range(len(company_sectors))}
    num_classes=len(sector_to_integer_idx)
    y=np.array([sector_to_integer_idx[company_sector.strip()] for company_sector in sectors])
    y=to_categorical(y)
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(descriptions)
    sequences=tokenizer.texts_to_sequences(descriptions)
    maxlen=max(len(seq) for seq in sequences)
    vocab_size=len(tokenizer.word_index)+1
    X=pad_sequences(sequences,maxlen=maxlen,padding='post')
    return X,y,maxlen,vocab_size,num_classes

class TextProcessor:
    def __init__(self):
        self.lemmatizer=WordNetLemmatizer()
    
    def clean_text(self,text):
        text=text.lower()
        text=REPLACE_BY_SPACE_RE.sub(' ',text)
        text=BAD_SYMBOLS_RE.sub('',text)
        words=[word for word in text.split() if word not in STOPWORDS]
        words=[self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    
    @staticmethod
    def iniatialize_tokenization(descriptions:list,sectors:list):
        encode_sectors=LabelEncoder()
        company_sectors_idx=encode_sectors.fit_transform(sectors)
        y=to_categorical(np.array(company_sectors_idx))
        tokenizer=Tokenizer(nb_words=MAX_NB_WORDS,char_level=False)
        tokenizer.fit_on_texts(descriptions)
        return tokenizer,y

    def prepare_data(self,dfset,tokenizer):
        sequences=tokenizer.texts_to_sequences(dfset['description'])
        maxlen=max(len(seq) for seq in sequences)
        vocab_size=len(tokenizer.word_index)+1
        X=pad_sequences(sequences,maxlen=maxlen,padding='post')
        return X,maxlen,vocab_size
    
    def __call__(self, dfset:pd.DataFrame,tokenizer):
        dfset['description']=dfset['description'].apply(self.clean_text)
        return self.prepare_data(dfset,tokenizer)
    

class SectorCategorizerNet:
    def __init__(self,num_classes,maxlen,vocab_size,embedding_matrix=None):
        self.run_id=datetime.now().strftime("%m%d%Y_%H%M%S")
        self.results_folder=self.__class__.__name__+os.path.sep+self.run_id
        self.checkpoints=os.path.join(self.results_folder,'Checkpoints')
        os.makedirs(self.results_folder,exist_ok=True)
        os.makedirs(self.checkpoints,exist_ok=True)
        
        self.model=Sequential()
        self.model.add(layers.Input(shape=(maxlen,),name='Text_Input_Layer',dtype='int32'))
        if embedding_matrix is not None:
            self.model.add(layers.Embedding(
                input_dim=vocab_size,
                output_dim=EMBD_DIM,
                weights=[embedding_matrix],
                input_length=maxlen,
                trainable=False
            ))
        else:
            self.model.add(layers.Embedding(
                input_dim=vocab_size,
                output_dim=EMBD_DIM,
                input_length=maxlen,
                trainable=True
            ))

        # CNN-LSTM potentially be added here
        self.model.add(layers.Conv1D(filters=64,kernel_size=3,activation='relu'))
        self.model.add(layers.MaxPooling1D(pool_size=4))
        self.model.add(layers.LSTM(128,dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
        self.model.add(layers.LSTM(64))
        
        self.model.add(layers.Dense(32,activation='relu',kernel_regularizer=keras.regularizers.l2(1e-2)))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(num_classes,activation='softmax'))
        
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    def train(self,X_train,Y_train,X_val,Y_val,save=True):
        history=self.model.fit(
            X_train,
            Y_train,
            epochs=50,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(self.checkpoints,'model_epoch_{epoch:02d}_accuracy_{val_accuracy:.2f}.keras'),
                    monitor='val_accuracy',
                    verbose=True,
                    save_best_only=True,
                    mode='max',
                    save_freq='epoch'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    mode='min',
                    restore_best_weights=True
                )
            ],
            validation_data=(X_val,Y_val)
        )
        self.model.summary()
        
        if save:
            self.model.save(os.path.join(self.results_folder,f'mission_statement_model_{statistics.mean(history.history["accuracy"])}.keras'))       
        
        val_loss,val_accuracy=self.model.evaluate(X_val,Y_val)
        print(f'[INFO]Validation Loss:{val_loss}\tValidation Accuracy:{val_accuracy}')

    def __call__(self, *args: argparse.Any, **kwds: argparse.Any) -> argparse.Any:
        pass
    
class SectorCategorizzerNet2:
    def __init__(self,num_classes,maxlen,vocab_size,embeddings_matrix=None) -> None:
        self.model=Sequential()
        self.maxlen=maxlen
        self.vocab_size=vocab_size
        self.num_classes=num_classes
        if embeddings_matrix is not None:
            self.model.add(layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=EMBD_DIM,
                weights =[embeddings_matrix],
                input_length=self.maxlen,
                embeddings_regularizer=keras.regularizers.l2(1e-2),
                trainable=False
            ))
        else:
            self.model.add(
                layers.Embedding(
                    input_dim=self.vocab_size,
                    output_dim=EMBD_DIM,
                    input_length=maxlen,
                    embeddings_regularizer=keras.regularizers.l2(1e-2),
                    trainable=True
                )
            )
            
            self.model.add(layers.Bidirectional(layers.CuDNNLSTM(64,dropout=0.2,recurrent_dropout=0.2)))
            self.model.add(layers.Dense(32,activation='relu',kernel_regularizer=keras.regularizers.l2(1e-2)))
            self.model.add(layers.Dropout(0.1))
            self.model.add(layers.Dense(self.num_classes,activation='softmax'))
    
    def train(self,X_train,Y_train,X_val,Y_val,save=False):
        history=self.model.fit(
            X_train,
            Y_train,
            epochs=100,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(self.checkpoints,'model_epoch_{epoch:02d}_accuracy_{val_accuracy:.2f}.keras'),
                    monitor='val_accuracy',
                    verbose=True,
                    save_best_only=True,
                    mode='max',
                    save_freq='epoch'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    mode='min',
                    restore_best_weights=True
                )
            ],
            validation_data=(X_val,Y_val)
        )        
        self.model.summary()
        
        if save:
            self.model.save(os.path.join(self.results_folder,f'mission_statement_model_{statistics.mean(history.history["accuracy"])}.keras'))       
        
        val_loss,val_accuracy=self.model.evaluate(X_val,Y_val)
        print(f'[INFO]Validation Loss:{val_loss}\tValidation Accuracy:{val_accuracy}')   

if __name__=='__main__':
    parser=argparse.ArgumentParser(prog="Company-Sector-Categorizer")
    parser.add_argument("--db-path",type=str,help="Path to the sqlite db that contains company descriptions",required=True)
    parser.add_argument("--table-name",type=str,help="Table name that contains the descriptions")
    parser.add_argument("--GloVembd",action='store_true',default=False)
    args=parser.parse_args()
    
    preprocessor=TextProcessor()
    db=Database(args.db_path)
    data=db.fetch_all_as_dataframe(args.table_name)
    db.close()

    tokenizer,y=TextProcessor.iniatialize_tokenization(descriptions=data['description'],sectors=data['sector'])

    df_train,df_test,Ytrain,Ytest=train_test_split(data,y,test_size=0.2,random_state=42,stratify=y)
    Xtrain,maxlen_train,vocabsize_train=preprocessor(df_train,tokenizer)
    Xtest,maxlen_test,vocabsize_test=preprocessor(df_test,tokenizer)
    
    
    maxlen=max(maxlen_train,maxlen_test)
    vocab_size=max(vocabsize_train,vocabsize_test)
    num_classes=y.shape[1]
    embeddings_matrix=None
    
    if args.GloVembd:
        embeddings_matrix=get_glove_embedding_layer(folder_for_GloVe='./data/Embeddings',word_index=tokenizer.word_index,embedding_dim=EMBD_DIM)
    
    model=SectorCategorizerNet(num_classes=num_classes,maxlen=maxlen,vocab_size=vocab_size,embedding_matrix=embeddings_matrix)
    model.train(X_train=Xtrain,Y_train=Ytrain,X_val=Xtest,Y_val=Ytest,save=True)
    