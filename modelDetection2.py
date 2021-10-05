import json
import plotly.graph_objs as go
import plotly
import numpy as np
import os
import time
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.layers import Input, Dense,LSTM,GRU,RepeatVector,TimeDistributed,Bidirectional,Conv1D, MaxPooling1D, UpSampling1D,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier,LocalOutlierFactor
from gensim import utils
import gensim.parsing.preprocessing as gsp
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from sklearn.decomposition import TruncatedSVD
import h5py as h5
import pywt
import keras
import uuid
import itertools
from statsmodels.tsa.arima_model import ARIMA
import seaborn as sns
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score,precision_recall_curve,average_precision_score,precision_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from nltk.cluster.kmeans import KMeansClusterer
from scipy.spatial import distance
import math


plt.rcParams.update({'figure.max_open_warning': 0})
matplotlib.use('Agg')

def outliersPlot(df,value,pred,firstOutliers):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=df.index,
            y=df[value],
            name="Observed values"
        ))
    for i in range(firstOutliers):
        ind =df.sort_values("outlier_score", ascending=False).index[i]
        val = df.sort_values("outlier_score", ascending=False).loc[ind][value]
        fig.add_annotation(
            x=ind,
            y=val,
            text="Outlier {}".format(i+1),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="#636363",
            ax=90,
            ay=-90,
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=1,
            bgcolor="#ff7f0e",
            opacity=0.8)
      
    fig.update_annotations(dict(
            showarrow=True
))
    
    fig.add_trace(go.Scatter(
            x=df.index,
            y=df[pred],
            name="Predicted values"
        ))

    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def sub_curve(df,ind,subseqlength):
    Y= df.loc[ind,"ts1":"ts{}".format(subseqlength)].values
    x=np.array(range(subseqlength))
    plt.figure()
    plt.plot(x, Y)
    plt.title(ind)
    plotfile = os.path.join('static', str(time.time()) + '.png')
    plt.savefig(plotfile)
    plt.clf()
    return plotfile

def split_sequence(sequence, n_steps):
    X = []
    y = []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def changePointDetectModel(df,date,value,lookback,epochs,Lstm = True,gru = True,CONV1D = True,cnn_lstm = True,BiLstm= True, Arima = True,firstOutliers=10):
    df = df[[date,value]].copy()
    df.set_index(date, inplace=True)
    df.index = pd.to_datetime(df.index)    
    df_result = df.copy()
    score_list = []
    pred_list = []
    if Lstm:
        modelLSTM = Sequential()
        modelLSTM.add(LSTM(64, return_sequences=True, input_shape=(lookback, 1)))
        modelLSTM.add(LSTM(128, return_sequences=True))
        modelLSTM.add(LSTM(256))
        modelLSTM.add(Dense(1))
        modelLSTM.compile(optimizer= "adam", loss='mse') 
        callback_lstm = [keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=20,verbose=0),\
                    keras.callbacks.ModelCheckpoint(filepath='lstmTS.h5',monitor='loss',save_best_only=True,\
                                                  save_weights_only=True,mode='min',verbose=0)]
        

    if gru:
        modelGRU = Sequential()
        modelGRU.add(GRU(64, return_sequences=True, input_shape=(lookback, 1)))
        modelGRU.add(GRU(128, return_sequences=True))
        modelGRU.add(GRU(256))
        modelGRU.add(Dense(1))
        modelGRU.compile(optimizer= "adam", loss='mse') 
        callback_GRU = [keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=20,verbose=0),\
                    keras.callbacks.ModelCheckpoint(filepath='GRUTS.h5',monitor='loss',save_best_only=True,\
                                                  save_weights_only=True,mode='min',verbose=0)]
        

        
    if CONV1D:
        modelConv1D = Sequential()
        kernel_size = lookback//2
        modelConv1D.add(Conv1D(64,kernel_size=kernel_size,activation='relu',input_shape=(lookback, 1)))
        modelConv1D.add(MaxPooling1D(pool_size=2))
        modelConv1D.add(Flatten())
        modelConv1D.add(Dense(1))
        modelConv1D.compile(optimizer= "adam", loss='mse')
        callback_conv1D = [keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=20,verbose=0),\
                    keras.callbacks.ModelCheckpoint(filepath='conv1DTS.h5',monitor='loss',save_best_only=True,\
                                                  save_weights_only=True,mode='min',verbose=0)]
      
    
    if cnn_lstm:
        lookback_cnn_lstm = lookback
        if (lookback%2) != 0:
            lookback_cnn_lstm = lookback+1
                     
        n_seq = 2
        n_steps = lookback_cnn_lstm//2
        while (n_steps%2==0) and (n_steps > 2) :
            n_seq = n_seq*2
            n_steps = n_steps//2
            

        kernel = n_steps//2
        model_cnn_lstm = Sequential()
        model_cnn_lstm.add(TimeDistributed(Conv1D(64, kernel_size=kernel, activation='relu'),input_shape=(None, n_steps, 1)))
        model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model_cnn_lstm.add(TimeDistributed(Flatten())) 
        model_cnn_lstm.add(LSTM(128, activation='relu')) 
        model_cnn_lstm.add(Dense(1))
        model_cnn_lstm.compile(optimizer='adam', loss='mse')

        callback_CNN_lstm = [keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=20,verbose=1),\
                            keras.callbacks.ModelCheckpoint(filepath='CNN_lstmTS.h5',monitor='loss',save_best_only=True,\
                                                          save_weights_only=True,mode='min',verbose=1)]
        
        
    if BiLstm:
        modelBiLSTM = Sequential()
        modelBiLSTM.add(Bidirectional(LSTM(128, activation='relu'), input_shape=(lookback, 1))) 
        modelBiLSTM.add(Dense(1))
        modelBiLSTM.compile(optimizer= "adam", loss='mse') 
        callback_Bi_lstm = [keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=20,verbose=0),\
                    keras.callbacks.ModelCheckpoint(filepath='Bi_lstmTS.h5',monitor='loss',save_best_only=True,\
                                                  save_weights_only=True,mode='min',verbose=0)]


    
    if Arima:
        p = d = q = range(0, 3)
        pdq = list(itertools.product(p, d, q))
        bestAIC = np.inf
        bestParam = None
        for param in pdq:
            try:
                mod = ARIMA(df, order=param)
                results = mod.fit()
                if results.aic < bestAIC:
                    bestAIC = results.aic
                    bestParam = param
            except:
                continue
                
        modelARIMA = ARIMA(df, order=bestParam)
        result = modelARIMA.fit()
        predARIMA = result.predict(start = bestParam[1] ,end = df.shape[0]-1 ,typ = "levels")
        df_result["Arima_pred"] = predARIMA
        df_result["Arima_score"] = (df[value] - predARIMA).abs()
        score_list.append("Arima_score")
        pred_list.append("Arima_pred")

    adf_result = adfuller(df)
    if adf_result[1] > 0.05:
        df2 = df.diff(periods=1).dropna()
        X, y = split_sequence(df2.values, lookback)
        
        if Lstm:
            history = modelLSTM.fit(X, y, epochs=epochs, callbacks=callback_lstm, verbose=0)
            trainPredict = modelLSTM.predict(X)
            ind = df2.index[lookback:].copy()
            trainPredictDf2 = pd.DataFrame(trainPredict,index = ind,columns=["Lstm_pred"])
            df3=df2.join(trainPredictDf2,how="left")
            df4 = df.shift(1).copy()
            df4.rename(columns={value:"Lstm_pred"},inplace =True)
            df5 = df3[["Lstm_pred"]] + df4
            df6=df.join(df5,how="left")
            df_result["Lstm_pred"] = df6["Lstm_pred"]
            df_result["Lstm_score"] = (df6[value] - df6["Lstm_pred"]).abs()
            score_list.append("Lstm_score")
            pred_list.append("Lstm_pred")

            
            
        if gru:
            history = modelGRU.fit(X, y, epochs=epochs, callbacks=callback_GRU, verbose=0)
            trainPredict = modelGRU.predict(X)
            ind = df2.index[lookback:].copy()
            trainPredictDf2 = pd.DataFrame(trainPredict,index = ind,columns=["gru_pred"])
            df3=df2.join(trainPredictDf2,how="left")
            df4 = df.shift(1).copy()
            df4.rename(columns={value:"gru_pred"},inplace =True)
            df5 = df3[["gru_pred"]] + df4
            df6=df.join(df5,how="left")
            df_result["GRU_score"] = (df6[value] - df6["gru_pred"]).abs()
            df_result["gru_pred"] = df6["gru_pred"]
            score_list.append("GRU_score")
            pred_list.append("gru_pred")

            
            
        if CONV1D:
            history = modelConv1D.fit(X, y, epochs=epochs, callbacks=callback_conv1D, verbose=0)
            trainPredict = modelConv1D.predict(X)
            ind = df2.index[lookback:].copy()
            trainPredictDf2 = pd.DataFrame(trainPredict,index = ind,columns=["cnn_pred"])
            df3=df2.join(trainPredictDf2,how="left")
            df4 = df.shift(1).copy()
            df4.rename(columns={value:"cnn_pred"},inplace =True)
            df5 = df3[["cnn_pred"]] + df4
            df6=df.join(df5,how="left")
            df_result["Conv1D_score"] = (df6[value] - df6["cnn_pred"]).abs()
            df_result["cnn_pred"] = df6["cnn_pred"]
            score_list.append("Conv1D_score")
            pred_list.append("cnn_pred")
            
            
            
        if cnn_lstm:
            X_new, y_new = split_sequence(df2.values, lookback_cnn_lstm)
            X_new = X_new.reshape((X_new.shape[0], n_seq, n_steps, 1))
            history = model_cnn_lstm.fit(X_new, y_new, epochs=epochs, callbacks=callback_CNN_lstm, verbose=0)
            trainPredict = model_cnn_lstm.predict(X_new)
            ind = df2.index[lookback_cnn_lstm:].copy()
            trainPredictDf2 = pd.DataFrame(trainPredict,index = ind,columns=["cnn_lstm_pred"])
            df3=df2.join(trainPredictDf2,how="left")
            df4 = df.shift(1).copy()
            df4.rename(columns={value:"cnn_lstm_pred"},inplace =True)
            df5 = df3[["cnn_lstm_pred"]] + df4
            df6=df.join(df5,how="left")
            df_result["CNN_LSTM_score"] = (df6[value] - df6["cnn_lstm_pred"]).abs()
            df_result["cnn_lstm_pred"] = df6["cnn_lstm_pred"]
            score_list.append("CNN_LSTM_score")
            pred_list.append("cnn_lstm_pred")
            
        if BiLstm:
            history = modelBiLSTM.fit(X, y, epochs=epochs, callbacks=callback_Bi_lstm, verbose=0)
            trainPredict = modelBiLSTM.predict(X)
            ind = df2.index[lookback:].copy()
            trainPredictDf2 = pd.DataFrame(trainPredict,index = ind,columns=["Bi_lstm_pred"])
            df3=df2.join(trainPredictDf2,how="left")
            df4 = df.shift(1).copy()
            df4.rename(columns={value:"Bi_lstm_pred"},inplace =True)
            df5 = df3[["Bi_lstm_pred"]] + df4
            df6=df.join(df5,how="left")
            df_result["BiLstm_score"] = (df6[value] - df6["Bi_lstm_pred"]).abs()
            df_result["Bi_lstm_pred"] = df6["Bi_lstm_pred"]
            score_list.append("BiLstm_score")
            pred_list.append("Bi_lstm_pred")


 
    else:
        X, y = split_sequence(df.values,lookback)
        
        if Lstm:
            history = modelLSTM.fit(X, y, epochs=epochs, callbacks=callback_lstm, verbose=0)
            trainPredict = modelLSTM.predict(X)
            ind = df.index[lookback:].copy()
            trainPredictDf2 = pd.DataFrame(trainPredict,index = ind,columns=["lstm_pred"])
            df6=df.join(trainPredictDf2,how="left")
            df_result["Lstm_score"] = (df6[value] - df6["lstm_pred"]).abs()
            df_result["lstm_pred"] = df6["lstm_pred"]
            score_list.append("Lstm_score")
            pred_list.append("lstm_pred")
            
        if gru:
            history = modelGRU.fit(X, y, epochs=epochs, callbacks=callback_GRU, verbose=0)
            trainPredict = modelGRU.predict(X)
            ind = df.index[lookback:].copy()
            trainPredictDf2 = pd.DataFrame(trainPredict,index = ind,columns=["gru_pred"])
            df6=df.join(trainPredictDf2,how="left")
            df_result["GRU_score"] = (df6[value] - df6["gru_pred"]).abs()
            df_result["gru_pred"] = df6["gru_pred"]
            score_list.append("GRU_score")
            pred_list.append("gru_pred")

            
        if CONV1D:
            history = modelConv1D.fit(X, y, epochs=epochs, callbacks=callback_lstm, verbose=0)
            trainPredict = modelConv1D.predict(X)
            ind = df.index[lookback:].copy()
            trainPredictDf2 = pd.DataFrame(trainPredict,index = ind,columns=["cnn_pred"])
            df6=df.join(trainPredictDf2,how="left")
            df_result["Conv1D_score"] = (df6[value] - df6["cnn_pred"]).abs()
            df_result["cnn_pred"] = df6["cnn_pred"]
            score_list.append("Conv1D_score")
            pred_list.append("cnn_pred")
            
            
        if cnn_lstm:
            X_new, y_new = split_sequence(df.values, lookback_cnn_lstm)
            X_new = X_new.reshape((X_new.shape[0], n_seq, n_steps, 1))
            history = model_cnn_lstm.fit(X_new, y_new, epochs=epochs, callbacks=callback_CNN_lstm, verbose=0)
            trainPredict = model_cnn_lstm.predict(X_new)
            ind = df.index[lookback:].copy()
            trainPredictDf2 = pd.DataFrame(trainPredict,index = ind,columns=["cnn_lstm_pred"])
            df6=df.join(trainPredictDf2,how="left")
            df_result["CNN_LSTM_score"] = (df6[value] - df6["cnn_lstm_pred"]).abs()
            df_result["cnn_lstm_pred"] = df6["cnn_lstm_pred"]
            score_list.append("CNN_LSTM_score")
            pred_list.append("cnn_lstm_pred")
            
        if BiLstm:
            history = modelBiLSTM.fit(X, y, epochs=epochs, callbacks=callback_Bi_lstm, verbose=0)
            trainPredict = modelBiLSTM.predict(X)
            ind = df.index[lookback:].copy()
            trainPredictDf2 = pd.DataFrame(trainPredict,index = ind,columns=["Bi_lstm_pred"])
            df6=df.join(trainPredictDf2,how="left")
            df_result["BiLstm_score"] = (df6[value] - df6["Bi_lstm_pred"]).abs()
            df_result["Bi_lstm_pred"] = df6["Bi_lstm_pred"]
            score_list.append("BiLstm_score")
            pred_list.append("Bi_lstm_pred")


            

    if not score_list:
        return None
      
    for c in score_list:
        scaler = StandardScaler()
        scaler.fit(df_result[c].values.reshape(-1, 1))
        df_result[c] = scaler.transform(df_result[c].values.reshape(-1, 1))
        del scaler
        
    df_result["outlier_score"]=df_result[score_list].mean(axis=1)
    df_result["pred"]=df_result[pred_list].mean(axis=1)

    return outliersPlot(df_result,value,"pred",firstOutliers)


def knnDetector(df,k=50):
    X = df.copy()
    Y = range(df.shape[0])
    knn = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
    knn.fit(X, Y)
    distances, _ = knn.kneighbors()
    X["knn_score"] = 0.0
    for i in range(X.shape[0]):
        X.loc[X.index[i],"knn_score"] = distances[i].mean()

    return X["knn_score"]

def subSeqDetectModel(df, date, value, subseqlength,numberOfEpochs=100,numberOfNeighbors=100,numberOfTrees=500, IF=True,knn=True,LOF=True,AE=True,Conv1D_AE=True,LSTM_AE=True,kmeans = True,pca = True,firstOutliers=20):
    df = df[[date,value]].copy()
    df.set_index(date, inplace=True)
    df.index = pd.to_datetime(df.index)
    if subseqlength%2 !=0:
        subseqlength=subseqlength+1
        
    dico = {}
    for i in range(0,df.shape[0],subseqlength):
        """try:
            ind = df.index.values[i+(subseqlength//2)]
        except:
            ind = df.index.values[i]
        dico[ind] = df.values[i:i+subseqlength]"""
        ind = df.index.values[i:i+subseqlength]
        ind = tuple(ind)
        if len(ind) == subseqlength:
            dico[ind] = df.values[i:i+subseqlength]
    
    newDf = pd.DataFrame.from_dict(dico.items())
    newDf.columns = ["Date","RawTs"]
    newDf.index = newDf["Date"]
    del newDf["Date"]
    """if (df.shape[0]%subseqlength !=0):
        newDf.drop(index =newDf.index[-1],inplace=True)"""

    # Creation of a new dataframe df_last from newDf, where each value in the subsequence corresponds to a column
    X = np.zeros((newDf.shape[0],subseqlength + 1),dtype='O')
    for row,i in enumerate(newDf.index):
        X[row,0]= i
        for c in range(subseqlength):
            X[row,c+1]= newDf.loc[newDf.index == i].values[0][0][c][0]
    
    df_last = pd.DataFrame(X)
    df_last.columns = ["Time"] + ["ts{}".format(i+1) for i in range(subseqlength)]
    #df_last["Time"] = pd.to_datetime(df_last["Time"])
    L = ["ts{}".format(i+1) for i in range(subseqlength)]
    for c in L:
        df_last[c]=df_last[c].astype("float")
    
    df_last.index = df_last["Time"]
    del df_last["Time"]
    # Creation of a new dataframe df_last_dwt of Wavelet coefficients, we apply a DWT transformation to df_last
    df_last_dwt = df_last.copy()
    for i in range(df_last_dwt.shape[0]):
        cA, cD = pywt.dwt(df_last_dwt.loc[df_last_dwt.index == df_last_dwt.index[i]].values, 'db1')
        C = np.concatenate((cA.T,cD.T),axis=0)
        df_last_dwt.loc[df_last_dwt.index == df_last_dwt.index[i]] = C.T
    # Standard scaling of columns
    X_std = df_last_dwt.copy()
    for c in X_std.columns:
        scaler = StandardScaler()
        scaler.fit(X_std[c].values.reshape(-1, 1))
        X_std[c] = scaler.transform(X_std[c].values.reshape(-1, 1))
        del scaler
    df_results=df_last.copy()
    # Calculation of the outlier score
    score_list=[]
    if IF:
        algoDetector = IsolationForest(n_estimators=numberOfTrees,n_jobs=-1)
        algoDetector.fit(X_std)
        df_results["score_IF"] = -algoDetector.score_samples(X_std) 
        score_list.append("score_IF") 
    
    if knn:
        if X_std.shape[0]> numberOfNeighbors:
            k=numberOfNeighbors
        else:
            k = X_std.shape[0]//2
        
        df_results["knn_score"] = knnDetector(X_std,k)
        score_list.append("knn_score")
    
    if LOF:
        if X_std.shape[0]> numberOfNeighbors:
            k=numberOfNeighbors
        else:
            k = X_std.shape[0]//2
        
        # Calculation of the outlier score
        LOF = LocalOutlierFactor(n_neighbors=k)
        LOF.fit(X_std)
        df_results["LOF_score"] = - LOF.negative_outlier_factor_
        score_list.append("LOF_score")
        
    if pca:
        N = subseqlength//2
        pca_mod = PCA(n_components=N)
        pca_mod.fit(X_std)
        df2_pca = pca_mod.transform(X_std)
        df2_inverse = pca_mod.inverse_transform(df2_pca)
        df_results["PCA_score"] = ((X_std - df2_inverse) ** 2).sum(axis=1)
        score_list.append("PCA_score")
    
    if AE:
        X_train, X_val = train_test_split(X_std, test_size=0.10)
        input_data = Input(shape=(subseqlength,))
        encoded = Dense(subseqlength - 1, activation='relu')(input_data)
        encoded = Dense(subseqlength - 2, activation='relu')(encoded)
        encoded = Dense(subseqlength - 3, activation='relu')(encoded)
        decoded = Dense(subseqlength - 2, activation='relu')(encoded)
        decoded = Dense(subseqlength - 1, activation='relu')(decoded)
        decoded = Dense(subseqlength, activation=None)(decoded)
        autoencoder = Model(input_data, decoded)
        encoder = Model(input_data, encoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        callback_AE = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=60,verbose=1),\
                        keras.callbacks.ModelCheckpoint(filepath='AE_test.h5',monitor='val_loss',save_best_only=True,\
                                                  save_weights_only=True,mode='min',verbose=0)]

        history = autoencoder.fit(X_train, X_train,
                    epochs=numberOfEpochs,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(X_val, X_val),
                    callbacks=callback_AE,
                    verbose = 0)
        # Calculation of the outlier score
        X_recons = autoencoder.predict(X_std)
        Error =X_std- X_recons
        df_results["score_AE"] = ((Error)**2).sum(axis=1)
        score_list.append("score_AE")
        
    if kmeans:
        def distanceMahalanobis(u,v,df=X_std):
            covMatrix = pd.DataFrame.cov(df)
            try:
                inVcovMatrix = np.linalg.inv(covMatrix)
            except:
                inVcovMatrix = np.linalg.pinv(covMatrix)
                
            return distance.mahalanobis(u, v, inVcovMatrix)
        
        data = X_std.values
        clusterResult = X_std.copy()
        score_list_kmeans = []
        for k in range(7,9):
            kclusterer = KMeansClusterer(k, distance=distanceMahalanobis, repeats=7,avoid_empty_clusters=True)
            assigned_clusters = kclusterer.cluster(data, assign_clusters=True)
            clusterResult["clusterLabel"] = assigned_clusters
            clusterResult["score1"] = 0.0 
            clusterResult["score2"] = 0.0
            for row in range(clusterResult.shape[0]):
                label = clusterResult.iloc[row,]["clusterLabel"]
                centroid = kclusterer.means()[int(label)]
                cluster = clusterResult[clusterResult["clusterLabel"]==label].loc[:,"ts1":"ts{}".format(subseqlength)]
                num = cluster.shape[0]
                #clusterResult.loc[clusterResult.index[row],"score2"] = - np.log(num/clusterResult.shape[0])
                clusterResult.iloc[row,-1] = - np.log(num/clusterResult.shape[0])
                #u = clusterResult.loc[clusterResult.index[row],"ts1":"ts{}".format(subseqlength)]
                u = clusterResult.iloc[row,]["ts1":"ts{}".format(subseqlength)]
                covMatrixCluster = pd.DataFrame.cov(cluster)
                try:
                    inVcovMatrixCluster = np.linalg.inv(covMatrixCluster)
                except:
                    inVcovMatrixCluster = np.linalg.pinv(covMatrixCluster)
                    
                clusterResult.loc[clusterResult.index[row],"score1"] = distance.mahalanobis(u, centroid, inVcovMatrixCluster)
        
            for c in ["score1","score2"]:
                scaler = StandardScaler()
                scaler.fit(clusterResult[c].values.reshape(-1, 1))
                clusterResult[c] = scaler.transform(clusterResult[c].values.reshape(-1, 1))
                del scaler
        
            clusterResult["score_{}".format(k)]=clusterResult[["score1","score2"]].mean(axis=1)
            score_list_kmeans.append("score_{}".format(k))
            del clusterResult["score1"]
            del clusterResult["score2"]
            del clusterResult["clusterLabel"]
            
        df_results["score_kMeans"] = clusterResult[score_list_kmeans].mean(axis=1)
        score_list.append("score_kMeans")
        
    if Conv1D_AE:
        X_train, X_val = train_test_split(df_last, test_size=0.10)
        X_train=X_train.values.reshape(X_train.shape[0],X_train.shape[1],1)
        X_val=X_val.values.reshape(X_val.shape[0],X_val.shape[1],1)
        
        input_data1D = Input((subseqlength,1))
        encoded1D = Conv1D(32, 5, activation='relu',padding='same')(input_data1D)
        encoded1D = MaxPooling1D(2)(encoded1D)
        encoded1D = Conv1D(64, 3, activation='relu',padding='same')(encoded1D)
        decoded1D = Conv1D(64,3, activation='relu', padding='same')(encoded1D)
        decoded1D = UpSampling1D(2)(decoded1D)
        decoded1D = Conv1D(32, 5, activation='relu',padding='same')(decoded1D)
        decoded1D = Conv1D(1,5, activation=None, padding='same')(decoded1D)
        autoencoder1D = Model(input_data1D, decoded1D)
        autoencoder1D.compile(optimizer='adam', loss='mean_squared_error')

        callback_Conv1DAE = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=60,verbose=1),\
                    keras.callbacks.ModelCheckpoint(filepath='Conv_1DAE_API.h5',monitor='val_loss',save_best_only=True,\
                                                  save_weights_only=True,mode='min',verbose=1)]
        
        history1D = autoencoder1D.fit(X_train, X_train,
                epochs=numberOfEpochs,
                batch_size=32,
                shuffle=True,
                validation_data=(X_val, X_val),
                callbacks=callback_Conv1DAE)
        
        X_recons_conv1D = autoencoder1D.predict(df_last.values.reshape(df_last.shape[0],df_last.shape[1],1))
        X_recons_conv1D=X_recons_conv1D.reshape(X_recons_conv1D.shape[0],X_recons_conv1D.shape[1])
        Error1D =df_last- X_recons_conv1D
        df_results["score_AE_conv1D"] = ((Error1D)**2).sum(axis=1)
        score_list.append("score_AE_conv1D")
        
    if LSTM_AE:
        #X_train, X_val = train_test_split(df_last, test_size=0.10)
        #X_train=X_train.values.reshape(X_train.shape[0],X_train.shape[1],1)
        #X_val=X_val.values.reshape(X_val.shape[0],X_val.shape[1],1)
        
        autoencoderLSTM = Sequential()
        autoencoderLSTM.add(LSTM(128, activation='relu',return_sequences=True,input_shape=(subseqlength,1)))
        autoencoderLSTM.add(LSTM(64, activation='relu', return_sequences=False))
        autoencoderLSTM.add(RepeatVector(subseqlength))
        autoencoderLSTM.add(LSTM(64, activation='relu', return_sequences=True))
        autoencoderLSTM.add(LSTM(128, activation='relu', return_sequences=True))
        autoencoderLSTM.add(TimeDistributed(Dense(1)))
        autoencoderLSTM.compile(optimizer='adam', loss='mse')
        
        callback_LSTM_AE = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=60,verbose=1),\
                    keras.callbacks.ModelCheckpoint(filepath='LSTM_AE_API.h5',monitor='val_loss',save_best_only=True,\
                                                  save_weights_only=True,mode='min',verbose=1)]

        historyLSTM = autoencoderLSTM.fit(df_last.values.reshape(df_last.shape[0],df_last.shape[1],1), 
                                          df_last.values.reshape(df_last.shape[0],df_last.shape[1],1),
                                            epochs=numberOfEpochs,
                                            batch_size=32,
                                            shuffle=True,
                                            callbacks=callback_LSTM_AE)
        
        X_recons_LSTM = autoencoderLSTM.predict(df_last.values.reshape(df_last.shape[0],df_last.shape[1],1))
        X_recons_LSTM=X_recons_LSTM.reshape(X_recons_LSTM.shape[0],X_recons_LSTM.shape[1])
        ErrorLSTM =df_last- X_recons_LSTM
        df_results["score_AE_LSTM"] = ((ErrorLSTM)**2).sum(axis=1)
        score_list.append("score_AE_LSTM")
     
    if not score_list:
        return None
    
    # Normalization of outlier scores
    for c in score_list:
        scaler = StandardScaler()
        scaler.fit(df_results[c].values.reshape(-1, 1))
        df_results[c] = scaler.transform(df_results[c].values.reshape(-1, 1))
        del scaler

    # The final score is equal to the average of all scores
    df_results["final_score"]=df_results[score_list].mean(axis=1)
    #ind = df_results.sort_values("final_score",ascending=False).index[0]
    result =[]
    #if not os.path.isdir('static'):
        #os.mkdir('static')
    #else:
        # Remove old plot files
        #for filename in glob.glob(os.path.join('static', '*.png')):
            #os.remove(filename)
 
    #plt.figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=df.index,
            y=df[value]))

    for k in df_results.sort_values("final_score",ascending=False).index[0:firstOutliers]:
        #result.append(sub_curve(df_results,k,subseqlength))
        result.append(dict(
                type="rect",
                # x-reference is assigned to the x-values
                xref="x",
                # y-reference is assigned to the plot paper [0,1]
                yref="paper",
                x0=str(k[0]),
                y0=0,
                x1=str(k[subseqlength-1]),
                y1=1,
                fillcolor="LightSalmon",
                opacity=0.5,
                layer="below",
                line_width=0,
            ))
        
    fig.update_layout(shapes=result)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def clean_text(d):
    d = d.lower()
    d = utils.to_unicode(d)
    d = gsp.strip_tags(d)
    d = gsp.strip_punctuation(d)
    d = gsp.strip_multiple_whitespaces(d)
    d = gsp.strip_numeric(d)
    d = gsp.remove_stopwords(d)
    d = gsp.strip_short(d)
    d = gsp.stem_text(d)
    return d

def knnCosineDetector(df,k=50):
    X = df.copy()
    Y = X.index
    knn = KNeighborsClassifier(n_neighbors=k,metric=cosine,algorithm="ball_tree",n_jobs=-1)
    knn.fit(X, Y)
    distances, _ = knn.kneighbors()
    X["knn_score"] = 0.0
    for i in range(X.shape[0]):
        X.loc[X.index[i],"knn_score"] = distances[i].mean()

    return X["knn_score"]


def textDetectModel(df,text_col,knn=True,LOF=True,LSA=True,AE=True,firstOutliers=20,numberOfNeighbors=100,epochs=30):
    df["clean_text"] = df[text_col].map(lambda x: clean_text(x))
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df["clean_text"])
    X_train = X.toarray()
    X_train_pd = pd.DataFrame(X_train,index=df.index)
    score_list = []
    if knn:
        if X_train_pd.shape[0] > numberOfNeighbors:
            k = numberOfNeighbors
        else:
            k = X_train_pd.shape[0] //2
            
        df["knn_score"] = knnCosineDetector(X_train_pd,k)
        score_list.append("knn_score")
        
    if LOF:
        if X_train_pd.shape[0] > numberOfNeighbors:
            k = numberOfNeighbors
        else:
            k = X_train_pd.shape[0] //2
            
        LOF = LocalOutlierFactor(n_neighbors=k,metric=cosine,algorithm="ball_tree",n_jobs=-1)
        LOF.fit(X_train_pd)
        df["LOF_score"] = - LOF.negative_outlier_factor_
        score_list.append("LOF_score")
        
    if LSA:
        svd = TruncatedSVD(n_components=100)
        X_train_pd_red = svd.fit_transform(X_train_pd)
        X_reconstruction= svd.inverse_transform(X_train_pd_red)
        Error_LSA = X_train_pd - X_reconstruction
        df["score_LSA"] = ((Error_LSA)**2).sum(axis=1)
        score_list.append("score_LSA")
        
    if AE:
        input_data = Input(shape=(X_train.shape[1],))
        encoded = Dense(X_train.shape[1]//4, activation='relu')(input_data)
        encoded = Dense(X_train.shape[1]//16, activation='relu')(encoded)
        encoded = Dense(X_train.shape[1]//32, activation='relu')(encoded)
        decoded = Dense(X_train.shape[1]//16, activation='relu')(encoded)
        decoded = Dense(X_train.shape[1]//4, activation='relu')(decoded)
        decoded = Dense(X_train.shape[1], activation=None)(decoded)
        autoencoder = Model(input_data, decoded)
        encoder = Model(input_data, encoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        callback_AE = [keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=60,verbose=1),\
                        keras.callbacks.ModelCheckpoint(filepath='AE_text.h5',monitor='loss',save_best_only=True,\
                                                  save_weights_only=True,mode='min',verbose=1)]

        history = autoencoder.fit(X_train, X_train,
                    epochs=epochs,
                    batch_size=64,
                    shuffle=True,
                    callbacks=callback_AE,
                    verbose = 1)
        X_recons = autoencoder.predict(X_train)
        Error =X_train- X_recons
        df["score_AE"] = ((Error)**2).sum(axis=1)
        score_list.append("score_AE")
        
    if not score_list:
        return None
    
    # Normalization of outlier scores
    for c in score_list:
        scaler = StandardScaler()
        scaler.fit(df[c].values.reshape(-1, 1))
        df[c] = scaler.transform(df[c].values.reshape(-1, 1))
        del scaler

    # The final score is equal to the average of all scores    
    df["Outlier_score"]=df[score_list].mean(axis=1)
    result = df[[text_col,"Outlier_score"]].sort_values("Outlier_score",ascending=False).head(firstOutliers)
    result = result.to_html(classes="table table-striped")
    titles=df.columns.values
    return result,titles

def supervisedOutlier(df,form):
    feature_list = df.columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(df.columns)-set(num_cols))
    selected_num_features = []
    selected_cat_features = []
    for field, val in form.data.items():
        if field in feature_list:
            if val == True:
                if field in num_cols:
                    selected_num_features.append(field)
                elif field in cat_cols:
                    selected_cat_features.append(field)
                    
    selected_features = selected_num_features + selected_cat_features
    
    """feature_list = df.columns
    selected_features = []
    for field, val in form.data.items():
        if field in feature_list:
            if val == True:
                selected_features.append(field)"""
            
    label_col = form.label_col.data
                        
    if label_col not in selected_features:
        selected_features.append(label_col)
        
    df = df[selected_features].copy()
    RF_OS = form.data["RF_OS"]
    RF_SM = form.data["RF_SM"]
    RF_US_ENS = form.data["RF_US_ENS"]
    XGB_OS = form.data["XGB_OS"]
    XGB_SM = form.data["XGB_SM"]
    XGB_US_ENS = form.data["XGB_US_ENS"]
    num_estimators = form.data["num_estimators"]
    
    X = df.drop([label_col], axis=1)
    Y = df[label_col]
    le = LabelEncoder()
    Y_enc =le.fit_transform(Y)
    Y_enc = pd.Series(Y_enc,index = Y.index,name=label_col)
    c1 = Y_enc.loc[Y_enc==le.classes_[0]].shape[0]
    c2 = Y_enc.loc[Y_enc==le.classes_[1]].shape[0]
    if c1 < c2:
        min_class = le.classes_[0]
        maj_class = le.classes_[1]
    else:
        min_class = le.classes_[1]
        maj_class = le.classes_[0]
    
    if min_class < maj_class:
        index_min_label = 0
    else:
        index_min_label = 1    

    for c in selected_cat_features:
        dummies = pd.get_dummies(X[c], drop_first=True,prefix=c)
        X = pd.concat([X, dummies], axis=1)
        del X[c]      
        
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_enc, test_size=0.3,random_state=2727,stratify=Y_enc)
    X_TRAIN = pd.concat([X_train, Y_train], axis=1)
    
    minority_samples = X_TRAIN[X_TRAIN[label_col]==min_class]
    majority_samples = X_TRAIN[X_TRAIN[label_col]==maj_class]
    
    roc_curve_dict = {}
    confusion_dict = {}
    
    if RF_OS:
        minority_oversampled = resample(minority_samples,
                          replace=True,
                          n_samples=len(majority_samples),
                          random_state=2727)
        oversampled = pd.concat([majority_samples, minority_oversampled])
        y_train_over = oversampled[label_col]
        X_train_over = oversampled.drop(label_col, axis=1)
        model_oversampled = RandomForestClassifier(n_estimators=100,n_jobs=-1)
        model_oversampled.fit(X_train_over, y_train_over)
        oversampled_pred = model_oversampled.predict(X_test)
        oversampled_pred_prob = model_oversampled.predict_proba(X_test)
        roc_curve_dict["Random Forest(Over Sampling)"] = []
        roc_curve_dict["random"] = []
        fpr, tpr, thresh = roc_curve(Y_test, oversampled_pred_prob[:,index_min_label], pos_label=min_class)
        random_probs = [0 for i in range(len(Y_test))]
        p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=min_class)
        auc_score_oversampled = roc_auc_score(Y_test, oversampled_pred_prob[:,index_min_label])
        roc_curve_dict["Random Forest(Over Sampling)"].append(fpr)
        roc_curve_dict["Random Forest(Over Sampling)"].append(tpr)
        roc_curve_dict["Random Forest(Over Sampling)"].append(auc_score_oversampled)
        roc_curve_dict["random"].append(p_fpr)
        roc_curve_dict["random"].append(p_tpr)
        
        cnf_matrix = confusion_matrix(Y_test,oversampled_pred)
        confusion_dict["Random Forest(Over Sampling)"] = [cnf_matrix]
        rec_over = recall_score(Y_test, oversampled_pred, pos_label = min_class)
        pre_over = precision_score(Y_test, oversampled_pred, pos_label = min_class)
        confusion_dict["Random Forest(Over Sampling)"].append(rec_over)
        confusion_dict["Random Forest(Over Sampling)"].append(pre_over)
        
    if RF_SM:
        sm = SMOTE(random_state=2727)
        X_train_sm, Y_train_sm = sm.fit_sample(X_train, Y_train)
        model_smote = RandomForestClassifier(n_estimators=100,n_jobs=-1)
        model_smote.fit(X_train_sm, Y_train_sm)
        smote_pred = model_smote.predict(X_test)
        smote_pred_prob = model_smote.predict_proba(X_test)
        roc_curve_dict["Random Forest(SMOTE)"] = []
        roc_curve_dict["random"] = []
        fpr_sm, tpr_sm, thresh_sm = roc_curve(Y_test, smote_pred_prob[:,index_min_label], pos_label=min_class)
        random_probs = [0 for i in range(len(Y_test))]
        p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=min_class)
        auc_score_sm = roc_auc_score(Y_test, smote_pred_prob[:,index_min_label])
        roc_curve_dict["Random Forest(SMOTE)"].append(fpr_sm)
        roc_curve_dict["Random Forest(SMOTE)"].append(tpr_sm)
        roc_curve_dict["Random Forest(SMOTE)"].append(auc_score_sm)
        roc_curve_dict["random"].append(p_fpr)
        roc_curve_dict["random"].append(p_tpr)
        
        cnf_matrix = confusion_matrix(Y_test,smote_pred)
        confusion_dict["Random Forest(SMOTE)"] = [cnf_matrix]
        #f1_over = f1_score(Y_test, smote_pred)
        #confusion_dict["Random Forest(SMOTE)"].append(f1_over)
        rec_over = recall_score(Y_test, smote_pred, pos_label = min_class)
        pre_over = precision_score(Y_test, smote_pred, pos_label = min_class)
        confusion_dict["Random Forest(SMOTE)"].append(rec_over)
        confusion_dict["Random Forest(SMOTE)"].append(pre_over)
        
    if RF_US_ENS:
        #num_estimators = 600
        estimators_list = []
        for i in range(num_estimators):
            majority_samples_under = resample(majority_samples,
                                replace = False,
                                n_samples = len(minority_samples))
            undersampled = pd.concat([majority_samples_under, minority_samples])
            y_train_underSample = undersampled[label_col]
            X_train_underSample = undersampled.drop(label_col, axis=1)
            estimator = RandomForestClassifier(n_estimators=100,n_jobs=-1)
            estimator.fit(X_train_underSample, y_train_underSample)
            estimators_list.append(estimator)
            
        predictions = np.asarray([clf.predict(X_test) for clf in estimators_list]).T
        maj_vote = np.apply_along_axis(lambda x:np.argmax(np.bincount(x)),axis=1,arr=predictions)
        probas = np.asarray([clf.predict_proba(X_test) for clf in estimators_list])
        avg_proba = np.average(probas,axis=0)
        roc_curve_dict["Random Forest(Under Sampling)"] = []
        roc_curve_dict["random"] = []
        #maj_vote_probs = np.argmax(avg_proba,axis=1)
        fpr_under, tpr_under, thresh_under = roc_curve(Y_test, avg_proba[:,index_min_label], pos_label=min_class)
        random_probs = [0 for i in range(len(Y_test))]
        p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=min_class)
        auc_score_under = roc_auc_score(Y_test, avg_proba[:,index_min_label])
        #auc_score_sm = roc_auc_score(Y_test, smote_pred_prob[:,1])
        roc_curve_dict["Random Forest(Under Sampling)"].append(fpr_under)
        roc_curve_dict["Random Forest(Under Sampling)"].append(tpr_under)
        roc_curve_dict["Random Forest(Under Sampling)"].append(auc_score_under)
        roc_curve_dict["random"].append(p_fpr)
        roc_curve_dict["random"].append(p_tpr)
        
        cnf_matrix = confusion_matrix(Y_test,maj_vote)
        confusion_dict["Random Forest(Under Sampling)"] = [cnf_matrix]
        #f1_over = f1_score(Y_test, maj_vote)
        #confusion_dict["Random Forest(Under Sampling)"].append(f1_over)
        rec_over = recall_score(Y_test, maj_vote, pos_label = min_class)
        pre_over = precision_score(Y_test, maj_vote, pos_label = min_class)
        confusion_dict["Random Forest(Under Sampling)"].append(rec_over)
        confusion_dict["Random Forest(Under Sampling)"].append(pre_over)
 
    if XGB_OS:
        minority_oversampled = resample(minority_samples,
                          replace=True,
                          n_samples=len(majority_samples),
                          random_state=2727)
        oversampled = pd.concat([majority_samples, minority_oversampled])
        y_train_over = oversampled[label_col]
        X_train_over = oversampled.drop(label_col, axis=1)
        model_oversampled = XGBClassifier(n_estimators=100,n_jobs=-1)
        model_oversampled.fit(X_train_over, y_train_over)
        oversampled_pred = model_oversampled.predict(X_test)
        oversampled_pred_prob = model_oversampled.predict_proba(X_test)
        roc_curve_dict["XGBoost(Over Sampling)"] = []
        roc_curve_dict["random"] = []
        fpr, tpr, thresh = roc_curve(Y_test, oversampled_pred_prob[:,index_min_label], pos_label=min_class)
        random_probs = [0 for i in range(len(Y_test))]
        p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=min_class)
        auc_score_oversampled = roc_auc_score(Y_test, oversampled_pred_prob[:,index_min_label])
        roc_curve_dict["XGBoost(Over Sampling)"].append(fpr)
        roc_curve_dict["XGBoost(Over Sampling)"].append(tpr)
        roc_curve_dict["XGBoost(Over Sampling)"].append(auc_score_oversampled)
        roc_curve_dict["random"].append(p_fpr)
        roc_curve_dict["random"].append(p_tpr)
        
        cnf_matrix = confusion_matrix(Y_test,oversampled_pred)
        confusion_dict["XGBoost(Over Sampling)"] = [cnf_matrix]
        #f1_over = f1_score(Y_test, oversampled_pred)
        #confusion_dict["XGBoost(Over Sampling)"].append(f1_over)
        rec_over = recall_score(Y_test, oversampled_pred, pos_label = min_class)
        pre_over = precision_score(Y_test, oversampled_pred, pos_label = min_class)
        confusion_dict["XGBoost(Over Sampling)"].append(rec_over)
        confusion_dict["XGBoost(Over Sampling)"].append(pre_over)

    if XGB_SM:
        sm = SMOTE(random_state=2727)
        X_train_sm, Y_train_sm = sm.fit_sample(X_train, Y_train)
        model_smote = XGBClassifier(n_estimators=100,n_jobs=-1)
        model_smote.fit(X_train_sm, Y_train_sm)
        smote_pred = model_smote.predict(X_test)
        smote_pred_prob = model_smote.predict_proba(X_test)
        roc_curve_dict["XGBoost(SMOTE)"] = []
        roc_curve_dict["random"] = []
        fpr_sm, tpr_sm, thresh_sm = roc_curve(Y_test, smote_pred_prob[:,index_min_label], pos_label=min_class)
        random_probs = [0 for i in range(len(Y_test))]
        p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=min_class)
        auc_score_sm = roc_auc_score(Y_test, smote_pred_prob[:,index_min_label])
        roc_curve_dict["XGBoost(SMOTE)"].append(fpr_sm)
        roc_curve_dict["XGBoost(SMOTE)"].append(tpr_sm)
        roc_curve_dict["XGBoost(SMOTE)"].append(auc_score_sm)
        roc_curve_dict["random"].append(p_fpr)
        roc_curve_dict["random"].append(p_tpr)
        
        cnf_matrix = confusion_matrix(Y_test,smote_pred)
        confusion_dict["XGBoost(SMOTE)"] = [cnf_matrix]
        #f1_over = f1_score(Y_test, smote_pred)
        #confusion_dict["XGBoost(SMOTE)"].append(f1_over)
        rec_over = recall_score(Y_test, smote_pred, pos_label = min_class)
        pre_over = precision_score(Y_test, smote_pred, pos_label = min_class)
        confusion_dict["XGBoost(SMOTE)"].append(rec_over)
        confusion_dict["XGBoost(SMOTE)"].append(pre_over)

    if XGB_US_ENS:
        #num_estimators = 600
        estimators_list = []
        for i in range(num_estimators):
            majority_samples_under = resample(majority_samples,
                                replace = False,
                                n_samples = len(minority_samples))
            undersampled = pd.concat([majority_samples_under, minority_samples])
            y_train_underSample = undersampled[label_col]
            X_train_underSample = undersampled.drop(label_col, axis=1)
            estimator = XGBClassifier(n_estimators=100,n_jobs=-1)
            estimator.fit(X_train_underSample, y_train_underSample)
            estimators_list.append(estimator)
            
        predictions = np.asarray([clf.predict(X_test) for clf in estimators_list]).T
        maj_vote = np.apply_along_axis(lambda x:np.argmax(np.bincount(x)),axis=1,arr=predictions)
        probas = np.asarray([clf.predict_proba(X_test) for clf in estimators_list])
        avg_proba = np.average(probas,axis=0)
        #maj_vote_probs = np.argmax(avg_proba,axis=1)
        roc_curve_dict["XGBoost(Under Sampling)"] = []
        roc_curve_dict["random"] = []
        fpr_under, tpr_under, thresh_under = roc_curve(Y_test, avg_proba[:,index_min_label], pos_label=min_class)
        random_probs = [0 for i in range(len(Y_test))]
        p_fpr, p_tpr, _ = roc_curve(Y_test, random_probs, pos_label=min_class)
        auc_score_under = roc_auc_score(Y_test, avg_proba[:,index_min_label])
        roc_curve_dict["XGBoost(Under Sampling)"].append(fpr_under)
        roc_curve_dict["XGBoost(Under Sampling)"].append(tpr_under)
        roc_curve_dict["XGBoost(Under Sampling)"].append(auc_score_under)
        roc_curve_dict["random"].append(p_fpr)
        roc_curve_dict["random"].append(p_tpr)
        
        cnf_matrix = confusion_matrix(Y_test,maj_vote)
        confusion_dict["XGBoost(Under Sampling)"] = [cnf_matrix]
        #f1_over = f1_score(Y_test, maj_vote)
        #confusion_dict["XGBoost(Under Sampling)"].append(f1_over)
        rec_over = recall_score(Y_test, maj_vote, pos_label = min_class)
        pre_over = precision_score(Y_test, maj_vote, pos_label = min_class)
        confusion_dict["XGBoost(Under Sampling)"].append(rec_over)
        confusion_dict["XGBoost(Under Sampling)"].append(pre_over)
  
    if not roc_curve_dict:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=roc_curve_dict["random"][0],
            y=roc_curve_dict["random"][1],
            mode='lines+markers',
            name='Random'))
    del roc_curve_dict["random"]
    for mod in roc_curve_dict.keys():
        fig.add_trace(go.Scatter(
            x=roc_curve_dict[mod][0],
            y=roc_curve_dict[mod][1],
            mode='lines',
            name= mod + " with AUC score {:.3f}".format(roc_curve_dict[mod][2])))
        
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    confusion_list = []
    if not os.path.isdir('static'):
        os.mkdir('static')
    else:
        # Remove old plot files
        for filename in glob.glob(os.path.join('static', '*.png')):
            os.remove(filename)
 
    plt.figure()
    for mod in confusion_dict.keys():
        cm = confusion_dict[mod][0]
        title = "{} - Recall:{:.3f} - Precision:{:.3f}".format(mod,confusion_dict[mod][1],confusion_dict[mod][2])
        sns.heatmap(pd.DataFrame(cm), annot = True, cmap = 'Blues', fmt = 'd')
        plt.xlabel('Predicted')
        plt.ylabel('Expected')
        plt.title(title)
        plotfile = os.path.join('static', str(time.time()) + '.png')
        plt.savefig(plotfile)
        plt.clf()
        confusion_list.append(plotfile)

    """for mod in confusion_dict.keys():
        cm = [[y for y in x] for x in confusion_dict[mod][0]]
        labels = [str(maj_class), str(min_class)]
        title = "Confusion Matrix of {} with F1 score {}".format(mod,confusion_dict[mod][1])
        data = go.Heatmap(z=cm, y=labels, x=labels)
        annotations = []
        for i, row in enumerate(cm):
            for j, value in enumerate(row):
                annotations.append(
                {
                "x": labels[i],
                "y": labels[j],
                "font": {"color": "white"},
                "text": str(value),
                "xref": "x1",
                "yref": "y1",
                "showarrow": False
                }
                )
        layout = {
            "title": title,
            "xaxis": {"title": "Predicted value"},
            "yaxis": {"title": "Expected value"},
            "annotations": annotations
            }
        figCM = go.Figure(data=data, layout=layout)
        graphJSONCM = json.dumps(figCM, cls=plotly.utils.PlotlyJSONEncoder)
        confusion_list.append(graphJSONCM)"""
    
    return graphJSON,confusion_list

def numericalOutlier(df, form):
    feature_list = df.columns
    selected_features = []
    for field, val in form.data.items():
        if field in feature_list:
            if val == True:
                selected_features.append(field)
                    
    df = df[selected_features].copy()
    IF = form.data["IF"]
    knn = form.data["knn"]
    LOF = form.data["LOF"]
    AE = form.data["AE"]
    kmeans = form.data["kmeans"]
    firstOutliers = form.data["firstOutliers"]

    index = df.index
    df_results=df.copy()
    score_list=[]
    if IF:
        algoDetector = IsolationForest(n_estimators=700,n_jobs=-1)
        algoDetector.fit(df)
        df_results["score_IF"] = -algoDetector.score_samples(df) 
        score_list.append("score_IF") 
    
    if knn:
        if df.shape[0]> 100:
            k=100
        else:
            k = df.shape[0]//2
        
        df_results["knn_score"] = knnDetector(df,k)
        score_list.append("knn_score")
    
    if LOF:
        if df.shape[0]> 100:
            k=100
        else:
            k = df.shape[0]//2
        
        # Calculation of the outlier score
        LOF = LocalOutlierFactor(n_neighbors=k)
        LOF.fit(df)
        df_results["LOF_score"] = - LOF.negative_outlier_factor_
        score_list.append("LOF_score")
    
    if AE:
        X_train, X_val = train_test_split(df, test_size=0.10)
        input_data = Input(shape=(df.shape[1],))
        encoded = Dense(df.shape[1] - 1, activation='relu')(input_data)
        encoded = Dense(df.shape[1] - 2, activation='relu')(encoded)
        encoded = Dense(df.shape[1] - 3, activation='relu')(encoded)
        decoded = Dense(df.shape[1] - 2, activation='relu')(encoded)
        decoded = Dense(df.shape[1] - 1, activation='relu')(decoded)
        decoded = Dense(df.shape[1], activation=None)(decoded)
        autoencoder = Model(input_data, decoded)
        encoder = Model(input_data, encoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        callback_AE = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=60,verbose=0),\
                        keras.callbacks.ModelCheckpoint(filepath='AE_test.h5',monitor='val_loss',save_best_only=True,\
                                                  save_weights_only=True,mode='min',verbose=0)]

        history = autoencoder.fit(X_train, X_train,
                    epochs=70,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(X_val, X_val),
                    callbacks=callback_AE,
                    verbose = 0)
        # Calculation of the outlier score
        X_recons = autoencoder.predict(df)
        Error =df- X_recons
        df_results["score_AE"] = ((Error)**2).sum(axis=1)
        score_list.append("score_AE")
        
    if kmeans:  
        def distanceMahalanobis(u,v,df=df):
            covMatrix = pd.DataFrame.cov(df)
            try:
                inVcovMatrix = np.linalg.inv(covMatrix)
            except:
                inVcovMatrix = np.linalg.pinv(covMatrix)
                
            return distance.mahalanobis(u, v, inVcovMatrix)
        
        data = df.values
        clusterResult = df.copy()
        features = df.columns
        score_list_kmeans = []
        for k in range(8,9):
            kclusterer = KMeansClusterer(k, distance=distanceMahalanobis, repeats=7,avoid_empty_clusters=True)
            #kclusterer = KMeansClusterer(k, distance=distanceMahalanobis, repeats=7)
            assigned_clusters = kclusterer.cluster(data, assign_clusters=True)
            clusterResult["clusterLabel"] = assigned_clusters
            clusterResult["score1"] = 0.0 
            clusterResult["score2"] = 0.0
            for row in range(clusterResult.shape[0]):
                label = clusterResult.iloc[row,]["clusterLabel"]
                centroid = kclusterer.means()[int(label)]
                cluster = clusterResult[clusterResult["clusterLabel"]==label][features]
                num = cluster.shape[0]
                clusterResult.loc[clusterResult.index[row],"score2"] = - np.log(num/clusterResult.shape[0])
                u = clusterResult.iloc[row,][features]
                covMatrixCluster = pd.DataFrame.cov(cluster)
                try:
                    inVcovMatrixCluster = np.linalg.inv(covMatrixCluster)
                except:
                    inVcovMatrixCluster = np.linalg.pinv(covMatrixCluster)
                    
                clusterResult.loc[clusterResult.index[row],"score1"] = distance.mahalanobis(u, centroid, inVcovMatrixCluster)
        
            for c in ["score1","score2"]:
                scaler = StandardScaler()
                scaler.fit(clusterResult[c].values.reshape(-1, 1))
                clusterResult[c] = scaler.transform(clusterResult[c].values.reshape(-1, 1))
                del scaler
        
            clusterResult["score_{}".format(k)]=clusterResult[["score1","score2"]].mean(axis=1)
            score_list_kmeans.append("score_{}".format(k))
            del clusterResult["score1"]
            del clusterResult["score2"]
            del clusterResult["clusterLabel"]
            
        df_results["score_kMeans"] = clusterResult[score_list_kmeans].mean(axis=1)
        score_list.append("score_kMeans")

    if not score_list:
        return None
    
    # Normalization of outlier scores
    for c in score_list:
        scaler = StandardScaler()
        scaler.fit(df_results[c].values.reshape(-1, 1))
        df_results[c] = scaler.transform(df_results[c].values.reshape(-1, 1))
        del scaler

    # The final score is equal to the average of all scores
    df_results["Outlier_score"]=df_results[score_list].mean(axis=1)
    result = df_results[["Outlier_score"]].sort_values("Outlier_score",ascending=False).head(firstOutliers)
    result = result.to_html(classes="table table-striped")
    titles=df_results.columns.values
    return result,titles



def mixedDataOutlier(df, form):
    feature_list = df.columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(df.columns)-set(num_cols))
    selected_num_features = []
    selected_cat_features = []
    for field, val in form.data.items():
        if field in feature_list:
            if val == True:
                if field in num_cols:
                    selected_num_features.append(field)
                elif field in cat_cols:
                    selected_cat_features.append(field)
                    
    knn = form.data["knn"]
    LOF = form.data["LOF"]
    pca = form.data["pca"]
    AE = form.data["AE"]
    IF = form.data["IF"]
    numberOfNeighbors = form.data["numberOfNeighbors"]
    numberOfTrees = form.data["numberOfTrees"]
    epochs = form.data["epochs"]
    #kmeans = form.data["kmeans"]
    firstOutliers = form.data["firstOutliers"]
    df = df[selected_num_features + selected_cat_features].copy()
    df_results=df.copy()
    score_list=[]
    if pca or AE:
        df2 = df.copy() 
        if selected_cat_features:
            for c in selected_cat_features:
                ni = df2[c].unique().shape[0] - 1
                dummies = pd.get_dummies(df2[c], drop_first=True,prefix=c)
                for b_c in dummies.columns:
                    fi = dummies[b_c].loc[dummies[b_c]==1].shape[0]/dummies.shape[0]
                    std_dummies = math.sqrt(ni*fi*(1-fi))
                    dummies[b_c] = dummies[b_c]/std_dummies
        
                df2 = pd.concat([df2, dummies], axis=1)
                del df2[c]
            
        for c in selected_num_features:
            scaler = StandardScaler()
            scaler.fit(df2[c].values.reshape(-1, 1))
            df2[c] = scaler.transform(df2[c].values.reshape(-1, 1))
            del scaler
            
        if pca:
            N = df2.shape[1]//2
            pca_mod = PCA(n_components=N)
            pca_mod.fit(df2)
            df2_pca = pca_mod.transform(df2)
            df2_inverse = pca_mod.inverse_transform(df2_pca)
            df_results["PCA_score"] = ((df2 - df2_inverse) ** 2).sum(axis=1)
            score_list.append("PCA_score")
            
        if AE:
            X_train, X_val = train_test_split(df2, test_size=0.10)
            n_features = df2.shape[1]
            input_data = Input(shape=(n_features,))
            encoded = Dense(n_features - 1, activation='relu')(input_data)
            encoded = Dense(n_features - 2, activation='relu')(encoded)
            encoded = Dense(n_features - 3, activation='relu')(encoded)
            decoded = Dense(n_features - 2, activation='relu')(encoded)
            decoded = Dense(n_features - 1, activation='relu')(decoded)
            decoded = Dense(n_features, activation=None)(decoded)
            autoencoder = Model(input_data, decoded)
            encoder = Model(input_data, encoded)
            autoencoder.compile(optimizer='adam', loss='mean_squared_error')

            callback_AE = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=7,verbose=0),\
                                    keras.callbacks.ModelCheckpoint(filepath='AE_test.h5',monitor='val_loss',save_best_only=True,\
                                                              save_weights_only=True,mode='min',verbose=0)]

            history = autoencoder.fit(X_train, X_train,
                        epochs=epochs,
                        batch_size=64,
                        shuffle=True,
                        validation_data=(X_val, X_val),
                        callbacks=callback_AE,
                        verbose = 0)
            X_recons = autoencoder.predict(df2)
            Error =df2- X_recons
            df_results["score_AE"] = ((Error)**2).sum(axis=1)
            score_list.append("score_AE")

    
    
    if knn or LOF or IF:
        #or kmeans
        df3 = df.copy()
        df_num = df3[num_cols].copy()
        if selected_cat_features:
            dummies_cols = []
            for c in cat_cols:
                dummies = pd.get_dummies(df3[c], drop_first=True,prefix=c)
                for b_c in dummies.columns:
                    dummies_cols.append(b_c)
        
                df3 = pd.concat([df3, dummies], axis=1)
                del df3[c]
           
        
            df_cat = df3[dummies_cols].copy()
        
            N_cat = len(cat_cols)
            pca = PCA(n_components=N_cat)
            pca.fit(df_cat)
            df_cat_pca = pca.transform(df_cat)
            df_cat_pca_pd = pd.DataFrame(df_cat_pca, columns = ["C{}".format(i) for i in range(N_cat)])
            dfNew = df_num.join(df_cat_pca_pd)
        else:
            dfNew = df_num.copy()
        
        for c in dfNew.columns:
            scaler = StandardScaler()
            scaler.fit(dfNew[c].values.reshape(-1, 1))
            dfNew[c] = scaler.transform(dfNew[c].values.reshape(-1, 1))
            del scaler
            
        if knn:
            if dfNew.shape[0]> numberOfNeighbors:
                k=numberOfNeighbors
            else:
                k = dfNew.shape[0]//2
            
            X = dfNew.copy()
            Y = dfNew.index
            knn_mod = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
            knn_mod.fit(X, Y)
            distances, _ = knn_mod.kneighbors()
            df_results["knn_score"] = 0.0
            for i in range(df_results.shape[0]):
                df_results.loc[df_results.index[i],"knn_score"] = distances[i].mean()
            
            score_list.append("knn_score")
            
        if LOF:
            if dfNew.shape[0]> numberOfNeighbors:
                k=numberOfNeighbors
            else:
                k = dfNew.shape[0]//2
        
            LOF = LocalOutlierFactor(n_neighbors=k)
            LOF.fit(dfNew)
            df_results["LOF_score"] = - LOF.negative_outlier_factor_
            score_list.append("LOF_score")
            
        if IF:
            algoDetector = IsolationForest(n_estimators=numberOfTrees,n_jobs=-1)
            algoDetector.fit(dfNew)
            df_results["score_IF"] = -algoDetector.score_samples(dfNew) 
            score_list.append("score_IF")
            
            
        """if kmeans:  
            def distanceMahalanobis(u,v,df=dfNew):
                covMatrix = pd.DataFrame.cov(df)
                #inVcovMatrix = np.linalg.inv(covMatrix)
                inVcovMatrix = np.linalg.pinv(covMatrix)
                return distance.mahalanobis(u, v, inVcovMatrix)
        
            data = dfNew.values
            clusterResult = dfNew.copy()
            features = dfNew.columns
            score_list_kmeans = []
            for k in range(7,9):
                kclusterer = KMeansClusterer(k, distance=distanceMahalanobis, repeats=7,avoid_empty_clusters=True)
                assigned_clusters = kclusterer.cluster(data, assign_clusters=True)
                clusterResult["clusterLabel"] = assigned_clusters
                clusterResult["score1"] = 0.0 
                clusterResult["score2"] = 0.0
                for row in range(clusterResult.shape[0]):
                    label = clusterResult.iloc[row,]["clusterLabel"]
                    centroid = kclusterer.means()[int(label)]
                    cluster = clusterResult[clusterResult["clusterLabel"]==label][features]
                    num = cluster.shape[0]
                    clusterResult.loc[clusterResult.index[row],"score2"] = - np.log(num/clusterResult.shape[0])
                    u = clusterResult.iloc[row,][features]
                    covMatrixCluster = pd.DataFrame.cov(cluster)
                    inVcovMatrixCluster = np.linalg.inv(covMatrixCluster)
                    clusterResult.loc[clusterResult.index[row],"score1"] = distance.mahalanobis(u, centroid, inVcovMatrixCluster)
        
                for c in ["score1","score2"]:
                    scaler = StandardScaler()
                    scaler.fit(clusterResult[c].values.reshape(-1, 1))
                    clusterResult[c] = scaler.transform(clusterResult[c].values.reshape(-1, 1))
                    del scaler
        
                clusterResult["score_{}".format(k)]=clusterResult[["score1","score2"]].mean(axis=1)
                score_list_kmeans.append("score_{}".format(k))
                del clusterResult["score1"]
                del clusterResult["score2"]
                del clusterResult["clusterLabel"]
            
            df_results["score_kMeans"] = clusterResult[score_list_kmeans].mean(axis=1)
            score_list.append("score_kMeans")"""
            
    if not score_list:
        return None
    
    for c in score_list:
        scaler = StandardScaler()
        scaler.fit(df_results[c].values.reshape(-1, 1))
        df_results[c] = scaler.transform(df_results[c].values.reshape(-1, 1))
        del scaler

    # The final score is equal to the average of all scores
    df_results["Outlier_score"]=df_results[score_list].mean(axis=1)
    result = df_results[["Outlier_score"]].sort_values("Outlier_score",ascending=False).head(firstOutliers)
    result = result.to_html(classes="table table-striped")
    titles=df_results.columns.values
    return result,titles
