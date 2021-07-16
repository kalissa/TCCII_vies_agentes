import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from utils.CleanText import pre_process


def count_vec(max_features):
    if max_features:
        return CountVectorizer(max_features=max_features, token_pattern=r'\S+')
    else:
        return CountVectorizer(token_pattern=r'\S+')


def createKNNModel(n_neighbors, x, y):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x, y)
    return knn


def hypertuning(x, y, n_neighbors=50, cv=5):
    knn = KNeighborsClassifier()
    param_grid = [
        {
            'n_neighbors': np.arange(1, n_neighbors)
        }
    ]
    knn_gscv = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy')
    knn_gscv.fit(x, y)

    return knn_gscv.best_params_, knn_gscv.best_score_


def remove_duplicates(df, collumn):
    return df.drop_duplicates(subset=collumn, keep="last")


def build_knn(frases, labels, max_features, k_number):
    count_vectorizer = count_vec(max_features)

    features = count_vectorizer.fit_transform(frases.values)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    best_params = hypertuning(X_train, y_train)

    if k_number:
        print(f'tem k = {k_number}')
    else:
        k_number = best_params[0]['n_neighbors']

    knn = createKNNModel(
        k_number,
        X_train,
        y_train)

    # Calculate the accuracy of the model
    y_knn_predict = knn.predict(X_test)

    print("KNN Accuracy With MaxFeatures : " + str(accuracy_score(y_test, y_knn_predict)) + " " + str(max_features))
    print("KNN f1 With MaxFeatures : " + str(f1_score(y_test, y_knn_predict)) + " " + str(max_features))
    print(
        "KNN precision With MaxFeatures : " + str(precision_score(y_test, y_knn_predict)) + " " + str(max_features))
    print("KNN AUROC: With MaxFeatures", str(roc_auc_score(y_test, y_knn_predict)) + " " + str(max_features))
    print("KNN recall With MaxFeatures : " + str(recall_score(y_test, y_knn_predict)) + " " + str(max_features))
    print("\n")

    return knn, count_vectorizer


df = pd.read_csv(os.path.join(Path().absolute(), "../data/Base de Treino KNN.csv"))

### Pre Process
df = remove_duplicates(df, 'frase')
df['clean_text'] = df.apply(pre_process, axis=1)
df['clean_text'].replace("", float("NaN"), inplace=True)
df.dropna(subset=["clean_text"], inplace=True)
frases = df['clean_text']

df_tweets_com_xingamentos = df.loc[df['xingamento'] == 1]
sexual_explicit = df_tweets_com_xingamentos['SEXUALLY_EXPLICIT'].dropna()

print(f'Total de Xingamentos: {df_tweets_com_xingamentos.shape[0]}')
print(f'Total de Sexual: {sexual_explicit.loc[lambda x: x == 1].shape[0]}')

tweets_df = pd.read_csv(os.path.join(Path().absolute(), "../data/termos-teste-classificadores.csv"))
tweets_df = remove_duplicates(tweets_df, 'frase')
tweets_df = tweets_df.reset_index(drop=True)
tweets_df['clean_text'] = tweets_df.apply(pre_process, axis=1)
tweets_df['clean_text'].replace("", float("NaN"), inplace=True)
tweets_df.dropna(subset=["clean_text"], inplace=True)
tweets = tweets_df['clean_text']

knn_for_xingamento, tdidf_vectorizer_for_xingamento = build_knn(frases, df['xingamento'], 96, 3)

knn_for_sexual_explicit, tdidf_vectorizer_for_sexual_explicit = build_knn(df_tweets_com_xingamentos['clean_text'],
                                                                          sexual_explicit, 48, 3)

vector_tweets = tdidf_vectorizer_for_xingamento.transform(tweets.values)

# Predict Output for xingamentos
xingamento_predictions = knn_for_xingamento.predict(vector_tweets)

f = open(os.path.join(Path().absolute(), 'Analise KNN.csv'), 'w')

csv_writer = csv.DictWriter(f, fieldnames=('frase', 'frase limpa', 'xingamento', 'sexual'), dialect='excel')
csv_writer.writeheader()

xingamentos = []

for prediction, tweet in zip(xingamento_predictions, tweets):
    original_tweet = tweets_df.loc[tweets_df['clean_text'] == tweet]['frase'].values[0],
    infos = {
        "frase": original_tweet[0],
        "clean_text": tweet,
        "xingamento": prediction
    }
    xingamentos.append(infos)


def extract_sexual_explicity(frases, sexual_explicit, xingamentos, tweets):
    knn_for_sexual_explicit, tdidf_vectorizer_for_sexual_explicit = build_knn(frases, sexual_explicit, 48, 3)
    sexual_explicit_vector = tdidf_vectorizer_for_sexual_explicit.transform(tweets)
    predictions_for_sexual_explicit = knn_for_sexual_explicit.predict(sexual_explicit_vector)
    for prediction, tweet in zip(predictions_for_sexual_explicit, tweets):
        next(item for item in xingamentos if item["clean_text"] == tweet).update({'SEXUALLY_EXPLICIT': prediction})
    return xingamentos


tweets_com_xingamentos = []

for tweet in tweets:
    tweet_com_xingamento = [item for item in xingamentos if item["clean_text"] == tweet and item["xingamento"] == 1]
    if tweet_com_xingamento:
        tweets_com_xingamentos.append(tweet)

xingamentos = extract_sexual_explicity(df_tweets_com_xingamentos['clean_text'], sexual_explicit, xingamentos,
                                       tweets_com_xingamentos)

for xingamento in xingamentos:
    print(f'frase: {xingamento["frase"]} '   f'xingamento: {xingamento["xingamento"]}')
    if xingamento['xingamento'] == 0:
        row = {
            'frase': xingamento['frase'],
            'frase limpa': xingamento['clean_text'],
            'xingamento': 0,
            'sexual': 0
        }
        csv_writer.writerow(row)
    else:
        row = {
            'frase': xingamento['frase'],
            'frase limpa': xingamento['clean_text'],
            'xingamento': xingamento['xingamento'],
            'sexual': int(xingamento['SEXUALLY_EXPLICIT']) if 'SEXUALLY_EXPLICIT' in xingamento else 0
        }
        csv_writer.writerow(row)
