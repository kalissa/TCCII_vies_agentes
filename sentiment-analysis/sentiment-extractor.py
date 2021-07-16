import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from detoxify import Detoxify

from utils.CleanText import pre_process


def remove_duplicates(df):
    return df.drop_duplicates(subset='clean_text', keep="last")


def get_cognitive_sentiment(text):
    payload = {
        'documents': [
            {
                'language': 'pt-br',
                'id': 1,
                'text': text
            }
        ]
    }

    headers = {'Content-type': 'application/json', 'Ocp-Apim-Subscription-Key': 'KEY'}
    time.sleep(1.5)
    r = session.post(
        'https://teste-nlp.cognitiveservices.azure.com/text/analytics/v3.0/sentiment',
        json=payload,
        headers=headers)
    print(r)
    return r.json()['documents'][0]


def get_toxity(toxity):
    if toxity >= 0.7426:
        return "xingamento"
    else:
        return "neutro"


def has_sexual_splicity(text):
    list_of_sexual_words = ["rola", "pau", "cu", "peitos", "anus", "vtnc", "pelada", "giromba", "pauzao", "pelada",
                            "caralho", "bundinha", "vagabunda", "mamando", "buceta", "caralhuda", "peladinho", "bunda",
                            "tmnc", "piroca", "pirocao", "chupando", "fdp", "cuzinho", "foda-se", "pirocuda"",sugada",
                            "nudes", "pika", "chupa", "peituda", "gostosa", "gostosinha", "xoxota", "cheirosa",
                            "cherosa", "69", "bucetao", "trolhao", "gostoso", "pauzuda", "mamadinha", "mama", "nude",
                            "safada", "gostosona"]

    contains_word = []
    for word in text.split(" "):
        if word in list_of_sexual_words:
            contains_word.append("1")
        else:
            contains_word.append("0")

    if "1" in contains_word:
        return "1"
    else:
        return "0"


def get_google_sentiment(text):
    payload = {
        'comment': {
            'text': text
        },
        'languages': [
            'pt'
        ],
        'requestedAttributes': {
            'TOXICITY': {},
            'SEVERE_TOXICITY': {},
            'IDENTITY_ATTACK': {},
            'INSULT': {},
            'PROFANITY': {},
            'THREAT': {}
        }
    }
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    time.sleep(1.5)
    r = session.post(
        'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key=GOOGLE_KEY',
        json=payload,
        headers=headers)
    if r and r.status_code == 200:
        return r.json()['attributeScores']


train_df = pd.read_csv(os.path.join(Path().absolute(), "data", 'clean-data-base-final.csv'))

train_df['data'] = pd.to_datetime(train_df['data_hora_tweet']).dt.date

train_df = train_df.replace(np.nan, '', regex=True)

train_df['clean_text'] = train_df.apply(pre_process, axis=1)

train_df = remove_duplicates(train_df)

sample_data = train_df.sort_values('data')

frases = sample_data['clean_text'].tolist()

session = requests.Session()
informations = []
infos = {}

print('Quantidade de tweets a serem analisados: ' + '\n' + str(sample_data.shape[0]))

f = open(os.path.join(Path().absolute()) + 'Base Analisada.csv', 'w')

csv_writer = csv.DictWriter(f,
                            fieldnames=(
                                'termo',
                                'frase',
                                'nosso_sentimento',
                                'google_TOXICITY',
                                'google_SEVERE_TOXICITY',
                                'google_IDENTITY_ATTACK',
                                'google_INSULT',
                                'google_PROFANITY',
                                'google_THREAT',
                                'SEXUALLY_EXPLICIT',
                                'microsoft_coginitive_sentiment',
                                'microsoft_coginitive_sentiment_positive_score',
                                'microsoft_coginitive_sentiment_neutral_score',
                                'microsoft_coginitive_sentiment_negative_score',
                                'tweet_origin_id',
                                'tweet_id',
                                'data_hora_tweet',
                                'data'
                            ),
                            dialect='excel')
csv_writer.writeheader()

try:
    for i in range(len(frases)):
        results = Detoxify('multilingual').predict(frases)
        frase = frases[i]
        if frase:
            print("Frase {}: {}".format(i + 1, frase))
            print(
                "Original Frase {}: {}".format(i + 1, train_df.loc[train_df['clean_text'] == frase]['text'].values[0]))
            googleSentiment = get_google_sentiment(frase)
            our_toxity = get_toxity(round(googleSentiment['TOXICITY']['summaryScore']['value'], 4))
            cognitiveSentiment = get_cognitive_sentiment(frase)

            if has_sexual_splicity(frase) == "1" and our_toxity == "neutro":
                our_toxity = "xingamento"

            if googleSentiment:
                infos = {
                    "termo": train_df.loc[train_df['clean_text'] == frase]['termo'].values[0],
                    "frase": train_df.loc[train_df['clean_text'] == frase]['text'].values[0],
                    "detoxify_toxicity": round(results['toxicity'][i], 4),
                    "nosso_sentimento": our_toxity,
                    "google_TOXICITY": round(googleSentiment['TOXICITY']['summaryScore']['value'], 4),
                    "google_THREAT": round(googleSentiment['THREAT']['summaryScore']['value'], 4),
                    "google_SEVERE_TOXICITY": round(googleSentiment['SEVERE_TOXICITY']['summaryScore']['value'], 4),
                    "google_IDENTITY_ATTACK": round(googleSentiment['IDENTITY_ATTACK']['summaryScore']['value'], 4),
                    "google_INSULT": round(googleSentiment['INSULT']['summaryScore']['value'], 4),
                    "google_PROFANITY": round(googleSentiment['PROFANITY']['summaryScore']['value'], 4),
                    "SEXUALLY_EXPLICIT": has_sexual_splicity(frase),
                    "microsoft_coginitive_sentiment": cognitiveSentiment['sentiment'],
                    "microsoft_coginitive_sentiment_positive_score": cognitiveSentiment['confidenceScores']['positive'],
                    "microsoft_coginitive_sentiment_neutral_score": cognitiveSentiment['confidenceScores']['neutral'],
                    "microsoft_coginitive_sentiment_negative_score": cognitiveSentiment['confidenceScores']['negative'],
                    'tweet_origin_id': train_df.loc[train_df['clean_text'] == frase]['tweet_origin_id'].values[0],
                    'tweet_id': train_df.loc[train_df['clean_text'] == frase]['tweet_id'].values[0],
                    'data_hora_tweet': train_df.loc[train_df['clean_text'] == frase]['data_hora_tweet'].values[0],
                    'data': train_df.loc[train_df['clean_text'] == frase]['data'].values[0]
                }
                row = {
                    'termo': infos['termo'],
                    'frase': infos['frase'],
                    'detoxify_toxicity': infos['detoxify_toxicity'],
                    'google_TOXICITY': infos['google_TOXICITY'],
                    'nosso_sentimento': infos['nosso_sentimento'],
                    'google_THREAT': infos['google_THREAT'],
                    'google_SEVERE_TOXICITY': infos['google_SEVERE_TOXICITY'],
                    'google_IDENTITY_ATTACK': infos['google_IDENTITY_ATTACK'],
                    'google_INSULT': infos['google_INSULT'],
                    'google_PROFANITY': infos['google_PROFANITY'],
                    'SEXUALLY_EXPLICIT': infos['SEXUALLY_EXPLICIT'],
                    'microsoft_coginitive_sentiment': infos['microsoft_coginitive_sentiment'],
                    'microsoft_coginitive_sentiment_positive_score': infos[
                        'microsoft_coginitive_sentiment_positive_score'],
                    'microsoft_coginitive_sentiment_neutral_score': infos[
                        'microsoft_coginitive_sentiment_neutral_score'],
                    'microsoft_coginitive_sentiment_negative_score': infos[
                        'microsoft_coginitive_sentiment_negative_score'],
                    'tweet_origin_id': infos['tweet_origin_id'],
                    'tweet_id': infos['tweet_id'],
                    'data_hora_tweet': infos['data_hora_tweet'],
                    'data': infos['data']
                }
                print("Escrevendo no arquivo")
                csv_writer.writerow(row)
        else:
            print("Tweet vazio")
    f.close()

except:
    print("Ocorreu um erro")
    f.close()
