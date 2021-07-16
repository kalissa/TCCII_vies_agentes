import csv
import os
import time
import webbrowser
from pathlib import Path

import tweepy
import yaml

with open(os.path.join(Path().absolute().parent) + "/config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)

consumer_key = cfg["twitter"]['consumer_key']
consumer_secret = cfg["twitter"]["consumer_secret"]

callback_uri = 'obb'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

redirect_url = auth.get_authorization_url()

print(redirect_url)

webbrowser.open(redirect_url)

user_pin_input = input("Qual o valor do PIN ?")

auth.get_access_token(user_pin_input)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, timeout=300)


def bots_search(file_name):
    termos_de_busca = [
        "magazineluiza", "magalu", "lu da magalu", "mascote magazine luiza", "bot magalu", "robo da magalu",
        "CasasBahia", "casas bahia", "baianinho", "cb", "mascote das casas bahia", "baianinho das casas bahia"
    ]
    informations = []
    for termo in termos_de_busca:
        print(termo)
        try:
            for tweet in tweepy.Cursor(api.search,
                                       q=termo,
                                       result_type='recent',
                                       timeout=999999,
                                       tweet_mode='extended',
                                       since="2021-06-04",
                                       until="2021-06-11",
                                       lang='pt').items(500000):

                if hasattr(tweet, 'in_reply_to_status_id_str'):
                    if tweet.user.name not in ['Lu do Magalu (em 🏡)', 'Casas Bahia']:
                        print("Buscando: " + str(termo))
                        try:
                            infos = {
                                "termo": termo,
                                "user": tweet.user.screen_name,
                                "text": tweet.full_text.replace('\n', ' '),
                                "tweet_origin_id": tweet.in_reply_to_status_id_str,
                                "tweet_id": tweet.id,
                                "data_do_tweet": tweet.created_at
                            }
                            informations.append(infos)
                        except tweepy.TweepError as e:
                            print("Problema ao buscar tweets" + str(e))
                            time.sleep(60)
                            continue
                        except StopIteration:
                            break
                else:
                    print("Nao tem in_reply_to_status_id_str: " + str(tweet))
        except tweepy.TweepError as e:
            print("Problema ao buscar tweets " + str(e))
            continue
        except StopIteration:
            break
    with open(os.path.join(Path().absolute() + file_name), 'w') as f:
        csv_writer = csv.DictWriter(f, fieldnames=(
            'termo', 'user', 'text', 'tweet_origin_id', 'tweet_id', 'data_do_tweet'))
        csv_writer.writeheader()
        for information in informations:
            print(information['termo'])
            row = {'termo': information['termo'],
                   'user': information['user'],
                   'text': information['text'],
                   'tweet_origin_id': "\' " + str(information['tweet_origin_id']) + " \'",
                   'tweet_id': "\' " + str(information['tweet_id']) + " \'",
                   'data_do_tweet': "\' " + str(information['data_do_tweet']) + " \'",
                   }
            print("Escrevendo no arquivo")
            csv_writer.writerow(row)


file_name = 'bot-termos-04-06-to-11-06.csv'

bots_search(file_name)
