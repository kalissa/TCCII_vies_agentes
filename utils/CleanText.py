import os
import re
import string
from pathlib import Path
from unicodedata import normalize

import preprocessor as preprocessor
from nltk import word_tokenize


def pre_process(row):
    text = row['frase']
    text = replace_links(text)
    text = clean_text(text)
    text = remove_punctuation(text)
    text = remove_digits(text)
    text = to_lower_case(text)
    text = ajusta_xingamentos(text)
    text = standardized_tokens(text)
    text = remove_stop_words(text)
    text = remove_word_with_just_one_letter(text)
    return text


def get_stop_words():
    bots_related_stop_words = open(os.path.join(Path().absolute().parent) + "/utils/bots_related_stop_words.txt", "r").read().split(
        "\n")
    twitter_slangs = open(os.path.join(Path().absolute().parent) + "/utils/twitter_slangs.txt", "r").read().split("\n")
    retails_stops_words = open(os.path.join(Path().absolute().parent) + "/utils/retails_stops_words.txt", "r").read().split("\n")
    stopwords_br = open(os.path.join(Path().absolute().parent) + "/utils/stopwords_br.txt", "r").read().split("\n")

    return bots_related_stop_words + twitter_slangs + retails_stops_words + stopwords_br


def replace_links(text):
    text = re.sub(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))*',
        '',
        text,
        flags=re.MULTILINE)
    text = text.replace('link', '')
    return text


def standardized_tokens(text):
    text_tokens = tokenize(text)
    dic_abbreviation = get_abbreviation_dictionary()

    if len(text_tokens) == 1:
        if text_tokens[0] in dic_abbreviation:
            return dic_abbreviation[text_tokens[0]]
        else:
            return text_tokens[0]
    else:
        standardized_tokens = list()
        for token in text_tokens:
            if (token.lower() in dic_abbreviation):
                standardized_tokens.append(dic_abbreviation[token.lower()])
            else:
                standardized_tokens.append(token)
        return (" ").join(standardized_tokens)


def get_abbreviation_dictionary():
    return {
        'dps': 'depois',
        'd': 'de',
        'aqles': 'aqueles',
        'kd': 'cade',
        'ctz': 'certeza',
        'dnv': 'denovo',
        'tava': 'estava',
        'hj': 'hoje',
        'mt': 'muito',
        'mto': 'mto',
        'mlhr': 'melhor',
        'noiz': 'nos',
        'noz': 'nos',
        'els': 'eles',
        'eh': 'e',
        'vc': 'voce',
        'vcs': 'voces',
        'tb': 'tambem',
        'tambm': 'tambem',
        'tbm': 'tambem',
        'obg': 'obrigado',
        'dms': 'demais',
        'gnt': 'gente',
        'q': 'que',
        'cmg': 'comigo',
        'p': 'para',
        'ta': 'esta',
        'to': 'estou',
        'vdd': 'verdade',
        'rs': 'risos',
        'sfd': 'safado',
        'rd': 'rodado',
        'pdp': 'claro',
        'dmr': 'demorou',
        'tamo': 'estamos',
        'ent': 'entao',
        'mlk': 'moleque',
        'nd': 'nada',
        'naum': 'nao',
        'nn': 'nao',
        'ñ': 'nao',
        'n': 'nao',
        'lol': 'risos',
        'tão': 'estao',
        'msm': 'mesmo',
        'pra': 'para',
        'ngm': 'ninguem',
        'bj': 'beijo',
        'bjo': 'beijo',
        'bjs': 'beijos',
        'smp': 'sempre',
        'agr': 'agora',
        'amg': 'amigo',
        'bb': 'bebe',
        'blz': 'beleza',
        'sdds': 'saudades',
        'sdd': 'saudade',
        'pq': 'porque',
        'qm': 'quem',
        'td': 'tudo',
        'fdase': 'se-foder',
        'vsf': 'se-foder',
        'vtnc': 'tomar-no-cu',
        'pqp': 'puta-que-pariu',
        'putaquepario': 'puta-que-pariu',
        'pnc': 'pau-no-cu',
        'krl': 'caralho',
        'crlh': 'caralho',
        'karalho': 'caralho',
        'caraio': 'caraio',
        'fdp': 'filho-da-puta',
        'nude': 'nudes',
        'ndess': 'nudes',
        'nudess': 'nudes',
        'nudesz': 'nudes',
        'nuds': 'nudes',
        'poha': 'porra',
        'msg': 'mensagem',
        'vlw': 'valeu',
        'vagbunda': 'vagabunda',
        'urg': 'urgente',
        'tambeim': 'tambem',
        'rsrsrsrsr': 'risos',
        'rsrsrs': 'risos',
        'rsrs': 'risos',
        'putaquepariu': 'puta-que-pariu',
        'pika': 'pica'
    }


def clean_text(text):
    preprocessor.set_options(preprocessor.OPT.HASHTAG, preprocessor.OPT.SMILEY, preprocessor.OPT.MENTION)
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = preprocessor.clean(text, )
    return text


def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def remove_digits(text):
    text = text.translate(str.maketrans('', '', string.digits))
    return text


def to_lower_case(text):
    text = text.lower()
    return text


def ajusta_xingamentos(text):
    if 'tomar no cu' in text:
        text = text.replace('tomar no cu', 'tomar-no-cu')
    if 'toma no cu' in text:
        text = text.replace('toma no cu', 'tomar-no-cu')
    if 'toma no seu cu' in text:
        text = text.replace('toma no seu cu', 'tomar-no-cu')
    if 'toma no teu cu' in text:
        text = text.replace('toma no teu cu', 'tomar-no-cu')
    if 'se foder' in text:
        text = text.replace('se foder', 'se-foder')
    if 'se fuder' in text:
        text = text.replace('se fuder', 'se-foder')
    if 'se-foderd' in text:
        text = text.replace('se-foderd', 'se-foder')
    if 'puta que pariu' in text:
        text = text.replace('puta que pariu', 'puta-que-pariu')
    if 'pau no cu' in text:
        text = text.replace('pau no cu', 'pau-no-cu')
    if 'pau no seu cu' in text:
        text = text.replace('pau no seu cu', 'pau-no-cu')
    if 'filho da puta' in text:
        text = text.replace('filho da puta', 'filho-da-puta')
    if 'filha da puta' in text:
        text = text.replace('filha da puta', 'filha-da-puta')
    return text


def remove_stop_words(text):
    stopwords = get_stop_words()
    text_tokens = tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords]
    text = (" ").join(tokens_without_sw)
    return text


def remove_word_with_just_one_letter(text):
    text_tokens = tokenize(text)
    text_bigger_than_len_1 = [word for word in text_tokens if len(word) > 1]
    text = (" ").join(text_bigger_than_len_1)
    return text


def tokenize(text):
    return word_tokenize(text, language='portuguese')
