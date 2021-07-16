import os
from pathlib import Path

import multidict as multidict
import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud

from utils.CleanText import pre_process

df = pd.read_csv(os.path.join(Path().absolute(), "knn", 'Base de Treino KNN.csv'))

frases_antes_da_limpeza = df['frase']

df['frase'] = df.apply(pre_process, axis=1)

frases = df['frase']


def getFrequencyDictForText(sentences):
    fullTermsDict = multidict.MultiDict()
    tmpDict = {}

    for text in sentences:
        val = tmpDict.get(text, 0)
        tmpDict[text.lower()] = val + 1
    for key in tmpDict:
        fullTermsDict.add(key, tmpDict[key])

    dict([(k, v) for k, v in fullTermsDict.items() if len(k) > 0])
    sorted(fullTermsDict.items(), key=lambda x: x[1], reverse=True)

    return fullTermsDict


def makeImage(text, file_name):
    wordcloud = WordCloud(width=3000,
                          height=2000,
                          random_state=1,
                          background_color='black',
                          max_words=500,
                          colormap='Set2',
                          collocations=False)
    # generate word cloud
    wordcloud.generate_from_frequencies(text)

    # show
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    wordcloud.to_file(os.path.join(Path().absolute() + file_name + ".png"))


makeImage(getFrequencyDictForText(frases), 'depois-pre-processamento')
makeImage(getFrequencyDictForText(frases_antes_da_limpeza), 'antes-pre-processamento')
