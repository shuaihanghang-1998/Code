import nltk

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

with open('trial2.txt', 'r') as myfile:
    data2=myfile.read().replace('\n', ' ')

sp=set(stopwords.words("english"))
variable = nltk.word_tokenize(data2)
sp2 = [stemmer.stem(w) for w in sp]
variable2 = [stemmer.stem(w) for w in variable]
filtered_sentence = [w for w in variable2 if not w in sp2]
a=" ".join(filtered_sentence)
sent = nltk.sent_tokenize(a)
b="\n".join(sent)
open('naya.txt','w').write(b)
