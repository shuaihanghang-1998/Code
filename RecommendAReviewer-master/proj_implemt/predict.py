import pandas as pd
import numpy as np 

from sklearn.manifold import TSNE
from gensim.models.ldamodel import LdaModel
# from gensim.utils.SaveLoad import save
# NLTK
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re

df = pd.read_csv("predictionInput.csv")
# Just the csv, similar to that given in training, but only with a single paper
# Removing numerals:
df['paper_text_tokens'] = df.paper_text.map(lambda x: re.sub(r'\d+', '', x))
# Lower case:
df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: x.lower())
df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: RegexpTokenizer(r'\w+').tokenize(x))

snowball = SnowballStemmer("english")
df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: [snowball.stem(token) for token in x])
stop_en = stopwords.words('english')
df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: [t for t in x if t not in stop_en])
df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: [t for t in x if len(t) > 1])
#df.to_hdf('processed.hdf','mydata',mode='w') 

#df = pd.read_hdf("processed.hdf")
from gensim import corpora, models
ldamodel =  LdaModel.load('ldaModel.model')
texts = df['paper_text_tokens'].values
dictionary = corpora.Dictionary.load('dictionary')
np.random.seed(2017)
doc_bow = dictionary.doc2bow(texts[0]) 
# model.show_topics()

# print topic 28
# model.print_topic(109, topn=20)
print "here !!!!!!!!!!!!!!!!!!!!!"

# print dictionary.items()
print ldamodel[doc_bow]


# another way
for i in range(1, ldamodel.num_topics):
    print ldamodel.print_topic(i)

# and another way, only prints top words
# for t in range(0, model.num_topics-1):
#     print 'topic {}: '.format(t) + ', '.join([v[1] for v in model.show_topic(t, 20)])


