import pandas as pd
import numpy as np 
import sanitize
from sklearn.manifold import TSNE
from gensim.models.ldamodel import LdaModel
# from gensim.utils.SaveLoad import save
# NLTK
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
#import io
#with io.open("nips_reviewer_data/abstracts.txt", "r", encoding="utf-8") as my_file:
#     f = my_file.readlines() 
#f = open('/home/harshv834/Recommend-a-Reviewer/proj_implemt/nips_reviewer_data/abstracts.txt','r').readlines()
#for item in f:
#	item = str(item.replace('/n',' '))
#df = pd.DataFrame(f,columns=['paper_text'])
df= pd.read_csv('nips_reviewer_data/abstracts.csv')

# Just the csv, similar to that given in training, but only with a single paper
# Removing numerals:

for i in range(len(df['paper_text'])):
	df['paper_text'][i] = ''.join(i for i in df['paper_text'][i] if ord(i)<128)
df['paper_text_tokens'] = df.paper_text.map(lambda x: re.sub(r'\d+', '', x))
# Lower case:
df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: x.lower())
df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: RegexpTokenizer(r'\w+').tokenize(x))



snowball = SnowballStemmer("english")
#for i in df['paper_text_tokens']:
#	for item in i:
#		item  = ''.join(j for j in item if ord(j)<128)
#		for j in item:
#			if ord(j)>128:
#				print j

#df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: [''.join(i) for i in token for token in x if ord(i)<128])
df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: [snowball.stem((token).encode("utf-8")) for token in x])
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
lda_topics = []
for i in range(len(texts)):
	doc_bow = dictionary.doc2bow(texts[i])
	lda_topics.append(ldamodel[doc_bow])
arr = np.array(lda_topics).astype(float)
print arr.shape
arr = arr[:,:,1]
np.savetxt('paper_vec.txt',arr) 
# model.show_topics()

# print topic 28
# model.print_topic(109, topn=20)
print "here !!!!!!!!!!!!!!!!!!!!!"

# print dictionary.items()
#print ldamodel[doc_bow]


# another way
#for i in range(1, ldamodel.num_topics):
#    print ldamodel.print_topic(i)

# and another way, only prints top words
# for t in range(0, model.num_topics-1):
#     print 'topic {}: '.format(t) + ', '.join([v[1] for v in model.show_topic(t, 20)])


