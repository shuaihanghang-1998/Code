import pandas as pd
import numpy as np

# LDA, tSNE
from sklearn.manifold import TSNE
from gensim.models.ldamodel import LdaModel
# from gensim.utils.SaveLoad import save
# NLTK
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
# Visualization
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib
# import seaborn as sns
# # Bokeh
# from bokeh.io import output_notebook
# from bokeh.plotting import figure, show
# from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider
# from bokeh.layouts import column
# from bokeh.palettes import all_palettes
# from pathlib import Path

#if Path("processed.hdf"):
df = pd.read_csv("papers.csv")
# print(df.paper_text[0][:500])
# Removing numerals:
df['paper_text_tokens'] = df.paper_text.map(lambda x: re.sub(r'\d+', '', x))
# Lower case:
df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: x.lower())
# print(df['paper_text_tokens'][0][:500])
df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: RegexpTokenizer(r'\w+').tokenize(x))
# print(df['paper_text_tokens'][0][:25])

snowball = SnowballStemmer("english")
df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: [snowball.stem(token) for token in x])
# print(df['paper_text_tokens'][0][:25])
stop_en = stopwords.words('english')

STOPWORDS = """
a about above across after afterwards again against all almost alone along already also although always am among amongst amoungst amount an and another any anyhow anyone anything anyway anywhere are around as at back be
became because become becomes becoming been before beforehand behind being below beside besides between beyond bill both bottom but by call can
cannot cant co computer con could couldnt cry de describe
detail did do doc doesn done down due during
each eg eight either eleven else elsewhere empty enough etc even ever every everyone everything everywhere except few fifteen
fify fill find fire first five for former formerly forty found four from front full further get give go
had has hasnt have he hence her here hereafter hereby herein hereupon hers herself him himself his how however http hundred i ie
if in inc indeed interest into is it its itself keep last latter latterly least less ltd
just
kg km
ll
made many may me meanwhile might mill mine more moreover most mostly move much must my myself name namely
neither never nevertheless next nine no nobody none noone nor not nothing now nowhere nt of off
often on once one only onto or org other others otherwise our ours ourselves out over own part per
perhaps please pdf put rather re
quite
rather really regarding
same see seem seemed seeming seems serious several she should show side since sincere six sixty so some somehow someone something sometime sometimes somewhere still such system take ten
than that the their them themselves then thence there thereafter thereby therefore therein thereupon these they thick thin third this those though three through throughout thru thus to together too top toward towards twelve twenty two un under
until up unless upon us used using
various very very via
was we well were what whatever when whence whenever where whereafter whereas whereby wherein whereupon wherever whether which while whither who whoever whole whom whose why will with within without would www
xls
yet you
your yours yourself yourselves
i ii iii iv v vi vii viii ix x xi xii xiii xiv xv xvi xvii xviii xix xx xxi xxii xxiii xxiv xxv xxvi xxvii xxviii xxix xxx
"""
STOPWORDS = frozenset(w.encode('utf8') for w in STOPWORDS.split() if w)

for word in STOPWORDS:
	stop_en.append(unicode(snowball.stem(word))) 

extraWords = ['figure', 'sample', 'paper', 'fig', 'conv']
for word in extraWords:
	stop_en.append(unicode(snowball.stem(word))) 

# print stop_en

df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: [t for t in x if t not in stop_en])
# print(df['paper_text_tokens'][0][:25])
df['paper_text_tokens'] = df.paper_text_tokens.map(lambda x: [t for t in x if len(t) > 1])
# print(df['paper_text_tokens'][0][:25])
#df.to_hdf('processed.hdf','mydata',mode='w') 

#df = pd.read_hdf("processed.hdf")
from gensim import corpora, models
np.random.seed(2017)
texts = df['paper_text_tokens'].values
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = models.ldamodel.LdaModel(corpus, id2word=dictionary,
                                    num_topics=127, passes=5, minimum_probability=0)

dictionary.save('dictionary')
ldamodel.save('ldaModel.model')


