import pandas as pd
import numpy as np
import nltk
from string import digits


data = pd.read_csv('nips_reviewer_data/training.csv')
rev = open('nips_reviewer_data/reviewers.txt','r').readlines()
paper_auth = data['author'].tolist()
rev_list = []
for item in rev:
     rev_list.append(item.replace('\t','').replace('\n','').translate(None,digits))
indices = []
for item in rev_list:
     indices.append([i for i,x in enumerate(paper_auth) if x == item])
	
authors = []
for item in indices:
	authors.append(arr[item].sum(axis=0)*1.0/len(item))
authors = np.array(authors)
np.savetxt('author_vec.txt',authors)

