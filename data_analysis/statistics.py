import matplotlib.pyplot as plt
import pymongo
import pandas as pd
import seaborn as sns
from pymongo import MongoClient, DESCENDING
import string
import re

def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist

if __name__ == '__main__':

	client = MongoClient('localhost', 27017)
	db = client.Yelp
	collection=db['Reviews']
	data = collection.find({},{"stars":1, "text":1})
	df = pd.DataFrame(list(data))

	#count of punctuation
	puncs = []
	for row in df["text"]:
		row = row.replace(',','')
		row = row.replace('.','')
		punc_count = len(re.findall('['+''.join(string.punctuation)+']',row))
		puncs.append(punc_count)
	df["punc"]= puncs
	g = sns.FacetGrid(data=df, col="stars")
	g.map(plt.hist, "punc", color="m", bins=50)
	plt.xlim(0,150)
	plt.show()
	
	#reviews based on fans of user
	fans = []
	data2 = collection.find({},{"text":1,"stars":1, "user_id.fans": 1})
	df2=pd.DataFrame(list(data2))
	df2["fans"]=df2["user_id"].map(lambda x:x["fans"])

	for row in df2["user_id"]:
		fans.append(df["fans"])
	g = sns.FacetGrid(data=df2, col="stars")
	g.map(plt.hist, "fans", color="steelblue", bins=50)
	plt.xlim(0,600)
	plt.ylim(0,50)
	plt.show()


	#count of words in capitals
	count = []
	for row in df['text']:
		punc = row.encode('utf-8')
		punc = punc.translate(None, string.punctuation) #remove punctuation		

		tokens = punc.split(" ")
		upper_count = sum(1 for c in tokens if c.isupper() and c is not 'I') #count uppercase words

		word_count = len(punc.split()) #initial word count
		ratio = upper_count/float(word_count)
		count.append(ratio)
	df["count"]= count
	g = sns.FacetGrid(data=df, col="stars")
	g.map(plt.hist, "count", color="steelblue", bins=20)
	plt.show()

	#count of price references
	price_count = []
	for row in df['text']:
		count = row.count('cent') + row.count('cents') + row.count('dollar') + row.count('dollars') + row.count('$')
		price_count.append(count)
	df["price"]= price_count
	g = sns.FacetGrid(data=df, col="stars")
	g.map(plt.hist, "price", color="steelblue", bins=10)
	plt.show()

	#count of unique words
	count = []
	for row in df['text']:
		punc = row.encode('utf-8')
		punc = punc.translate(None, string.punctuation) #remove punctuation		
		unique =' '.join(unique_list(punc.split())) #remove duplicate words

		word_count = len(punc.split()) #initial word count
		unique_word_count = len(unique.split()) #unique word count
		ratio = unique_word_count/float(word_count)
		count.append(ratio)
	df["count"]= count
	g = sns.FacetGrid(data=df, col="stars")
	g.map(plt.hist, "count", color="steelblue", bins=20)
	plt.show()
														
