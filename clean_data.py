import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 2.1 (3 pts.) 
def lowerCase(df):
	df["Text"] = df["Text"].str.lower()
	return df

def  clean_text(df):
    df["Text"] = df["Text"].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", str(elem) ))  
    # remove numbers
    df["Text"] = df["Text"].apply(lambda elem: re.sub(r"\d+", "", elem))
    return df

# 2.2 (3 pts.) 
def tokenizeDF(df):
	df["Text"] = df["Text"].map(lambda elem: word_tokenize(str(elem)))
	return df

def lemmatize(df):
    def lemmatize_helper(text):
        return [WordNetLemmatizer().lemmatize(i) for i in text]
    df["Text"] = df["Text"].apply(lambda elem: lemmatize_helper(elem))
    return df

# 2.3 (4 pts.) 	
def removeStop(df, stopeng):
	df["Text"] = df["Text"].map(lambda x: [word for word in x if word not in (stopeng)])
	return df


# 2.4 (4 pts.) 
def keepAlpha(df):
	alpahbets = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
	def keepAlpha_helper(a):
		data_list=[]
		for word in a:
			if len(word) == 0 or len(word) == 1:
				continue
			hasChar = False
			for character in word:
				if character in alpahbets:
					hasChar = True
			if hasChar:
				data_list.append(word)
		return data_list	
	df["Text"] = df["Text"].map(lambda x: keepAlpha_helper(x))
	return df

# 2.5 (3 pts.) 
def joinText(df):
	df["Text"] = df["Text"].map(lambda x: ' '.join(x))
	return df



def main():
    df = pd.read_csv("post.csv")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download("wordnet")
    df = lowerCase(df)
    df = clean_text(df)
    df = tokenizeDF(df)
    df = lemmatize(df)
    stopeng = set(stopwords.words("english"))
    df = removeStop(df,stopeng)
    df = keepAlpha(df)
    df = joinText(df)
    df.to_csv("cleaned_post.csv", index=False)



if __name__ == "__main__":
    main()