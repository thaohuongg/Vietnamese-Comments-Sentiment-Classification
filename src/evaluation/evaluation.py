import io
import pickle
import json
import gensim
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import sys
sys.path.append("..") 
from model import model

def turn_into_vector(file_name, stopwords):
	comments = {}
	with open(file_name, 'r+', encoding='utf8') as f:
		json_data = f.read()
		comments = json.loads(json_data)
	X = []
	y = []
	for stat, com_list in comments.items():
		for comment in com_list:
			comment = ViTokenizer.tokenize(comment)
			comment = gensim.utils.simple_preprocess(comment)
			for word in comment:
				if word in stopwords:
					comment.remove(word)
			X.append(comment)
			y.append(stat)
	return X,y
	
def main():
	stopwords = set()
	with open (r'stopwords.txt', 'r', encoding="utf8") as file:
		line = file.readline() 
		while line:
			line = line = line.strip('\n')
			stopwords.add(line)
			line = file.readline()
	X_test, y_test = turn_into_vector(r'comments2.json',stopwords)
	
	print ('Loading pickle files')
	def dummy_func(doc):
		return doc
	tfidf_vect = TfidfVectorizer(
		decode_error="replace",
		vocabulary=pickle.load(open(r"../../preprocessed_data/tfidf_vocab.pickle", "rb")),
		tokenizer=dummy_func,
		preprocessor=dummy_func,
		token_pattern=None)
	with open (r'../../preprocessed_data/NaiveBayesClassifier.pickle', 'rb') as file:
		clf = pickle.load(file)
	print('Loaded successfully')
	
	X_test_tfidf =  tfidf_vect.fit_transform(X_test)
		
	accuracy = model.predict(clf, X_test_tfidf, y_test)
	print ('Accurancy = %.2f%%' %  accuracy)
	
if __name__ == '__main__':
	main()