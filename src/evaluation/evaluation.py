import io
import pickle
import json
import gensim
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import xlwt 
from xlwt import Workbook
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
	data = []
	for stat, com_list in comments.items():
		for comment in com_list:
			data.append(comment)
			comment = ViTokenizer.tokenize(comment)
			comment = gensim.utils.simple_preprocess(comment)
			for word in comment:
				if word in stopwords:
					comment.remove(word)
			X.append(comment)
			y.append(stat)
	return X, y, data
	
def to_xls(clf, X_test, data, name):
	y_predict = clf.predict(X_test)
	wb = Workbook() 
	sheet1 = wb.add_sheet('Sheet 1') 
	sheet1.write(0,0,'comments')
	sheet1.write(0,1,'predict')
	for i in range (1,len(data)+1):
		sheet1.write(i, 0, data[i-1])
		sheet1.write(i, 1, y_predict[i-1])
	wb.save(name) 
	
def main():
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
	
	#----------------------------------------------------
	
	X_test, y_test, data = turn_into_vector(r'comments.json',stopwords)
	X_test_tfidf =  tfidf_vect.fit_transform(X_test)
	to_xls(clf, X_test_tfidf, data)
	accuracy = model.predict(clf, X_test_tfidf, y_test, 'prediction1.xls')
	print ('Accurancy = %.2f%%' %  accuracy)
	
	#-----------------------------------------------------
	'''
	with open (r'../../preprocessed_data/test_pos_comments.pickle', 'rb') as file:
		test_pos_comments = pickle.load(file)
		
	with open (r'../../preprocessed_data/test_neg_comments.pickle', 'rb') as file:
		test_neg_comments = pickle.load(file)
		
	data = test_pos_comments + test_neg_comments
	
	with open (r'../../preprocessed_data/X_test.pickle', 'rb') as file:
		X_test = pickle.load(file)
	X_test_tfidf =  tfidf_vect.fit_transform(X_test)
	to_xls(clf, X_test_tfidf, data, 'prediction.xls')
	'''
	
if __name__ == '__main__':
	main()