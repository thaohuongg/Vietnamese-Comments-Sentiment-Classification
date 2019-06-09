import dill
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def tfidf(X_data):
	def dummy_func(doc):
		return doc
	tfidf_vect = TfidfVectorizer(
		analyzer='word',
		tokenizer=dummy_func,
		preprocessor=dummy_func,
		token_pattern=None) 
	tfidf_vect.fit(X_data)	
	return tfidf_vect

def classifier(X_train, y_train):
	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	return clf

def predict (clf, X_test, y_test):
	y_pred = clf.predict(X_test)
	return accuracy_score(y_test, y_pred)*100

def main():
	print ('Loading pickle files')
	
	with open (r'../../preprocessed_data/X_train.pickle', 'rb') as file:
		X_train = pickle.load(file)
	with open (r'../../preprocessed_data/X_test.pickle', 'rb') as file:
		X_test = pickle.load(file)
	with open (r'../../preprocessed_data/y_train.pickle', 'rb') as file:
		y_train = pickle.load(file)
	with open (r'../../preprocessed_data/y_test.pickle', 'rb') as file:
		y_test = pickle.load(file)
			
	print ('Len(X_train) = {}\nLen(X_test) = {}'.format(len(X_train), len(X_test)))
	tfidf_vect = tfidf(X_train)
	with open (r'../../preprocessed_data/tfidf_vocab.pickle', 'wb') as file:
		pickle.dump(tfidf_vect.vocabulary_, file, pickle.HIGHEST_PROTOCOL)
		
	print ('Save ok')
	
	X_train_tfidf =  tfidf_vect.transform(X_train)
	X_test_tfidf =  tfidf_vect.transform(X_test)
	
	clf = classifier(X_train_tfidf, y_train)
	
	with open (r'../../preprocessed_data/NaiveBayesClassifier.pickle', 'wb') as file:
		pickle.dump(clf, file, pickle.HIGHEST_PROTOCOL)
		
	accuracy = predict(clf, X_test_tfidf, y_test)
	print ('Accurancy = %.2f%%' %  accuracy) #Accurancy = 73.80%
	
	
if __name__ == '__main__':
	main()