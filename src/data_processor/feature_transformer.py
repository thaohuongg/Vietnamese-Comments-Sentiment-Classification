import pickle
from pyvi import ViTokenizer
import gensim

#def turn_into_vector(stopwords):
def turn_into_vector():
	print ('Turning into vectors')
	def get_data(pickle_path, label):
		with open (pickle_path, 'rb') as file:
			comments = pickle.load(file)
		X = []
		y = []
		for comment in comments:
			comment = ViTokenizer.tokenize(comment)
			comment = gensim.utils.simple_preprocess(comment)
			X.append(comment)
			y.append(label)
		return X,y
	
	X_train_pos, y_train_pos = get_data(r'../../preprocessed_data/train_pos_comments.pickle', 'pos')
	#print (X_train_pos[2])
	X_train_neg, y_train_neg = get_data(r'../../preprocessed_data/train_neg_comments.pickle', 'neg')
	
	X_train = X_train_pos + X_train_neg 
	y_train = y_train_pos + y_train_neg 
	
	X_test_pos, y_test_pos = get_data(r'../../preprocessed_data/test_pos_comments.pickle', 'pos')
	X_test_neg, y_test_neg = get_data(r'../../preprocessed_data/test_neg_comments.pickle', 'neg')
	
	X_test = X_test_pos + X_test_neg 
	y_test = y_test_pos + y_test_neg 
	
	print (len(X_train), len(X_test))
	
	with open (r'../../preprocessed_data/X_train.pickle', 'wb') as file:
		pickle.dump(X_train, file)
		
	with open (r'../../preprocessed_data/y_train.pickle', 'wb') as file:
		pickle.dump(y_train, file)
		
	with open (r'../../preprocessed_data/X_test.pickle', 'wb') as file:
		pickle.dump(X_test, file)
		
	with open (r'../../preprocessed_data/y_test.pickle', 'wb') as file:
		pickle.dump(y_test, file)
	
	print('Turned into vectors successfully')
	
def main():
	turn_into_vector()
	
if __name__ == '__main__':
	main()