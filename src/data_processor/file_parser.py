import json
import os
import io
import pickle
import random

def get_all_file(path):
    list_of_file = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            list_of_file.append(os.path.join(path, name))
    return list_of_file

def multiple_list(list, times):
	new_list = []
	for i in range (times):
		new_list = new_list + list
	return new_list

def clean_data(path1, path2, sizes):
	pos_comments = []
	neg_comments = []
	print ('Cleaning data...')
	'''
	list_tgdd = get_all_file(path1)
	for file in list_tgdd:
		with open(file, 'r+', encoding='utf8') as f:
			json_data = f.read()
			comments = json.loads(json_data)
			for comment in comments:
				star = comment['start']
				if star <= 3:
					neg_comments.append(comment['comment'])
				elif star > 3:
					pos_comments.append(comment['comment'])
	list_tiki = get_all_file(path2)
	c = 0
	for file in list_tiki:
		with open(file, 'r+', encoding='utf8') as f:
			json_data = f.read()
			try:
				comments = json.loads(json_data)
			except json.decoder.JSONDecodeError:
				bug = False
			for comment in comments['reviews']:
				c += 1
				star = comment['start']
				print (comment['comment'], c)
				if star <= 3:
					neg_comments.append(comment['comment'])
				elif star > 3:
					pos_comments.append(comment['comment'])
	
	print ('Pickling')
	with open (r'../../preprocessed_data/pos_comments.pickle', 'wb') as file:
		pickle.dump(pos_comments, file)
		
	with open (r'../../preprocessed_data/neg_comments.pickle', 'wb') as file:
		pickle.dump(neg_comments, file)
	print('Done pickling')
	'''
	#print(pos_comments[2])
	
	print ('Loading from pickle files')
	with open (r'../../preprocessed_data/pos_comments.pickle', 'rb') as file:
		pos_comments = pickle.load(file)
		
	with open (r'../../preprocessed_data/neg_comments.pickle', 'rb') as file:
		neg_comments = pickle.load(file)
	print ('Successfully loaded')
	print (len(pos_comments), len(neg_comments))
	
	pos_comments = set(pos_comments)
	neg_comments = set(neg_comments)
	
	print (len(pos_comments), len(neg_comments))	
	
	pos_comments = list(pos_comments)
	neg_comments = list(neg_comments)
	
	train_pos_comments = pos_comments[:sizes[0]]
	train_neg_comments = neg_comments[:sizes[1]]
	
	train_pos_comments = multiple_list(train_pos_comments, 11)
	train_neg_comments = multiple_list(train_neg_comments, 11)
	
	print (len(train_pos_comments), len(train_neg_comments))
	
	test_pos_comments = pos_comments[-sizes[2]:]
	test_neg_comments = neg_comments[-sizes[3]:]
	
	test_pos_comments = multiple_list(test_pos_comments, 11)
	test_neg_comments = multiple_list(test_neg_comments, 11) 
	
	
	with open (r'../../preprocessed_data/train_neg_comments.pickle', 'wb') as file:
		pickle.dump(train_neg_comments, file)
		
	with open (r'../../preprocessed_data/train_pos_comments.pickle', 'wb') as file:
		pickle.dump(train_pos_comments, file)
		
	with open (r'../../preprocessed_data/test_neg_comments.pickle', 'wb') as file:
		pickle.dump(test_neg_comments, file)
		
	with open (r'../../preprocessed_data/test_pos_comments.pickle', 'wb') as file:
		pickle.dump(test_pos_comments, file)
	
	print ('Done cleaning')
	
def main():
	sizes = [29000,8500, 1000, 1000]
	clean_data(r'../../raw_data/All_From_TGDD', r'../../raw_data/All_From_Tiki', sizes)	
	
	
if __name__ == '__main__':
	main()
	