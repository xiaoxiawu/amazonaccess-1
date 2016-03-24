from __future__ import division

import pandas as pd
import numpy as np
from pandas.core.series import Series


# show statistics of dataframe columns
def show_basic_statics(df, col_list = None):
	print 'Showing Statistics'

	if not col_list :
		col_list = df.columns
	for colname in col_list:
		col = df[colname].values
		num_unique = np.unique(col).shape[0]
		min_elem = np.amin(col)
		max_elem = np.amax(col)
		mean = np.mean(col)
		std = np.std(col)
		print 'Col:{0:s}'.format(colname)
		print 'unique:', num_unique, 'min:', min_elem, 'max:', max_elem, 'mean:', mean, 'std:', std

# check the combination of different columns
def check_two_columns(df, col1, col2):
	print 'Checking {0:s} and {1:s}'.format(col1, col2)

	c1_c2_list = df[col1, col2].values.tolist()
	c1_c2_tuple = [tuple(c1_c2) for c1_c2 in c1_c2_list]

	num_unique_c1_c2 = len(set(c1_c2_tuple))
	num_unique_c1 = np.unique(df[col1].values).shape[0]
	num_unique_c2 = np.unique(df[col2].values).shape[0]

	print 'c1:', num_unique_c1, 'c2:', num_unique_c2, 'comb:', num_unique_c1_c2

def merge_two_columns(df, col1, col2, col_new=col1+'_COMB_'+col2, remove='none', hasher=None):
	print 'Combining {0:s} and {1:s} into {2:s}'.format(col1, col2, col_new)
	
	c1_c2_list = df[col1, col2].values.tolist()
	c1_c2_tuple = [tuple(c1_c2) for c1_c2 in c1_c2_list]
	c1_c2_set = set(c1_c2_tuple)

	c1_c2_tuple_dict = dict()
	i = 0
	for c1_c2 in c1_c2_set:
		c1_c2_tuple_dict[c1_c2] = i
		i+=1

	col_new_data = np.zeros(df[col1].shape)
	for i in xrange(col_new_data.shape[0]):
	    col_new_data[i] = c1_c2_tuple_dict[c1_c2_tuple[i]]

	if hasher:
		col_new_data = hasher(col_new_data)

	if remove == 'both':
		df = df.drop[col1, col2]
	elif remove == 'col1':
		df = df.drop[col1]
	elif remove == 'col2':
		df = df.drop[col2]

	df[col_new] = Series(col_new_data, index = df.index)


# generating countable
import cPickle as pickle

COUNT_TABLE_PATH = './cache/count_tb.pickle'

def get_count_table(df, col_list, file_path = COUNT_TABLE_PATH):
	# check if cache already exists
	try:
	    with open( file_path, 'rb') as f:
	        count_tables = pickle.load(f)

	    return count_tables
	except IOError:
	    num_col = len(col_list)

	    one_count_table = dict()
	    for i in xrange(num_col):
	        col_i = col_list[i]
	        col_i_data = df[col_i].values
	        col_i_set = np.unique(col_i_data)

	        if col_i in one_count_table:
	            continue
	        one_count_table[col_i] = dict()
	        for i_elem in col_i_set:
	            sub_i_ind = np.argwhere(col_i_data == i_elem)
	            one_count_table[col_i][i_elem] = sub_i_ind.shape[0]

	    two_count_table = dict()
	    for i in xrange(num_col):
	        col_i = col_list[i]
	        col_i_data = df[col_i].values
	        col_i_set = np.unique(col_i_data)

	        for j in xrange(num_col):
	            if j == i:
	                continue
	            col_j = col_list[j]
	            col_j_data = df[col_j].values

	            tuple_col_ij = tuple(col_i, col_j)
	            if tuple_col_ij in two_count_table:
	                continue
	            two_count_table[tuple_col_ij] = dict()
	            for i_elem in col_i_set:
	                sub_i_ind = np.argwhere(col_i_data == i_elem)
	                sub_i_ind = sub_i_ind.reshape(sub_i_ind.shape[0])
	                sub_j_data = col_j_data[sub_i_ind]
	                sub_j_set = np.unique(sub_j_data)

	                two_count_table[tuple_col_ij][i_elem] = {'unique': len(sub_j_set)}
	                for j_elem in sub_j_set:
	                    sub_j_ind = np.argwhere(sub_j_data == j_elem)
	                    two_count_table[tuple_col_ij][i_elem][j_elem] = sub_j_ind.shape[0]

	    count_tables = [one_count_table, two_count_table]

	    with open( file_path, 'wb') as f:
	        pickle.dump(count_tables, f, pickle.HIGHEST_PROTOCOL)

	    return count_tables


def two_degree_counts(df, col_i, col_j, operation, count_tables = None, file_path = COUNT_TABLE_PATH):
	# if two columns are the same just return one_degree_count
    if col_i == col_j:
        return one_degree_counts(df, col_i, count_tables, file_path)

    if not count_tables:
        try:
            with open( file_path, 'rb') as f:
                count_tables = pickle.load(f)
                one_count_table = count_tables[0]
                two_count_table = count_tables[1]
        except IOError:        
            one_count_table = dict()
            two_count_table = dict()


    if col_i in one_count_table:
        i_table = one_count_table[col_i]
    else:
        i_table = None

    tuple_col_ij = tuple(col_i, col_j)
    if tuple_col_ij in two_count_table:
        ij_table = two_count_table[tuple_col_ij]
    else:
        ij_table = None

    if i_table == None or ij_table == None:
        count_tables = get_count_table(df, [col_i, col_j], file_path)
        one_count_table = count_tables[0]
        two_count_table = count_tables[1]
        i_table = one_count_table[col_i]
        ij_table = two_count_table[tuple_col_ij]

    col_i_data = df[col_i].values
    col_j_data = df[col_j].values
    if operation == 'per': 	# 'per': percentage of (elem_i, elem_j) in all (elem_i, col_j)  
        vfunc = np.vectorize(lambda x,y: float(ij_table[x][y])/i_table[x])
        col_new = vfunc(col_i_data, col_j_data)
    elif operation == 'num':	# 'num': number of different kinds of (elem_i, col_j) 
        vfunc = np.vectorize(lambda x: ij_table[x]['unique'])
        col_new = vfunc(col_i_data)

    return col_new




def one_degree_counts(df, col_i, count_tables = None, file_path = COUNT_TABLE_PATH):
    if not count_tables:
        try:
            with open( file_path, 'rb') as f:
                count_tables = pickle.load(f)
                one_count_table = count_tables[0]
        except IOError:        
            one_count_table = dict()


    if col_i in one_count_table:
        i_table = one_count_table[col_i]
    else:
        i_table = None

    if i_table == None:
        count_tables = get_count_table(df, [col_i], file_path)
        one_count_table = count_tables[0]
        i_table = one_count_table[col_i]

    vfunc = np.vectorize(lambda x: table[x])
    col_new = vfunc(col_i_data)

    return col_new

# data preprocessing and generate different set of features
from sklearn import preprocessing
# put all rare event into one group
def combine_rare(Xtrain, Xtest, col_list, rare_line=1):
	for col in col_list:
		col_data_train = Xtrain[:, col]
		col_data_test = Xtest[:, col]
		col_data = np.vstack(col_data_train, col_data_test)
		if issubclass(col_data.dtype.type, np.integer):
			le = preprocessing.LabelEncoder()
			le.fit(col_data)
			col_data = le.transform(col_data)
			max_label = np.amax(col_data)
			counts = np.bincount(col_data)
			rare_cats = np.argwhere(counts <= rare_line)
			rare_cats = rare_cats.reshape(rare_cats.shape[0])
			rare_positions = [np.argwhere(dummy2 == rare_cat)[0,0] for rare_cat in rare_cats]
			col_data[rare_positions] = max_label+1
			Xtrain[:, col] = col_data[:Xtrain.shape[0]]
			Xtest[:, col] = col_data[Xtrain.shape[0]:]
		else:
			print 'col:{0:d} not integer'.format(col)

# perform in place numerical transform on selected col, here X is numpy matrix
def numeric_transform(Xtrain, Xtest, col_list, operation='log', standardize=False):
	if operation == 'log':
		vfunc = np.vectorize(lambda x: np.log(x))
	elif operation == 'log+1':
		vfunc = np.vectorize(lambda x: np.log(x+1))
	elif operation == 'exp':
		vfunc = np.vectorize(lambda x: np.exp(x))
	elif operation == 'exp-1':
		vfunc = np.vectorize(lambda x: np.exp(x)-1)
	elif operation == 'none':
		vfunc = None
	else:
		vfunc = None
		print 'Unkown operation not performed'

	for col in col_list:
		if vfunc:
			Xtrain[:, col] = vfunc(Xtrain[:, col])
			Xtest[:, col] = vfunc(Xtest[:, col])
		if standardize:
			col_data_train = Xtrain[:, col]
			col_data_test = Xtest[:, col]
			col_data = np.vstack(col_data_train, col_data_test)
			col_mean = np.mean(col_data)
			col_std = np.std
			Xtrain[:, col] = 1./col_std * (Xtrain[:, col] - col_mean)
			Xtest[:, col] = 1./col_std * (Xtest[:, col] - col_mean)




# we consider model as a combination of feature and classifier
# classifier need to implement several methods
# many parts of the implementation only suitable for binary classification
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
# template class mostly just list methods to be implemented
class MyClassifer(object):
	def __init__(self, params):
		raise NotImplementedError
	def update_params(self, updates):
		raise NotImplementedError
	def fit(self, Xtrain, ytrain):
		raise NotImplementedError
	# def predict(self, Xtest, option):
	# 	raise NotImplementedError
	def predict_proba1(self, Xtest, option):
		raise NotImplementedError

# logistic regression
from sklearn.linear_model import LogisticRegression
class MyLogisticReg(MClassifer):
	def __init__(self, params=dict()):
		self._params = params
		self._lr = LogisticRegression(**(self._params))

	def update_params(self, updates):
		self._params.update(updates)
		self._lr = LogisticRegression(**(self._params))

	def fit(self, Xtrain, ytrain):
		self._lr.fit(Xtrain, ytrain)

	# def predict(self, Xtest, option = None):
	# 	return self._lr.predict(Xtest)

	def predict_proba1(self, Xtest, option = None):
		return self._lr.predict_proba(Xtest)[:, 1]


# k-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
class MyKnn(MClassifer):
	def __init__(self, params=dict()):
		self._params = params
		self._knn = KNeighborsClassifier(**(self._params))

	def update_params(self, updates):
		self._params.update(updates)
		self._knn = KNeighborsClassifier(**(self._params))

	def fit(self, Xtrain, ytrain):
		self._knn.fit(Xtrain, ytrain)

	# def predict(self, Xtest, option = None):
	# 	return self._knn.predict(Xtest)

	def predict_proba1(self, Xtest, option = None):
		return self._knn.predict_proba(Xtest)[:, 1]

# extremelyrandomforest
from sklearn.ensemble import ExtraTreesClassifier
class MyExtraTree(MClassifer):
	def __init__(self, params=dict()):
		self._params = params
		self._extree = ExtraTreesClassifier(**(self._params))

	def update_params(self, updates):
		self._params.update(updates)
		self._extree = ExtraTreesClassifier(**(self._params))

	def fit(self, Xtrain, ytrain):
		self._extree.fit(Xtrain, ytrain)

	# def predict(self, Xtest, option = None):
	# 	return self._extree.predict(Xtest)

	def predict_proba1(self, Xtest, option = None):
		return self._extree.predict_proba(Xtest)[:, 1]

# xgboost
import xgboost as xgb
class MyXGBoost(MClassifer):
	def __init__(self, params=dict()):
		self._params = params
		if 'num_round' in params:
			self._num_round = params['num_round']
			del self._params['num_round']
		else:
			 self._num_round = None
		self._xgb = None

	def update_params(self, updates):
		self._params = self._params.update(updates)
		if 'num_round' in updates:
			self._num_round = updates['num_round']
			del self._params['num_round']

	def fit(self, Xtrain, ytrain):
		dtrain = xgb.DMatrix( Xtrain, label=ytrain)
		if self._num_round:
			self._xgb = xgb.train(self._params, dtrain, self._num_round)
		else:
			self._xgb = xgb.train(self._params, dtrain)

	def predict_proba1(self, Xtest, option = dict()):
		dtest= xgb.DMatrix(Xtest)
		if 'ntree_limit' not in option:
			return self._xgb.predict(dtest)
		else:
			return self._xgb.predict(dtest, ntree_limit=option['ntree_limit'])


# cv_score related functions

def strat_cv_predict_proba(myclassifier, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=roc_auc_score):
		y = np.zeros(ytrain.shape, float)

		sk_cv_score = np.zeros(nfolds, float)
		k = 0
		skfold = StratifiedKFold(ytrain, n_folds=nfolds, random_state=randstate)
		for train_index, test_index in skfold:
			sk_Xtrain, sk_Xtest = Xtrain[train_index], Xtrain[test_index]
			sk_ytrain, sk_ytest = ytrain[train_index], ytrain[test_index]
			myclassifier.fit(sk_Xtrain, sk_ytrain)
			sk_ypred = myclassifier.predict_proba(sk_Xtest)
			y[test_index] = sk_ypred
			sk_cv_score[k] = score_func(sk_ytest, sk_ypred)
			k += 1

		return y, sk_cv_score

def cv_score(myclassifier, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=roc_auc_score):
	k_cv_score = np.zeros(ytrain.shape, float)

	k = 0
	kfold = KFold(ytrain, n_folds=nfolds, random_state=randstate)
	for train_index, test_index in kfold:
		k_Xtrain, k_Xtest = Xtrain[train_index], Xtrain[test_index]
		k_ytrain, k_ytest = ytrain[train_index], ytrain[test_index]
		myclassifier.fit(k_Xtrain, k_ytrain)
		k_ypred = myclassifier.predict_proba(sk_Xtest)
		k_cv_score[k] = score_func(k_ytest, k_ypred)
		k += 1

	return k_cv_score

def strat_cv_score(myclassifier, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=roc_auc_score):
		sk_cv_score = np.zeros(nfolds, float)

		k = 0
		skfold = StratifiedKFold(ytrain, n_folds=nfolds, random_state=randstate)
		for train_index, test_index in skfold:
			sk_Xtrain, sk_Xtest = Xtrain[train_index], Xtrain[test_index]
			sk_ytrain, sk_ytest = ytrain[train_index], ytrain[test_index]
			myclassifier.fit(sk_Xtrain, sk_ytrain)
			sk_ypred = myclassifier.predict_proba(sk_Xtest)
			sk_cv_score[k] = score_func(sk_ytest, sk_ypred)
			k += 1

		return sk_cv_score

import itertools
# here param_grid just need to contain the parameters required to be updated
def cv_grid_search(myclassifier, param_grid, Xtrain, ytrain, nfolds=10, randstate=SEED, score_func=roc_auc_score, criterion = 'max'):
	# can be adapted to use sklearn CVGridSearch
	# org_params = myclassifier._params
	# org_param_grid = dict{}
	# for key, value in org_params:
	# 	org_param_grid[key] = [value]
	# org_param_grid.update(param_grid)
	param_names = param_grid.keys()
	param_pools = param_grid.values()
	num_param = len(param_names)
	param_set_list = []
	mean_score_list = []
	best_param_set = None
	best_mean_score = None
	for param_valuelist in itertools.product(*param_pools):
		param_set = dict()
		for k in xrange(num_param):
			param_set[param_names[k]] = param_valuelist[k]

		param_set_list.append(param_set)

		myclassifier.update(param_set)
		cur_scores = cv_score(myclassifier, Xtrain, ytrain, nfolds, randstate, score_func)	
		cur_mean_score = np.mean(cur_scores)

		mean_score_list.append(cur_mean_score)

		if not best_paramset:
			best_param_set = param_set
			best_mean_score = cur_mean_score
		elif criterion == 'max' and cur_mean_score > best_mean_score:
			best_param_set = param_set
			best_mean_score = cur_mean_score
		elif criterion == 'min' and cur_mean_score < best_mean_score:
			best_param_set = param_set
			best_mean_score = cur_mean_score

	return best_param_set, best_mean_score, param_set_list, mean_score_list


def main():
	pass

if __name__ == "__main__":
	main()
# define model as a combination of feature and classifier
# class MyMdoel(object):

# class MyFeature(object):
# 	self.feature_name = None
# 	self.feature_dmatrix = None

# 	def __init__(self, name = ):


# class MyFeatureList(object):
# 	self.feature_name_list = []
# 	self.feature_dmatrix = None

# 	def __init__(self):
# 		raise NotImplementedError

# 	def add(myfeature):
# 		self.feature_name_list.append(myfeature.feature_name)


# 	def remove(myfeature)

# from sklearn import ensemble
# from sklearn import linear_model

# SEED = 1234

# dfAmTrain = pd.read_csv('./train.csv')
# dfAmTest = pd.read_csv('./test.csv')