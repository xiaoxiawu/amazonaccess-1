from __future__ import division

import pandas as pd
import numpy as np
from pandas.core.series import Series
import os

SEED = 1234

# show statistics of dataframe columns
def show_basic_statics(df, col_list = list()):
    print 'Showing Statistics\n'

    if not col_list :
        col_list = list(df.columns.values)
    for colname in col_list:
        col = df[colname].values
        num_unique = np.unique(col).shape[0]
        min_elem = np.amin(col)
        max_elem = np.amax(col)
        mean = np.mean(col)
        std = np.std(col)
        print 'Col:{0:s}, dtype:{1:s}'.format(colname, col.dtype)
        print 'unique:', num_unique, 'min:', min_elem, 'max:', max_elem, 'mean:', mean, 'std:', std
        print

# check the combination of different columns
def check_two_columns(df, col1, col2):
    print 'Checking {0:s} and {1:s}'.format(col1, col2)

    c1_c2_list = df[[col1, col2]].values.tolist()
    c1_c2_tuple = [tuple(c1_c2) for c1_c2 in c1_c2_list]

    num_unique_c1_c2 = len(set(c1_c2_tuple))
    num_unique_c1 = np.unique(df[col1].values).shape[0]
    num_unique_c2 = np.unique(df[col2].values).shape[0]

    print '{0:s}:'.format(col1), num_unique_c1, '{0:s}:'.format(col2), num_unique_c2, 'comb:', num_unique_c1_c2

def merge_two_cat_columns(df, col1, col2, col_new=None, remove='none', hasher=None):
    if not col_new:
        col_new=col1+'_COMB_'+col2
    print 'Combining {0:s} and {1:s} into {2:s}'.format(col1, col2, col_new)

    if col_new in list(df.columns.values):
        print 'Overwriting exisiting {0:s}'.format(col_new)
        # df.drop([col_new])

    c1_data = df[col1].values
    c2_data = df[col2].values
    c1_c2_tuple = zip(c1_data, c2_data)
    c1_c2_set = set(c1_c2_tuple)

    c1_c2_tuple_dict = dict()
    i = 0
    for c1_c2 in c1_c2_set:
        c1_c2_tuple_dict[c1_c2] = i
        i+=1

    col_new_data = np.zeros(df[col1].shape, np.int)
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

from sklearn import preprocessing
# put all rare event into one group
def combine_rare(df, col_list = list(), new_name_list = list(), rare_line=1):
    if not col_list :
        col_list = list(df.columns.values)
    if not new_name_list :
        new_name_list = ['CR_'+col for col in col_list]

    for col, new_name in zip(col_list, new_name_list):
        col_data = df[col].values
        if issubclass(col_data.dtype.type, np.integer):
            le = preprocessing.LabelEncoder()
            le.fit(col_data)
            col_data = le.transform(col_data)
            max_label = np.amax(col_data)
            counts = np.bincount(col_data)
            rare_cats = np.argwhere(counts <= rare_line)
            rare_cats = rare_cats.reshape(rare_cats.shape[0])
            rare_positions = [np.argwhere(col_data == rare_cat)[0,0] for rare_cat in rare_cats]
            col_data[rare_positions] = max_label+1
            df[new_name] = Series(col_data, index = df.index)
        else:
            print 'col:{0:s} not integer'.format(col)


import cPickle as pickle

# generating countable

class MyCountTable(object):
    def __init__(self):
        self.one_count_table = dict()
        self.two_count_table = dict()
        self._file_path = None

    def save_count_tables(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'wb') as f:
            pickle.dump([self.one_count_table, self.two_count_table], f, pickle.HIGHEST_PROTOCOL)
        self._file_path = file_path

    def load_count_tables(self, file_path = None):
        if not file_path:
            file_path = self._file_path
        if file_path:
            try:
                with open( file_path, 'rb') as f:
                    self.one_count_table, self.two_count_table = pickle.load(f)
            except IOError:
                print 'Loading count table file failed: file not found.'
        else:
            print 'Loading count table file failed: file not saved.'

    # count table operation related with data frame
    def df_generate_count_tables(self, df, col_list=list(), file_path = None):
        if not col_list:
            col_list = list(df.columns.values)

        n_col = len(col_list)

        for i in xrange(n_col):
            col_i = col_list[i]
            col_i_data = df[col_i].values
            col_i_set = np.unique(col_i_data)

            if col_i in self.one_count_table:
                continue
            self.one_count_table[col_i] = dict()
            for i_elem in col_i_set:
                sub_i_ind = np.argwhere(col_i_data == i_elem)
                self.one_count_table[col_i][i_elem] = sub_i_ind.shape[0]

        for i in xrange(n_col):
            col_i = col_list[i]
            col_i_data = df[col_i].values
            col_i_set = np.unique(col_i_data)

            for j in xrange(n_col):
                if j == i:
                    continue
                col_j = col_list[j]
                col_j_data = df[col_j].values

                tuple_col_ij = (col_i, col_j)
                if tuple_col_ij in self.two_count_table:
                    continue
                self.two_count_table[tuple_col_ij] = dict()
                for i_elem in col_i_set:
                    sub_i_ind = np.argwhere(col_i_data == i_elem)
                    sub_i_ind = sub_i_ind.reshape(sub_i_ind.shape[0])
                    sub_j_data = col_j_data[sub_i_ind]
                    sub_j_set = np.unique(sub_j_data)

                    self.two_count_table[tuple_col_ij][i_elem] = {'unique': len(sub_j_set)}
                    for j_elem in sub_j_set:
                        sub_j_ind = np.argwhere(sub_j_data == j_elem)
                        self.two_count_table[tuple_col_ij][i_elem][j_elem] = sub_j_ind.shape[0]

        if file_path:
            self.save_count_tables(file_path)

    # only called when directed from df_two_degree_counts or df_one_degree_counts
    # helper function to load and generate count table when necessary
    def _df_get_count_tables(self, df, col_list = list(), file_path = None):
        if not col_list:
            col_list = list(df.columns.values)

        if not file_path:
            file_path = self._file_path

        # first try local count tables
        flag = 0
        for col in col_list:
            if col not in self.one_count_table:
                flag = 1
                break

        if flag == 0:
            return

        # if not good try dumped count tables
        if file_path:
            self.load_count_tables(file_path)
            flag = 0
            for col in col_list:
                if col not in self.one_count_table:
                    flag = 1
                    break

        # generate countables if necessary
        if flag == 1:
                self.df_generate_count_tables(df, col_list, file_path)


    def df_two_degree_counts(self, df, col_i, col_j, operation, file_path = None):
        # if two columns are the same just return one_degree_count
        if col_i == col_j:
            return self.df_one_degree_counts(df, col_i, file_path)

        self._df_get_count_tables(df, [col_i, col_j], file_path)

        i_table = one_count_table[col_i]
        ij_table = two_count_table[(col_i, col_j)]

        col_i_data = df[col_i].values
        col_j_data = df[col_j].values
        if operation == 'per':  # 'per': percentage of (elem_i, elem_j) in all (elem_i, col_j)  
            vfunc = np.vectorize(lambda x,y: float(ij_table[x][y])/i_table[x])
            col_new = vfunc(col_i_data, col_j_data)
        elif operation == 'num':    # 'num': number of different kinds of (elem_i, col_j) 
            vfunc = np.vectorize(lambda x: ij_table[x]['unique'])
            col_new = vfunc(col_i_data)

        return col_new

    def df_one_degree_counts(df, col_i, file_path = None):
        self._df_get_count_tables(df, [col_i], file_path)

        i_table = one_count_table[col_i]

        col_i_data = df[col_i].values
        vfunc = np.vectorize(lambda x: i_table[x])
        col_new = vfunc(col_i_data)

        return col_new

    # fset  version of countable
    def fset_generate_count_tables(self, myfset, col_list=list(), file_path = None):
        if not col_list:
            col_list = range(len(myfset.fname_list))

        # n_col = len(col_list)

        for col_i in col_list:
            col_i_name = myfset.fname_list[col_i]
            col_i_ind = myfset.find_list[col_i]
            col_i_data_train = myfset.Xtrain[:, col_i_ind]
            col_i_data_test = myfset.Xtest[:, col_i_ind]
            col_i_data = np.hstack((col_i_data_train, col_i_data_test))
            col_i_set = np.unique(col_i_data)

            if col_i_name in self.one_count_table:
                continue
            self.one_count_table[col_i_name] = dict()
            for i_elem in col_i_set:
                sub_i_ind = np.argwhere(col_i_data == i_elem)
                self.one_count_table[col_i_name][i_elem] = sub_i_ind.shape[0]

        for col_i in col_list:
            col_i_name = myfset.fname_list[col_i]
            col_i_ind = myfset.find_list[col_i]
            col_i_data_train = myfset.Xtrain[:, col_i_ind]
            col_i_data_test = myfset.Xtest[:, col_i_ind]
            col_i_data = np.hstack((col_i_data_train, col_i_data_test))
            col_i_set = np.unique(col_i_data)

            for col_j in col_list:
                if col_j == col_i:
                    continue
                col_j_name = myfset.fname_list[col_j]
                col_j_ind = myfset.find_list[col_j]
                col_j_data_train = myfset.Xtrain[:, col_j_ind]
                col_j_data_test = myfset.Xtest[:, col_j_ind]
                col_j_data = np.hstack((col_j_data_train, col_j_data_test))

                tuple_col_ij = (col_i_name, col_j_name)
                if tuple_col_ij in self.two_count_table:
                    continue
                self.two_count_table[tuple_col_ij] = dict()
                for i_elem in col_i_set:
                    sub_i_ind = np.argwhere(col_i_data == i_elem)
                    sub_i_ind = sub_i_ind.reshape(sub_i_ind.shape[0])
                    sub_j_data = col_j_data[sub_i_ind]
                    sub_j_set = np.unique(sub_j_data)

                    self.two_count_table[tuple_col_ij][i_elem] = {'unique': len(sub_j_set)}
                    for j_elem in sub_j_set:
                        sub_j_ind = np.argwhere(sub_j_data == j_elem)
                        self.two_count_table[tuple_col_ij][i_elem][j_elem] = sub_j_ind.shape[0]

        if file_path:
            self.save_count_tables(file_path)    

    def _fset_get_count_tables(self, myfset, col_list=list(), file_path = None):
        if not col_list:
            col_list = range(len(myfset.fname_list))

        if not file_path:
            file_path = self._file_path

        # first try local count tables
        flag = 0
        for col in col_list:
            col_name = myfset.fname_list[col]
            if col_name not in self.one_count_table:
                flag = 1
                break

        if flag == 0:
            return

        # if not good try dumped count tables
        if file_path:
            self.load_count_tables(file_path)
            flag = 0
            for col in col_list:
                col_name = myfset.fname_list[col]
                if col_name not in self.one_count_table:
                    flag = 1
                    break

        # generate countables if necessary
        if flag == 1:
                self.fset_generate_count_tables(myfset, col_list, file_path)

    def fset_two_degree_counts(self, myfset, col_i, col_j, operation, file_path = None):
        # if two columns are the same just return one_degree_count
        if col_i == col_j:
            return self.fset_one_degree_counts(myfset, col_i, file_path)

        self._fset_get_count_tables(myfset, [col_i, col_j], file_path)

        col_i_name = myfset.fname_list[col_i]
        col_j_name = myfset.fname_list[col_j]
        col_i_ind = myfset.find_list[col_i]
        col_j_ind = myfset.find_list[col_j]


        i_table = self.one_count_table[col_i_name]
        ij_table = self.two_count_table[(col_i_name, col_j_name)]

        col_i_data_train = myfset.Xtrain[:, col_i_ind]
        col_i_data_test = myfset.Xtest[:, col_i_ind]
        col_j_data_train = myfset.Xtrain[:, col_j_ind]
        col_j_data_test = myfset.Xtest[:, col_j_ind]
        if operation == 'per':  # 'per': percentage of (elem_i, elem_j) in all (elem_i, col_j)  
            vfunc = np.vectorize(lambda x,y: float(ij_table[x][y])/i_table[x])
            col_new_train = vfunc(col_i_data_train, col_j_data_train)
            col_new_test = vfunc(col_i_data_test, col_j_data_test)
        elif operation == 'num':    # 'num': number of different kinds of (elem_i, col_j) 
            vfunc = np.vectorize(lambda x: ij_table[x]['unique'])
            col_new_train = vfunc(col_i_data_train)
            col_new_test = vfunc(col_i_data_test)

        return col_new_train, col_new_test

    def fset_one_degree_counts(self, myfset, col_i, file_path = None):
        self._fset_get_count_tables(myfset, [col_i], file_path)

        col_i_name = myfset.fname_list[col_i]
        col_i_ind = myfset.find_list[col_i]

        i_table = self.one_count_table[col_i_name]

        col_i_data_train = myfset.Xtrain[:, col_i_ind]
        col_i_data_test = myfset.Xtest[:, col_i_ind]
        vfunc = np.vectorize(lambda x: i_table[x])
        col_new_train = vfunc(col_i_data_train)
        col_new_test = vfunc(col_i_data_test)

        return col_new_train, col_new_test

# data preprocessing and generate different set of features

# perform in place combination of rare events
def np_combine_rare(Xtrain, Xtest, col_list = list(), rare_line=1):
    if Xtrain.shape[1] != Xtest.shape[1]:
        print 'Xtrain, Xtest shape not match.'
        return

    if not col_list :
        col_list = range(Xtrain.shape[1])

    n_train = Xtrain.shape[0]
    for col in col_list:
        col_data_train = Xtrain[:, col]
        col_data_test = Xtest[:, col]
        col_data = np.hstack((col_data_train, col_data_test))
        if issubclass(col_data.dtype.type, np.integer):
            le = preprocessing.LabelEncoder()
            le.fit(col_data)
            col_data = le.transform(col_data)
            max_label = np.amax(col_data)
            counts = np.bincount(col_data)
            rare_cats = np.argwhere(counts <= rare_line)
            rare_cats = rare_cats.reshape(rare_cats.shape[0])
            rare_positions = [np.argwhere(col_data == rare_cat)[0,0] for rare_cat in rare_cats]
            col_data[rare_positions] = max_label+1
            Xtrain[:, col] = col_data[:n_train]
            Xtest[:, col] = col_data[n_train:]
        else:
            print 'col:{0:s} not integer'.format(col)

# perform in place numerical transform on selected col, here X is numpy matrix, currently do not support sparse matrix
def np_numeric_transform(Xtrain, Xtest, col_list = list(), operation='log', standardize=False):
    if Xtrain.shape[1] != Xtest.shape[1]:
        print 'Xtrain, Xtest shape not match.'
        return

    if not col_list:
        col_list = range(Xtrain.shape[1])

    if operation == 'log':
        vfunc = np.vectorize(lambda x: np.log(x))
    elif operation == 'log1p':
        vfunc = np.vectorize(lambda x: np.log1p(x))
    elif operation == 'exp':
        vfunc = np.vectorize(lambda x: np.exp(x))
    elif operation == 'expm1':
        vfunc = np.vectorize(lambda x: np.expm1(x))
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
            col_data = np.hstack((col_data_train, col_data_test))
            col_mean = np.mean(col_data)
            # print col_mean
            col_std = np.std(col_data)
            Xtrain[:, col] = 1./col_std * (Xtrain[:, col] - col_mean)
            Xtest[:, col] = 1./col_std * (Xtest[:, col] - col_mean)




# we consider model as a combination of feature and classifier
# classifier need to implement several methods
# many parts of the implementation only suitable for binary classification
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
# template class mostly just list methods to be implemented
class MyClassifier(object):
    def __init__(self, params):
        raise NotImplementedError
    def update_params(self, updates):
        raise NotImplementedError
    def fit(self, Xtrain, ytrain):
        raise NotImplementedError
    # def predict(self, Xtest, option):
    #   raise NotImplementedError
    def predict_proba(self, Xtest, option):
        raise NotImplementedError

# logistic regression
from sklearn.linear_model import LogisticRegression
class MyLogisticReg(MyClassifier):
    def __init__(self, params=dict()):
        self._params = params
        self._lr = LogisticRegression(**(self._params))

    def update_params(self, updates):
        self._params.update(updates)
        self._lr = LogisticRegression(**(self._params))

    def fit(self, Xtrain, ytrain):
        self._lr.fit(Xtrain, ytrain)

    # def predict(self, Xtest, option = None):
    #   return self._lr.predict(Xtest)

    def predict_proba(self, Xtest, option = None):
        return self._lr.predict_proba(Xtest)[:, 1]


# k-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
class MyKnn(MyClassifier):
    def __init__(self, params=dict()):
        self._params = params
        self._knn = KNeighborsClassifier(**(self._params))

    def update_params(self, updates):
        self._params.update(updates)
        self._knn = KNeighborsClassifier(**(self._params))

    def fit(self, Xtrain, ytrain):
        self._knn.fit(Xtrain, ytrain)

    # def predict(self, Xtest, option = None):
    #   return self._knn.predict(Xtest)

    def predict_proba(self, Xtest, option = None):
        return self._knn.predict_proba(Xtest)[:, 1]

# extremelyrandomforest
from sklearn.ensemble import ExtraTreesClassifier
class MyExtraTree(MyClassifier):
    def __init__(self, params=dict()):
        self._params = params
        self._extree = ExtraTreesClassifier(**(self._params))

    def update_params(self, updates):
        self._params.update(updates)
        self._extree = ExtraTreesClassifier(**(self._params))

    def fit(self, Xtrain, ytrain):
        self._extree.fit(Xtrain, ytrain)

    # def predict(self, Xtest, option = None):
    #   return self._extree.predict(Xtest)

    def predict_proba(self, Xtest, option = None):
        return self._extree.predict_proba(Xtest)[:, 1]

# xgboost
import xgboost as xgb
class MyXGBoost(MyClassifier):
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

    def predict_proba(self, Xtest, option = dict()):
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
        skfold = StratifiedKFold(n = ytrain.shape[0], n_folds=nfolds, random_state=randstate)
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
    k_cv_score = np.zeros(nfolds, float)

    k = 0
    kfold = KFold(n = ytrain.shape[0], n_folds=nfolds, random_state=randstate)
    for train_index, test_index in kfold:
        k_Xtrain, k_Xtest = Xtrain[train_index], Xtrain[test_index]
        k_ytrain, k_ytest = ytrain[train_index], ytrain[test_index]
        myclassifier.fit(k_Xtrain, k_ytrain)
        k_ypred = myclassifier.predict_proba(k_Xtest)
        k_cv_score[k] = score_func(k_ytest, k_ypred)
        k += 1

    return k_cv_score

def strat_cv_score(myclassifier, Xtrain, ytrain, nfolds=5, randstate=SEED, score_func=roc_auc_score):
        sk_cv_score = np.zeros(nfolds, float)

        k = 0
        skfold = StratifiedKFold(n = ytrain.shape[0], n_folds=nfolds, random_state=randstate)
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
    #   org_param_grid[key] = [value]
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



# define model as a combination of feature and classifier
# class MyMdoel(object):

# class MyFeature(object):
#   self.feature_name = None
#   self.feature_dmatrix = None

#   def __init__(self, name = ):


class MyFeatureSet(object):
    # self.set_name = None
    def __init__(self):
        self.fname_list = list()
        self.find_list = list()
        self.Xtrain = None
        self.Xtest = None
        self._file_path = None


    def generate_feature_set(self, file_path):
        raise NotImplementedError

    def fetch_feature_set(self, file_path = None):
        if (not self.Xtrain) or (not self.Xtrain):
            if self._file_path:
                self.load_feature_set(self._file_path)
            elif file_path:
                self.load_feature_set(file_path) 
            else:
                # print 'Feature set not available.'
                self.generate_feature_set(file_path)

        return self.Xtrain, self.Xtest

    def load_feature_set(self, file_path = None):
        if not file_path:
            file_path = self._file_path
        if file_path:
            try:
                with open( file_path, 'rb') as f:
                    self.fname_list, self.find_list, self.Xtrain, self.Xtest = pickle.load(f)
            except IOError:
                print 'Loading feature set file failed: file not found.'
                self.generate_feature_set(file_path)
        else:
            print 'Loading featue set file failed: file not saved.'

    def save_feature_set(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'wb') as f:
            pickle.dump([self.fname_list, self.find_list, self.Xtrain, self.Xtest], f, pickle.HIGHEST_PROTOCOL)
        self._file_path = file_path


import scipy
# concatenate feature set in feature set list, sparsify if one of the feature set is sparse
def concat_feature_set(myfset_list, sparsify = False):  
    if not sparsify:
        for myfset in myfset_list:
            if scipy.sparse.issparse(myfset.Xtrain):
                sparsify = True
                break

    if sparsify:
        hstack_func = scipy.sparse.hstack
    else:
        hstack_func = np.hstack

    newfname_list = None
    newfind_list = None
    newXtrain = None
    newXtest = None

    n_fset = len(myfset_list)
    for i in xrange(n_fset):
        myfset = myfset_list[i]

        if i == 0:
            newfname_list = myfset.fname_list
            
            newfind_list = myfset.find_list
            
            newXtrain = myfset.Xtrain
            
            newXtest = myfset.Xtest

        else:
            newfname_list = newfname_list + myfset.fname_list

            prev_total = newfind_list[-1]
            curfind_list = [prev_total + find for find in myfset.find_list[1:]]
            newfind_list = newfind_list + curfind_list

            newXtrain = hstack_func((newXtrain, myfset.Xtrain))

            newXtest = hstack_func((newXtest, myfset.Xtest))
 

    if sparsify:
        newXtrain = newXtrain.tocsr()
        newXtest = newXtest.tocsr()

    return newfname_list, newfind_list, newXtrain, newXtest








def main():
    pass

if __name__ == "__main__":
    main()
#   def remove(myfeature)

# from sklearn import ensemble
# from sklearn import linear_model



# dfAmTrain = pd.read_csv('./train.csv')
# dfAmTest = pd.read_csv('./test.csv')