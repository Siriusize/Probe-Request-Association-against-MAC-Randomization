########## ie_model.py ##########

from inspect import isfunction

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from vmacinfer.model.prob.prob_model_common import *


def longest_common_substring_length(s1, s2):
    m, n = len(s1), len(s2)
    L = np.zeros((m + 1, n + 1), dtype=int)
    longest = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                L[i, j] = L[i - 1, j - 1] + 1
                if L[i, j] > longest:
                    longest = L[i, j]
            else:
                L[i, j] = 0
    return longest


def longest_common_subsequence_length(s1, s2):
    m, n = len(s1), len(s2)
    L = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i, j] = 0
            elif s1[i - 1] == s2[j - 1]:
                L[i, j] = L[i - 1, j - 1] + 1
            else:
                L[i, j] = max(L[i - 1, j], L[i, j-1])
    return L[m, n]


class IEAssocModel(ProbAssocModel):
    
    def __init__(self, model=None, ie_diff_func=None, **model_args):
        if model == 'decision_tree':
            self.model = DecisionTreeModelWrapper(**model_args)
        elif model == 'logistic':
            self.model = LogisticRegressionModelWrapper(**model_args)
        elif model == 'linear':
            self.model = LinearRegressionModelWrapper(**model_args)
        else:
            print('No valid model is defined.')
            
        self.compute_ie_diff = self._get_compute_ie_diff_func(ie_diff_func)
        if self.compute_ie_diff is None:
            print('No valid "compute_ie_diff" is defined.')
            
    @classmethod
    def _get_compute_ie_diff_func(cls, ie_diff_func):
        if isfunction(ie_diff_func):
            return ie_diff_func
        elif ie_diff_func is None or ie_diff_func == 'equality':
            return cls.compute_ie_diff_equality
        elif ie_diff_func == 'substring':
            return cls.compute_ie_diff_common_substring
        elif ie_diff_func == 'subsequence':
            return cls.compute_ie_diff_common_subsequence
        else:
            return None

    @staticmethod
    def compute_ie_diff_equality(ie_from, ie_to):
        # 1：both are nan | a == b
        # 0: either is nan
        # -1: both are not nan & a != b
        either_nans = pd.isnull(ie_from) | pd.isnull(ie_to)
        ie_nan = np.array(~either_nans, dtype=int)  # 0 if either is nan, 1 otherwise
        equals = (ie_from == ie_to)
        ie_same = (equals - 0.5) * 2  # 1 if both are the same, -1 otherwise
        ie_diff = ie_nan * ie_same
        ie_diff = (ie_diff + 1) / 2
        return ie_diff
    
    @staticmethod
    def compute_ie_diff_common_substring(ie_from, ie_to):
        shape = ie_from.shape
        ie_from_, ie_to_ = ie_from.reshape(-1), ie_to.reshape(-1)
        size = len(ie_from_)
        ie_diff = np.zeros(size)
        for i in range(size):
            s1, s2 = ie_from_[i], ie_to_[i]
            if pd.isnull(s1) and pd.isnull(s2):
                score = 1
            elif pd.isnull(s1) or pd.isnull(s2):
                score = 0.5
            elif len(s1) == 0 and len(s2) == 0:
                score = 1
            elif len(s1) == 0 or len(s2) == 0:
                score = 0
            else:
                common_len = longest_common_substring_length(s1, s2)
                score = common_len / len(s1)
            ie_diff[i] = score
        ie_diff = ie_diff.reshape(shape)
        return ie_diff
    
    @staticmethod
    def compute_ie_diff_common_subsequence(ie_from, ie_to):
        shape = ie_from.shape
        ie_from_, ie_to_ = ie_from.reshape(-1), ie_to.reshape(-1)
        size = len(ie_from_)
        ie_diff = np.zeros(size)
        for i in range(size):
            s1, s2 = ie_from_[i], ie_to_[i]
            if pd.isnull(s1) and pd.isnull(s2):
                score = 1
            elif pd.isnull(s1) or pd.isnull(s2):
                score = 0.5
            elif len(s1) == 0 and len(s2) == 0:
                score = 1
            elif len(s1) == 0 or len(s2) == 0:
                score = 0
            else:
                common_len = longest_common_subsequence_length(s1, s2)
                score = common_len / len(s1)
            ie_diff[i] = score
        ie_diff = ie_diff.reshape(shape)
        return ie_diff
    
#     @staticmethod
#     def compute_ie_diff(ie_from, ie_to):
#         # 1：both are not nan & a == b
#         # 0: both are nan
#         # -1: a != b or either is nan
#         both_nans = pd.isnull(ie_from) & pd.isnull(ie_to)
#         ie_nan = np.array(~both_nans, dtype=int)  # 0 if both are nan, 1 otherwise
#         equals = (ie_from == ie_to)
#         ie_same = (equals - 0.5) * 2  # 1 if both are the same, -1 otherwise
#         ie_diff = ie_nan * ie_same
#         return ie_diff
        
    def compute_feat_vecs(self, pkt_df, **kwargs):
        X = pkt_df['ie'].to_numpy()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X
    
    def compute_delta_feat_vecs(self, X_from, X_to, **kwargs):
        if self.compute_ie_diff is None:
            raise ValueError('No valid compute_ie_diff is defined in model.')
        if X_from.shape != X_to.shape:
            raise ValueError('The shape of X_from (%s) does not match to the shape of X_to (%s).' % (str(X_from.shape), str(X_to.shape)))
        if X_from.ndim == 1:  # so does X_to
            X_from = X_from.reshape(1, -1)
            X_to = X_to.reshape(1, -1)
        DeltaX = self.compute_ie_diff(X_from, X_to)
        return DeltaX
    
    def load_training_data(self, pkt_df_from=None, pkt_df_to=None, 
                           X_from=None, X_to=None, t_from=None, t_to=None, 
                           DeltaX=None, delta_t=None, y=None, 
                           **kwargs):
        self.DeltaX_train, self.delta_t_train, self.y_train = self.prepare_data(pkt_df_from=pkt_df_from, pkt_df_to=pkt_df_to, 
                                                                                X_from=X_from, X_to=X_to, t_from=t_from, t_to=t_to, 
                                                                                DeltaX=DeltaX, delta_t=delta_t, y=y, 
                                                                                **kwargs)
    
    @require_training_data
    def train(self, **kwargs):
        self.model.fit(self.DeltaX_train, self.y_train)
        print('IE model is trained.')

    @require_model
    def predict(self, pkt_df_from=None, pkt_df_to=None, 
                X_from=None, X_to=None, t_from=None, t_to=None, 
                DeltaX=None, delta_t=None, 
                **kwargs):
        DeltaX, delta_t = self.prepare_data(pkt_df_from=pkt_df_from, pkt_df_to=pkt_df_to, 
                                           X_from=X_from, X_to=X_to, t_from=t_from, t_to=t_to, 
                                           DeltaX=DeltaX, delta_t=delta_t)
        
        is_single = len(DeltaX.shape) == 1
        if is_single:
            DeltaX = DeltaX.reshape(1, -1)
        
        decisions = self.model.predict(DeltaX)[:, 1]
        
        if is_single:
            return decisions[0]
        else:
            return decisions

    @require_model
    def predict_proba(self, pkt_df_from=None, pkt_df_to=None, 
                      X_from=None, X_to=None, t_from=None, t_to=None, 
                      DeltaX=None, delta_t=None, **kwargs):
        DeltaX, delta_t = self.prepare_data(pkt_df_from=pkt_df_from, pkt_df_to=pkt_df_to, 
                                            X_from=X_from, X_to=X_to, t_from=t_from, t_to=t_to, 
                                            DeltaX=DeltaX, delta_t=delta_t)
            
        is_single = len(DeltaX.shape) == 1
        if is_single:
            DeltaX = DeltaX.reshape(1, -1)
        
        probs = self.model.predict_proba(DeltaX)
        probs = probs[:, 1]
        
        if is_single:
            return probs[0]
        else:
            return probs 


class DecisionTreeModelWrapper:

    def __init__(self, **model_args):
        self.model = DecisionTreeClassifier(**model_args)

    def fit(self, DeltaX, y):
        self.model.fit(DeltaX, y)

    def predict(self, DeltaX):
        return self.model.predict(DeltaX)

    def predict_proba(self, DeltaX):
        return self.model.predict_proba(DeltaX)


class LogisticRegressionModelWrapper:

    def __init__(self, **model_args):
        self.model = LogisticRegression(**model_args)

    def fit(self, DeltaX, y):
        self.model.fit(DeltaX, y)

    def predict(self, DeltaX):
        return self.model.predict_proba(DeltaX) >= 0.5

    def predict_proba(self, DeltaX):
        return self.model.predict_proba(DeltaX)


class LinearRegressionModelWrapper:

    def __init__(self, **model_args):
        self.model = LinearRegression(**model_args)

    def fit(self, DeltaX, y):
        self.model.fit(DeltaX, y)

    def predict(self, DeltaX):
        return self.model.predict(DeltaX) >= 0.5

    def predict_proba(self, DeltaX):
        prob = self.model.predict(DeltaX)
        return np.vstack([1 - prob, prob]).T
