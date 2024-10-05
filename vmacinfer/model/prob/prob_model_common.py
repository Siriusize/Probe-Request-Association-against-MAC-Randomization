from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
import warnings

import numpy as np
import pandas as pd


def require_model(func):
        
    def wrapper(cls, *args, **kwargs):
        if cls.model is None or (isinstance(cls.model, Iterable) and len(cls.model) == 0):
            print('Model is not defined.')
            return None
        else:
            return func(cls, *args, **kwargs)

    return wrapper


def require_training_data(func):
        
    def wrapper(cls, *args, **kwargs):
        if cls.DeltaX_train is None or cls.y_train is None:
            print('Training data is not available.')
            return None
        else:
            return func(cls, *args, **kwargs)

    return wrapper


class ProbAssocModel:
    
#     def __init__(self):
#         self.model = None
#         self.DeltaX_train = None
#         self.y_train = None

#     def prepare_data(self, *members):
#         """
#         Check whether the number of elements in the given members are equal, and (if so) convert all members to ndarrays if they are not.
#         """
#         num_members = len(members)
#         lens = [len(member) for member in members]
#         for i in range(1, num_members):
#             if lens[i] != lens[i - 1]:
#                 raise ValueError('The lengths are not equal. (lens = %s).' % (str(lens)))
#         ndarray_members = [np.array(member) for member in members]
#         return tuple(ndarray_args)

    @abstractmethod
    def load_training_data(self, pkt_df_from=None, pkt_df_to=None, **kwargs): pass
    
    @abstractmethod
    def train(self, **kwargs): pass
    
    @abstractmethod
    def predict(self, pkt_df_from=None, pkt_df_to=None, **kwargs): pass

    @abstractmethod
    def predict_proba(self, pkt_df_from=None, pkt_df_to=None, **kwargs): pass
    
    @staticmethod
    @abstractmethod
    def compute_feat_vecs(pkt_df, **kwargs): pass
    
    @staticmethod
    @abstractmethod
    def compute_delta_feat_vecs(X_from, X_to, **kwargs): pass
    
    @staticmethod
    def compute_ts_vecs(pkt_df):
        return np.array(pkt_df[('basic', 'timestamp')])
    
    @staticmethod
    def compute_delta_ts_vecs(t_from, t_to):
        if t_from is None or t_to is None:
#             warnings.warn('Either t_from or t_to is missing. Return delta_t as None.')
            delta_t = None
        else:
            delta_t = t_to - t_from
        return delta_t
    
    def prepare_data(self, pkt_df_from=None, pkt_df_to=None, X_from=None, X_to=None, t_from=None, t_to=None, DeltaX=None, delta_t=None, y=None, **kwargs):
        if DeltaX is None:
            if X_from is None or X_to is None:
                if pkt_df_from is None or pkt_df_to is None:
                    raise ValueError('Cannot obtain DeltaX because both X and pkt_df are missing.')
                # convert pkt_df to X
                X_from = self.compute_feat_vecs(pkt_df_from, **kwargs)
                X_to = self.compute_feat_vecs(pkt_df_to, **kwargs)
                # process dimension
                if len(X_from.shape) <= 1:  # The shape is () if it is a scaler, (?, ) if it is a 1D array.
                    X_from = X_from.reshape((1, -1))
                if len(X_to.shape) <= 1:
                    X_to = X_to.reshape((1, -1))
                # Till now, both X_from and X_to should be 2D.
                if len(X_from) != len(X_to):
                    if len(X_from) == 1:
                        X_from = np.repeat(X_from, len(X_to), axis=0)
                    elif len(X_to) == 1:
                        X_to = np.repeat(X_to, len(X_from), axis=0)
                    else:
                        raise ValueError('X_from and X_to do not have same lengths (X_from: %d, X_to: %d).' % (len(X_from), len(X_to)))
                # Till now, both X_from and X_to should have the same length.
            # convert X to DeltaX
            DeltaX = self.compute_delta_feat_vecs(X_from, X_to, **kwargs)
        if delta_t is None:
            if t_from is None or t_to is None:
                if pkt_df_from is None or pkt_df_to is None:
                    warnings.warn('Cannot obtain delta_t because both X and pkt_df are missing.')  # Do not throw an error because it is allowed to have no delta_t.
                    t_from, t_to = None, None
                else:
                    t_from = self.compute_ts_vecs(pkt_df_from)
                    t_to = self.compute_ts_vecs(pkt_df_to)
            delta_t = self.compute_delta_ts_vecs(t_from, t_to)
        if y is None:
            return DeltaX, delta_t
        else:
            y = np.array(y)
            return DeltaX, delta_t, y

#     def generate_feat_diff_data(self, feat_from_data, feat_to_data, compute_feat_diff):
#         """
#         Generate feat-diff data
#         """
#         n = len(feat_from_data)
#         feat_diff_list = [None] * n
#         for i in range(n):
#             feat_from = feat_from_data.iloc[i]
#             feat_to = feat_to_data.iloc[i]
#             feat_diff = compute_feat_diff(feat_from, feat_to)
#             feat_diff_list[i] = feat_diff
#         feat_diff_df = pd.DataFrame(feat_diff_list, columns=feat_diff_list[0].index)
#         return feat_diff_df
