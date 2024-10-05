########## unified_model.py ##########

import time
# import torch.nn as nn

from model.prob.prob_model_common import *


class UnifiedAssocModel(ProbAssocModel):

    def __init__(self, ie_model=None, seq_model=None, sigtrans_model=None, X_dims=None, time_decay_factor=None):
        self.ie_model = ie_model
        self.seq_model = seq_model
        self.sigtrans_model = sigtrans_model
        self.X_dims = X_dims
        self.time_decay_factor = time_decay_factor

    @property
    def model(self):
        if self.ie_model is None and self.seq_model is None and self.sigtrans_model is None:
            return None
        else:
            return [self.ie_model, self.seq_model, self.sigtrans_model]
    
    def extract_sub_X(self, X):
        X_ie = X[:, :self.X_dims[0]]
        X_seq = X[:, self.X_dims[0]:self.X_dims[0] + self.X_dims[1]]
        X_sigtrans = X[:, self.X_dims[0] + self.X_dims[1]:]
        return X_ie, X_seq, X_sigtrans
        
    def compute_feat_vecs(self, pkt_df, **kwargs):
        if pkt_df.ndim == 1:
            n = 1
        else:
            n = pkt_df.shape[0]
        if self.ie_model is not None:
            X_ie = self.ie_model.compute_feat_vecs(pkt_df, **kwargs).reshape(-1, self.X_dims[0])
        else:
            X_ie = np.zeros((n, self.X_dims[0]))
        if self.seq_model is not None:
            X_seq = self.seq_model.compute_feat_vecs(pkt_df, **kwargs).reshape(-1, self.X_dims[1])
        else:
            X_seq = np.zeros((n, self.X_dims[1]))
        if self.sigtrans_model is not None:
            X_sigtrans = self.sigtrans_model.compute_feat_vecs(pkt_df, **kwargs).reshape(-1, self.X_dims[2])
        else:
            X_sigtrans = np.zeros((n, self.X_dims[2]))
        X = np.hstack([X_ie, X_seq, X_sigtrans])
        return X
    
    def compute_delta_feat_vecs(self, X_from, X_to, **kwargs):
        X_ie_from, X_seq_from, X_sigtrans_from = self.extract_sub_X(X_from)
        X_ie_to, X_seq_to, X_sigtrans_to = self.extract_sub_X(X_to)
        X_sigtrans_from = X_sigtrans_from.astype(np.float64)
        X_sigtrans_to = X_sigtrans_to.astype(np.float64)
        
        if self.ie_model is not None:
            DeltaX_ie = self.ie_model.compute_delta_feat_vecs(X_ie_from, X_ie_to, **kwargs).reshape(X_ie_from.shape)
        else:
            DeltaX_ie = np.zeros(X_ie_from.shape)
        if self.seq_model is not None:
            DeltaX_seq = self.seq_model.compute_delta_feat_vecs(X_seq_from, X_seq_to, **kwargs).reshape(X_seq_from.shape)
        else:
            DeltaX_seq = np.zeros(X_seq_from.shape)
        if self.sigtrans_model is not None:
            DeltaX_sigtrans = self.sigtrans_model.compute_delta_feat_vecs(X_sigtrans_from, X_sigtrans_to, **kwargs).reshape(-1, 2)
        else:
            DeltaX_sigtrans = np.zeros(X_sigtrans_from.shape)
        DeltaX = np.hstack([DeltaX_ie, DeltaX_seq, DeltaX_sigtrans])
        return DeltaX
    
    def load_training_data(self, pkt_df_from=None, pkt_df_to=None, 
                           X_from=None, X_to=None, t_from=None, t_to=None, 
                           DeltaX=None, delta_t=None, y=None, 
                           **kwargs):
        self.DeltaX_train, self.delta_t_train, self.y_train = self.prepare_data(pkt_df_from=pkt_df_from, pkt_df_to=pkt_df_to, 
                                                                                X_from=X_from, X_to=X_to, t_from=t_from, t_to=t_to, 
                                                                                DeltaX=DeltaX, delta_t=delta_t, 
                                                                                y=y)
        DeltaX_ie, DeltaX_seq, DeltaX_sigtrans = self.extract_sub_X(self.DeltaX_train)
        DeltaX_sigtrans = DeltaX_sigtrans.astype(int)
        
        if self.ie_model is not None:
            print("Preprocessing data for IE model...")
            self.ie_model.load_training_data(DeltaX=DeltaX_ie, delta_t=self.delta_t_train, y=self.y_train)
        if self.seq_model is not None:
            print("Preprocessing data for sequence number model...")
            self.seq_model.load_training_data(DeltaX=DeltaX_seq, delta_t=self.delta_t_train, y=self.y_train)
        if self.sigtrans_model is not None:
            print("Preprocessing data for signal transition model...")
            self.sigtrans_model.load_training_data(DeltaX=DeltaX_sigtrans, delta_t=self.delta_t_train, y=self.y_train)
    
#     @require_training_data
    def train(self, **kwargs):
        if self.ie_model is not None:
            print("Training IE model...")
            self.ie_model.train()
        if self.seq_model is not None:
            print("Training Sequence number model...")
            self.seq_model.train()
        if self.sigtrans_model is not None:
            print("Training signal transition model...")
            self.sigtrans_model.train()

    @require_model
    def predict(self, pkt_df_from=None, pkt_df_to=None, **kwargs):
        pass

    def predict_proba(self, pkt_df_from=None, pkt_df_to=None, 
                      X_from=None, X_to=None, t_from=None, t_to=None, 
                      DeltaX=None, delta_t=None, 
                      **kwargs):
        DeltaX, delta_t = self.prepare_data(pkt_df_from=pkt_df_from, pkt_df_to=pkt_df_to, 
                                            X_from=X_from, X_to=X_to, t_from=t_from, t_to=t_to, 
                                            DeltaX=DeltaX, delta_t=delta_t)
        DeltaX_ie, DeltaX_seq, DeltaX_sigtrans = self.extract_sub_X(DeltaX)
        DeltaX_seq = DeltaX_seq.astype(int)
        DeltaX_sigtrans = DeltaX_sigtrans.astype(int)

        # IE model
        if self.ie_model is not None and DeltaX_ie is not None:
            t1 = time.time()
            pred_y_probs_ie = self.ie_model.predict_proba(DeltaX=DeltaX_ie, delta_t=delta_t)
            t2 = time.time()
#             print('ie: %dms' % ((t2 - t1) * 1000))
        else:
            pred_y_probs_ie = 1
            
        # Sequence model
        if self.seq_model is not None and DeltaX_seq is not None:
            t3 = time.time()
            pred_y_probs_seq = self.seq_model.predict_proba(DeltaX=DeltaX_seq, delta_t=delta_t)
            t4 = time.time()
#             print('seq: %dms' % ((t4 - t3) * 1000))
        else:
            pred_y_probs_seq = 1
            
        # Signal transition model
        if self.sigtrans_model is not None and DeltaX_sigtrans is not None:
            t5 = time.time()
            pred_y_probs_sigtrans = self.sigtrans_model.predict_proba(DeltaX=DeltaX_sigtrans, delta_t=delta_t)                 
            t6 = time.time()
#             print('sigtrans: %dms' % ((t6 - t5) * 1000))
        else:
            pred_y_probs_sigtrans = 1
            
        pred_y_probs = pred_y_probs_ie * pred_y_probs_seq * pred_y_probs_sigtrans

        if self.time_decay_factor is not None:
            time_decay = np.exp(-self.time_decay_factor * delta_t)
            pred_y_probs = pred_y_probs * time_decay

        return pred_y_probs


