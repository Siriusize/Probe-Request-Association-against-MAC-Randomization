########## seq_model.py ##########

from scipy import integrate
from sklearn.mixture import GaussianMixture

from vmacinfer.model.prob.prob_model_common import *


class SeqAssocModel(ProbAssocModel):
    
    def __init__(self, interval_gran=np.inf, n_modes=1,
                 seq_upper_bound=np.inf, rand_seed=1, **model_args):
        self.interval_gran = interval_gran
        self.n_modes = n_modes
        self.seq_upper_bound = seq_upper_bound
        self.submodels = None
        self.rand_seed = rand_seed

        self.DeltaX_train = None
        self.delta_t_train = None
        self.y_train = None
        
    @property
    def model(self):
        return self.submodels
    
    @staticmethod
    def _compute_delta_seqns(seqn_from, seqn_to):
        delta_seqns = seqn_to - seqn_from
#         delta_seqns = delta_seqns.flatten()
        delta_seqns[delta_seqns < 0] += 4096
        return delta_seqns
    
    @staticmethod
    def _compute_delta_ts(time_from, time_to):
        return time_to - time_from
        
    @staticmethod
    def compute_feat_vecs(pkt_df, **kwargs):
        X = np.array(pkt_df[('seq', 'seq')])
        X = X.reshape(-1, 1)
        return X
    
    @staticmethod
    def compute_delta_feat_vecs(X_from, X_to, **kwargs):
        if X_from.shape != X_to.shape:
            raise ValueError('The shape of X_from (%s) does not match to the shape of X_to (%s).' % (str(X_from.shape), str(X_to.shape)))
#         delta_ts = SeqAssocModel._compute_delta_ts(X_from[:, 0], X_to[:, 0])
#         delta_seqns = SeqAssocModel._compute_delta_seqns(X_from[:, 1], X_to[:, 1])
#         DeltaX = np.vstack([delta_ts, delta_seqns]).T
        DeltaX = SeqAssocModel._compute_delta_seqns(X_from, X_to)
        return DeltaX
    
    def load_training_data(self, pkt_df_from=None, pkt_df_to=None, 
                           X_from=None, X_to=None, T_from=None, T_to=None, 
                           DeltaX=None, DeltaT=None, y=None, 
                           **kwargs):
        self.DeltaX_train, self.delta_t_train, self.y_train = self.prepare_data(pkt_df_from=pkt_df_from, pkt_df_to=pkt_df_to, 
                                                                                X_from=X_from, X_to=X_to, T_from=T_from, T_to=T_to, 
                                                                                DeltaX=DeltaX, DeltaT=DeltaT, y=y, 
                                                                                **kwargs)

    def _compute_bucket_id(self, delta_t, min_delta_t=-1):
        if isinstance(delta_t, int):
            return delta_t // self.interval_gran if delta_t > min_delta_t else -1
        else:
            small_inds = delta_t <= min_delta_t
            bucket_ids = delta_t // self.interval_gran
            bucket_ids[small_inds] = -1
            return bucket_ids

    def _select_submodel(self, delta_t):
        bucket_id = self._compute_bucket_id(delta_t)
        submodel_id = bucket_id
        return self.submodels[submodel_id]

    def _segregate_data(self, DeltaX, delta_t):
        bucket_ids = self._compute_bucket_id(delta_t)
        DeltaX_buckets = {}
        for bucket_id in set(bucket_ids):
            inds = np.where(bucket_ids == bucket_id)[0]
            DeltaX_buckets[bucket_id] = DeltaX[inds]
        return DeltaX_buckets
    
    @require_training_data
    def train(self, **kwargs):
        np.random.seed(self.rand_seed)
#         mask = seq_diff < self.seq_upper_bound
#         mask = mask.reshape(-1)
#         bounded_seq_diff = seq_diff[mask]
        DeltaX_positive = self.DeltaX_train[np.where(self.y_train)]  # only use the data with positive labels (associated)
        delta_t_positive = self.delta_t_train[np.where(self.y_train)]  # only use the data with positive labels (associated)
        print('DeltaX_positive = %d' % len(DeltaX_positive))

        # segregate data
        DeltaX_buckets = self._segregate_data(DeltaX_positive, delta_t_positive)

        # train
        self.submodels = {}
        # counts = {}
        for bucket_id, DeltaX_in_bucket in DeltaX_buckets.items():
            n_samples = len(DeltaX_in_bucket)
            if n_samples < self.n_modes:
                continue
            # submodel = GaussianMixture(n_components=self.n_modes, covariance_type='full', verbose=0, tol=1e-3)
            submodel = ProbDistriSeqSubModelProxy(delta_seq_range=(0, 4096),
                                                  n_components=self.n_modes, covariance_type='full', verbose=0, tol=1e-3)
            submodel.fit(DeltaX_in_bucket.reshape(-1, 1))
            self.submodels[bucket_id] = submodel
            # counts[bucket_id] = n_samples
#             print('GMM (clus_id = %d) is trained.' % clus_id)
#         total_count = np.sum(list(counts.values()))
        # The weight is unnecessary. The prob has already been P(\Delta s | \Delta t). 
        # That is, given \Delta t, we can pick a model and predict the probability of \Delta s.
#         self.weights = {bucket_id: 1 / len(counts) / (count / total_count) for bucket_id, count in counts.items()}
        print('Model is trained (n_submodels = %d).' % len(self.submodels))
#         print('weights: %s' % str(self.weights))

    @require_model
    def predict(self, pkt_df_from=None, pkt_df_to=None, 
                X_from=None, X_to=None, t_from=None, t_to=None, 
                DeltaX=None, delta_t=None, 
                **kwargs):
        pass
        
    @require_model
    def predict_proba(self, pkt_df_from=None, pkt_df_to=None, 
                      X_from=None, X_to=None, t_from=None, t_to=None, 
                      DeltaX=None, delta_t=None, 
                      **kwargs):
        DeltaX, delta_t = self.prepare_data(pkt_df_from=pkt_df_from, pkt_df_to=pkt_df_to,
                                            X_from=X_from, X_to=X_to, t_from=t_from, t_to=t_to,
                                            DeltaX=DeltaX, delta_t=delta_t)

        bucket_ids = self._compute_bucket_id(delta_t)
        probs = np.zeros(len(DeltaX))
        for i in range(len(DeltaX)):
            bucket_id = bucket_ids[i]
            submodel = self.submodels.get(bucket_id, None)
            if submodel is None:
                prob = 0.
            else:
                # log_prob = submodel.score_samples(DeltaX[i].reshape(-1, 1))
                # prob = np.exp(log_prob)# * self.weights[clus_id]
                delta_x = int(DeltaX[i])
                prob = submodel.predict_proba(np.array(delta_x).reshape(1, 1))[0]
            probs[i] = prob
        return probs


class ProbDensitySeqSubModelProxy:

    def __init__(self, **embedded_model_kwargs):
        self.embedded_model = GaussianMixture(**embedded_model_kwargs)

    def fit(self, X):
        self.embedded_model.fit(X)

    def predict_proba(self, X):
        return np.exp(self.embedded_model.score_samples([[X]])[0])


class ProbDistriSeqSubModelProxy:

    def __init__(self, delta_seq_range, **embedded_model_kwargs):
        self.delta_seq_range = delta_seq_range      # [a, b)
        self.embedded_model = GaussianMixture(**embedded_model_kwargs)
        self.prob_distribution = np.zeros(delta_seq_range[1] - delta_seq_range[0])

    def fit(self, X):
        self.embedded_model.fit(X)
        for i, val in enumerate(range(self.delta_seq_range[0], self.delta_seq_range[1])):
            self.prob_distribution[i] = integrate.quad(self.predict_proba_density, val - 0.5, val + 0.5)[0]
        self.prob_distribution /= np.sum(self.prob_distribution)    # normalization

    def predict_proba_density(self, X):     # X is a 2D array
        if isinstance(X, int) or isinstance(X, float):
            X = np.array([[X]])
            single = True
        probas = np.exp(self.embedded_model.score_samples(X))
        if single:
            return probas[0]
        else:
            return probas

    def predict_proba(self, X):     # X is a 2D array
        X = X.flatten()
        probas = np.zeros(len(X))
        in_range_mask = (X >= self.delta_seq_range[0]) & (X < self.delta_seq_range[1])
        X_in_range = X[in_range_mask]
        probas_in_range = np.zeros(len(X_in_range))
        for i, x in enumerate(X_in_range):
            probas_in_range[i] = self.prob_distribution[x - self.delta_seq_range[0]]
        probas[in_range_mask] = probas_in_range
        return probas


if __name__ == '__main__':
    # test the function of ProbDistriSeqSubModelWrapper
    submodel = ProbDistriSeqSubModelProxy(delta_seq_range=(0, 101), n_components=2, covariance_type='full', verbose=0, tol=1e-3)
    X = np.array([1, 3, 5, 18, 19, 25]).reshape(-1, 1)
    submodel.fit(X)
    X_test = np.array([list(range(-100, 200))]).reshape(-1, 1)
    probas = submodel.predict_proba(X_test)
    print(np.sum(probas))