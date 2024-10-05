########## sigtrans_model.py ##########

from warnings import catch_warnings
from sklearn.cluster import KMeans

from vmacinfer.model.prob.prob_model_common import *

DEFAULT_MIN_RSSI = -100
MAX_DELTA_TS = 10 * 60 * 1000


class RegionPartitioner:

    def __init__(self, n_regions=1, random_seed=1, **kwargs):
        self.n_regions = n_regions
        self.model = None
        self.random_seed = random_seed
        self.model_kwargs = kwargs

    def fit(self, X_train):
        np.random.seed(self.random_seed)
        X_train = np.nan_to_num(X_train, nan=DEFAULT_MIN_RSSI)
        self.model = KMeans(init='k-means++', n_clusters=self.n_regions, **self.model_kwargs)
        # self.model = SpectralClustering(n_clusters=self.n_regions, affinity='nearest_neighbors', assign_labels='kmeans')
        self.model.fit(X_train)

    def compute_region(self, X, min_det_aps=1):
        num_aps_from = self.compute_num_detected_aps(X)
        defect_inds = (num_aps_from < min_det_aps)
#         print('defect_inds: %d' % np.sum(defect_inds))
        try:
            X = np.nan_to_num(X, nan=DEFAULT_MIN_RSSI)
        except TypeError:
            print("the incorrect input:", str(X))
        regions = self.model.predict(X)
        regions[defect_inds] = -1
        return regions
    
    @staticmethod
    def compute_num_detected_aps(X):
        X_sig = X[:, 1:]
        nan_num = pd.isna(X_sig).sum(axis=1)
        dim = X_sig.shape[-1]
        notnan_num = dim - nan_num
        return notnan_num
    
        
class SigTransAssocModel(ProbAssocModel):
    """
    Note: DeltaX = [region_from, region_to]
    """
    
    def __init__(self, region_partitioner=None, interval_gran=MAX_DELTA_TS, random_seed=1):
        self.region_partitioner = region_partitioner
        self.interval_gran = interval_gran
        self.n_time_buckets = int(np.floor(MAX_DELTA_TS / interval_gran))
        self.random_seed = random_seed
        self.submodels = {}

    @property
    def model(self):
        return self.submodels

    @property
    def n_regions(self):
        return self.region_partitioner.n_regions
        
    def compute_feat_vecs(self, pkt_df, **kwargs):
        X = np.array(pkt_df['rssi'], dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X
    
    def compute_delta_feat_vecs(self, X_from, X_to, **kwargs):
        regions_from = self.region_partitioner.compute_region(X_from).reshape(-1, 1)
        regions_to = self.region_partitioner.compute_region(X_to).reshape(-1, 1)
        DeltaX = np.hstack([regions_from, regions_to]).astype(np.int64)  # important to convert to int as regions will be index for ndarray
        return DeltaX
    
    def load_training_data(self, pkt_df_from=None, pkt_df_to=None, 
                           X_from=None, X_to=None, t_from=None, t_to=None, 
                           DeltaX=None, delta_t=None, y=None, 
                           **kwargs):
        self.DeltaX_train, self.delta_t_train, self.y_train = self.prepare_data(pkt_df_from=pkt_df_from, pkt_df_to=pkt_df_to, 
                                                                                X_from=X_from, X_to=X_to, t_from=t_from, t_to=t_to, 
                                                                                DeltaX=DeltaX, delta_t=delta_t, y=y, 
                                                                                **kwargs)
        
    def _compute_bucket_id(self, delta_t, min_delta_t=-1):
        if isinstance(delta_t, int):
            return delta_t // self.interval_gran if delta_t >= min_delta_t else -1
        else:
            small_inds = delta_t < min_delta_t
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
            DeltaX_buckets[bucket_id] = DeltaX[inds, :]
        return DeltaX_buckets
    
    @require_training_data
    def train(self, **kwargs):
        np.random.seed(self.random_seed)

        DeltaX_positive = self.DeltaX_train[self.y_train]  # only use the data with positive labels (associated)
        delta_t_positive = self.delta_t_train[self.y_train]
        print('DeltaX_positive_train: %d' % len(DeltaX_positive))
        valid_inds = np.all(DeltaX_positive >= 0, axis=1)
        valid_DeltaX_positive = DeltaX_positive[valid_inds]
        valid_delta_t_positive = delta_t_positive[valid_inds]
        print('valid_DeltaX_positive_train: %d' % len(valid_DeltaX_positive))

        # segregate data
        DeltaX_buckets = self._segregate_data(DeltaX_positive, delta_t_positive)

        # train
        self.submodels = {}
        for bucket_id, DeltaX_in_bucket in DeltaX_buckets.items():
            submodel = RegionTransSubmodel(n_regions=self.n_regions)
            submodel.fit(DeltaX_in_bucket)
            self.submodels[bucket_id] = submodel
        print('Model is trained (n_submodels = %s).' % len(self.submodels))

    @require_model
    def predict(self, pkt_df_from=None, pkt_df_to=None, **kwargs):
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
                delta_x = DeltaX[i, :]
                prob = submodel.predict_proba(delta_x.reshape(-1, 2))[0]
            probs[i] = prob
        return probs


class RegionTransSubmodel:

    def __init__(self, n_regions):
        self.n_regions = n_regions
        self.trans_counts = np.zeros((n_regions, n_regions), dtype=int)
        self.joint_prob_matrix = np.ones((n_regions, n_regions)) / (n_regions * n_regions)
        self.trans_prob_matrix = np.ones((n_regions, n_regions)) / n_regions

    def fit(self, DeltaX):
        n = len(DeltaX)
        for i in range(n):
            region_from, region_to = DeltaX[i, :]
            self.trans_counts[region_from, region_to] += 1
        self.joint_prob_matrix = self.trans_counts / np.sum(self.trans_counts)
        self.trans_prob_matrix = self.trans_counts / np.sum(self.trans_counts, axis=1).reshape(-1, 1)

    def predict_proba(self, DeltaX):
        n = len(DeltaX)
        probs = np.zeros(n)
        for i in range(n):
            region_from, region_to = DeltaX[i, :]
            probs[i] = self.trans_prob_matrix[region_from, region_to]
        return probs