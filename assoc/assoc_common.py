import numpy as np
import pandas as pd

from common.utils.object_utils import pickle_object

class AssocPktGroup:
    def __init__(self, init_pkt=None, init_pkt_id=None):
        self.pkt_ids = np.array([], dtype=int)
        self.pkt_timestamps = np.array([], dtype=int)
        self.last_pkt = None
        if init_pkt is not None:
            self.append_pkt(init_pkt, init_pkt_id)

    def append_pkt(self, new_pkt, new_pkt_id=None):
        if new_pkt_id is None:
            new_pkt_id = new_pkt.name
        new_timestamp = new_pkt['basic', 'timestamp']
        if len(self.pkt_ids) == 0 or new_pkt_id > self.pkt_ids[-1]:
            self.pkt_ids = np.append(self.pkt_ids, new_pkt_id)
            self.pkt_timestamps = np.append(self.pkt_timestamps, new_timestamp)
            self.last_pkt = new_pkt
        else:
            inserted = False
            for i in range(len(self.pkt_ids) - 2, -1, -1):
                if self.pkt_ids[i] < new_pkt_id:
                    self.pkt_ids = np.insert(self.pkt_ids, i + 1, new_pkt_id)
                    self.pkt_timestamps = np.insert(self.pkt_timestamps, i + 1, new_timestamp)
                    inserted = True
                    break
            if not inserted:
                self.pkt_ids = np.insert(self.pkt_ids, 0, new_pkt_id)
                self.pkt_timestamps = np.insert(self.pkt_timestamps, 0, new_timestamp)

    def extend_pkt_grps(self, other_pkt_grp):
        if other_pkt_grp.last_pkt_id is None:
            return
        if self.last_pkt_id is None or self.last_pkt_id < other_pkt_grp.last_pkt_id:
            self.last_pkt = other_pkt_grp.last_pkt
        self.pkt_ids = np.hstack([self.pkt_ids, other_pkt_grp.pkt_ids])
        self.pkt_timestamps = np.hstack([self.pkt_timestamps, other_pkt_grp.pkt_timestamps])
        argsort_inds = np.argsort(self.pkt_ids)
        self.pkt_ids = self.pkt_ids[argsort_inds]
        self.pkt_timestamps = self.pkt_timestamps[argsort_inds]

    def get_pkt_id(self, ind):
        req_len = ind + 1 if ind >= 0 else -ind
        if self.pkt_ids is None or len(self.pkt_ids) < req_len:
            return None
        return self.pkt_ids[ind]

    def get_pkt_timestamp(self, ind):
        req_len = ind + 1 if ind >= 0 else -ind
        if self.pkt_timestamps is None or len(self.pkt_timestamps) < req_len:
            return None
        return self.pkt_timestamps[ind]

    @property
    def last_pkt_id(self):
        if self.pkt_ids is None or len(self.pkt_ids) == 0:
            return None
        return self.pkt_ids[-1]

    @property
    def last_timestamp(self):
        if self.pkt_timestamps is None or len(self.pkt_timestamps) == 0:
            return None
        return self.pkt_timestamps[-1]


def dump_assoc_result(assoc_pkt_grps, pkl_filepath, time_range=None):
    if time_range is not None:
        data = {'assoc_pkt_grps': assoc_pkt_grps, 'time_range': time_range}
    else:
        data = {'assoc_pkt_grps': assoc_pkt_grps}
    pickle_object(data, pkl_filepath, override=True)



