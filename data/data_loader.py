import binascii
import datetime
import os
import re
from collections import deque

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import env
from common.pcap import ieee80211


############### RAW DATA ###############

def _load_raw_pkt_df(filename, data_dirpath=None, sort=True):
    if data_dirpath is None or data_dirpath == 'env':
        data_dirpath = os.path.join(env.get_working_dir(), 'data')
    rawdata_filepath = os.path.join(data_dirpath, filename)
    print('Loading raw packets from %s...' % rawdata_filepath)
    raw_pkt_df = pd.read_csv(rawdata_filepath)# , dtype={"info_element": str})
    print('%d entries are loaded from %s.' % (len(raw_pkt_df), filename))

    # sort
    if sort:
        raw_pkt_df = raw_pkt_df.sort_values(['timestamp', 'seq'], ascending=True)

    return raw_pkt_df


def _filter_raw_pkt_df_by_time_range(raw_pkt_df, time_range=None, is_relative_time_range=False):
    has_valid_time_range = time_range is not None and len(time_range) == 2
    if is_relative_time_range == 'auto':
        if has_valid_time_range:
            is_relative_time_range = time_range[0] < 946684800000
        else:
            is_relative_time_range = False
    if not isinstance(is_relative_time_range, bool):
        raise TypeError('Invalid argument of "is_relative_time_range" (%s).' % str(is_relative_time_range))
        
    # select data within time range
    if has_valid_time_range:
        if is_relative_time_range:
            base_ts = raw_pkt_df.iloc[0]['timestamp']
            # modify to 0:0:0
            base_date = datetime.date.fromtimestamp(base_ts / 1000)
            base_ts = int(datetime.datetime(base_date.year, base_date.month, base_date.day).timestamp() * 1000)
            time_range = (base_ts + time_range[0], base_ts + time_range[1])
        fltr_raw_pkt_df = raw_pkt_df.loc[(raw_pkt_df['timestamp'] >= time_range[0]) & (raw_pkt_df['timestamp'] <= time_range[1])]
    else:
        fltr_raw_pkt_df = raw_pkt_df

    print('%d entries are obtained after time filtering (%.1f%%).'
          % (len(fltr_raw_pkt_df), len(fltr_raw_pkt_df) / len(raw_pkt_df) * 100))

    return fltr_raw_pkt_df


def _filter_raw_pkt_df_by_category(raw_pkt_df, category=None):
    if category == 'phy' or category == 'physical':
        # select all the data without randomization
        fltr_raw_pkt_df = raw_pkt_df[~raw_pkt_df['is_virtual_mac']]
    elif category == 'virt' or category == 'virtual':
        fltr_raw_pkt_df = raw_pkt_df[raw_pkt_df['is_virtual_mac']]
    elif category == 'all' or category is None:
        fltr_raw_pkt_df = raw_pkt_df
    else:
        raise TypeError('Invalid argument of "category" (%s).' % str(category))

    print('%d entries are obtained after category filtering (%.1f%%).'
          % (len(fltr_raw_pkt_df), len(fltr_raw_pkt_df) / len(raw_pkt_df) * 100))

    return fltr_raw_pkt_df


def _filter_raw_pkt_df_by_duration(raw_pkt_df):
    # generate valid mac addresses
    mac_grp_gdf = raw_pkt_df[['timestamp']].groupby(raw_pkt_df['tx_addr'])
    mac_grp_ts_agg_df = mac_grp_gdf['timestamp'].agg([
        ('probe_count', 'count'),
        ('sec_count', lambda c: (c // 1000).nunique()),
        ('duration_sec', lambda c: (c.max() - c.min()) / 1000),
    ])
    mac_grp_ts_agg_df = mac_grp_ts_agg_df[
        (mac_grp_ts_agg_df['sec_count'] > 1)
        # & (mac_grp_ts_agg_df['duration_sec'] >= 2)
        # & (mac_grp_ts_agg_df['duration_sec'] <= 3600)
        ]
    valid_macs = set(mac_grp_ts_agg_df.index)

    fltr_raw_pkt_df = raw_pkt_df[[mac in valid_macs for mac in raw_pkt_df['tx_addr']]]

    print('%d entries are obtained after duration filtering (%.1f%%).'
          % (len(fltr_raw_pkt_df), len(fltr_raw_pkt_df) / len(raw_pkt_df) * 100))

    return fltr_raw_pkt_df


# generate pkt from data frame
def _generate_pkt_df(raw_pkt_df, keep_origin_index=False):
    
    basic_indices = ['timestamp', 'tx_addr']
    basic_indices_parent = ['basic'] * len(basic_indices)
    
    ie_indices = ['ie%d' % ie_id for ie_id in ieee80211.IE_ID_LIST]
    ie_indices_parent = ['ie'] * len(ie_indices)
    
    seq_indices = ['seq']
    seq_indices_parent = ['seq']
    
    original_rssi_indices = [col for col in raw_pkt_df.columns if re.match('rssi@.+', col)]
    rssi_indices = [col.replace('rssi@', '') for col in original_rssi_indices]
    rssi_indices_parent = ['rssi'] * len(rssi_indices)
    
    def extract_basic_sub_df(raw_pkt_df):
        sub_df = raw_pkt_df[['timestamp', 'tx_addr']]
        sub_df.columns = [basic_indices_parent, basic_indices]
        return sub_df
    
    def extract_ie_sub_df(raw_pkt_df):
        all_ies = []
        for _, ie_hex in raw_pkt_df['info_element'].items():
            ie_dict = {}
            ie_bytes = binascii.unhexlify(ie_hex)
            _, ie = ieee80211.ieee80211_mgt_parse(ie_bytes, skip_last_len=0)
            for ie_id, ie_val in ie.items():
                ie_dict[('ie', 'ie%d' % ie_id)] = ie_val.hex()
            all_ies.append(ie_dict)
        sub_df = pd.DataFrame(all_ies, index=raw_pkt_df.index, columns=list(zip(ie_indices_parent, ie_indices)))
#         ordered_columns = sorted(sub_df.columns, key=lambda col: int(col[1].replace('ie', '')))
#         sub_df = sub_df[ordered_columns]
#         sub_df = sub_df.dropna(axis=1, how='all')
        return sub_df
    
    def extract_seq_sub_df(raw_pkt_df):
        sub_df = raw_pkt_df[['seq']]
        sub_df.columns = [seq_indices_parent, seq_indices]
        return sub_df
    
    def extract_rssi_sub_df(raw_pkt_df):
        sub_df = raw_pkt_df[original_rssi_indices]
        sub_df.columns = [rssi_indices_parent, rssi_indices]
        return sub_df
    
    basic_sub_df = extract_basic_sub_df(raw_pkt_df)
    ie_sub_df = extract_ie_sub_df(raw_pkt_df)
    seq_sub_df = extract_seq_sub_df(raw_pkt_df)
    rssi_sub_df = extract_rssi_sub_df(raw_pkt_df)

    pkt_df = pd.concat([basic_sub_df, ie_sub_df, seq_sub_df, rssi_sub_df], axis=1)
    if not keep_origin_index:
        pkt_df.reset_index(drop=True)
        
    return pkt_df


def _merge_pkts_within_interval(pkt_df, interval=1000):
    """
    Merge packets if
    1) they have the same tx_addr,
    2) they have the same ie,
    3) their time difference is less than interval
    :param pkt_df:
    :param interval:
    :return:
    """

    def within_interval(pkt_1, pkt_2):
        return abs(pkt_2[('basic', 'timestamp')] - pkt_1[('basic', 'timestamp')]) <= interval

    def need_merge(pkt_1, pkt_2):
        return pkt_1[('basic', 'tx_addr')] == pkt_2[('basic', 'tx_addr')]

    cache = deque()
    new_pkt_list = []
    for i, pkt in pkt_df.iterrows():
        while len(cache) > 0 and not within_interval(cache[0], pkt):
            new_pkt_list.append(cache.popleft())
        insert_flag = True
        for cache_pkt in cache:     # all the element in cache are within interval to pkt
            if need_merge(cache_pkt, pkt):
                insert_flag = False
                break
        if insert_flag:
            cache.append(pkt)
        # print("{}/{} merged".format(i, len(pkt_df)))
    while len(cache) > 0:       # remaining
        new_pkt_list.append(cache.popleft())
    new_pkt_df = pd.DataFrame(new_pkt_list)

    print('%d out of %d packets are obtained after merging (%.1f%%).'
          % (len(new_pkt_df), len(pkt_df), len(new_pkt_df) / len(pkt_df) * 100))

    return new_pkt_df


def prepare_rawdata(filename, data_dirpath=None,
                    time_range=None, is_relative_time_range='auto', category=None,
                    merge=True):
    """
    Load raw data from specified file, and filter data accordingly.
    :param filename:
    :param time_range:
    :param is_relative_time_range:
    :param category:
    :return: A DataFrame of filtered packet data.
    """
    # read in as data frame
    raw_pkt_df = _load_raw_pkt_df(filename, data_dirpath=data_dirpath)
    # differentiate virtual and real data
    raw_pkt_df = _filter_raw_pkt_df_by_category(raw_pkt_df, category=category)
    # filter out the pkts that is always there (from site beacon)
    raw_pkt_df = _filter_raw_pkt_df_by_duration(raw_pkt_df)
    # get data from a particular time range
    raw_pkt_df = _filter_raw_pkt_df_by_time_range(raw_pkt_df, time_range=time_range, is_relative_time_range=is_relative_time_range)
    pkt_df = _generate_pkt_df(raw_pkt_df)
    # pkt_df.to_csv("./data/hwdata/result/dp2.csv")

    if merge:
        pkt_df = _merge_pkts_within_interval(pkt_df, interval=4000)
    return pkt_df

############### END OF RAW DATA ###############


############### DATASET EXTACTION ###############

def _find_all_same_mac_precurs(cur_pkt, ctx_pkt_df, sort_by_time=True):
    cur_mac = cur_pkt['basic', 'tx_addr']
    ctx_pkt_df_wiz_same_mac = ctx_pkt_df[ctx_pkt_df['basic', 'tx_addr'] == cur_mac]
    if sort_by_time:
        ctx_pkt_df_wiz_same_mac = ctx_pkt_df_wiz_same_mac.sort_values(('basic', 'timestamp'), ascending=False)
    return ctx_pkt_df_wiz_same_mac


def _find_direct_precur(cur_pkt, ctx_pkt_df):
    ctx_pkt_df_wiz_same_mac = _find_all_same_mac_precurs(cur_pkt, ctx_pkt_df)
    if len(ctx_pkt_df_wiz_same_mac) > 0:
        return ctx_pkt_df_wiz_same_mac.iloc[0]
    else:
        return None


def _find_arbitrary_same_mac_precur(cur_pkt, ctx_pkt_df):
    ctx_pkt_df_wiz_same_mac = _find_all_same_mac_precurs(cur_pkt, ctx_pkt_df, sort_by_time=False)
    if len(ctx_pkt_df_wiz_same_mac) > 0:
        return ctx_pkt_df_wiz_same_mac.sample(n=1).iloc[0]
    else:
        return None


def _find_arbitrary_diff_mac_precur(cur_pkt, ctx_pkt_df):
    cur_mac = cur_pkt['basic', 'tx_addr']
    ctx_pkt_df_wiz_diff_mac = ctx_pkt_df[ctx_pkt_df['basic', 'tx_addr'] != cur_mac]
    if len(ctx_pkt_df_wiz_diff_mac) > 0:
        return ctx_pkt_df_wiz_diff_mac.sample(n=1).iloc[0]
    else:
        return None

    
def _find_arbitrary_precur(cur_pkt, ctx_pkt_df):
    if len(ctx_pkt_df) > 0:
        return ctx_pkt_df.sample(n=1).iloc[0]
    else:
        return None


def generate_random_trans_pair(pkt_df, num_samples, 
                               ctx_time_range=(1000, 300*1000), 
                               positive_ratio=0.5, random_seed=1):
    """
    Generate randomly sampld transition pairs.
    :param pkt_df:
    :param num_samples:
    :param ctx_time_range:
    :param positive_ratio:
    :param random_seed:
    :return: A generator of sampled transition pairs.
    """
    np.random.seed(random_seed)
    num_samples_positive = positive_ratio * num_samples
    num_generated, num_generated_positive = 0, 0
    while num_generated < num_samples:
        cur_pkt = pkt_df.sample(n=1).iloc[0]
        cur_ts = cur_pkt['basic', 'timestamp']
        ctx_pkt_df = pkt_df[
            (cur_ts - pkt_df['basic', 'timestamp'] > ctx_time_range[0]) &
            (cur_ts - pkt_df['basic', 'timestamp'] < ctx_time_range[1])
        ]
        # label = np.random.rand() < positive_ratio
        label = num_generated_positive < num_samples_positive
        if label:
            pkt_from = _find_direct_precur(cur_pkt, ctx_pkt_df)
        else:
            pkt_from = _find_arbitrary_diff_mac_precur(cur_pkt, ctx_pkt_df)
        if pkt_from is not None:
            yield (pkt_from, cur_pkt, label)
            num_generated += 1
            if label:
                num_generated_positive += 1


def generate_random_trans_pair_dataset(pkt_df, num_samples, 
                                       ctx_time_range=(1000, 300*1000), 
                                       positive_ratio=0.5, random_seed=1):
    dataset = list(generate_random_trans_pair(pkt_df, num_samples, ctx_time_range=ctx_time_range, positive_ratio=positive_ratio, random_seed=random_seed))
    _print_label_distribution([entry[2] for entry in dataset])
    return dataset


def generate_pkt_context(pkt_df, ctx_time_range, num_samples, random_seed=1):
    np.random.seed(random_seed)
    num_generated = 0
    while num_generated < num_samples:
        cur_pkt = pkt_df.sample(n=1).iloc[0]
        cur_ts = cur_pkt['basic', 'timestamp']
        ctx_pkt_df = pkt_df[
            (cur_ts - pkt_df['basic', 'timestamp'] > ctx_time_range[0]) &
            (cur_ts - pkt_df['basic', 'timestamp'] < ctx_time_range[1])
        ]
        cur_mac = cur_pkt['basic', 'tx_addr']
        ctx_pkt_df_wiz_same_mac = _find_all_same_mac_precurs(cur_pkt, ctx_pkt_df)
        if len(ctx_pkt_df_wiz_same_mac) > 0:
            direct_precur_id = ctx_pkt_df_wiz_same_mac.index[0]
            all_precur_ids = list(ctx_pkt_df_wiz_same_mac.index)
        else:
            direct_precur_id = -1
            all_precur_ids = []
        entry = (cur_pkt, ctx_pkt_df, direct_precur_id, all_precur_ids)
        yield entry
        num_generated += 1


def generate_pkt_context_dataset(pkt_df, num_samples, random_seed=1):
    dataset = list(generate_pkt_context(pkt_df, num_samples=num_samples, random_seed=random_seed))
    return dataset


# def generate_full_trans_pair(pkt_df, start_ts=None, duration=300*1000):
#     """
#     Generate all the transition pairs (time_1 < time_2) within a period of time.
#     :param pkt_df:
#     :param start_ts:
#     :param duration:
#     :return: A generator of all the transition pairs.
#     """
#     if start_ts is None:
#         start_ts = pkt_df.iloc[0]['basic', 'timestamp']
#     end_ts = start_ts + duration
#     fltr_pkt_df = pkt_df[(pkt_df['basic', 'timestamp'] >= start_ts) & (pkt_df['basic', 'timestamp'] < end_ts)].sort_values([('basic', 'timestamp')], ascending=True)
#     row_ind, n = 0, len(fltr_pkt_df)
#     for pkt_id_from, pkt_from in fltr_pkt_df.iterrows():
#         for i in range(row_ind + 1, n):
#             pkt_to = fltr_pkt_df.iloc[i]
#             label = pkt_from['basic', 'tx_addr'] == pkt_to['basic', 'tx_addr']
#             yield (pkt_from, pkt_to, label)
#         row_ind += 1


def generate_full_trans_pair(pkt_df, ctx_time_range, num_samples, trans_type='all', random_seed=1):
    for cur_pkt, ctx_pkt_df, direct_precur_id, all_precur_ids in generate_pkt_context(pkt_df, ctx_time_range, num_samples, random_seed):
        if trans_type == 'all':
            for id_from, pkt_from in ctx_pkt_df.iterrows():
                label = (id_from in all_precur_ids)
                yield (pkt_from, cur_pkt, label)
        elif trans_type == 'direct':
            if direct_precur_id > 0:
                pkt_from = ctx_pkt_df.loc[direct_precur_id]
                yield (pkt_from, cur_pkt, True)
        elif trans_type == 'positive':
            if direct_precur_id > 0:
                for id_from in all_precur_ids:
                    pkt_from = ctx_pkt_df.loc[id_from]
                    yield (pkt_from, cur_pkt, True)

                
def generate_full_trans_pair_dataset(pkt_df, ctx_time_range, num_samples, trans_type='all', random_seed=1):
    dataset = list(generate_full_trans_pair(pkt_df, ctx_time_range, num_samples, trans_type=trans_type, random_seed=random_seed))                        
    _print_label_distribution([entry[2] for entry in dataset])
    return dataset


def inject_ref_by_tx_addr(pkt_df, same_mac_max_interval=4000):
    """
    Assume the tx_addr is packet's device id.
    """
    # Add new columns
    pkt_df['ref', 'dev_id'] = None
    pkt_df['ref', 'order'] = None
    # Find packet with same tx_addr, and increase its order
    # device id is the mac address and the order is its order in the sequence
    last_mac_info_dict = {}
    for row_idx, row in pkt_df.iterrows():
        mac = row['basic', 'tx_addr']
        ts = row['basic', 'timestamp']
        if mac in last_mac_info_dict:
            last_info = last_mac_info_dict[mac]
            order = last_info[1]
            # we consider it one request upon different channels
            if ts - last_info[0] > same_mac_max_interval:
                order += 1
        else:
            order = 0
        pkt_df.at[row_idx, ('ref', 'dev_id')] = mac
        pkt_df.at[row_idx, ('ref', 'order')] = order
        last_mac_info_dict[mac] = (ts, order)
    return pkt_df
    
    
def extract_pkt_chain_df_by_dev_id(pkt_df):
    pkt_gdf = pkt_df.groupby(pkt_df[('ref', 'dev_id')])
    pkt_chain_df = pkt_gdf.apply(lambda df: list(df.index))
    pkt_chain_df = pkt_chain_df.reset_index(name='pkt_chain')
    return pkt_chain_df
    

def _print_label_distribution(labels):
    num_total = len(labels)
    num_positive = np.sum(labels)
    num_negative = num_total - num_positive
    print('positive: %d (%.2f%%)' % (num_positive, num_positive / num_total * 100))
    print('negative: %d (%.2f%%)' % (num_negative, num_negative / num_total * 100))

    
def extract_dataset(dataset, model):
    X_from = np.vstack([model.compute_feat_vecs(entry[0]) for entry in dataset])
    t_from = np.array([model.compute_ts_vecs(entry[0]) for entry in dataset])
    X_to = np.vstack([model.compute_feat_vecs(entry[1]) for entry in dataset])
    t_to = np.array([model.compute_ts_vecs(entry[1]) for entry in dataset])
    if len(dataset[0]) >= 3:
        y = np.array([entry[2] for entry in dataset])
    else:
        y = None
    return X_from, X_to, t_from, t_to, y

############### END OF DATASET EXTACTION ###############


if __name__ == '__main__':
    pkt_df = prepare_rawdata(filename='rssiraw_2019-07-20.csv', data_dirpath='env',
                             time_range=(1563588000000, 1563588000000 + 60000), category='phy')
    # pkt_df = prepare_rawdata(filename='rssiraw.csv', data_dirpath='env')
    print(pkt_df)
