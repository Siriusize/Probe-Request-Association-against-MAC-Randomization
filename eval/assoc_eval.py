import sys
import itertools

import numpy as np
import pandas as pd
from sklearn import metrics

from data import data_loader


# Packet chain (i.e., association result)
# Format: [[...], [...], [...], ...]


def build_pkt_chains_from_assoc_result(assoc_result, pkt_df=None):
    """
    If pkt_df is specified, only packets in the pkt_df will be considered.
    :param assoc_result:
    :param pkt_df:
    :return:
    """
    # pkt_chains = [np.array(pkt_grp.pkt_ids) for pkt_grp in assoc_result]
    pkt_chains = []
    for pkt_grp in assoc_result:
        chain = np.array(list(filter(lambda pkt_id: pkt_id in pkt_df.index, pkt_grp.pkt_ids)))
        if len(chain) > 0:
            pkt_chains.append(chain)
    return pkt_chains


def build_pkt_chain_df(pkt_chains):
    # df = pd.DataFrame(columns=['pkt_chain'])
    # for ind, chain in enumerate(pkt_chains):
    #     df.loc['pred_%d' % ind] = [chain]
    df = pd.DataFrame({'pkt_chain': pkt_chains})
    return df


def find_corresponding_true_chain_ids(pred_chain, true_pkt_chain_df):
    if len(pred_chain) == 0:
        return []
    intersect_lens = np.array([len(np.intersect1d(pred_chain, true_entry['pkt_chain'])) for _, true_entry in true_pkt_chain_df.iterrows()])
    true_chain_inds = np.where((intersect_lens == np.max(intersect_lens)))
    # union_lens = np.array([len(np.union1d(pred_chain, true_entry['pkt_chain'])) for _, true_entry in true_pkt_chain_df.iterrows()])
    # scores = intersect_lens / union_lens
    # true_chain_inds = np.where((scores == np.max(scores)) & (scores >= 0.25))
    # true_chain_inds = np.where((scores == np.max(scores)))
    true_chain_ids = true_pkt_chain_df.index[true_chain_inds]
    return true_chain_ids


def find_corresponding_true_chain_id(pred_chain, true_pkt_chain_df):
    true_chain_ids = find_corresponding_true_chain_ids(pred_chain, true_pkt_chain_df)
    if true_chain_ids is not None and len(true_chain_ids) > 0:
        return true_chain_ids[0]
    else:
        return None


def build_pkt_mapping(pred_pkt_chain_df, true_pkt_chain_df):
    """
    Build two mappings from the perspective of packet.
    The first one maps from pkt_id to the pred_chain from which the packet comes.
    The second one maps from pkt_id to true_chain from which the packet comes.
    :param pred_pkt_chain_df:
    :param true_pkt_chain_df:
    :return: pkt_from_pred_chain_mapping, pkt_from_true_chain_mapping
    """
    # pkt_id --> id of its true chain
    pkt_from_true_chain_mapping = {}
    for true_chain_id, row in true_pkt_chain_df.iterrows():
        true_chain = row['pkt_chain']
        for pkt_id in true_chain:
            pkt_from_true_chain_mapping[pkt_id] = true_chain_id

    # pkt_id --> id of its predicted chain
    pkt_from_pred_chain_mapping = {}
    for pred_chain_id, row in pred_pkt_chain_df.iterrows():
        pred_chain = row['pkt_chain']
        for pkt_id in pred_chain:
            pkt_from_pred_chain_mapping[pkt_id] = pred_chain_id

    return pkt_from_pred_chain_mapping, pkt_from_true_chain_mapping


def build_chain_mapping(pred_pkt_chain_df, true_pkt_chain_df, will_exclude_empty_pred=False):
    """

    :param pred_pkt_chain_df:
    :param true_pkt_chain_df:
    :return:
        pred_to_true_chain_mapping: Given a predicted pkt chain, which is its corresponding true chain?
        true_to_pred_chain_mapping: Given a true pkt chain, which is its corresponding predicted chain?
    """
    pred_to_true_chain_mapping = {}
    for pred_chain_id, row in pred_pkt_chain_df.iterrows():
        pred_chain = row['pkt_chain']
        if len(pred_chain) == 0:
            correspond_true_chain_id = None
        else:
            correspond_true_chain_id = find_corresponding_true_chain_id(pred_chain, true_pkt_chain_df)
        if will_exclude_empty_pred and correspond_true_chain_id is None:
            continue
        else:
            pred_to_true_chain_mapping[pred_chain_id] = correspond_true_chain_id

    true_to_pred_chain_mapping = {}
    for true_chain_id, row in true_pkt_chain_df.iterrows():
        true_to_pred_chain_mapping[true_chain_id] = []
    for pred_chain_id, true_chain_id in pred_to_true_chain_mapping.items():
        true_to_pred_chain_mapping[true_chain_id].append(pred_chain_id)

    return pred_to_true_chain_mapping, true_to_pred_chain_mapping


def build_labels_array(pkt_from_pred_chain_mapping, pkt_from_true_chain_mapping):
    pkt_ids = np.sort(list(pkt_from_pred_chain_mapping.keys()))
    pred_labels = np.zeros(len(pkt_ids), dtype=int)
    true_labels = np.zeros(len(pkt_ids), dtype=int)
    for i, pkt_id in enumerate(pkt_ids):
        pred_labels[i] = pkt_from_pred_chain_mapping[pkt_id]
        true_labels[i] = pkt_from_true_chain_mapping[pkt_id]
    return pred_labels, true_labels


def prepare_evaluation(assoc_result, true_pkt_chain_df, pkt_df=None):
    pred_pkt_chains = build_pkt_chains_from_assoc_result(assoc_result, pkt_df=pkt_df)

    pred_pkt_chain_df = build_pkt_chain_df(pred_pkt_chains)

    pkt_from_pred_chain_mapping, pkt_from_true_chain_mapping = \
        build_pkt_mapping(pred_pkt_chain_df, true_pkt_chain_df)

    pred_to_true_chain_mapping, true_to_pred_chain_mapping = \
        build_chain_mapping(pred_pkt_chain_df, true_pkt_chain_df, will_exclude_empty_pred=True)

    pred_labels, true_labels = build_labels_array(pkt_from_pred_chain_mapping, pkt_from_true_chain_mapping)

    return pred_pkt_chain_df, \
           pkt_from_pred_chain_mapping, pkt_from_true_chain_mapping, \
           pred_to_true_chain_mapping, true_to_pred_chain_mapping, \
           pred_labels, true_labels


# Precision and recall
# [REMARK] This seems to be equal to the purity
# def eval_chain_accuracy(true_to_pred_chain_mapping, pred_pkt_chain_df, true_pkt_chain_df):
#     tps, fps, fns = [], [], []
#     for true_chain_id, pred_chain_ids in true_to_pred_chain_mapping.items():
#         true_chain = true_pkt_chain_df.loc[true_chain_id, 'pkt_chain']
#         pred_chains = pred_pkt_chain_df.loc[pred_chain_ids, 'pkt_chain']
#         concat_pred_chain = []
#         valid_pred_chains = [chain for chain in pred_chains]
#         if len(valid_pred_chains) > 0:
#             concat_pred_chain = np.concatenate(valid_pred_chains).astype(np.int)
#         intersect_pkts = np.intersect1d(concat_pred_chain, true_chain)
#         tp = len(intersect_pkts)
#         # fp len(np.setdiff1d(concat_pred_chain, actual_chain))
#         fp = len(concat_pred_chain) - tp
#         # fn = len(np.setdiff1d(actual_chain, concat_pred_chain))
#         fn = len(true_chain) - tp
#         tps.append(tp)
#         fps.append(fp)
#         fns.append(fn)
#
#     total_tp, total_fp, total_fn = np.sum(tps), np.sum(fps), np.sum(fns)
#     precision = total_tp / (total_tp + total_fp)
#     recall = total_tp / (total_tp + total_fn)
#
#     # context = {'tps': tps, 'fps': fps, 'fns': fns}
#
#     print('tp = %d, fp = %d, fn = %s' % (total_tp, total_fp, total_fn))
#
#     return precision, recall


def eval_precision_and_recall(pred_labels, true_labels):
    # true <-> pred
    # tp: two packets are in the same true chain, and in the same predicted chain
    tp, fp, fn, tn = 0, 0, 0, 0
    for i, j in itertools.combinations(list(range(len(pred_labels))), 2):
        pred_equal = pred_labels[i] == pred_labels[j]
        true_equal = true_labels[i] == true_labels[j]
        if true_equal and pred_equal:
            tp += 1
        elif true_equal and not pred_equal:
            fn += 1
        elif not true_equal and pred_equal:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_measure_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_measure_score


def eval_adjusted_rand_index(pred_labels, true_labels):
    ari = metrics.adjusted_rand_score(true_labels, pred_labels)
    return ari


def eval_mutual_information(pred_labels, true_labels):
    nmi = metrics.normalized_mutual_info_score(true_labels, pred_labels)
    return nmi


def eval_entropy_analysis(pred_labels, true_labels):
    homogeneity, completeness, v_measure_score = \
        metrics.homogeneity_completeness_v_measure(true_labels, pred_labels)
    return homogeneity, completeness, v_measure_score


# Association accuracy
def eval_association_accuracy(pred_pkt_chain_df, true_pkt_chain_df):
    """
    Evaluate the accuracy of association, initialization and termination.
    :return: association accuracy, initialization accuracy and termination accuracy
    """

    def extract_assoc_pairs(pkt_chain_df):
        assoc_pairs = []
        init_list, term_list = [], []
        for chain_id, row in pkt_chain_df.iterrows():
            chain = row['pkt_chain']
            for i in range(len(chain) - 1):
                assoc_pairs.append((chain[i], chain[i + 1]))
            if len(chain) > 0:
                init_list.append(chain[0])
                term_list.append(chain[-1])
        return assoc_pairs, init_list, term_list

    pred_assoc_pairs, pred_init_list, pred_term_list = extract_assoc_pairs(pred_pkt_chain_df)
    true_assoc_pairs, true_init_list, true_term_list = extract_assoc_pairs(true_pkt_chain_df)

    # assoc_fp: a and b are not from the same device, they are associated
    # assoc_fn: a and b are from the same device, they are not associated
    assoc_tp, assoc_fp, assoc_fn = 0, 0, 0
    init_tp, init_fp, init_fn = 0, 0, 0
    term_tp, term_fp, term_fn = 0, 0, 0
    for pred_pair in pred_assoc_pairs:
        if pred_pair in true_assoc_pairs:
            assoc_tp += 1
        else:
            assoc_fp += 1
    for true_pair in true_assoc_pairs:
        if true_pair not in pred_assoc_pairs:
            assoc_fn += 1

    assoc_precision = assoc_tp / (assoc_tp + assoc_fn)
    assoc_recall = assoc_tp / (assoc_tp + assoc_fp)
    # init_precision = init_tp / (init_tp + init_fn)
    # init_recall = init_tp / (init_tp + init_fp)
    # term_precision = term_tp / (term_tp + term_fn)
    # term_recall = term_tp / (term_tp + term_fp)
    init_precision = 0
    init_recall = 0
    term_precision = 0
    term_recall = 0

    return assoc_precision, assoc_recall, \
           init_precision, init_recall, \
           term_precision, term_recall


# ID-switch related
def eval_id_switches(true_to_pred_chain_mapping, pkt_from_true_chain_mapping, pred_pkt_chain_df):
    id_switches, segs = [], []
    for true_id, pred_chains_ids in true_to_pred_chain_mapping.items():
        pred_chains = [pred_pkt_chain_df.loc[pred_id, 'pkt_chain'] for pred_id in pred_chains_ids]
        concat_pred_chain = []
        if len(pred_chains) > 0:
            #concat_pred_chain = np.concatenate(pred_chains).astype(np.int)
            concat_pred_chain = np.concatenate(pred_chains).astype(int)
        pred_chain_labels = [pkt_from_true_chain_mapping.get(pred_id, 1) for pred_id in concat_pred_chain]
        pred_chain_labels = np.array(pred_chain_labels)
        # print(pred_chain_labels)
        num_switch = np.sum(pred_chain_labels[:-1] != pred_chain_labels[1:])
        id_switches.append(num_switch)
        segs.append(len(pred_chains_ids))
    return id_switches, segs


def eval_purity(pred_to_true_chain_mapping, pred_pkt_chain_df, true_pkt_chain_df):
    intersect_counts, pred_chain_counts = [], []
    for pred_chain_id, true_chain_id in pred_to_true_chain_mapping.items():
        true_chain = true_pkt_chain_df.loc[true_chain_id, 'pkt_chain']
        pred_chain = pred_pkt_chain_df.loc[pred_chain_id, 'pkt_chain']
        intersect_counts.append(len(np.intersect1d(true_chain, pred_chain)))
        pred_chain_counts.append(len(pred_chain))
    purity = np.sum(intersect_counts) / np.sum(pred_chain_counts)
    return purity


def print_assoc_detail(true_to_pred_chain_mapping, true_pkt_chain_df, pred_pkt_chain_df, pkt_df):
    for true_chain_id, pred_chain_ids in true_to_pred_chain_mapping.items():
        true_chain = true_pkt_chain_df.loc[true_chain_id, 'pkt_chain']
        pred_chains = pred_pkt_chain_df.loc[pred_chain_ids, 'pkt_chain']
        concat_pred_chain = []
        valid_pred_chains = [chain for chain in pred_chains]
        if len(valid_pred_chains) > 0:
            concat_pred_chain = np.concatenate(valid_pred_chains).astype(np.int)
        union_pkts = np.sort(np.union1d(concat_pred_chain, true_chain))
        for pkt_id in union_pkts:
            if pkt_id in true_chain:
                found = False
                for i, pred_chain in enumerate(pred_chains):
                    if pkt_id in pred_chain:
                        print('#%d %s [%d]' % (i, pkt_df.loc[pkt_id][('basic', 'tx_addr')], pkt_id), end='')
                        found = True
                        break
                if not found:
                    print('(%s [%d])' % (pkt_df.loc[pkt_id][('basic', 'tx_addr')], pkt_id), end='')
            elif pkt_id in concat_pred_chain:
                print('*%s [%d]*' % (pkt_df.loc[pkt_id][('basic', 'tx_addr')], pkt_id), end='')
            print(' -> ', end='')
        print('END')


def evaluate_all(assoc_result, pkt_df, true_pkt_chain_df=None, print_detail=False):

    if true_pkt_chain_df is None:
        true_pkt_chain_df = data_loader.extract_pkt_chain_df_by_dev_id(pkt_df)

    pred_pkt_chain_df, \
    pkt_from_pred_chain_mapping, pkt_from_true_chain_mapping, \
    pred_to_true_chain_mapping, true_to_pred_chain_mapping, \
    pred_labels, true_labels = prepare_evaluation(assoc_result, true_pkt_chain_df, pkt_df=pkt_df)

    if print_detail:
        print_assoc_detail(true_to_pred_chain_mapping, true_pkt_chain_df, pred_pkt_chain_df, pkt_df)

    num_pkts = len(pkt_df)
    num_pred_devices = len(pred_pkt_chain_df)
    num_true_devices = len(true_pkt_chain_df)
    print('num_pkts = %d' % num_pkts)
    print('num_pred_devices = %d' % num_pred_devices)
    print('num_true_devices = %d' % num_true_devices)

    homogeneity, completeness, v_measure_score = eval_entropy_analysis(pred_labels, true_labels)
    print('homogeneity = %.3f, completeness = %.3f, v_measure_score = %.3f' % (homogeneity, completeness, v_measure_score))

    id_switches, segs = eval_id_switches(true_to_pred_chain_mapping, pkt_from_true_chain_mapping, pred_pkt_chain_df)
    average_id_switches = np.mean(id_switches)
    average_segs = np.mean(segs)
    print('average_id_switches = %.3f' % np.mean(id_switches))
    print('average_segs = %.3f' % np.mean(segs))

    purity = eval_purity(pred_to_true_chain_mapping, pred_pkt_chain_df, true_pkt_chain_df)
    print('purity = %.3f' % purity)

    return {'num_pkts': num_pkts, 'num_pred_devices': num_pred_devices, 'num_true_devices': num_true_devices,
            'homogeneity': homogeneity, 'completeness': completeness, 'v_measure_score': v_measure_score,
            'average_id_switches': average_id_switches, 'average_segs': average_segs,
            'purity': purity, "pred_label": pred_pkt_chain_df, "true_label": true_pkt_chain_df}


