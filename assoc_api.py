import os, sys
import time
import numpy as np 
import pandas as pd
from ortools.graph.python import min_cost_flow
from common.utils.object_utils import load_pickle
from model.prob.unified_model import UnifiedAssocModel
from assoc.assoc_common import AssocPktGroup, dump_assoc_result
from data.data_loader import prepare_rawdata, inject_ref_by_tx_addr
from eval.assoc_eval import evaluate_all

PRECISION = 0.000001

class MinCostFlowAssociator:
    def __init__(self, pkt_df, model,
                 interval=30 * 1000, unassoc_thresh=0, timeout_thresh=600 * 1000):
        # main input
        self.pkt_df = pkt_df  # to be processed
        self.model = model

        # parameters
        self.interval = interval
        self.unassoc_thresh = unassoc_thresh
        self.timeout_thresh = timeout_thresh

        self.ongoing_pkt_grps = []  # ongoing packet groups
        self.ultimate_pkt_grps = []  # finalized packet groups

        self.precision = PRECISION

        self.step_results = []
        self.ts = []


    def compute_cost_matrix_for_pkt_grp(self, pkt_grps, pkt_df_to, num_revisit=3):
        cost_vol = np.zeros((num_revisit, len(pkt_grps), len(pkt_df_to)))
        for k in range(1, num_revisit + 1):
            pkt_ids_from = np.array([pkt_grp.get_pkt_id(-k) for pkt_grp in pkt_grps])
            valid_row_inds = np.array([pkt_id is not None for pkt_id in pkt_ids_from])
            if np.sum(valid_row_inds) > 0:
                pkt_df_from = self.pkt_df.loc[pkt_ids_from[valid_row_inds]]
                cost_mat_ = self.compute_cost_matrix(self.model, pkt_df_from, pkt_df_to)
                cost_vol[k - 1, valid_row_inds, :] = cost_mat_
        return np.min(cost_vol, axis=0)

    @staticmethod
    def compute_cost_matrix(model, pkt_df_from, pkt_df_to):
        m, n = len(pkt_df_from), len(pkt_df_to)
        score_mat = np.zeros((m, n))
        i = 0
        for pkt_id, pkt_from in pkt_df_from.iterrows():
            time_diff = pkt_df_to[('basic', 'timestamp')] - pkt_from[('basic', 'timestamp')]
            time_valid_inds = time_diff >= 0
            if np.any(time_valid_inds):
                scores = model.predict_proba(pkt_from, pkt_df_to[time_valid_inds])
                score_mat[i, time_valid_inds] = scores
            i += 1
        # set the diag to 0 if pkt_df_from and pkt_df_to are the same
        # because we use <= in the line of pkt_from[('basic', 'timestamp')] <= pkt_df_to[('basic', 'timestamp')]
        # it may appear some cases that elements in the lower right part is not zero
        if pkt_df_from is pkt_df_to:
            score_mat = np.triu(score_mat, k=1)     # only get the upper triangle part (exclude diagonal)
        score_mat = np.log(score_mat)
        score_mat = np.nan_to_num(score_mat, neginf=-5000)
        cost_mat = -score_mat
        return cost_mat

    @staticmethod
    def construct_network(prev2cur_cost_mat, cur2cur_cost_mat, unassoc_cost):
        if prev2cur_cost_mat is not None and len(prev2cur_cost_mat) != 0:
            m, n = prev2cur_cost_mat.shape
        else:
            m = 0
            n = len(cur2cur_cost_mat)
        f = m + n * 2

        # nodes
        src_node_id = 0
        from_node_ids = list(range(1, m + n + 2))  # [1 : m]: prev nodes, [m + 1 : m + n]: cur nodes, m + n + 1: null node
        to_node_ids = list(range(m + n + 2, m + n * 2 + 3))  # [m + n + 2 : m + n * 2 + 1]: cur nodes, m + n * 2 + 2: null node
        sink_node_id = m + n * 2 + 3
        num_nodes = m + n * 2 + 4
        supplies = [f] + [0] * (f + 2) + [-f]

        # arcs
        start_nodes, end_nodes, capacities, costs = [], [], [], []
        ## src_node -> from_nodes
        start_nodes += [src_node_id] * (m + n + 1)
        end_nodes += from_node_ids
        capacities += [1] * (m + n) + [n]
        costs += [0] * (m + n + 1)
        ## from_nodes -> to_nodes
        ### prev + cur nodes -> cur nodes
        for i in range(m + n):
            from_node_id = from_node_ids[i]
            start_nodes += [from_node_id] * (n + 1)
            end_nodes += to_node_ids
            capacities += [1] * (n + 1)
            if i < m:
                costs += prev2cur_cost_mat[i, :].tolist() + [unassoc_cost]
            else:
                costs += cur2cur_cost_mat[i - m, :].tolist() + [unassoc_cost]
        ### null node -> cur nodes
        start_nodes += [from_node_ids[-1]] * (n + 1)
        end_nodes += to_node_ids  # [:-1]
        capacities += [1] * n + [f]
        costs += [unassoc_cost] * (n + 1)
        # to_nodes -> sink_node
        start_nodes += to_node_ids
        end_nodes += [sink_node_id] * (n + 1)
        capacities += [1] * n + [m + n]
        costs += [0] * (n + 1)

        # print(start_nodes)
        # print(end_nodes)
        # print(capacities)
        # print(costs)
        return start_nodes, end_nodes, capacities, costs, supplies

    @staticmethod
    def solve_min_cost_flow(start_nodes, end_nodes, capacities, costs, supplies, precision):
        ## convert to unit costs
        unit_costs = [int(np.round(c / precision)) if not pd.isna(c) else 0
                      for c in costs]

        # Instantiate a SimpleMinCostFlow solver
        smcf = min_cost_flow.SimpleMinCostFlow()
        for i in range(len(start_nodes)):
            smcf.add_arcs_with_capacity_and_unit_cost(start_nodes[i], end_nodes[i], capacities[i], unit_costs[i])
        for i in range(len(supplies)):
            smcf.set_nodes_supplies(i, supplies[i])

        rt = smcf.solve()
        if rt != smcf.OPTIMAL:
            print('The solution is not OPTIMAL.')
        solution = []
        for i in range(smcf.num_arcs()):
            solution.append((smcf.tail(i), smcf.head(i),
                             smcf.flow(i), smcf.capacity(i)))
        return solution

    @staticmethod
    def parse_min_cost_flow_solution(mcf_solution, m, n):
        prev2cur_assoc_ind_pairs, cur2cur_assoc_ind_pairs = [], []
        for entry in mcf_solution:
            from_node, to_node, flow, capacity = entry
            if flow > 0:
                if 1 <= from_node <= m:
                    if (m + n + 2) <= to_node <= (m + n * 2 + 1):
                        prev2cur_assoc_ind_pairs.append((from_node - 1, to_node - m - n - 2))
                    elif to_node == (m + n * 2 + 2):
                        prev2cur_assoc_ind_pairs.append((from_node - 1, None))
                elif m + 1 <= from_node <= m + n:
                    if (m + n + 2) <= to_node <= (m + n * 2 + 1):
                        cur2cur_assoc_ind_pairs.append((from_node - m - 1, to_node - m - n - 2))
                    elif to_node == (m + n * 2 + 2):
                        cur2cur_assoc_ind_pairs.append((from_node - m - 1, None))
                elif from_node == (m + n + 1):
                    if (m + n + 2) <= to_node <= (m + n * 2 + 1):
                        prev2cur_assoc_ind_pairs.append((None, to_node - m - n - 2))
        return prev2cur_assoc_ind_pairs, cur2cur_assoc_ind_pairs

    def associate_by_mcf(self, prev_pkt_grps, cur_pkt_df):
        print('Constructing cost matrix...')
        t0 = time.time()

        m = len(prev_pkt_grps)
        if prev_pkt_grps is not None and m > 0:
            prev_pkt_df = pd.DataFrame([pkt_grp.last_pkt for pkt_grp in prev_pkt_grps])
            prev2cur_cost_mat = self.compute_cost_matrix(self.model, prev_pkt_df, cur_pkt_df)
            print('[*] prev2cur cost matrix: %s' % str(prev2cur_cost_mat.shape))
        else:
            prev2cur_cost_mat = None
            m = 0

        n = len(cur_pkt_df)
        cur2cur_cost_mat = self.compute_cost_matrix(self.model, cur_pkt_df, cur_pkt_df)
        print('[*] cur2cur cost matrix: %s' % str(cur2cur_cost_mat.shape))

        t1 = time.time()
        print('Association matrix constructed in %.2fs' % (t1 - t0))

        print('Solving min-cost flow problem...')
        t0 = time.time()
        unassoc_cost = -np.log(self.unassoc_thresh)
        mcf_network = self.construct_network(prev2cur_cost_mat, cur2cur_cost_mat, unassoc_cost)
        mcf_solution = self.solve_min_cost_flow(*mcf_network, precision=self.precision)
        t1 = time.time()
        print('Problem solved in %.2fs' % (t1 - t0))

        prev2cur_assoc_ind_pairs, cur2cur_assoc_ind_pairs = self.parse_min_cost_flow_solution(mcf_solution, m, n)

        return prev2cur_assoc_ind_pairs, cur2cur_assoc_ind_pairs

    def run_one_step(self, prev_pkt_grps, cur_pkt_df):

        print('[*] number of PREV packet groups: %d' % len(prev_pkt_grps))
        print('[*] number of CUR packets: %d' % len(cur_pkt_df))

        t0 = time.time()

        # association
        prev2cur_assoc_ind_pairs, cur2cur_assoc_ind_pairs = self.associate_by_mcf(prev_pkt_grps, cur_pkt_df)

        # generate associate pkt groups
        new_pkt_grps = []
        for prev_pkt_ind, cur_pkt_ind in prev2cur_assoc_ind_pairs:
            if prev_pkt_ind is not None and cur_pkt_ind is not None:
                ## PREV -> CUR
                cur_pkt_id = cur_pkt_df.index[cur_pkt_ind]
                cur_pkt = cur_pkt_df.loc[cur_pkt_id]
                prev_pkt_grp = prev_pkt_grps[prev_pkt_ind]
                prev_pkt_grp.append_pkt(cur_pkt)
                new_pkt_grps.append(prev_pkt_grp)
            elif prev_pkt_ind is None:
                ## NULL -> CUR (new packet group)
                cur_pkt_id = cur_pkt_df.index[cur_pkt_ind]
                cur_pkt = cur_pkt_df.loc[cur_pkt_id]
                new_pkt_grp = AssocPktGroup(cur_pkt)
                new_pkt_grps.append(new_pkt_grp)
            elif cur_pkt_ind is None:
                ## PREV -> NULL
                new_pkt_grps.append(prev_pkt_grps[prev_pkt_ind])

        # CUR -> CUR
        cur2cur_assoc_id_pairs = [(cur_pkt_df.index[pair[0]] if pair[0] is not None else None,
                                   cur_pkt_df.index[pair[1]] if pair[1] is not None else None)
                                  for pair in cur2cur_assoc_ind_pairs]
        # print(cur2cur_assoc_id_pairs)
        for cur_pkt_grp in new_pkt_grps:
            last_pkt_id = cur_pkt_grp.last_pkt_id
            while True:  # associate to form a chain
                found_last_pkt_assoc_pairs = [pair for pair in cur2cur_assoc_id_pairs if pair[0] == last_pkt_id]
                if len(found_last_pkt_assoc_pairs) != 1:
                    break  # unlikely
                last_pkt_assoc_pair = found_last_pkt_assoc_pairs[0]
                next_pkt_id = last_pkt_assoc_pair[1]
                if next_pkt_id is None:
                    break
                cur_pkt_grp.append_pkt(cur_pkt_df.loc[next_pkt_id])
                last_pkt_id = next_pkt_id

        t1 = time.time()
        print('Packet groups updated in %.2fs.' % (t1 - t0))

        return new_pkt_grps

    def finalize_timeout_pkt_grps(self, cur_ts=np.inf):
        new_ongoing_pkt_grps = []
        num_finalized, num_total = 0, 0
        for pkt_grp in self.ongoing_pkt_grps:
            if cur_ts - pkt_grp.last_timestamp > self.timeout_thresh:
                self.ultimate_pkt_grps.append(pkt_grp)
                num_finalized += 1
                self.ts.append(pkt_grp.pkt_timestamps)
            else:
                new_ongoing_pkt_grps.append(pkt_grp)
            num_total += 1
        print('[*] %d out of %d packet groups are finalized. ' % (num_finalized, num_total))
        self.ongoing_pkt_grps = new_ongoing_pkt_grps
        return None

    def run(self, export_start_ts=None, export_cycle=0, export_dirpath=None):
        """
        Execute the association on the specified pkt_df.
        :param export_cycle: The interval of exporting intermediate result.
        If it is 0, then nothing will be exported.
        :param export_dirpath: The path of folder for exporting intermediate result.
        :return: The final association result (i.e., a list of packet chains)
        """

        all_ts = self.pkt_df['basic', 'timestamp']
        min_ts, max_ts = all_ts.min(), all_ts.max()
        last_export_ts = init_ts = min_ts
        self.ongoing_pkt_grps = []
        num_total_iter = (max_ts + self.interval - min_ts) // self.interval
        iter_count = 0
        for start_ts in range(min_ts, max_ts + self.interval, self.interval):
            end_ts = start_ts + self.interval
            print('======== Iteration %d/%d ========' % (iter_count, num_total_iter))
            print('[*] time range: %d - %d (%dms)' % (start_ts, end_ts, end_ts - start_ts))
            self.finalize_timeout_pkt_grps(start_ts)
            cur_pkt_df = self.pkt_df.loc[(all_ts >= start_ts) & (all_ts < end_ts)]
            if len(cur_pkt_df) <= 0:
                print('Skip this iteration because no packet found in this period.')
                continue
            self.ongoing_pkt_grps = self.run_one_step(self.ongoing_pkt_grps, cur_pkt_df)
            print('[*] number of ongoing packet groups: %d' % len(self.ongoing_pkt_grps))

            iter_count += 1

            if export_cycle > 0:
                cur_iter_end_ts = end_ts
                if cur_iter_end_ts - last_export_ts >= export_cycle:
                    assoc_result_step = self.ultimate_pkt_grps + self.ongoing_pkt_grps
                    time_range_step = (init_ts, cur_iter_end_ts)
                    if export_dirpath is not None:
                        pkl_filename = str(time_range_step).replace(' ', '').replace('(', '').replace(')', '').replace(',', '_') \
                                       + '.pkl'
                        dump_assoc_result(assoc_result_step,
                                          os.path.join(export_dirpath, pkl_filename),
                                          time_range=time_range_step)
                    else:
                        print('The export_dirpath is not specified. Skip dumping result.')
                    last_export_ts = cur_iter_end_ts

        print('======== Cleaning up ========')
        self.finalize_timeout_pkt_grps()  # finalize remaining packet groups
        print('[*] number of ultimate packet groups: %d' % len(self.ultimate_pkt_grps))

        if export_dirpath is not None:
            pkl_filename = 'all.pkl'
            dump_assoc_result(self.ultimate_pkt_grps,
                              os.path.join(export_dirpath, pkl_filename),
                              time_range=(min_ts, max_ts))
        else:
            print('The export_dirpath is not specified. Skip dumping result.')

        return self.ultimate_pkt_grps

class vmac_assoc_api:
    def __init__(self, cor_model="Espresso",
                                        IE_DIM=35, SEQ_DIM=1, SIGTRANS_DIM=21,
                                        interval= 30*1000, # mini-batch setting to 30 * 1000ms
                                        unassoc_thres = 1e-6, # threshold for breaking association
                                        timeout_thres = 900 * 1000, # timeout in 15 minutes
                                        export_cycle = 10 * 60 * 1000,
                                        ie_pth="./model/prob/ie_model.pkl",
                                        seq_pth="./model/prob/seq_model.pkl",
                                        sigtrans_pth="./model/prob/sigtrans_model.pkl") -> None:
        self.interval = interval
        self.unassoc_thres = unassoc_thres
        self.export_cycle = export_cycle
        self.timeout_thres = timeout_thres
        
        ie_model = load_pickle(ie_pth)
        seq_model = load_pickle(seq_pth)
        # sigtrans_model = load_pickle(sigtrans_pth) # This is site dependent

        self.unified_model = UnifiedAssocModel(ie_model=ie_model, seq_model=seq_model, sigtrans_model=None,
                                      X_dims=(IE_DIM, SEQ_DIM, SIGTRANS_DIM), 
                                      time_decay_factor=0.00001   # time delay between packets
                                      )

    def assoc_run(self, input_pkt_df, export_dirpath, 
                        test_time_range): 
        mcf_processor = MinCostFlowAssociator(input_pkt_df, self.unified_model,
                                          interval=self.interval, unassoc_thresh=self.unassoc_thres,
                                          timeout_thresh=self.timeout_thres)
        assoc_result = mcf_processor.run(export_start_ts=test_time_range[0], export_cycle=self.export_cycle,
                                     export_dirpath=export_dirpath)
        assoc_eval_result = evaluate_all(assoc_result, test_pkt_df)

        return assoc_eval_result

def save_result(data_period, proc_time, eval_info, output_dir):
    result = [str(proc_time) + " seconds", str(eval_info['num_pred_devices']), 
                str(eval_info['num_true_devices']), str(eval_info['v_measure_score'])
            ]
    
    result = pd.DataFrame(result, index=('Processing time', "Predicted device number", 
                                            "True device number", 'V-measure'))
    result.to_csv(output_dir + "eval.csv", header=False)
    print("Results are saved in {}".format(output_dir + "eval_result.csv"))


if __name__ == "__main__":
    file_pth = "sample_input.csv"
    output_pth = "./data/result/"
    
    # Time range to assoc
    test_time_range = (0 * 3600 * 1000, 24 * 3600 * 1000) 

    # Load model
    assoc_model = vmac_assoc_api()

    # Preprocessing
    test_pkt_df = prepare_rawdata(filename=file_pth,
                                  time_range=test_time_range, 
                                  is_relative_time_range='auto', 
                                  category='phy') # Test on physical MAC for verification
    test_pkt_df.to_csv(os.path.join(output_pth, "df_info.csv"))
    test_pkt_df = inject_ref_by_tx_addr(test_pkt_df)
    all_ts = test_pkt_df['basic', 'timestamp']
    min_ts, max_ts = all_ts.min(), all_ts.max()
        
    start_ts = time.time()
    eval_info = assoc_model.assoc_run(test_pkt_df, output_pth, test_time_range)
    end_ts = time.time()

    eval_info["pred_label"].to_csv(os.path.join(output_pth, "Pred.csv"))
    eval_info["true_label"].to_csv(os.path.join(output_pth, "GT.csv"))

    save_result(round((max_ts - min_ts)/3600/1000, 1), round(end_ts - start_ts, 1), eval_info, output_pth)





    
    