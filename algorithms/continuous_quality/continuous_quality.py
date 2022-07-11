from itertools import product
from multiprocessing import Pool, get_context
from scipy.stats import truncnorm 

from algorithms.base_matcher import BaseMatcher
from data_loader.instance_loader import InstanceLoader


class ContinuousQualityMatcher(BaseMatcher):
    
    def __init__(self, threshold: float = 0.25, process_num: int = 1):
        self.threshold = threshold
        self.process_num = process_num

        # hardcoded
        mu_C = 0.44
        mu_K = 0.0
        sigma_C = 0.19
        sigma_K = 0.28

        a1, b1 = (- mu_C) / sigma_C, (1 - mu_C) / sigma_C           # Defining C limits
        a2, b2 = (- mu_K) / sigma_K, (1 - mu_K) / sigma_K           # Defining K limits
        self.Cmodel = truncnorm(a1, b1, scale=sigma_C, loc=mu_C)    # Creating C model
        self.Kmodel = truncnorm(a2, b2, scale=sigma_K, loc=mu_K)    # Creating K model

    def get_matches(self, source: InstanceLoader, target: InstanceLoader, dataset_name: str):

        matches = dict()
        src_table = source.table.name
        trg_table = target.table.name

        with get_context("spawn").Pool(self.process_num) as process_pool:
            for (col_src_name, col_src_obj), (col_trg_name, col_trg_obj) in \
                    product(source.table.columns.items(), target.table.columns.items()):

                sim = self.continuous_quality(self, col_src_obj.data, col_trg_obj.data, process_pool)

                matches[((src_table, col_src_name), (trg_table, col_trg_name))] = sim

        matches = dict(filter(lambda elem: elem[1] > self.threshold, matches.items()))

        sorted_matches = {k: v for k, v in sorted(matches.items(), key=lambda item: item[1], reverse=True)}

        return sorted_matches

    @staticmethod
    def continuous_quality(self, list1: list, list2: list, process_pool: Pool):
        A = set(list1)
        B = set(list2)
        max_l = max(len(A),len(B))
        min_l = min(len(A),len(B))

        k_value = float(min_l) / max_l if max_l else 0
        c_value = float(len(A&B)) / len(A) if len(A) else 0

        return self.Cmodel.cdf(c_value) * self.Kmodel.cdf(k_value)
