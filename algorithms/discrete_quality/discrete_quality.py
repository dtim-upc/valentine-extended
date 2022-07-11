from itertools import product
from multiprocessing import Pool, get_context
import Levenshtein as Lv

from algorithms.base_matcher import BaseMatcher
from data_loader.instance_loader import InstanceLoader


class DiscreteQualityMatcher(BaseMatcher):

    def __init__(self, process_num: int = 1, L: int = 4): 
        self.process_num = process_num
        self.L = L

    def get_matches(self, source: InstanceLoader, target: InstanceLoader, dataset_name: str):

        matches = dict()
        src_table = source.table.name
        trg_table = target.table.name

        with get_context("spawn").Pool(self.process_num) as process_pool:
            for (col_src_name, col_src_obj), (col_trg_name, col_trg_obj) in \
                    product(source.table.columns.items(), target.table.columns.items()):

                sim = self.discrete_quality(self, col_src_obj.data, col_trg_obj.data, process_pool)

                matches[((src_table, col_src_name), (trg_table, col_trg_name))] = sim

        matches = dict(filter(lambda elem: elem[1] > 0.0, matches.items()))  # Remove the pairs with zero similarity

        sorted_matches = {k: v for k, v in sorted(matches.items(), key=lambda item: item[1], reverse=True)}

        return sorted_matches

    @staticmethod
    def discrete_quality(self, list1: list, list2: list, process_pool: Pool):
        A = set(list1)
        B = set(list2)
        max_l = max(len(A),len(B))
        min_l = min(len(A),len(B))

        k_value = float(min_l) / max_l if max_l else 0
        c_value = float(len(A&B)) / len(A) if len(A) else 0

        if c_value * k_value == 1.0:
            return 1

        for i in range(0,self.L):
            if c_value >= 1-i/self.L and k_value >= 1/2**i:
                return float(self.L-i+1)/self.L
        return 0
