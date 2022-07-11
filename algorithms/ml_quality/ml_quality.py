from itertools import product
from multiprocessing import Pool, get_context
import os
import pandas as pd
import json

from algorithms.base_matcher import BaseMatcher
from data_loader.instance_loader import InstanceLoader
from data_loader.data_objects.table import Column
from utils.utils import get_project_root
import pickle 


class ML_ModelQuality(BaseMatcher):

    def __init__(self, threshold: float = 0.5, process_num: int = 1):
        self.threshold = threshold
        self.process_num = process_num
        self.project_root_path = get_project_root()
        self.jar_filepath = self.project_root_path + "/algorithms/ml_quality/artifact/NextiaJD.jar"

        # Load ML model from file
        model_filepath = self.project_root_path + "/algorithms/ml_quality/artifact/ML_Best_model.pkl"
        with open(model_filepath, 'rb') as file:
            self.model = pickle.load(file)

    def get_matches(self, source: InstanceLoader, target: InstanceLoader, dataset_name: str):
        matches = dict()
        src_table = source.table.name
        trg_table = target.table.name

        command_string = 'java -jar %s --computeDistances --pathA %s --pathB %s --output distances' % \
            (self.jar_filepath, self.project_root_path + '/' + source.data_path, self.project_root_path + '/' + target.data_path)

        os.system(command_string)

        with get_context("spawn").Pool(self.process_num) as process_pool:
            for (col_src_name, col_src_obj), (col_trg_name, col_trg_obj) in \
                    product(source.table.columns.items(), target.table.columns.items()):
                sim = self.ml_model_quality(self, col_src_obj, col_trg_obj, process_pool)

                matches[((src_table, col_src_name), (trg_table, col_trg_name))] = sim

        matches = dict(filter(lambda elem: elem[1] > self.threshold, matches.items()))

        sorted_matches = {k: v for k, v in sorted(matches.items(), key=lambda item: item[1], reverse=True)}

        return sorted_matches

    @staticmethod
    def ml_model_quality(self, col1: Column, col2: Column, process_pool: Pool):
        list1 = col1.data
        list2 = col2.data

        A = set(list1)
        B = set(list2)
        max_l = max(len(A), len(B))
        min_l = min(len(A), len(B))
        k_value = float(min_l) / max_l if max_l else 0

        profiles = []
        profile_filepath = self.project_root_path + "/distances/distances.json"
        with open(profile_filepath) as file:
            for jsonProfile in file:
                profile = json.loads(jsonProfile)
                profiles.append(profile)
        
        profile_dict = [elem for elem in profiles if elem['att_name'] == col1.name and elem['att_name_2'] == col2.name]
        
        if not profile_dict:
            return 0 # Els atributs integer s'han tret del ground truth 
        
        profile_dict = profile_dict[0]

        df = pd.DataFrame([profile_dict])
        df['K'] = k_value
        df['freqWordCleanContainment'] = df['freqWordContainment']
        
        X = df[['frequency_avg', 'frequency_min', 'frequency_max', 'frequency_sd',
       'val_pct_std', 'frequency_2qo', 'frequency_4qo', 'frequency_6qo',
       'frequency_1qo', 'frequency_3qo', 'frequency_5qo', 'frequency_7qo',
       'entropy', 'freqWordContainment', 'freqWordContainment',
       'freqWordSoundexContainment', 'PctAlphabetic', 'PctDateTime',
       'PctAlphanumeric', 'PctNumeric', 'PctNonAlphanumeric', 'PctPhones',
       'PctEmail', 'PctIP', 'PctURL', 'PctUsername', 'PctGeneral', 'PctOthers',
       'PctSpaces', 'PctPhrases', 'cardinality', 'incompleteness',
       'len_max_word', 'len_min_word', 'len_avg_word', 'firstWord', 'lastWord',
       'wordsCntAvg', 'wordsCntMin', 'wordsCntMax', 'wordsCntSd',
       'numberWords', 'uniqueness', 'isempty', 'val_pct_min', 'val_pct_max',
       'frequency_IQR', 'isBinary', 'isempty_2', 'name_dist', 'K']]

        Ypredict = self.model.predict(X)
        return Ypredict[0]
