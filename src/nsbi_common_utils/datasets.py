import os
import pandas as pd
import numpy as np
import uproot
import pathlib
from typing import Any, Dict, List, Literal, Optional, Union


class datasets:

    def __init__(self, 
                config, 
                branches_to_load: List):

        self.config                     = config
        if len(branches_to_load) == 0:
            raise Exception(f"Empty branch list.")
        self.branches_to_load           = branches_to_load
        self.branches_all               = self.branches_to_load

    def load_datasets_from_config(self,
                                load_systematics = False):

        dict_datasets = {}
        dict_datasets["Nominal"] = {}

        for dict_sample in self.config["Samples"]:

            weight_branch       = [dict_sample["Weight"]] if "Weight" in dict_sample.keys() else []

            path_to_root_file   = dict_sample["SamplePath"]
            tree_name           = dict_sample["Tree"]
            sample_name         = dict_sample["Name"]
            branches_to_load = self.branches_to_load + weight_branch
            dict_datasets["Nominal"][sample_name] = self._load_dataset(path_to_root_file, 
                                                                    tree_name, 
                                                                    branches_to_load)

            if "Weight" in dict_sample.keys():
                dict_datasets["Nominal"][sample_name] = dict_datasets["Nominal"][sample_name].rename(columns={dict_sample['Weight']: "weights"})
            else:
                dict_datasets["Nominal"][sample_name]["weights"] = 1.0

        if load_systematics:
            for dict_syst in self.config["Systematics"]:

                syst_name = dict_syst["Name"]
                syst_type = dict_syst["Type"]
                if syst_type == "NormPlusShape":
                    for direction in ["Up", "Dn"]:
                        syst_name_var        = syst_name + "_" + direction
                        dict_datasets[syst_name_var] = {}
                        for dict_sample in dict_syst[direction]:
                            path_to_root_file   = dict_sample["Path"]
                            sample_name         = dict_sample["SampleName"]
                            tree_name           = dict_sample["Tree"]
                            weight_branch       = [dict_sample["Weight"]] if "Weight" in dict_sample.keys() else []
                            branches_to_load = self.branches_to_load + weight_branch
                            dict_datasets[syst_name_var][sample_name] = self._load_dataset(path_to_root_file, 
                                                                                    tree_name, 
                                                                                    branches_to_load)


                            if "Weight" in dict_sample.keys():
                                dict_datasets[syst_name_var][sample_name] = dict_datasets[syst_name_var][sample_name].rename(columns={dict_sample['Weight']: "weights"})
                            else:
                                dict_datasets[syst_name_var][sample_name]["weights"] = 1.0

        return dict_datasets

    def _load_dataset(self,
                    path_to_root_file  : str,
                    tree_name         : str,
                    branches_to_load  : list):

        with uproot.open(f"{path_to_root_file}:{tree_name}") as tree:
            dataframe = tree.arrays(branches_to_load, library="pd")

        return dataframe

    def add_appended_branches(self, 
                              branches: List):

        self.branches_all           = self.branches_to_load + branches


    def save_datasets(self,
                    dict_datasets,
                    save_systematics = False):

        for dict_sample in self.config["Samples"]:

            path_to_root_file   = dict_sample["SamplePath"]
            tree_name           = dict_sample["Tree"]
            sample_name         = dict_sample["Name"]
            self._save_dataset(dict_datasets["Nominal"][sample_name], 
                                path_to_root_file, 
                                tree_name)

        if save_systematics:
            for dict_syst in self.config["Systematics"]:

                syst_name = dict_syst["Name"]
                syst_type = dict_syst["Type"]
                if syst_type == "NormPlusShape":
                    for direction in ["Up", "Dn"]:
                        syst_name_var        = syst_name + "_" + direction
                        if syst_name_var not in dict_datasets.keys(): continue
                        for dict_sample in dict_syst[direction]:
                            path_to_root_file   = dict_sample["Path"]
                            sample_name         = dict_sample["SampleName"]

                            if sample_name not in dict_datasets[syst_name_var].keys(): continue

                            tree_name           = dict_sample["Tree"]
                            self._save_dataset(dict_datasets[syst_name_var][sample_name],
                                                path_to_root_file, 
                                                tree_name)


    # def _save_dataset(self, 
    #                 dataset,
    #                 path_to_root_file  : str,
    #                 tree_name         : str):

    #     if "weights" not in self.branches_all:
    #         self.branches_all =  self.branches_all + ["weights"]
    #     dataset = dataset[self.branches_all]

    #     tmp_path = path_to_root_file + ".tmp"
    #     with uproot.open(path_to_root_file) as fin, uproot.recreate(tmp_path) as fout:
    #         for tree_name_in_file, tree_obj in fin.items(): 
    #             tree_name_in_file = tree_name_in_file.split(";")[0]
    #             if tree_name_in_file == tree_name:
    #                 continue 
    #             try:
    #                 fout[tree_name_in_file] = tree_obj  
    #             except Exception:
    #                 print(f"Unable to save.")
    #                 pass

    #         fout[tree_name] = dataset

    #     os.replace(tmp_path, path_to_root_file)  

    def _save_dataset(self,
                    dataset, 
                    path_to_root_file: str, 
                    tree_name: str):

        if "weights" not in self.branches_all:
            self.branches_all =  self.branches_all + ["weights"]
        dataset = dataset[self.branches_all]

        tmp_path = path_to_root_file + ".tmp"

        with uproot.open(path_to_root_file) as fin, uproot.recreate(tmp_path) as fout:
            for _tree_name, classname in fin.classnames().items():
                _tree_name = _tree_name.split(";")[0]
                if _tree_name == tree_name:
                    continue
                if classname == "TTree":
                    arrs = fin[_tree_name].arrays(library="ak")
                    fout[_tree_name] = arrs

            fout[tree_name] = dataset

        os.replace(tmp_path, path_to_root_file)
