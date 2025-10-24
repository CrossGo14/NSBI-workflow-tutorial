import os
import pandas as pd
import numpy as np
import uproot
import copy
import pathlib
from typing import Any, Dict, List, Literal, Optional, Union

from nsbi_common_utils.configuration import ConfigManager

class datasets:

    def __init__(self, 
                config_path: Union[pathlib.Path, str], 
                branches_to_load: List):

        self.config              = ConfigManager(file_path_string = config_path)
        
        if len(branches_to_load) == 0:
            raise Exception(f"Empty branch list.")
        self.branches_to_load           = list(branches_to_load)
        self.branches_all               = list(self.branches_to_load)

    def load_datasets_from_config(self,
                                load_systematics = False):

        dict_datasets = {}
        dict_datasets["Nominal"] = {}

        for dict_sample in self.config.config["Samples"]:

            weight_branch       = [dict_sample["Weight"]] if "Weight" in dict_sample.keys() else []

            path_to_root_file   = dict_sample["SamplePath"]
            tree_name           = dict_sample["Tree"]
            sample_name         = dict_sample["Name"]
            branches_to_load    = list(self.branches_to_load)
            if weight_branch[0] not in branches_to_load:
                branches_to_load += weight_branch
                
            dict_datasets["Nominal"][sample_name] = load_dataframe_from_root(path_to_root_file, 
                                                                            tree_name, 
                                                                            branches_to_load)

            dict_datasets["Nominal"][sample_name]["sample_name"] = sample_name

            if "Weight" in dict_sample.keys():
                dict_datasets["Nominal"][sample_name] = dict_datasets["Nominal"][sample_name].rename(columns={dict_sample['Weight']: "weights"})
            else:
                dict_datasets["Nominal"][sample_name]["weights"] = 1.0

        if load_systematics:
            systematics_dict_list = self.config.config.get("Systematics", [{}])
            for dict_syst in systematics_dict_list:
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
                            branches_to_load    = list(self.branches_to_load)
                            if weight_branch[0] not in branches_to_load:
                                branches_to_load += weight_branch
                            dict_datasets[syst_name_var][sample_name] = load_dataframe_from_root(path_to_root_file, 
                                                                                                tree_name, 
                                                                                                branches_to_load)

                            dict_datasets[syst_name_var][sample_name]["sample_name"] = sample_name

                            if "Weight" in dict_sample.keys():
                                dict_datasets[syst_name_var][sample_name] = dict_datasets[syst_name_var][sample_name].rename(columns={dict_sample['Weight']: "weights"})
                            else:
                                dict_datasets[syst_name_var][sample_name]["weights"] = 1.0

        return dict_datasets

    def add_appended_branches(self, 
                              branches: List):

        self.branches_all           = self.branches_to_load + branches


    def save_datasets(self,
                    dict_datasets,
                    save_systematics = False):

        for dict_sample in self.config.config["Samples"]:

            path_to_root_file   = dict_sample["SamplePath"]
            tree_name           = dict_sample["Tree"]
            sample_name         = dict_sample["Name"]
            self._save_dataset(dict_datasets["Nominal"][sample_name], 
                                path_to_root_file, 
                                tree_name)

        if save_systematics:
            systematics_dict_list = self.config.config.get("Systematics", [{}])
            for dict_syst in systematics_dict_list:

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

    def filter_region_by_type(self,
                             dataset: Dict[str, Dict[str, pd.DataFrame]],
                             region: str) -> Dict[str, Dict[str, pd.DataFrame]]:

        for type_name, type_dict in dataset.items():
            dataset[type_name] = self.filter_region_dataset(type_dict, region = region)

        return dataset

    def filter_region_dataset(self,
                              dataset: Dict[str, pd.DataFrame],
                              region: str) -> Dict[str, pd.DataFrame]:

        region_filters = self.config.get_channel_filters(channel_name = region)
        for sample_name, sample_dataframe in dataset.items():
            dataset[sample_name] = sample_dataframe.query(region_filters).copy()
        return dataset

    def merge_dataframe_dict_for_training(self, 
                                        dataset_dict, 
                                        label_sample_dict: Union[dict[str, int], None] = None,
                                        samples_to_merge = []):

        if len(samples_to_merge) == 0:
            raise Exception

        list_dataframes = []
        for sample_name, dataset in dataset_dict.items():
            if sample_name not in samples_to_merge: continue
            list_dataframes.append(dataset)

        dataset = pd.concat(list_dataframes)

        if label_sample_dict is not None:

            dataset = self._add_normalised_weights_and_train_label_class(dataset, 
                                                                        label_sample_dict)

        return dataset

    def _add_normalised_weights_and_train_label_class(self,
                                                    dataset, 
                                                    label_sample_dict: dict[str, int]):

        dataset['weights_normed']       = dataset['weights'].to_numpy()
        dataset['train_labels']         = -999

        for sample_name, label in label_sample_dict.items():

            mask_sample_name                                     = np.isin(dataset["sample_name"], [sample_name])

            dataset.loc[mask_sample_name, "train_labels"]        = label

        train_labels_unique = np.unique(dataset.train_labels)

        for train_label in train_labels_unique:

            mask_train_label                                     = np.isin(dataset["train_labels"], [train_label])

            total_train_weight                                   = dataset.loc[mask_train_label, "weights"].sum()

            dataset.loc[mask_train_label, "weights_normed"]      = dataset.loc[mask_train_label, "weights_normed"] / total_train_weight

        return dataset


def save_dataframe_as_root(dataset        : pd.DataFrame,
                           path_to_save   : str,
                           tree_name      : str) -> None:

    with uproot.recreate(f"{path_to_save}") as ntuple:

        arrays = {col: dataset[col].to_numpy() for col in dataset.columns}

        ntuple[tree_name] = arrays
        

def load_dataframe_from_root(path_to_load      : str,
                           tree_name         : str,
                           branches_to_load  : list) -> pd.DataFrame:

    with uproot.open(f"{path_to_load}:{tree_name}") as tree:
            dataframe = tree.arrays(branches_to_load, library="pd")

    return dataframe
