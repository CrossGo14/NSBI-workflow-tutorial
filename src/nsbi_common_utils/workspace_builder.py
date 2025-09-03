import json, math
import numpy as np
import pandas as pd
from collections.abc import Callable as CABC
import pathlib
from nsbi_common_utils import configuration, datasets

class WorkspaceBuilder:
    """Collects functionality to build a workspace"""

    def __init__(self, config_path: Union[pathlib.Path, str]) -> None:
        """Creates a workspace corresponding to configuration file"""
        self.config_path = config_path
        self.config = nsbi_common_utils.configuration(file_path_string = config_path)
        self.config_dict = self.config.config

    def normfactor_modifiers(self, 
                             region_name: str, 
                             sample_name: str) -> list[dict[str, Any]]:
        '''
        returns the modifier list of all normfactors affecting a sample in a region
        '''
        list_dict_norm_factors = self.config.config.get("NormFactors", [])
        modifiers = []
        for norm_factor_dict in list_dict_norm_factors:
            norm_factor_name = norm_factor_dict["Name"]
            norm_factor_data = norm_factor_dict.get("Data", None)
            regions_affected = norm_factor_dict.get("Region", None)
            if regions_affected is not None:
                if region_name not in regions_affected:
                    continue
            samples_affected = norm_factor_dict.get("Samples", None)
            if samples_affected is not None:
                if sample_name not in samples_affected:
                    continue
                else:
                    modifiers.append({"name": norm_factor_name, 
                                      "data": norm_factor_data, 
                                      "type": "normfactor"})
        return modifiers

    def normplusshape_modifiers(self, 
                                dataset           : pd.DataFrane, 
                                region            : dict[str, Any], 
                                sample            : dict[str, Any], 
                                systematic_dict   : dict[str, Any],
                                nominal_data      : np.array):

        syst_name                      = systematic_dict["Name"]
        
        channel_name    = region["Name"]
        sample_name     = sample["Name"]
        sample_path     = sample["SamplePath"]
        region_variable = region[""]

        variation_data = {}
        
        for direction in ["Up", "Dn"]:

            key_syst            = syst_name + '_' + direction
            
            weights             = dataset[key_syst]["weights"].to_numpy()
            
            feature_var         = np.clip(dataset[key_syst][sample_name][region_variable],
                                                          np.amin(region_binning), np.amax(region_binning))
            
            syst_var_data, _       = np.histogram(feature_var, weights = weights, bins = region_binning)

            variation_data[direction] = syst_var_data / nominal_data
        
        modifiers = [{"name": syst_name,
                      "type": "histosys",
                      "data": {"hi_data": variation_data["Up"],
                               "lo_data": variation_data["Dn"]}}]

        return modifiers

    def sys_modifiers(self, dataset: pd.DataFrame, 
                      region: dict[str, Any], 
                      sample: dict[str, Any],
                      nominal_data) -> list[dict[str, Any]]:

        modifiers = []
        for systematic_dict in self.config.get("Systematics", []):
            syst_name = systematic_dict["Name"]
            syst_type = systematic_dict["Type"]

            regions_affected = systematic_dict.get("Regions", None)
            if regions_affected is not None:
                if region_name not in regions_affected:
                    continue
            samples_affected = systematic_dict.get("Samples", None)
            if samples_affected is not None:
                if sample_name not in samples_affected:
                    continue
                else:
                    if systematic["Type"] == "NormPlusShape":
                        modifiers += self.normplusshape_modifiers(
                            dataset, region, sample, systematic_dict, nominal_data
                        )
                    else:
                        raise NotImplementedError(
                            "not supporting other systematic types yet"
                        )
        return modifiers
        

    def channels(self) -> List[Dict[str, Any]]:
        """Returns the channel information: yields/density ratio models per sample and modifiers.

        Returns:
            List[Dict[str, Any]]: channels for workspace
        """
        channels = []
        for region in self.config_dict["Regions"]:
            channel = {}
            channel_name = region["Name"]
            channel_type = region["Type"]
            channel.update({"name": channel_name,
                            "type": channel_type})
            if region.get("Variable", None) is not None:
                type_of_fit  = "binned"
            else:
                type_of_fit  = "unbinned"
            channel.update({"type": type_of_fit})
            if type_of_fit == "binned":
                region_binning      = np.array(region["Binning"])
                region_variable     = region["Variable"]
                region_filters      = region["Filter"]
                branches_to_load    = [region_variable] 
                
                samples = []
                for sample_dict in self.config_dict["Samples"]:

                    current_sample = {}
                    
                    sample_name     = sample_dict["Name"]
                    current_sample.update({"name": sample_name})
                    
                    sample_path     = sample_dict["SamplePath"]
                    weight_var      = sample_dict.get("Weight", None)
                    branches_to_load_sample  = branches_to_load
                    if weight_var is not None:
                        branches_to_load_sample.append(weight_var)
                        
                    datasets            = nsbi_common_utils.dataset.dataset(self.config_path,
                                                                        branches_to_load =  branches_to_load_sample)
                    datasets_incl       = datasets.load_datasets_from_config(load_systematics = False)
                    dataset_region_dict = datasets.filter_region_by_type(datasets_incl, 
                                                                         region = channel_name)

                    feature_var         = np.clip(dataset_region_dict["Nominal"][sample_name][region_variable],
                                                  np.amin(region_binning), np.amax(region_binning))
                    if weight_var is not None:
                        weights = dataset_region_dict["Nominal"][weight_var].to_numpy()
                    else:
                        weights = np.ones_like(feature_var)
                        
                    sample_data, _       = np.histogram(feature_var, weights = weights, bins = region_binning)

                    current_sample.update({"data": sample_data})
 
                    modifiers = []

                    # modifiers can have region and sample dependence, which is checked
                    # check if normfactors affect sample in region, add modifiers as needed
                    nf_modifier_list = self.normfactor_modifiers(channel_name, sample_name)

                    modifiers += nf_modifier_list

                    # check if systematics affect sample in region, add modifiers as needed
                    sys_modifier_list = self.sys_modifiers(dataset_region_dict, region, sample_dict, sample_data)
                    modifiers += sys_modifier_list

                    current_sample.update({"modifiers": modifiers})                    
                
            channel.update({"samples": samples})
            channels.append(channel)
            
        return channels

    def measurements(self):
        
        measurements = []
        measurement = {}
        measurement.update({"name": self.config["General"]["Measurement"]})
        config_dict = {}

        # get the norm factor initial values / bounds / constant setting
        parameters_list = []
        for nf in self.config.get("NormFactors", []):
            nf_name = nf["Name"]  # every NormFactor has a name
            init = nf.get("Nominal", None)
            bounds = nf.get("Bounds", None)

            parameter = {"name": nf_name}
            if init is not None:
                parameter.update({"inits": [init]})
            if bounds is not None:
                parameter.update({"bounds": [bounds]})

            parameters_list.append(parameter)

        for sys in self.config.get("Systematics", []):
            # when there are many more systematics than NormFactors, it would be more
            # efficient to loop over fixed parameters and exclude all NormFactor related
            # ones to set all the remaining ones to constant (which are systematics)
            sys_name = sys["Name"]  # every systematic has a name

        parameters = {"parameters": parameters_list}
        config_dict.update(parameters)
        # POI defaults to "" (interpreted as "no POI" by pyhf) if not specified
        config_dict.update({"poi": self.config["General"].get("POI", "")})
        measurement.update({"config": config_dict})
        measurements.append(measurement)
        return measurements


    def build(self) -> Dict[str, Any]:
        """
        Constructs a workspace.

        Returns:
            Dict[str, Any]
        """
        ws: Dict[str, Any] = {}  # the workspace

        # channels
        channels = self.channels()
        ws.update({"channels": channels})

        # measurements
        measurements = self.measurements()
        ws.update({"measurements": measurements})

        # # build observations
        # observations = self.observations()
        # ws.update({"observations": observations})

        # workspace version
        ws.update({"version": "1.0.0"})

        return ws

    def add_systematic(
        self,
        channel: str,
        sample: str,
        name: str,
        typ: str,
        data: str,
        expr: str = "lam"
    ):
        """
        Register one modifier for one sample in one channel.
        `data` can be
          – a numpy array / list of per‑bin shifts,
          – OR a callable(nominal_bin_array)→array.
        """
        self._systematics\
            .setdefault(channel, {})\
            .setdefault(sample, [])\
            .append({
                "name": name,
                "typ": typ,
                "data": data,
                "expr": expr
            })

    def build_histogram_channel(
        self,
        hists: dict[str, np.ndarray],
        data: np.ndarray,
        channel: str = "SR",
    ):
        """
        Turn a { sample: hist_array } dict into a pyhf‐style channel JSON,
        picking up any systematics previously added with add_systematic().
        """
        ch = {
            "name": channel,
            "samples": []
        }

        # loop over every sample you gave me
        for samp, vals in hists.items():
            sample_dict = {
                "name": samp,
                "data": vals.tolist(),
                "modifiers": []
            }

            # see if the user registered any systematics for this (channel, sample)
            for mod in self._systematics.get(channel, {}).get(samp, []):
                # evaluate the .data if it's a function
                raw = mod["data"]
                if callable(raw):
                    raw = raw(vals)
                # make sure it ends up as a Python list in the JSON
                sample_dict["modifiers"].append({
                    "name": mod["name"],
                    "typ": mod["typ"],
                    "data": (raw.tolist() if isinstance(raw, np.ndarray)
                             else raw),
                    "expr": mod["expr"]
                })

            ch["samples"].append(sample_dict)

        # store it
        self.channel_specs.append(ch)
        self.observations.append({
            "name": channel,
            "data": data.tolist()
        })

    def build_workspace(
        self,
        list_of_pois,
        list_of_unconstrained_nps,
        list_of_constrained_nps,
        initial_values
    ):
        return {
            "version":       self.version,
            "channels":      self.channel_specs,
            "observations":  self.observations,
            "measurements": [
                {
                    "name":    "measurement",
                    "config": {
                        "pois":                            list_of_pois,
                        "unconstrained_nuisance_parameters": list_of_unconstrained_nps,
                        "constrained_nuisance_parameters":   list_of_constrained_nps,
                        "inits":                            initial_values
                    }
                }
            ]
        }

    def dump_workspace(self, ws: dict, outpath: str = "workspace.json"):
        with open(outpath, "w") as f:
            json.dump(ws, f, indent=2)
        print(f"Wrote {outpath}")
