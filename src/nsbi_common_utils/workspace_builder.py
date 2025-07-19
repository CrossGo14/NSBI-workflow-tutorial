# utils/workspace.py  (create a helper once and re‑use)
import json, math
import numpy as np

lambda kappa_t, alpha_over_pi: jnp.cos(alpha_over_pi * jnp.pi)**2 * kappa_t**2

class workspace_builder:

    def __init__(self, version="1.0.0"):

        self.version = version
        self.channel_list = []
        self.data_observed = []

    def build_histogram_channel(
        self,
        hists: dict[str, np.ndarray],
        data: np.ndarray,
        channel: str = "SR",
        parameter_name: str = dict[str, "mu"],
        modifier_functions: dict[str, lambda x: x]
    ):
        """
        Convert {sample: histogram} dictionary for a given channel -> JSON workspace dictionary
        hists: dictionary of histograms, with sample names as keys
        data: observed data histogram
        channel: channel name
        parameter_name: name of the parameter 
        modifier_functions: 
        """
        channel_spec = {"name": channel, 
                        "samples": []}
    
        for samp, vals in hists.items():
    
            channel_spec["samples"].append(
                {"name": samp, 
                 "data": vals.tolist(), 
                 "modifiers": modifier_functions[samp]}
            )
            
        self.channel_list.append(channel)
        self.data_observed.append({"name": channel, "data": data.tolist()})
        
    
    def build_workspace(self, list_of_pois, list_of_unconstrained_nps, list_of_constrained_nps, intial_value_all_params):
        
        workspace = {
            "version":       self.version,
            "channels":      self.channel_list,
            "observations":  self.data_observed,
            "measurements":  [
                {
                    "name": "measurement",
                    "config": {
                        
                        "pois": list_of_pois,
                        "unconstrained_nuisance_parameters": list_of_unconstrained_nps,
                        "constrained_nuisance_parameters": list_of_constrained_nps,
                        "inits": intial_value_all_params
                }
            ],
        }
        
        return workspace
    
    
    def dump_workspace(self,
                       ws: dict, 
                       outpath: str = "workspace.json"):
        
        with open(outpath, "w") as f:
            json.dump(ws, f, indent=2)
        print(f"✔  Wrote {outpath}")
