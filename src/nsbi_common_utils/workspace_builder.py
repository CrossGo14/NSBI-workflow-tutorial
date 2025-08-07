import json, math
import numpy as np
from collections.abc import Callable as CABC

class WorkspaceBuilder:
    """Collects functionality to build a workspace"""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Creates a workspace corresponding to configuration file"""
        self.config = config

    def channels(self) -> List[Dict[str, Any]]:
        """Returns the list of channel dictionaries, containing model information

        Returns:
            List[Dict[str, Any]]: channels for the custom NSBI workspace
        """
        regions      = self.config["Regions"]
        channels = []
        for region in regions:
            channel_dict = {}
            channel_dict.update(
                {
                    "name"      : region["Name"],
                    "type"      : region["Type"]
                }
            )
            samples = []

            for sample in self.config["Samples"]:

            

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
