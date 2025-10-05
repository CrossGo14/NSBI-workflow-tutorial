import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import Dict, Union, Any

class Model:
    """
    The RooFit equivalent in the NSBI case - class that defines the core model to be passed to fitting algotithms
    """
    def __init__(self, 
                 workspace: Dict[Any, Any],
                 measurement: str):

        self.workspace                                  = workspace
        self.measurements_dict: list[Dict[str, Any]]    = workspace["measurements"]

        for measurement in self.measurements_dict:
            measurement_name = measurement.get("name")
            if measurement_name == measurement:
                self.measurement_name                   = measurement_name
                self.poi                                = measurement["poi"]
        
        self.channels_binned                            = self._get_channel_list(type="binned")
        self.channels_unbinned                          = self._get_channel_list(type="unbinned")
        self.all_channels                               = self.channels_binned + self.channels_unbinned

        self.all_samples                                = self._get_samples_list()
        
        self.list_parameters, \
            self.list_parameters_types                  = self._get_parameters()
        
        self.list_syst_normplusshape                    = self._get_list_syst_for_interp() 
        self.list_normfactors, self.norm_sample_map     = self._get_norm_factors() 

        self.index_normparam_map: Dict[str, list[str]]  = self._make_map_index_norm()


        self.yield_array_dict, _                        = self._get_nominal_expected_arrays(type = "binned")
        self.unbinned_total_dict, \
            self.ratios_array_dict                      = self._get_nominal_expected_arrays(type = "unbinned")

        self.weight_arrays_unbinned                     = self._get_asimov_weights_array(type = "unbinned")

        self.combined_var_up_binned, \
            self.combined_var_dn_binned                 = self._get_systematic_data(type="binned")
        
        self.combined_var_up_unbinned, \
            self.combined_var_dn_unbinned, \
                self.combined_tot_up_unbinned, \
                    self.combined_tot_dn_unbinned       = self._get_systematic_data(type="unbinned")

    def _get_norm_factors(self):
        """Assume same normfactor across channels for now (TODO: Add support for normfactor per channel)"""
        dict_sample_normfactors = {sample_name: [] for sample_name in self.all_samples}
        for channel in self.all_channels[:1]:
            channel_index = self._index_of_region(channel_name=channel)
            for sample in self.all_samples:
                sample_index = self._index_of_samples(channel_name=channel, sample_name=sample)
                modifier_list = self.workspace["channels"][channel_index]["samples"][sample_index]["modifiers"]
                for modifier in modifier_list:
                    if modifier["type"] == "normfactor":
                        modifier_name = modifier["name"]
                        if modifier_name in dict_sample_normfactors[sample]: continue
                        dict_sample_normfactors[sample].append(modifier_name)

        return dict_sample_normfactors

    def _get_list_syst_for_interp(self):
        """Get the list of subset of systematics that need interpolation."""
        mask_normplusshape  = (self.list_parameters_types == "normplusshape")
        list_normplusshape  = self.list_parameters[mask_normplusshape].copy()
        return list_normplusshape

    def _get_channel_list(self, 
                          type: Union[Str, None] = None) -> list:
        """Get the channel list to be used in the measurement"""
        list_channels = []
        channels: list[Dict[str, Any]] = self.workspace["channels"]
        for channel_dict in channels:
            if type is not None:
                if channel_dict.get("type") != type: continue 
            list_channels.append(channel_dict.get("name"))
        return list_channels
    
    def _get_samples_list(self):
        """Get the sample list from the first channel"""
        list_samples = []
        channels: list[Dict[str, Any]] = self.workspace["channels"]
        for channel_dict in channels:
            samples: list[Dict[str, Any]] = channel_dict["samples"]
            for sample_dict in samples:
                list_samples.append(sample_dict.get("name"))
                break
        return list_samples
    
    def _get_asimov_weights_array(self):
        """
        Get the Asimov weight vector for fitting
        """
        weight_array = np.array([])
        for channel in self.channels_unbinned:
            channel_index       = self._index_of_region(channel)
            weights             = np.load(self.workspace["channels"][channel_index]["weights"])
            weight_array        = np.append([weight_array, weights])
        return weight_array
    
    def _get_parameters(self):
        """Get a list of all parameters."""
        list_param_names = []
        list_param_types = []
        channels: list[Dict[str, Any]] = self.workspace["channels"]
        for channel_dict in channels:
            samples: list[Dict[str, Any]] = channel_dict["samples"]
            for sample_dict in samples:
                modifiers_list: list[Dict[str, Any]] = sample_dict["modifiers"]
                for modifier in modifiers_list:
                    modifier_name = modifier.get("name")
                    modifier_type = modifier.get("type")
                    if modifier_name not in list_param_names:
                        list_param_names.append(modifier_name)
                        list_param_types.append(modifier_type)
        return list_param_names, list_param_types
    
    @jax.jit
    def calculate_parameterized_yields(self, param_vec, hist_yields, hist_vars):

        nu_tot = 0.0
        
        for process in self.all_samples:
            # This will not work in the general case where model is non-linear in POI, needs modifications (TO-DO)
            nu_tot += param_vec[param_index] * hist_yields[process] * hist_vars[process]

        for process in self.fixed_processes:
            nu_tot += hist_yields[process] * hist_vars[process]

        return nu_tot
    
    @jax.jit
    def _calculate_parameterized_ratios(self, param_vec, 
                                        nu_nominal, nu_vars, 
                                        ratios, ratio_vars):

        dnu_dx = jnp.zeros_like(self.weight_arrays_unbinned)

        for process in self.all_samples:
            param_index = tuple(self.process_index[process])
            dnu_dx += param_vec[param_index] * nu_vars[process] * nu_nominal[process] * ratios[process] * ratio_vars[process]

        for process in self.fixed_processes:
            dnu_dx += nu_vars[process] * nu_nominal[process] * ratios[process] * ratio_vars[process]
            
        return jnp.log( dnu_dx )
    
    def model(self, param_array):
        """
        Output model to pass onto inference algorithms
        """
        param_array_interpolation           = param_vec[self.num_unconstrained_params:]
        list_syst
        nll                                 = self.nll_function(param_array, 
                                                                param_array_interpolation,
                                                                self.combined_var_up_binned,
                                                                self.combined_var_dn_binned  ,
                                                                self.combined_var_up_unbinned,
                                                                self.combined_var_dn_unbinned,
                                                                self.combined_tot_up_unbinned,
                                                                self.combined_tot_dn_unbinned)
        return nll
    
    @jax.jit
    def nll_function(self, 
                     param_vec                  : list, 
                     param_vec_interpolation    : list,
                     combined_var_up_binned     : np.array(),
                     combined_var_dn_binned     : np.array(),
                     combined_var_up_unbinned   : np.array(),
                     combined_var_dn_unbinned   : np.array(),
                     combined_tot_up_unbinned   : np.array(),
                     combined_tot_dn_unbinned   : np.array()):
        """
        Optimized function for NLL computations
        """
        norm_modifiers          = {}
        hist_vars_binned        = {}
        hist_vars_unbinned      = {}
        ratio_vars_unbinned     = {}

        for process in self.all_samples:
            
            norm_modifiers[process]     = calculate_norm_variations(param_vec, self.index_normparam_map)
            hist_vars_binned[process]   = calculate_combined_var(  param_vec_interpolation, 
                                                                        combined_var_up_binned[process],
                                                                        combined_var_dn_binned[process]    )

            hist_vars_unbinned[process]  = calculate_combined_var(  param_vec_interpolation, 
                                                                        combined_tot_up_unbinned[process],
                                                                        combined_tot_dn_unbinned[process]    )   

            ratio_vars_unbinned[process] = calculate_combined_var( param_vec_interpolation, 
                                                                        combined_var_up_unbinned[process],
                                                                        combined_var_dn_unbinned[process]    )        

        nu_binned = self.calculate_parameterized_yields(param_vec, 
                                                        self.yield_array_dict, 
                                                        hist_vars_binned)

        data_hist = self.data_hist_channel

        llr_tot_binned = pois_loglikelihood(data_hist, nu_binned)

        nu_tot_unbinned = self.calculate_parameterized_yields(param_vec, 
                                                                  self.unbinned_total_dict, 
                                                                  hist_vars_unbinned)

        llr_tot_unbinned = self._calculate_parameterized_ratios(param_vec, 
                                                               self.unbinned_total_dict, 
                                                               hist_vars_unbinned, 
                                                               self.ratios_array_dict, 
                                                               ratio_vars_unbinned) \
                            - jnp.log(nu_tot_unbinned)
        
        llr_tot = llr_tot_binned + llr_tot_unbinned + jnp.sum(param_vec_interpolation**2)        

        return llr_tot
    
    

    def _get_nominal_expected_arrays(self, type:str):
        """
        Get an array of expected event yields or ratios
        """
        data_expected   = {sample_name : np.array([]) for sample_name in self.all_samples}
        ratio_expected  = {sample_name : np.array([]) for sample_name in self.all_samples}

        if type == "binned":
            channels_list       =   self.channels_binned
        elif type == "unbinned": 
            channels_list       =   self.channels_unbinned 

        for sample_name in self.all_samples:

            for channel_name in channels_list:

                channel_index           = self._index_of_region(channel_name = channel_name)
                sample_index            = self._index_of_sample(channel_name = channel_name,
                                                                    sample_name  = sample_name)
            
                if type == "binned":
                    sample_data             = np.array(self.workspace["channels"][channel_index]["samples"][sample_index]["data"])
                    sample_ratio            = np.array([])
                elif type == "unbinned":
                    sample_data             = np.array(self.workspace["channels"][channel_index]["samples"][sample_index]["data"])
                    sample_ratio            = np.load(self.workspace["channels"][channel_index]["samples"][sample_index]["ratios"])

                data_expected[sample_name]  =   np.append([data_expected[sample_name], sample_data])
                ratio_expected[sample_name] =   np.append([ratio_expected[sample_name], sample_ratio])

        return data_expected, ratio_expected
    
    def _get_systematic_data(self, type: str) -> Dict[str, np.array()]:
        """
        Builds a rectangular array with (N_syst, N_datapoints) dimensions, where N_datapoints is the number of bins in binned channels and number of events in unbinned channels. 
        Concatenates all binned or all unbinned channels into one big array for array-based computations.

        type -> choose if building array for "unbinned" channels or "binned"
        """
        if type == "binned":
            base_array_for_size     = self.yield_array_dict[sample_name]
            channel_list            = self.channels_binned
        elif type == "unbinned":
            base_array_for_size     = self.ratios_array_dict[sample_name]
            base_tot_for_size       = self.unbinned_total_dict[sample_name]
            channel_list            = self.channels_unbinned

        combined_var_up             = {sample_name: np.ones((self.list_syst_normplusshape, 
                                                             base_array_for_size)) for sample_name in self.all_samples}
        combined_var_dn             = combined_var_up.copy()

        if type == "unbinned":
            combined_tot_up             = {sample_name: np.ones((self.list_syst_normplusshape, 
                                                             base_tot_for_size)) for sample_name in self.all_samples}
            combined_tot_dn             = combined_tot_up.copy()

        for sample_name in self.all_samples:

            for count, systematic_name in enumerate(self.list_syst_normplusshape):

                var_up_array_syst = np.array([])
                var_dn_array_syst = np.array([])

                if type == "unbinned":

                    var_up_tot_syst = np.array([])
                    var_dn_tot_syst = np.array([])

                for channel_name in channel_list:

                    channel_index           = self._index_of_region(channel_name = channel_name)
                    sample_index            = self._index_of_sample(channel_name = channel_name,
                                                                    sample_name  = sample_name)
                    
                    modifier_index          = self._index_of_systematic(channel_name    = channel_name,
                                                                        sample_name     = sample_name,
                                                                        systematic_name = systematic_name)
                    
                    modifier_dict           = self.workspace["channels"][channel_index]["samples"][sample_index]["modifiers"][modifier_index]

                    if type == "binned":
                        var_array_up_channel = modifier_dict["data"]["hi_data"]
                        var_array_dn_channel = modifier_dict["data"]["lo_data"]

                    elif type == "unbinned":
                        var_array_up_channel = np.load(modifier_dict["data"]["hi_ratio"])
                        var_total_up_channel = np.load(modifier_dict["data"]["hi_data"])

                        var_array_dn_channel = np.load(modifier_dict["data"]["lo_ratio"])
                        var_total_dn_channel = np.load(modifier_dict["data"]["lo_data"])

                        var_up_tot_syst       = np.append([var_up_tot_syst, var_total_up_channel])
                        var_dn_tot_syst       = np.append([var_dn_tot_syst, var_total_dn_channel])

                    var_up_array_syst       = np.append([var_up_array_syst, var_array_up_channel])
                    var_dn_array_syst       = np.append([var_dn_array_syst, var_array_dn_channel])

                combined_var_up[sample_name][count] = var_up_array_syst
                combined_var_dn[sample_name][count] = var_dn_array_syst

                if type == "unbinned":
                    combined_tot_up[sample_name][count] = var_up_tot_syst
                    combined_tot_dn[sample_name][count] = var_dn_tot_syst
                    return combined_var_up, combined_var_dn, combined_tot_up, combined_tot_dn

        return combined_var_up, combined_var_dn

        
    def _index_of_modifiers(self, 
                          channel_name: str,
                          sample_name: str,
                          systematic_name: str) -> Optional[int]:
        """
        Get the index associated with a systematic, in a specific sample of a particular channel
        """
        channel_index                       = self._index_of_region(channel_name)
        sample_index                        = self._index_of_samples(sample_name)
        modifiers: list[dict[str, Any]]     = self.workspace["channels"][channel_index]["samples"][sample_index]["modifiers"]
        for count, modifier in modifiers:
            if modifier.get("name") == systematic_name:
                return count
        return None

    def _index_of_samples(self, 
                          channel_name: str,
                          sample_name: str) -> Optional[int]:
        """
        Get the index associated with a sample, in a particular channel
        """
        channel_index = self._index_of_region(channel_name)
        samples: list[dict[str, Any]] = self.workspace["channels"][channel_index]["samples"]
        for count, sample in samples:
            if sample.get("name") == sample_name:
                return count
        return None

    def _index_of_region(self, channel_name: str) -> Optional[int]:
        """
        Get the index associated with a particular channel in the workspace
        """
        channels: list[dict[str, Any]] = self.workspace["channels"]
        for count, channel in enumerate(channels):
            if channel.get("name") == channel_name:
                return count
        return None

    
# poynomial interpolation, same as HistFactory
@jax.jit
def poly_interp(tuple_input):
    """
    Function for polynomial interpolation
    """
    alpha, pow_up, pow_down = tuple_input
    
    logHi         = jnp.log(pow_up)
    logLo         = jnp.log(pow_down)
    pow_up_log    = jnp.multiply(pow_up, logHi)
    pow_down_log  = -jnp.multiply(pow_down, logLo)
    pow_up_log2   =  jnp.multiply(pow_up_log, logHi)
    pow_down_log2 = -jnp.multiply(pow_down_log, logLo)

    S0 = (pow_up + pow_down) / 2.0
    A0 = (pow_up - pow_down) / 2.0
    S1 = (pow_up_log  + pow_down_log) / 2.0
    A1 = (pow_up_log  - pow_down_log) / 2.0
    S2 = (pow_up_log2 + pow_down_log2) / 2.0
    A2 = (pow_up_log2 - pow_down_log2) / 2.0

    a1 = ( 15 * A0 -  7 * S1 + A2)      / 8.0
    a2 = (-24 + 24 * S0 -  9 * A1 + S2) / 8.0
    a3 = ( -5 * A0 +  5 * S1 - A2)      / 4.0
    a4 = ( 12 - 12 * S0 +  7 * A1 - S2) / 4.0
    a5 = (  3 * A0 -  3 * S1 + A2)      / 8.0
    a6 = ( -8 +  8 * S0 -  5 * A1 + S2) / 8.0

    return alpha * (a1 + alpha * ( a2 + alpha * ( a3 + alpha * ( a4 + alpha * ( a5 + alpha * a6 ) ) ) ) )

# exponential function for extrapolation   
@jax.jit
def exp_extrap(tuple_input):
    """
    Function for exponential extrapolation
    """
    alpha, varUp, varDown = tuple_input

    return jnp.where(alpha>1.0, (varUp)**alpha, (varDown)**(-alpha)) - 1.0

# loop over systematic uncertainty variations to calculate net effect
@jax.jit
def calculate_combined_var(param_vec, combined_var_up, combined_var_down):

    def calculate_variations(carry, param_val):
        
        param, combined_var_up_NP, combined_var_down_NP = param_val
        
        combined_var_array_alpha = carry
    
        # Strategy 5 of RooFit:
        combined_var_array_alpha += combined_var_array_alpha * jax.lax.cond(jnp.abs(param)<=1.0, 
                                                                            poly_interp, 
                                                                            exp_extrap, 
                                                                            (param, combined_var_up_NP, combined_var_down_NP))            
        return combined_var_array_alpha, None

    # Prepare loop_tuple for jax.lax.scan 
    loop_tuple = (param_vec, combined_var_up, combined_var_down)

    # Loop over systematic variations to calculate net effect
    combined_var_array, _ = jax.lax.scan(calculate_variations, jnp.ones_like(combined_var_up[0]), loop_tuple)

    return combined_var_array

@jax.jit
def calculate_norm_variations(param_vec, index_normparam_map):

    for sample in self.all_samples

# Compute the poisson likelihood ratio
@jax.jit
def pois_loglikelihood(data_hist, nu_hist):
    return -2 * jnp.sum( data_hist * jnp.log(nu_hist) - nu_hist)