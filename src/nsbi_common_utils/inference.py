import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.optimizers import Nadam
import matplotlib.pyplot as plt
from iminuit import Minuit

def perform_fit(model_nll, 
                initial_values, 
                list_parameters, 
                fit_strategy=2, 
                freeze_params=[]):
            
    m = Minuit(model_nll, initial_values, 
                grad=None, name=tuple(list_parameters))
    
    m.errordef = Minuit.LEAST_SQUARES
    strategy = fit_strategy
    if len(freeze_params)>=1:
        for param in freeze_params:
            m.fixed[param] = True
    m.strategy = strategy
    mg = m.migrad()
    print(f'fit: \n {mg}')

class nsbi_inference:

    def __init__(self, channels_binned, channels_unbinned, asimov_param_vec, num_unconstrained_params, 
                all_processes, fixed_processes, floating_processes, process_index, 
                ratios, ratio_variations, weights, data_hist_channel, hist_channels, hist_channel_variations, list_params_all):

        self.num_unconstrained_params       = num_unconstrained_params
        self.all_processes                  = all_processes
        self.fixed_processes                = fixed_processes
        self.floating_processes             = floating_processes
        self.process_index                  = process_index
        self.ratios                         = ratios
        self.weights                        = weights
        self.channels_binned                = channels_binned
        self.channels_unbinned              = channels_unbinned
        self.asimov_param_vec               = asimov_param_vec
        self.data_hist_channel              = data_hist_channel
        self.hist_channels                  = hist_channels
        self.hist_channel_variations        = hist_channel_variations
        self.list_params_all                = list_params_all
        self.ratio_variations               = ratio_variations
        
        if self.num_unconstrained_params == len(self.list_params_all):

            self.jitted_NLL_function = jax.jit(self.full_nll_function_noConst)
            self.grad_jitted_NLL_function = jax.jit(jax.grad(self.jitted_NLL_function))

        else:
            self.jitted_NLL_function = jax.jit(self.full_nll_function)
            self.grad_jitted_NLL_function = jax.jit(jax.grad(self.jitted_NLL_function))



    # parameterized yields calculation
    def calculate_parameterized_yields(self, param_vec, hist_yields, hist_vars):

        nu_tot = 0.0
        
        for process in self.floating_processes:

            # This will not work in the general case where model is non-linear in POI, needs modifications (TO-DO)
            param_index = tuple(self.process_index[process])
            nu_tot += param_vec[param_index] * hist_yields[process] * hist_vars[process]

        for process in self.fixed_processes:
            nu_tot += hist_yields[process] * hist_vars[process]

        return nu_tot

    # parameterized log-likelihood ratio calculation
    def calculate_parameterized_ratios(self, param_vec, nu_nominal, nu_vars, ratios, ratio_vars):

        dnu_dx = jnp.zeros_like(self.weights)

        for process in self.floating_processes:
            param_index = tuple(self.process_index[process])
            dnu_dx += param_vec[param_index] * nu_vars[process] * nu_nominal[process] * ratios[process] * ratio_vars[process]

        for process in self.fixed_processes:
            dnu_dx += nu_vars[process] * nu_nominal[process] * ratios[process] * ratio_vars[process]
            
        return jnp.log( dnu_dx )



    # compute the full summed log-likelihood
    def full_nll_function(self, param_vec):

        llr_tot = 0.0

        self.hist_vars = {}
        for channel in self.channels_binned:

            self.hist_vars[channel] = {}
            for process in self.all_processes:
                
                self.hist_vars[channel][process] = calculate_combined_var(param_vec[self.num_unconstrained_params:], 
                                                                            self.hist_channel_variations[channel][process]['up'],
                                                                            self.hist_channel_variations[channel][process]['dn'])           

            nu_hist_channel = self.calculate_parameterized_yields(param_vec, 
                                                                  self.hist_channels[channel], 
                                                                  self.hist_vars[channel])

            data_hist_channel = self.data_hist_channel[channel]

            llr_tot += pois_loglikelihood(data_hist_channel, nu_hist_channel)

        self.ratio_vars = {}
        for channel in self.channels_unbinned:

            # Trivially removing alpha dependence for first tests
            self.hist_vars[channel] = {}
            self.ratio_vars[channel] = {}
            for process in self.all_processes:
                self.hist_vars[channel][process]   = calculate_combined_var(param_vec[self.num_unconstrained_params:], 
                                                                            self.hist_channel_variations[channel][process]['up'],
                                                                            self.hist_channel_variations[channel][process]['dn'])
                
                self.ratio_vars[channel][process]  = calculate_combined_var(param_vec[self.num_unconstrained_params:], 
                                                                            self.ratio_variations[channel][process]['up'],
                                                                            self.ratio_variations[channel][process]['dn'])

            nu_tot_unbinned = self.calculate_parameterized_yields(param_vec, 
                                                                  self.hist_channels[channel], 
                                                                  self.hist_vars[channel])

            data_hist_channel = self.data_hist_channel[channel]

            llr_tot += pois_loglikelihood(data_hist_channel, nu_tot_unbinned)

            llr_pe = self.calculate_parameterized_ratios(param_vec, self.hist_channels[channel], self.hist_vars[channel], 
                                                        self.ratios[channel], self.ratio_vars[channel]) \
                    - jnp.log(nu_tot_unbinned)

            llr_tot += jnp.sum(jnp.multiply(self.weights, -2 * llr_pe))

        llr_tot += jnp.sum(param_vec[self.num_unconstrained_params:]**2)

        return llr_tot
    
    # compute the full summed log-likelihood without constrained NPs
    def full_nll_function_noConst(self, param_vec):

        llr_tot = 0.0

        self.hist_vars = {}
        for channel in self.channels_binned:        

            self.hist_vars[channel] = {key: jnp.ones_like(value) for key, value in self.hist_channels[channel].items()}

            nu_hist_channel = self.calculate_parameterized_yields(param_vec, 
                                                                  self.hist_channels[channel], 
                                                                  self.hist_vars[channel])

            data_hist_channel = self.data_hist_channel[channel]

            llr_tot += pois_loglikelihood(data_hist_channel, nu_hist_channel)

        self.ratio_vars = {}
        for channel in self.channels_unbinned:

            self.hist_vars[channel] = {key: jnp.ones_like(value) for key, value in self.hist_channels[channel].items()}
            
            nu_tot_unbinned = self.calculate_parameterized_yields(param_vec, 
                                                                  self.hist_channels[channel], 
                                                                  self.hist_vars[channel])

            data_hist_channel = self.data_hist_channel[channel]

            llr_tot += pois_loglikelihood(data_hist_channel, nu_tot_unbinned)

            self.ratio_vars[channel] = {key: jnp.ones_like(value) for key, value in self.ratios[channel].items()}

            llr_pe = self.calculate_parameterized_ratios(param_vec, self.hist_channels[channel], self.hist_vars[channel], 
                                                        self.ratios[channel], self.ratio_vars[channel]) \
                    - jnp.log(nu_tot_unbinned)

            llr_tot += jnp.sum(jnp.multiply(self.weights, -2 * llr_pe))

        return llr_tot


    def perform_fit(self, fit_strategy=2, freeze_params=[]):
            
        m = Minuit(self.jitted_NLL_function, self.asimov_param_vec, 
                    grad=self.grad_jitted_NLL_function, name=tuple(self.list_params_all))
        
        m.errordef = Minuit.LEAST_SQUARES
        strategy = fit_strategy
        if len(freeze_params)>=1:
            for param in freeze_params:
                m.fixed[param] = True
        m.strategy = strategy
        mg = m.migrad()
        print(f'fit: \n {mg}')

        self.pulls_global_fit = jnp.array(m.values)


    def plot_NLL_scan(self, parameter_name, parameter_label='', bound_range=(0.0, 3.0), 
                      fit_strategy=2, isConstrainedNP=False, freeze_params=[], doStatOnly = False):

            
        m = Minuit(self.jitted_NLL_function, self.asimov_param_vec, 
                    grad=self.grad_jitted_NLL_function, name=tuple(self.list_params_all))
            
        m.errordef = Minuit.LEAST_SQUARES
        m.strategy = fit_strategy
        if len(freeze_params)>=1:
            for param in freeze_params:
                m.fixed[param] = True
        scan_points, NLL_value, _ = m.mnprofile(parameter_name, bound=bound_range, subtract_min=True)
        if doStatOnly: 
            label_stat_syst = 'Stat+Syst'
            label_stat_only = 'Stat Only'

            m_StatOnly = Minuit(self.jitted_NLL_function, self.pulls_global_fit, 
                                grad=self.grad_jitted_NLL_function, name=tuple(self.list_params_all))
            m_StatOnly.errordef = Minuit.LEAST_SQUARES
            m_StatOnly.strategy = fit_strategy
            for param in self.list_params_all[self.num_unconstrained_params:]:
                m_StatOnly.fixed[param] = True
            scan_points_StatOnly, NLL_value_StatOnly, _ = m_StatOnly.mnprofile(parameter_name, bound=bound_range, subtract_min=True)
            
            plt.plot(scan_points_StatOnly, NLL_value_StatOnly, label = label_stat_only, color = 'b', linestyle='--')
            plt.plot(scan_points, NLL_value, label = label_stat_syst, color = 'b')
            
            plt.legend()
            
        else:
            label_stat_syst = ''
            plt.plot(scan_points, NLL_value, label = label_stat_syst)
            
        if not isConstrainedNP:
            plt.axis(ymin=0.0)
        else:
            plt.axis(ymin=0.0, ymax=2.0) # Constrained NPs should be bounded by +-1 uncertainty
        if parameter_label=='':
            parameter_label = parameter_name
        plt.xlabel(parameter_label)
        if isConstrainedNP:
            plt.ylabel(r"$t_\alpha$")
        else:
            plt.ylabel(r"$t_\mu$")

    