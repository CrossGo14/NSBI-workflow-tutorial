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

class inference:

    def __init__(self, model_nll, 
                initial_values, 
                list_parameters):
        
        self.model_nll = model_nll
        self.initial_values = initial_values
        self.list_parameters = list_parameters

    def perform_fit(self, 
                    fit_strategy=2, 
                    freeze_params=[]):
                
        m = Minuit(self.model_nll, self.initial_values, 
                    grad=None, name=tuple(self.list_parameters))
        
        m.errordef = Minuit.LEAST_SQUARES
        strategy = fit_strategy
        if len(freeze_params)>=1:
            for param in freeze_params:
                m.fixed[param] = True
        m.strategy = strategy
        mg = m.migrad()
        print(f'fit: \n {mg}')
    
    
    # def plot_NLL_scan(self, parameter_name, parameter_label='', bound_range=(0.0, 3.0), 
    #                   fit_strategy=2, isConstrainedNP=False, freeze_params=[], doStatOnly = False):
    
            
    #     m = Minuit(self.jitted_NLL_function, self.asimov_param_vec, 
    #                 grad=self.grad_jitted_NLL_function, name=tuple(self.list_params_all))
            
    #     m.errordef = Minuit.LEAST_SQUARES
    #     m.strategy = fit_strategy
    #     if len(freeze_params)>=1:
    #         for param in freeze_params:
    #             m.fixed[param] = True
    #     scan_points, NLL_value, _ = m.mnprofile(parameter_name, bound=bound_range, subtract_min=True)
    #     if doStatOnly: 
    #         label_stat_syst = 'Stat+Syst'
    #         label_stat_only = 'Stat Only'
    
    #         m_StatOnly = Minuit(self.jitted_NLL_function, self.pulls_global_fit, 
    #                             grad=self.grad_jitted_NLL_function, name=tuple(self.list_params_all))
    #         m_StatOnly.errordef = Minuit.LEAST_SQUARES
    #         m_StatOnly.strategy = fit_strategy
    #         for param in self.list_params_all[self.num_unconstrained_params:]:
    #             m_StatOnly.fixed[param] = True
    #         scan_points_StatOnly, NLL_value_StatOnly, _ = m_StatOnly.mnprofile(parameter_name, bound=bound_range, subtract_min=True)
            
    #         plt.plot(scan_points_StatOnly, NLL_value_StatOnly, label = label_stat_only, color = 'b', linestyle='--')
    #         plt.plot(scan_points, NLL_value, label = label_stat_syst, color = 'b')
            
    #         plt.legend()
            
    #     else:
    #         label_stat_syst = ''
    #         plt.plot(scan_points, NLL_value, label = label_stat_syst)
            
    #     if not isConstrainedNP:
    #         plt.axis(ymin=0.0)
    #     else:
    #         plt.axis(ymin=0.0, ymax=2.0) # Constrained NPs should be bounded by +-1 uncertainty
    #     if parameter_label=='':
    #         parameter_label = parameter_name
    #     plt.xlabel(parameter_label)
    #     if isConstrainedNP:
    #         plt.ylabel(r"$t_\alpha$")
    #     else:
    #         plt.ylabel(r"$t_\mu$")
    
