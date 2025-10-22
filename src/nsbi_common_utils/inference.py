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
    """
    Thin wrapper around iminuit to:
      1) run a global fit of a user-provided negative log-likelihood (NLL),
      2) scan/profile the NLL along a chosen parameter and plot the profile.

    Parameters
    ----------
    model_nll : callable
        Any arbitrary optimized function returning NLL to minimize. 
        Signature must match Minuit's expectation, i.e.
        f(theta0, theta1, ...) and return a scalar NLL.
    initial_values : list[float]
        Starting values for all parameters in the same order as `list_parameters`.
    list_parameters : list[str]
        Names of parameters (order matters; must match model_nll argument order).
    num_unconstrained_params : int
        Count of parameters considered "POIs/statistical" (unconstrained),
        with remaining treated as constrained/nuisance.
    """
    def __init__(self, 
                model_nll, 
                initial_values: list[float], 
                list_parameters: list[str],
                num_unconstrained_params: int):
        
        self.model_nll = model_nll
        self.initial_values = initial_values
        self.list_parameters = list_parameters
        self.num_unconstrained_params = num_unconstrained_params

    def perform_fit(self, 
                    fit_strategy=2, 
                    freeze_params=[]):
        """
        Run MIGRAD and store best-fit parameter values.

        Parameters
        ----------
        fit_strategy : int
            Minuit strategy (0 = fast, 1 = default, 2 = robust).
        freeze_params : list[str] | None
            List of parameter names to fix during the global fit.
        """

        # Instantiate the iminuit object
        m = Minuit(self.model_nll, 
                    self.initial_values, 
                    grad=None, 
                    name=tuple(self.list_parameters))
        
        m.errordef = Minuit.LEAST_SQUARES
        strategy = fit_strategy

        # Freeze parameters in freeze_params list to initial values
        if len(freeze_params)>=1:
            for param in freeze_params:
                m.fixed[param] = True

        m.strategy = strategy

        # Run the fit with MIGRAD
        mg = m.migrad()

        # Store best-fit values in parameter order
        self.pulls_global_fit = np.array(m.values)

        # Displays results of the global fit
        print(f'fit: \n {mg}')
    
    
    def plot_NLL_scan(self, 
                      parameter_name: str = '', 
                      parameter_label: str = '', 
                      bound_range: tuple[float] = (0.0, 3.0), 
                      fit_strategy: int = 2, 
                      freeze_params: list[str] =[], 
                      doStatOnly: bool = False,
                      isConstrainedNP: bool = False,
                      ax: plt.Axes | None = None):
        """
        Profile the NLL along `parameter_name` and plot the scan.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter to scan (must be in `list_parameters`).
        parameter_label : str
            Pretty label for the x-axis. Defaults to `parameter_name` if empty.
        bound_range : (float, float)
            Scan bounds for profile fit.
        fit_strategy : int
            Minuit strategy for the profile scans.
        freeze_params : list[str] | None
            Parameters to fix for both scans (Stat+Syst and StatOnly).
        doStatOnly : bool
            If True, also produce a "Stat Only" curve by fixing nuisance params
            (those after `num_unconstrained_params`) at their global-fit values.
        isConstrainedNP : bool
            If True, change the y-axis label to t_alpha; else use t_mu.
        use_likelihood_errordef : bool
            See `perform_fit`—should match how your NLL is defined.
        ax : matplotlib.axes.Axes | None
            Optional axis to draw on. If None, a new figure/axis is created.
        """
        if ax is None:
            fig, ax = plt.subplots()

        m = Minuit(self.model_nll, 
                   self.initial_values, 
                   grad=None, 
                   name=tuple(self.list_parameters))
            
        m.errordef = Minuit.LEAST_SQUARES
        m.strategy = fit_strategy

        for param in freeze_params:
            m.fixed[param] = True

        # Profile fit: subtract_min=True returns \Delta NLL
        scan_points, NLL_value, _ = m.mnprofile(parameter_name, bound=bound_range, subtract_min=True)

        # Optionally plot a stat-only NLL curve 
        if doStatOnly: 
            if self.pulls_global_fit is None:
                raise RuntimeError(
                    "perform_fit() must be called before doStatOnly=True, "
                    "so nuisance parameters can be fixed at their global-fit values."
                )
            label_stat_syst = 'Stat+Syst'
            label_stat_only = 'Stat Only'

            # Re-initialize with global fit pulls so that fixed params are at the best-fit point
            m_StatOnly = Minuit(self.model_nll, 
                                self.pulls_global_fit, 
                                grad=None, 
                                name=tuple(self.list_parameters))
        
            m_StatOnly.errordef = Minuit.LEAST_SQUARES
            m_StatOnly.strategy = fit_strategy

            # Fix the same globally-frozen params
            for param in freeze_params:
                m_StatOnly.fixed[param] = True

            # Additionally fix nuisance parameters (constrained NPs)
            for param in self.list_parameters[self.num_unconstrained_params:]:
                m_StatOnly.fixed[param] = True
            
            scan_points_StatOnly, NLL_value_StatOnly, _ = m_StatOnly.mnprofile(parameter_name, 
                                                                               bound=bound_range, 
                                                                               subtract_min=True)
            
            ax.plot(
                scan_points_StatOnly,
                NLL_value_StatOnly,
                linestyle="--",
                label="Stat Only",
            )
            ax.plot(
                scan_points,
                NLL_value,
                label="Stat+Syst",
            )
            ax.legend()

        else:
            ax.plot(scan_points, NLL_value, label="")
            
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(parameter_label or parameter_name)
        ax.set_ylabel(r"$t_\alpha$" if isConstrainedNP else r"$t_\mu$")
        ax.set_title(f"NLL profile: {parameter_label or parameter_name}")
    
