import numpy as np
import matplotlib.pyplot as plt
#keras.models.Model.predict_proba = keras.models.Model.predict
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import mplhep as hep
from scipy import stats

hep.set_style("ATLAS")

def abline(slope, intercept):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--')

def fill_histograms_wError(data, weights, edges, histrange, epsilon, normalize=True):
        
    h, _ = np.histogram(data, edges, histrange, weights=weights)
    
    if normalize:
        i = np.sum(h)

        h = h/i
    
    h_err, _ = np.histogram(data, edges, histrange, weights=weights**2)
    
    if normalize:
        
        h_err = h_err/(i**2)
    
    return h, h_err


def plot_loss(history, path_to_figures=""):

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    
    plt.title('model loss', size=12)
    plt.ylabel('loss', size=12)
    plt.xlabel('epoch', size=12)
    plt.legend(['train', 'validation'], loc='upper left')

    plt.savefig(f'{path_to_figures}/loss_plot.png', bbox_inches='tight')
    plt.clf()
    
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.savefig(f'{path_to_figures}/accuracy_plot.png', bbox_inches='tight')
    plt.clf()

def plot_calibration_curve(data_den, weight_den, data_num, weight_num, data_denH, weight_denH, data_numH, weight_numH, path_to_figures="", nbins=100, epsilon=1.0e-20, label="Calibration Curve", score_range="standard"):
    
    data = np.concatenate([data_num,data_den,data_denH,data_numH]).flatten()

    xmin = np.amin(data)
    xmax = np.amax(data)

    edges = np.linspace(xmin, xmax, nbins + 1)
    histrange = (xmin,xmax)
    
    h_S, h_S_err = fill_histograms_wError(data_den, weight_den, edges, histrange, epsilon)
    h_X, h_X_err = fill_histograms_wError(data_num, weight_num, edges, histrange, epsilon)
    
    h_sum = (h_S+h_X)
    h_ratio = (h_X/h_sum)

    err = h_ratio**2*np.abs(h_S/h_X)*np.sqrt((h_X_err/(h_X)**2)+(h_S_err/(h_S)**2))
    
    h_SH, h_SH_err = fill_histograms_wError(data_denH, weight_denH, edges, histrange, epsilon)
    h_XH, h_XH_err = fill_histograms_wError(data_numH, weight_numH, edges, histrange, epsilon)
    
    h_sumH = (h_SH+h_XH)
    h_ratioH = (h_XH/h_sumH)

    errH = h_ratioH**2*np.abs(h_SH/h_XH)*np.sqrt((h_XH_err/(h_XH)**2)+(h_SH_err/(h_SH)**2))

    fig1 = plt.figure(1)
    plt.rc('xtick',labelsize=10)
    plt.rc('xtick',labelsize=10)
    frame1=fig1.add_axes((.1,.5,1.5,1.5),xticklabels=([]))

    hep.histplot([h_ratio,h_ratioH], bins=edges, yerr=[err,errH], label=['Training','Holdout'])
    
    abline(1.0,0.0)
    plt.title(label, fontsize=12)

    xmin = np.amin(data)
    xmax = np.amax(data)

    plt.axis(xmin=xmin, xmax=xmax, ymax=1.1, ymin=-0.1)

    plt.ylabel("Probability ratio", size=12)
    plt.legend(loc='lower right')
    hep.atlas.text(loc=1, text='Internal')

    slopeOne = (edges[:-1] + edges[1:]) / 2

    #Residual plot
    residue = (h_ratio-slopeOne)/err
    residueH = (h_ratioH-slopeOne)/errH
    frame2=fig1.add_axes((.1,.1,1.5,.4))

    plt.errorbar(slopeOne, residue, yerr=1.0, drawstyle='steps-mid')
    plt.errorbar(slopeOne, residueH, yerr=1.0, drawstyle='steps-mid')

    plt.xlabel("Predicted Score", size=12)
    plt.ylabel("Residue        ", size=12)
    plt.axis(xmin=xmin, xmax=xmax, ymin=-4.0,ymax= 4.0)

    abline(0.0,0.0)
    plt.savefig(f'{path_to_figures}/calib_plot_'+str(score_range)+'.png', bbox_inches='tight')
    plt.clf()


    
