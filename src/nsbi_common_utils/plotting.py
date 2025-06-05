import numpy as np
import matplotlib.pyplot as plt
#keras.models.Model.predict_proba = keras.models.Model.predict
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import mplhep as hep
from scipy import stats

hep.set_style("ATLAS")

# method for plotting lines
def abline(slope, intercept):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--', color='gray')

# General method for filling weighted histograms 
def fill_histograms_wError(data, weights, edges, histrange, normalize=True):
        
    h, _ = np.histogram(data, edges, histrange, weights=weights)
    
    if normalize:
        i = np.sum(h)

        h = h/i
    
    h_err, _ = np.histogram(data, edges, histrange, weights=weights**2)
    
    if normalize:
        
        h_err = h_err/(i**2)
    
    return h, h_err

# Diagnostics for training loss and accuracy 
def plot_loss(history, path_to_figures=""):

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    
    plt.title('model loss', size=12)
    plt.ylabel('loss', size=12)
    plt.xlabel('epoch', size=12)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig(f'{path_to_figures}/loss_plot.png', bbox_inches='tight')
    plt.clf()
    
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.show()
    plt.savefig(f'{path_to_figures}/accuracy_plot.png', bbox_inches='tight')
    plt.clf()
    

# Check for calibration of the NN output probability scores
def plot_calibration_curve(data_den, weight_den, data_num, weight_num, 
                           data_den_holdout, weight_den_holdout, data_num_holdout, weight_num_holdout, 
                           path_to_figures="", nbins=100, epsilon=1.0e-20, 
                           label="Calibration Curve", score_range="standard"):

    # Calculate the min and max range
    data = np.concatenate([data_num,data_den,data_den_holdout,data_num_holdout]).flatten()
    xmin = np.amin(data)
    xmax = np.amax(data)
    edges = np.linspace(xmin, xmax, nbins + 1)
    histrange = (xmin,xmax)

    # Fill the histograms of score function 
    hist_den, hist_den_err = fill_histograms_wError(data_den, weight_den, 
                                                    edges, histrange)
    hist_num, hist_num_err = fill_histograms_wError(data_num, weight_num, 
                                                    edges, histrange)

    # Calculate the bin-by-bin frequency (equivalent of NN output score function)
    hist_ratio = hist_num/(hist_den+hist_num)

    # Propagate the error 
    hist_ratio_err = hist_ratio**2 * np.abs(hist_den/hist_num) * np.sqrt((hist_num_err/(hist_num)**2)\
                                                          +(hist_den_err/(hist_den)**2))

    # Repeat for the holdout dataset
    hist_den_holdout, hist_den_holdout_err = fill_histograms_wError(data_den_holdout, weight_den_holdout, 
                                                                    edges, histrange)
    hist_num_holdout, hist_num_holdout_err = fill_histograms_wError(data_num_holdout, weight_num_holdout, 
                                                                    edges, histrange)
    
    hist_ratio_holdout = hist_num_holdout / ( hist_den_holdout + hist_num_holdout )

    hist_ratio_err_holdout = hist_ratio_holdout**2 * np.abs(hist_den_holdout/hist_num_holdout) * np.sqrt((hist_num_holdout_err/(hist_num_holdout)**2)\
                                                +(hist_den_holdout_err/(hist_den_holdout)**2))

    fig1 = plt.figure(1)
    plt.rc('xtick',labelsize=18)
    frame1=fig1.add_axes((.1,.9,.8,.8),xticklabels=([]))

    hep.histplot([hist_ratio, hist_ratio_holdout], bins=edges, yerr=[hist_ratio_err, hist_ratio_err_holdout], label=['Training', 'Holdout'])
    
    abline(1.0,0.0)
    plt.title(label, fontsize=18)

    xmin = np.amin(data)
    xmax = np.amax(data)

    plt.axis(xmin=xmin, xmax=xmax, ymax=1.1, ymin=-0.1)

    plt.ylabel("Probability ratio", size=18)
    plt.legend(loc='lower right', fontsize=18)

    slopeOne = (edges[:-1] + edges[1:]) / 2

    #Residual plot
    residue = (hist_ratio-slopeOne)/hist_ratio_err
    residue_holdout = (hist_ratio_holdout-slopeOne)/hist_ratio_err_holdout
    
    frame2=fig1.add_axes((.1,.5,.8,.4))

    plt.errorbar(slopeOne, residue, yerr=1.0, drawstyle='steps-mid')
    plt.errorbar(slopeOne, residue_holdout, yerr=1.0, drawstyle='steps-mid')

    plt.xlabel("Predicted Score", size=18)
    plt.ylabel("Residue", loc="center", size=18)
    plt.axis(xmin=xmin, xmax=xmax, ymin=-4.0,ymax= 4.0)

    abline(0.0,0.0)
    plt.show()
    plt.savefig(f'{path_to_figures}/calib_plot_'+str(score_range)+'.png', bbox_inches='tight')
    plt.clf()


def plot_calibration_curve_ratio(data_den, weight_den, data_num, weight_num, 
                                 data_den_holdout, weight_den_holdout, data_numH, weight_numH, 
                                 path_to_figures="", nbins=100, epsilon=1.0e-20, label="Calibration Curve"):

    # calculate the log-likelihood ratios from the score values
    data_den = np.log(data_den/(1.0-data_den))
    data_den_holdout = np.log(data_den_holdout/(1.0-data_den_holdout))
    
    data_num = np.log(data_num/(1.0-data_num))
    data_numH = np.log(data_numH/(1.0-data_numH))

    # Calculate the min and max range
    data = np.concatenate([data_num,data_den,data_den_holdout,data_numH]).flatten()
    xmin = np.amin(data)
    xmax = np.amax(data)
    edges = np.linspace(xmin, xmax, nbins + 1)
    histrange = (xmin,xmax)

    # Bin the log-likelihood ratios
    hist_den, hist_den_err = fill_histograms_wError(data_den, weight_den, edges, histrange)
    hist_num, hist_num_err = fill_histograms_wError(data_num, weight_num, edges, histrange)
    
    h_log = np.log(hist_num/hist_den)
    h_log_err = np.sqrt((hist_num_err/hist_num**2)+(hist_den_err/hist_den**2))
 
    hist_den_holdout, hist_den_holdout_err = fill_histograms_wError(data_den_holdout, weight_den_holdout, edges, histrange)
    hist_num_holdout, hist_num_holdout_err = fill_histograms_wError(data_numH, weight_numH, edges, histrange)
    
    h_log_holdout = np.log(hist_num_holdout/hist_den_holdout)
    h_log_holdout_err = np.sqrt((hist_num_holdout_err/hist_num_holdout**2)+(hist_den_holdout_err/hist_den_holdout**2))

    fig1 = plt.figure(1)
    plt.rc('xtick',labelsize=18)
    frame1=fig1.add_axes((.1,.9,.8,.8),xticklabels=([]))
    hep.histplot([h_log,h_log_holdout], yerr=[h_log_err,h_log_holdout_err], bins=edges, label=['Training','Holdout'])
    
    # plt.plot(edges, edges)
    abline(1.0,0.0)
    plt.title(label, fontsize=18)

    xmin = np.amin(data)
    xmax = np.amax(data)

    plt.axis(xmin=xmin, xmax=xmax)
    plt.ylabel("log of MC density ratio", size=18)
    plt.legend(loc='lower right', fontsize=18)
    # hep.atlas.text(loc=1, text='Internal')

    slopeOne = (edges[:-1] + edges[1:]) / 2

    #Ratio sub-plot
    ratio = np.where(slopeOne>=0, h_log/slopeOne, slopeOne/h_log)
    ratio_err = np.abs(ratio)*(h_log_err/np.sqrt(h_log**2))

    ratio_holdout = np.where(slopeOne>=0, h_log_holdout/slopeOne, slopeOne/h_log_holdout)
    ratio_err_holdout = np.abs(ratio_holdout)*(h_log_holdout_err/np.sqrt(h_log_holdout**2))

    frame2=fig1.add_axes((.1,.5,.8,.4),xticklabels=([]))

    plt.errorbar(slopeOne, ratio, yerr=ratio_err, drawstyle='steps-mid')
    plt.errorbar(slopeOne, ratio_holdout, yerr=ratio_err_holdout, drawstyle='steps-mid')

    plt.ylabel("Ratio", loc="center", size=18)
    plt.axis(xmin=xmin, xmax=xmax, ymin=0.5,ymax= 1.5)

    abline(0.0,1.0)

    #Residual plot
    residue = (h_log-slopeOne)/h_log_err
    residue_holdout = (h_log_holdout-slopeOne)/h_log_holdout_err
    frame3=fig1.add_axes((.1,.1,.8,.4))

    plt.errorbar(slopeOne, residue, yerr=1.0, drawstyle='steps-mid')
    plt.errorbar(slopeOne, residue_holdout, yerr=1.0, drawstyle='steps-mid')

    plt.xlabel("Predicted log of density ratio", size=18)
    plt.ylabel("Residue", loc="center", size=18)
    plt.axis(xmin=xmin, xmax=xmax, ymin=-4.0,ymax=4.0)

    abline(0.0,0.0)
    plt.show()
    plt.savefig(f'{path_to_figures}/calib_plot_llr.png', bbox_inches='tight')
    plt.clf()


def plot_reweighted(dataset, score_den, weight_den, score_num, weight_num, 
                    path_to_figures="", num=15, variables=['NN_MELA_incl_disc'], 
                    sample_name=['Bkg','Ref'], 
                    scale="linear", label='w/o Calibration'):

    # Separate the dataset into numerator p_c and denominator p_ref hypothesis
    data_den = dataset[dataset.train_labels==0].copy()
    data_den['score'] = score_den
    data_num = dataset[dataset.train_labels==1].copy()
    data_num['score'] = score_num

    # Get the original weights
    weight_den = np.ravel(data_den.weights)
    weight_num = np.ravel(data_num.weights)

    # Get the NN estimated score for the p_ref hypothesis
    score_den = np.ravel(data_den.score)
    ratio_den = score_den / (1.0 - score_den)

    # Calculate the reweighted weights for reference
    den_to_num_rwt = weight_den * ratio_den
    
    # Determine the number of rows of figures needed
    num_vars = len(variables)
    num_rows = (num_vars + 1) // 2  

    for variable in variables:
        
        var_den = np.ravel(data_den[variable])
        var_num = np.ravel(data_num[variable])
    
        concat = np.concatenate([var_den, var_num]).flatten()
        xmin = np.amin(concat)
        xmax = np.amax(concat)
    
        edges = np.linspace(xmin, xmax, num=num+1)
        histrange = (xmin, xmax)
    
        hist_den, hist_den_err = fill_histograms_wError(var_den, den_to_num_rwt, edges, histrange)
        hist_num, hist_num_err = fill_histograms_wError(var_num, weight_num, edges, histrange)
    
        hist_deno, hist_deno_err = fill_histograms_wError(var_den, weight_den, edges, histrange)
        hist_numo, hist_numo_err = fill_histograms_wError(var_num, weight_num, edges, histrange)
    
        fig1 = plt.figure(1)
        frame1=fig1.add_axes((.1,.5,.8,.8),xticklabels=([]))
    
        hep.histplot(hist_den, edges, yerr=np.sqrt(hist_den_err), 
                     label=str(sample_name[1])+'->'+str(sample_name[0])+' rwt', 
                     linewidth=2.0)
        
        hep.histplot(hist_num, edges, yerr=np.sqrt(hist_num_err), 
                     label=sample_name[0], linewidth=2.0)
        
        hep.histplot(hist_deno, edges, yerr=np.sqrt(hist_deno_err), 
                     label=sample_name[1], linewidth=2.0)
    
        # chi_sqrd = np.nansum((hist_den - hist_num)**2/(hist_num_err))
        # p_val = 1 - stats.chi2.cdf(chi_sqrd, num)
    
        plt.title(label, fontsize=18)
        plt.legend(loc='upper right', fontsize=18)
        plt.ylabel("Normalized events", size=18)
        plt.axis(xmin=xmin, xmax=xmax)
        if scale=='log': plt.yscale('log')
        # hep.atlas.text(loc=0, text='Internal')
    
    
        #Ratio plot
        rat = hist_den/hist_num
        frame2=fig1.add_axes((0.1,0.1,.8,.4))
        hep.histplot(rat, edges, yerr = np.abs(rat*np.sqrt((np.sqrt(hist_den_err)/hist_den)**2+(np.sqrt(hist_num_err)/hist_num)**2)), linewidth=2.0)
        plt.axis(xmin=xmin, xmax=xmax, ymin=0.5, ymax=1.5)
    
        plt.xlabel(variable, size=18)
        plt.ylabel("Ratio", loc='center', size=18)
        abline(0.0,1.0)

        plt.savefig(f'{path_to_figures}/reweighted_'+str(variable)+'.png', bbox_inches='tight')
        plt.show()
        plt.clf()

def plot_overfit(score_1, score_2, w_train, w_test, nbins=50, 
                 plotRange=[0.0,1.0], holdout_index=1, 
                 label='X', path_to_figures=""):

    bins = np.linspace(plotRange[0],plotRange[1],num=nbins)

    w_train = w_train/w_train.sum()
    w_test = w_test/w_test.sum()

    score_1_h, bins = np.histogram(score_1, bins=bins, weights=w_train)
    score_2_h, bins = np.histogram(score_2, bins=bins, weights=w_test)

    score_1_err, bins = np.histogram(score_1, bins=bins, weights=w_train**2)
    score_2_err, bins = np.histogram(score_2, bins=bins, weights=w_test**2)

    fig1 = plt.figure(1)
    frame1=fig1.add_axes((.1,.3,.6,.6),xticklabels=([]))
    hep.histplot(score_1_h, bins, yerr=np.sqrt(score_1_err), label='Holdout')
    hep.histplot(score_2_h, bins, yerr=np.sqrt(score_2_err), label='Train')

    plt.title("Overfit Plot for "+str(label), fontsize=18)
    plt.ylabel("Normalized", fontsize=18)
    plt.legend(loc='upper right', fontsize=18)
    plt.axis(xmin=0.0, xmax=1.0)

    #Residual plot
    difference = score_1_h - score_2_h
    frame2=fig1.add_axes((.1,.1,.6,.2))
    hep.histplot(difference, bins, yerr = np.sqrt(score_2_err+score_1_err))
    plt.axis(xmin=0.0, xmax=1.0, ymin=-0.0015, ymax=0.0015)

    plt.xlabel("NN Prediction", fontsize=18)
    plt.ylabel("Residue", loc="center", size=18)
    abline(0.0,0.0)
    plt.savefig(f'{path_to_figures}/overfit_'+str(holdout_index)+'_'+str(label)+'.png', bbox_inches='tight')
    plt.show()
    plt.clf()
