#################################################
### Histogram-based calibration strategy. 
### Base part of the code copied from https://github.com/smsharma/mining-for-substructure-lens
### New weighted quantiles method, and small changes
#################################################

import numpy as np

class HistogramCalibrator:
    
    def __init__(self, calibration_data_num, calibration_data_den, w_num, w_den, mode="dynamic", nbins=100, histrange=None, method="direct"):
        
        self.range, self.edges = self._find_binning(
            calibration_data_num, calibration_data_den, mode, nbins, histrange
        )

        self.hist_num, self.num_err = self._fill_histogram(calibration_data_num, w_num)
        
        self.method = method
        
        if self.method == "direct":
            self.hist_den, self.den_err = self._fill_histogram(calibration_data_den, w_den)
        else:
            self.hist_den, self.den_err = self._fill_histogram(calibration_data_num, w_num)+self._fill_histogram(calibration_data_den, w_den)
            
    def return_hist(self):
        return self.hist_num, self.hist_den, self.num_err, self.den_err, self.quant_binning
        
    def cali_pred(self, data):
        indices = self._find_bins(data)
        num = self.hist_num[indices]
        den = self.hist_den[indices]
        cal_pred = num/den
        
        if self.method == "direct":
            score  = cal_pred/(1+cal_pred)
            return score
        else:
            return cal_pred

    def _find_binning(self, data_num, data_den, mode, nbins, histrange):
        data = np.hstack((data_num, data_den)).flatten()
        if histrange is None:
            hmin = np.min(data)
            print(hmin)
            hmax = np.max(data)
            print(hmax)
        else:
            hmin, hmax = histrange

        if mode == "fixed":
            edges = np.linspace(hmin, hmax, nbins + 1)
        elif mode == "dynamic":
            #percentages = 100.0 * np.linspace(0.0, 1.0, nbins+1)
            edges = self.weighted_quantile(data, np.linspace(0.0, 1.0, nbins+1))
        elif mode == "dynamic_unweighted":
            percentages = 100.0 * np.linspace(0.0, 1.0, nbins+1)
            edges = np.percentile(data, percentages)
            
        else:
            raise RuntimeError("Unknown mode {}".format(mode))
        
        self.quant_binning = edges
        return (hmin, hmax), edges

    def _fill_histogram(self, data, weights, epsilon=1.0e-39):
        histo, _ = np.histogram(data, bins=self.edges, range=self.range, weights=weights)
        i = np.sum(histo)
        histo = histo / i
        
        err,_ = np.histogram(data, bins=self.edges, range=self.range, weights=weights**2)
        err = err/(i**2)
        
        return histo, err

    def _find_bins(self, data):
        indices = np.digitize(data, self.edges)
        #indices = np.searchsorted(self.edges, data)
        indices = np.clip(indices - 1, 0, len(self.edges) - 2)
        return indices
    
    def weighted_quantile(self, data, quantiles, sample_weight=None):
        
        values = np.array(data)
        quantiles = np.array(quantiles)
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        sample_weight = np.array(sample_weight)

        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]

        return np.interp(quantiles, weighted_quantiles, values)
