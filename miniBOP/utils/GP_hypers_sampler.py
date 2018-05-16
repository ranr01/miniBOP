##
# Copyright (C) 2018 Ran Rubin
# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from .slice_sample import slice_sample
import scipy.stats

class GP_hypers_sampler(object):
    def __init__(self,chooser,max_ls,noise_scale,amp2_scale,\
                 length_scale=0.25,length_alpha=1.):
        '''
        A class to sample the hyper parameters of a GP of a chooser object.
        Parameters are the parameters of the prior distribution of the GP
        hyperparameters.
        '''
        self.chooser = chooser
        self.max_ls = max_ls # We impose a maximal length sacle just to keep things sane
        self.v_noise = noise_scale
        self.A2 = amp2_scale
        self.lambda_length = length_scale
        self.alpha_length = length_alpha

    def set_vals(self,vals):
        self.vals = vals
        self.mean_bounds = (np.min(vals),np.max(vals))

    def sample_chooser_hypers(self):
        hypers = self._sample_noisy()
        self.chooser.mean  = hypers[0]
        self.chooser.amp2  = hypers[1]
        self.chooser.noise = hypers[2]
        #sample length scales
        self.chooser.ls = self._sample_ls()

    def _sample_ls(self):
        def logprob(ls):
            if np.any(ls < 0) or np.any(ls > self.max_ls):
                return -np.inf
            lp = self.chooser._GP_logprob(self.vals,self.chooser.amp2,ls,\
                                          self.chooser.noise,\
                                          self.chooser.mean)
            # Roll in length scales inverse gamma prior.
            ls_prior = np.sum(scipy.stats.invgamma.logpdf(ls,\
                                                          self.alpha_length,\
                                                          0.0,\
                                                          self.lambda_length))
            lp +=  ls_prior
            return lp

        return slice_sample(self.chooser.ls, logprob, compwise=True)

    def _sample_noisy(self):

        def logprob(hypers):
            mean  = hypers[0]
            amp2  = hypers[1]
            noise = hypers[2]

            # This is pretty hacky, but keeps things sane.
            #if mean > np.max(vals) or mean < np.min(vals):
            #    return -np.inf
            if mean > self.mean_bounds[1] or mean < self.mean_bounds[0]:
                return -np.inf

            if amp2 < 0 or noise < 0:
                return -np.inf

            lp = self.chooser._GP_logprob(self.vals,amp2,self.chooser.ls,noise,mean)
            # Roll in noise horseshoe prior.
            lp += np.log(np.log(1 + (self.v_noise/noise)**2))
            # Roll in amplitude lognormal prior
            lp -= 0.5*(np.log(amp2)/self.A2)**2

            return lp

        init_x = np.array([self.chooser.mean, self.chooser.amp2, self.chooser.noise])
        return slice_sample(init_x, logprob, compwise=False)
