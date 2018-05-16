# Copyright (C) 2018 Ran Rubin
#
# This code is written for research and educational purposes only to
# supplement the paper entitled
# "Practical Bayesian Optimization of Machine Learning Algorithms"
# by Snoek, Larochelle and Adams
# Advances in Neural Information Processing Systems, 2012
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
import numpy          as np
import scipy.linalg   as spla
import scipy.optimize as spo

import george
from george import kernels
from miniBOP.utils.GP_sample import GP_sample
from miniBOP.utils.GP_hypers_sampler import GP_hypers_sampler

from miniBOP.chooser.BOP_Chooser import BOP_Chooser,\
                                        _hyper_cube_bounds_check,\
                                        _reshape_and_bound_checked

def _smpl_f_barrier(smpl_f,min_std,z):
    def f(x):
        y,m,v = smpl_f(x,full_output=True)
        return y + (min_std/np.sqrt(v))**(z)
    return f

class FuBarVarCtrl_Chooser(BOP_Chooser):
    def __init__(self, D, covar=kernels.Matern52Kernel, mcmc_iters=1):
        BOP_Chooser.__init__(self,D,covar,mcmc_iters)
        self.z = 10 #Exponent of power law barrier function

    def _find_best_point(self,comp,vals):
        #estimate current minimum of function excluding points with low predicted variance
        # using a barier function
        mu,var = self.gp.predict(vals,comp,return_cov=False, return_var=True)
        min_std = self.rho*np.sqrt(self.noise)
        barier_mu = mu + (min_std/np.sqrt(var))**(self.z)
        best_point = comp[np.argmin(barier_mu)]
        best_estimated_mean = np.min(barier_mu)

        return best_point, best_estimated_mean

    def _acquisition_function(self,vals):
        return _reshape_and_bound_checked(\
                        _smpl_f_barrier(GP_sample(self.gp,vals),\
                                        self.rho*np.sqrt(self.noise),\
                                        self.z),\
                        self.D)

    def _test_sols_variance(self,vals,sols):
        ''' Since the barier function (softly) guaranties the sampled points cannot
        have low variance there is nothing for us to do here '''
        return sols
