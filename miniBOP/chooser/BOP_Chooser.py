##
# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
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

def _hyper_cube_bounds_check(x):
    '''returns True if x is inside the hyper cube [0,1]^D
       expects x in R^D
    '''
    if (x>1.).any() or (x<0.).any():
        return False
    else:
        return True

def _reshape_and_bound_checked(fun,D):
    '''
    A decorator to reshape input to a D dim. vector and, create a barier around
    the hyper cube [0,1]^D.
    '''
    def f(x):
        x = np.array(x).reshape(1,D)
        if _hyper_cube_bounds_check(x):
            return fun(x)
        else:
            return np.Inf
    return f

"""
Chooser module for the Gaussian process Sample Improvement Sampling with poll steps around
current minimum when no sample improvement is found.
To avoid sampling the same points repeatedly, points are viable candidates only
if the predicted variance at the point is greater than some threshold.

Slice sampling is used to sample Gaussian process hyperparameters (a la Snoek et al).
"""
class BOP_Chooser(object):

    def __init__(self, D, covar=kernels.Matern52Kernel, mcmc_iters=1):
        self.T_MCMC = int(mcmc_iters)
        self.D               = D

        self.covar = covar

        #Following 3 are now inside the hypers_sampler
        v_noise = 0.1  # horseshoe prior
        A2_scale  = 1    # zero-mean log normal prior
        max_ls      = 2    # top-hat prior on length scales
        self.hypers_sampler = GP_hypers_sampler(self,max_ls,v_noise,A2_scale)

        # parameters
        self.xatol       = 0.01 # x tol for optimization of smapled functions
        self.l_poll = 0.2  # size of poll step relative to estimated length scales.
        self.n_poll = D+1 # number of random candidate points for poll step
        self.rho = 0.5 # Minimal estimated SEM for function estimation
                                      # relative to estimated noise STD
        self.SEM_min = 1e-3 # Minimal estimated SEM for function estimation
        self.exclude_edge_points = True # Exclude smapled points too close to edges
        self.n_cand = 5*D  # number of initial points for minimizing the sampled GP function
        self.min_n_points_for_inference = 2*D # number of complete points before BO starts

        # internal state
        self._init_hypers = True

    def next(self,comp,vals,pend):
        n_complete = comp.shape[0]

        if n_complete < self.min_n_points_for_inference:
            candidate = np.random.rand(self.D)
            return candidate,'random'
        else:
            random_point = False

        # Perform the real initialization.
        if self._init_hypers:
            self._real_init(vals)
            self._init_hypers = False

        #precompute the GP with complete points
        gp = self.gp
        gp.compute(comp)

        # perform some MC steps to sample the GP hyperparameters
        if self.T_MCMC > 0:
            self.sample_hypers(vals)

        #set GP hypers to sampled values
        self._set_gp_params(self.amp2,self.ls,self.noise,self.mean)

        # sample pending results
        comp_pend,vals_pendvals = self._sample_pending(comp,vals,pend)

        #pre calculate the GP on the comp U pend set
        gp.compute(comp_pend)
        #find best point
        best_point,best_estimated_mean = self._find_best_point(comp_pend,vals_pendvals)

        # Sampling
        sols = self._sampling(comp_pend,vals_pendvals,best_point)

        #exclude local minimum points with no improvement
        sols = [sol for sol in sols if sol.fun < best_estimated_mean]

        #check if we should cosider poll
        poll = len(sols)==0
        if not poll:
            sols = self._test_sols_variance(vals_pendvals,sols)
            # check if poll is necessary
            poll = len(sols)==0

        if poll:
            # If no internal points with improvement were found,
            # get a set of poll points around best_point.
            x_poll = self._poll_points(best_point)
            # exclude out of bounds points
            ind = np.array([_hyper_cube_bounds_check(x) for x in x_poll])
            x_poll = x_poll[ind]
            random_point = x_poll.shape[0] == 0
            if not random_point:
                # calculate the predicted variance at each poll point
                mu,var = gp.predict(vals_pendvals,x_poll,\
                                return_cov=False, return_var=True)
                #exclude points with low predicted variance
                ind = self._check_min_var(np.sqrt(var))
                x_poll = x_poll[ind]
                var = var[ind]
                # if there are no high variance poll points return a random point
                random_point = x_poll.shape[0] == 0

        if random_point: # return a random point
            candidate = np.random.rand(self.D)
            return candidate,'random'
        elif poll: # return the poll point with the maximal predicted variance (is it a good idea?)
            ind = np.argmax(var)
            return x_poll[ind],'poll'
        else: # return the point with the greatest improvement.
            x = [r.x for r in sols]
            si = [-r.fun for r in sols]
            ind = np.argmax(si)
            return x[ind],'Thompson'

    def sample_hypers(self, vals):
        self.hypers_sampler.set_vals(vals)
        for t in range(self.T_MCMC):
            #this updates the value of all hypers to the new sample
            self.hypers_sampler.sample_chooser_hypers()
            #save sample
            self.hyper_history.append(\
                    np.hstack(([self.mean,self.amp2,self.noise],self.ls)))

    def _real_init(self, values):
        # Initial length scales.
        self.ls = np.ones(self.D)

        # Initial amplitude.
        self.amp2 = np.std(values)+1e-4

        # Initial observation noise.
        self.noise = 1e-3

        # Initial mean.
        self.mean = np.mean(values)

        # create George kernel and GP
        self.kernel = self.amp2 * self.covar(metric=(self.ls)**2,ndim=self.D)
        self.gp = george.GP(self.kernel,\
                            mean = self.mean,fit_mean=True,\
                            white_noise=np.log(self.noise),fit_white_noise=True)

        self.hyper_history = [np.hstack(([self.mean,self.amp2,self.noise],self.ls))]

    def _test_sols_variance(self,vals,sols):
        # calculate the predicted variance at each local minimum
        mu,var = self.gp.predict(vals,np.array([sol.x for sol in sols]),\
                            return_cov=False, return_var=True)
        #exclude points with low predicted variance
        sols = [sol for sol,large_var \
                    in zip(sols,self._check_min_var(np.sqrt(var))) \
                    if large_var]
        return sols

    def _find_best_point(self,comp,vals):
        #estimate current minimum of function excluding points with low predicted variance
        mu,var = self.gp.predict(vals,comp,return_cov=False, return_var=True)
        #best_point = comp_pend[np.argmin(mu)]
        ind = self._check_min_var(np.sqrt(var))
        if ind.any():
            best_point = comp[ind][np.argmin(mu[ind])]
            best_estimated_mean = np.min(mu[ind])
        else: #if all points have low variance ignore the variance rule
            best_point = comp[np.argmin(mu)]
            best_estimated_mean = np.min(mu)
        return best_point,best_estimated_mean

    def _poll_points(self,best_point):
        dx = np.dot(np.diag(self.l_poll*self.ls),\
                    np.random.randn(self.D,self.n_poll))
        return dx.T + best_point.reshape((1,self.D))

    def _sample_pending(self,comp,vals,pend):
        gp = self.gp

        # sample results of pending jobs
        if pend.shape[0] > 0:
            smpl_f = GP_sample(gp,vals)
            f_pend = smpl_f(pend)
            f_pend += np.sqrt(self.noise) * np.random.randn(*f_pend.shape)
            #now recompute the GP with the sampled results.
            #this reduces the predicted variance at pending points
            comp = np.vstack((comp,pend))
            vals = np.hstack((vals,f_pend))

        return comp,vals

    def _acquisition_function(self,vals):
        return _reshape_and_bound_checked(GP_sample(self.gp,vals),self.D)

    def _sampling(self,comp,vals,best_point):
        # Treat each cand point as a start point
        # for a local minimum search
        cand = np.random.rand(self.n_cand,self.D)
        #add the current best point to candidates in case it's from a small basin
        cand = np.vstack((cand,best_point))

        sols = []

        for c in cand:
            #for each cand we sample a new function.
            #This is not stricktly Thompson sampling but is
            #done to avoid sampling the same function many times
            #since sampling becomes more expensive as one samples more points
            smpl_f = self._acquisition_function(vals)
            try:
                sol = spo.minimize( smpl_f,\
                                    c,\
                                    method='Nelder-Mead',\
                                    options={'xatol': self.xatol})
            except spla.LinAlgError:
                continue

            # Local minimas of smpl_f tend to accumulate on edges of the hypercube
            # for 2 reasons: 1) The GP is extrapolating there so without a prior
            # to push it somewhere it can monotonically decreas. 2) In high dimension
            # a lot of volume is concentrated on the edges. Since we do not expect
            # to find the optimum at the edges we exclude edge points.
            if self.exclude_edge_points:
                eps = 2*self.xatol
                if not ((sol.x<eps).any() or (sol.x>1-eps).any()):
                    sols.append(sol)
            else:
                sols.append(sol)

        return sols

    def _check_min_var(self,x):
        return (x > self.rho*np.sqrt(self.noise)) & \
               (x > self.SEM_min)

    def _set_gp_params(self,amp2,ls,noise,mean):
        p = np.hstack(([mean,np.log(noise),np.log(amp2/self.D)],np.log(ls**2)))
        self.gp.set_parameter_vector(p)

    def _GP_logprob(self,vals,amp2,ls,noise,mean):
        self._set_gp_params(amp2,ls,noise,mean)
        return self.gp.log_likelihood(vals, quiet=True)
