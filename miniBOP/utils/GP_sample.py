##
# Copyright (C) 2018 Ran Rubin
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
import scipy.linalg as linalg

def update_cholesky(L_11,A_12,A_22):
    S_11 = L_11
    S_12 = linalg.solve_triangular(L_11,A_12,trans='T')
    S_22 = linalg.cholesky(A_22 - np.dot(S_12.T,S_12))
    return np.vstack((np.hstack([S_11,S_12]),\
                      np.hstack((np.zeros(S_12.T.shape),S_22))))


class GP_sample(object):
    def __init__(self,gp,y,eps=1e-3):
        '''
        A class to sample a function from a GP.
        
        Parameters:
        gp - A gp object
        y - (n_obs,) ndarray of obervations at points gp._x
        eps - The minimal L2 distance between sampled points.
        '''
        self._xs = None
        self._gp = gp
        self._y = y
        self.eps = eps
        self._vals = np.zeros(0)
        self._means = np.zeros(0)
        self._var = np.zeros(0)

    def __call__(self,new_points,full_output=False):
        '''
        Samples the sampled function at new_points

        Parameters:
        new_points - (n_samples,n_dims) ndarray of points in R^D

        returns:
        The value of the sampled function.
        if full_output = True also returns the predicted mean and variance at
        new_points.

        If any point in new_point if too close to one of the
        previously sampled points (|r_new-r_old|<self.eps) then the prviously sampled value for
        this point is returned.

        In the same manner if new_points contains points that are too
        close to each other then only one is sampled and the value of the other is set to the same
        sampled value.
        '''

        # If we try to sample exactly the same point again it adds a zero eigen value to our cov matrix.
        # So we remove new_points that are too close to previously sampled points (|r_new-r_old|<self.eps)
        new_points, old_points_ind, old_points_vals,\
        old_points_means, old_points_var = self._test_for_sampled_points(new_points)
        # for the same reason we do not want to have new_points that are too close to one another:
        new_points, dup_points_ind = self._test_dup_points(new_points)


        if new_points.shape[0]==0: # if all the points have been sampled before
            return self._return_from_call_(old_points_vals, old_points_means,\
                                           old_points_var,full_output)
        else:
            # calculate the predicted means at new_points and the covariance
            # between the new points and all previously sampled points.
            # means.shape = (n_samples,)
            # cov.shape = (n_samples + n_prev, n_samples)
            means, cov = self._update_points(new_points)

            n,n_new = cov.shape
            n_old = n - n_new

            if n_old > 0:
                #update the cholesky factor of the covariance with the new part of the covariance
                self._factor = update_cholesky(self._factor,cov[:n_old,:],cov[n_old:,:])
                # draw n_new i.i.d standard normal random numbers and add to previously drawn
                self._z = np.vstack((self._z,np.random.randn(n_new,1)))
            else: # this handle the case of the first evaluated points
                # calculate the cholesky factoriztion
                self._factor = linalg.cholesky(cov)
                # draw n_new i.i.d standard normal random numbers
                self._z = np.random.randn(n_new,1)
            #sample the values of the sampled points
            vals = means + np.dot(self._factor.T[n_old:,:],self._z).flatten()
            #keep these values in case we sample the same point again
            self._vals = np.hstack((self._vals,vals))
            self._means = np.hstack((self._means,means))
            var = np.diag(cov[n_old:,:])
            self._var = np.hstack((self._var,var))
            #print(vals)
            if len(dup_points_ind) > 0:
                vals = self._add_dup_points_vals(vals,dup_points_ind)
                means = self._add_dup_points_vals(means,dup_points_ind)
                var = self._add_dup_points_vals(var,dup_points_ind)
                #print(vals)
            if len(old_points_ind) > 0: # we had previously sampled points
                vals = self._add_old_points_vals(vals,old_points_ind,old_points_vals)
                means = self._add_old_points_vals(means,old_points_ind,old_points_means)
                var = self._add_old_points_vals(var,old_points_ind,old_points_var)
                #print(vals)
            return self._return_from_call_(vals,means,var,full_output)

    def _return_from_call_(self,vals,means,var,full_output):
        if full_output:
            return vals,means,var
        else:
            return vals

    def _init_points(self,new_points):
        '''
            This is run at the first call of __call__ (and _update_points).
            It performs the inference, calculate the predicted means and covariance,

            Returns the means and cov
        '''
        cache = True

        #inference
        self._gp.recompute()
        self._alpha = self._gp._compute_alpha(self._y, cache)

        xs,kernel,Kxs,mu,KinvKxs = self._calc_prediction_aux(new_points)

        # calc predicted covaraince
        cov = kernel.get_value(xs)
        cov -= np.dot(Kxs, KinvKxs)

        # We save these for use in later evaluations of the function
        self._xs = xs
        self._Kxs = Kxs

        return mu, cov

    def _update_points(self,new_points):
        '''
        Updates the points in which the function was evaluated and returns
        the predicted mean at new_points and the new part of the covariance matrix (n_old+n_new,n_new)
        '''

        if self._xs is None:
            mu, cov = self._init_points(new_points)

        else:
            xs,kernel,Kxs,mu,KinvKxs = self._calc_prediction_aux(new_points)

            # update previous points and previously computed correlations
            self._xs = np.vstack((self._xs,xs))
            self._Kxs = np.vstack((self._Kxs,Kxs))

            # calculate covariance matrix
            cov = kernel.get_value(self._xs,xs)
            cov -= np.dot(self._Kxs, KinvKxs)

        return mu, cov

    def _calc_prediction_aux(self,new_points):
        '''
        A helper function for the calculation of the predictive mean and covariacne.

        returns:
        xs - the new samples in the right format
        kernel - the self._gp.kernel
        Kxs - the kernel values between the new_points and the observation point
        mu - predictive means at new_points
        KinvKxs - K^-1*Kxs. Needed for covariance calculation.
        '''

        t = new_points
        xs = self._gp.parse_samples(t)
        kernel = self._gp.kernel
        # Compute the predictive mean.
        Kxs = kernel.get_value(xs, self._gp._x)
        mu = np.dot(Kxs, self._alpha) + self._gp._call_mean(xs)
        # Compute the first step of predictive covariance.
        KinvKxs = self._gp.solver.apply_inverse(Kxs.T)

        return xs,kernel,Kxs,mu,KinvKxs

    ############################ Bookkeeping functions to make sure no point is sampled twice #########

    def _add_old_points_vals(self,vals,old_points_ind,old_points_vals):
        n = len(vals) + len(old_points_vals)
        new_points_ind = {i for i in range(n)}.difference(old_points_ind)
        all_vals = np.zeros(n)
        all_vals[np.sort(list(new_points_ind))] = vals
        all_vals[np.sort(list(old_points_ind))] = old_points_vals

        return all_vals

    def _add_dup_points_vals(self,vals,dup_points_ind):
        n = len(vals) + len(dup_points_ind)
        new_points_ind = {i for i in range(n)}.difference({ind_pair[0] for ind_pair in dup_points_ind})
        all_vals = np.zeros(n)
        all_vals[np.sort(list(new_points_ind))] = vals
        for ind_pair in dup_points_ind:
            all_vals[ind_pair[0]] = all_vals[ind_pair[1]]

        return all_vals

    def _test_for_sampled_points(self,new_points):
        old_points_ind = set()
        old_points_vals = []
        old_points_means = []
        old_points_var = []
        points = new_points
        if self._xs is not None: # these are not the first sampled points
            for ind,p in enumerate(new_points):
                #distance to previously sampled points
                ds = linalg.norm(self._xs - p.reshape((1,-1)),axis=1)
                # find if it is close enough
                sampled_ind = np.nonzero(ds < self.eps)[0]
                if sampled_ind.shape[0] > 0:
                    old_points_ind.add(ind)
                    old_points_vals.append(self._vals[sampled_ind[0]])
                    old_points_means.append(self._means[sampled_ind[0]])
                    old_points_var.append(self._var[sampled_ind[0]])
            if len(old_points_ind)>0:
                new_points_ind = {i for i in range(new_points.shape[0])}.difference(old_points_ind)
                if len(new_points_ind) > 0:
                    points = new_points[np.sort(list(new_points_ind))]
                else:
                    points = np.zeros((0,new_points.shape[1]))
        return points, old_points_ind, np.array(old_points_vals),\
                np.array(old_points_means),  np.array(old_points_var)

    def _test_dup_points(self,new_points):
        dup_points_ind = []
        n_points = new_points.shape[0]
        points = new_points
        if n_points > 1:
            for i in range(n_points):
                for j in range(i+1,n_points):
                    if linalg.norm(points[i]-points[j])<self.eps:
                        dup_points_ind.append((j,i))
            if len(dup_points_ind)>0:
                j_ind = set(np.unique([ind[0] for ind in dup_points_ind]))
                new_point_ind = {i for i in range(n_points)}.difference(j_ind)
                points = points[np.sort(list(new_point_ind))]
        return points, dup_points_ind
