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

import ipyparallel as ipp
import numpy as np
from .ParOptimizer import ParOptimizer

'''
Implements a ParOptimizer that uses Ipyparallel to run jobs.


cluster_work_manager - a manager object to submit jobs and recieve results.
    For IPP_MinimintOptimizer the manager needs to implement:
    1. A get_ready_views() method that returns a load balanced view
    and a direct view for the cluster.
    2. An ipp_client property that references the ipyparallel client
    3. A void update_ready_engines() method that returns None

    Two managers are defiend below.
'''
class IPP_ParOptimizer(ParOptimizer):
    ######################### HELPER FUNCTIONS #################################
    def _init_cluster_work_manager(self,cluster_work_manager):
        # Ipyparallel objects
        self.cluster_work_manager = cluster_work_manager
        self.lbview, _ = self.cluster_work_manager.get_ready_views()
        self.ipp_client = self.cluster_work_manager.ipp_client
        self._TimeoutException = ipp.TimeoutError

    def _wait_for_distributed_job(self,wait_time):
        self.ipp_client.wait(self.pending, wait_time)

    def _get_finished_jobs(self):
        return self.pending.difference(self.ipp_client.outstanding)

    def _get_job_result(self, job_info):
        # we know these are done, so don't worry about blocking
        ar = job_info['async_res']
        result = ar.get()
        #sometimes returns list with one tuple
        if type(result)==list:
            result = result[0]
        return result

    def _get_job_duration(self,job_info):
        return job_info['async_res'].elapsed

    def _N_BO(self):
        ''' returns the number of running BO jobs '''
        return len([key for key,val in self.pending_work_dict.items() \
                  if val['type']==self._BO])
    def _N_sim(self):
        ''' returns the number of running sim jobs '''
        return len([key for key,val in self.pending_work_dict.items() \
                   if val['type']==self._SIMULATION])

    def _submit_sim_job(self,job_id,params):
        ''' Submits sim job and updates pending DB '''
        ar = self.lbview.apply_async(self.run_job,params)
        #BOOKKEEPING
        # update the pending set
        self.pending.add(ar.msg_ids[0])
        # update pending_work_dict
        self.pending_work_dict[ar.msg_ids[0]] = \
            {'async_res':ar,\
             'type':self._SIMULATION,\
             'job_id':job_id}

    def _submit_BO_job(self):
        ''' Submits BO job and updates pending DB '''
        #submit BO with current known and pending results
        ar = self.lbview.apply_async(\
                 self.select_point,\
                 np.array(self.BOWorkManager.complete),\
                 np.array(self.BOWorkManager.values),\
                 np.array(self.BOWorkManager.pending))
        #BOOKKEEPING
        # update the pending set
        self.pending.add(ar.msg_ids[0])
        # update work_dict
        self.pending_work_dict[ar.msg_ids[0]] = \
            {'async_res':ar,\
             'type':self._BO}

    def _update_max_running_jobs(self):
        self.cluster_work_manager.update_ready_engines()
        self.lbview, dview = self.cluster_work_manager.get_ready_views()
        return len(self.lbview)

class IpyparallelStaticManager(object):
    '''
    A minimal cluster_work_manager that just holds the reference of the ipyparallel
    client and implements the interface required bu IPP_ParOptimizer.
    '''
    def __init__(self,ipp_client):
        self.ipp_client = ipp_client

    def update_ready_engines(self):
        pass

    def get_ready_views(self):
        ''' get a load balanced view and a direct view with the current initialized engines'''
        return self.ipp_client.load_balanced_view(),\
               self.ipp_client.direct_view()


class _dummy_async_res(object):
    def wait(self):
        pass
    def successful(self):
        return True

class IpyparallelDynamicInitializer(object):
    '''
    A cluster_work_manager that can dynamically initialize and add engines to the
    working views when new engines join the cluster.
    The users supply a pre_init_function that runs first on each engine, and
    and init_function that runs second and recieves as arguments the engine id and
    a user supplied init_dict dictionary.
    '''
    def __init__(self,ipp_client):
        self.ipp_client = ipp_client
        self.ready_ids = {eid for eid in ipp_client.ids}
        self.init_dict = {}
        self.init_function = None
        self.pre_init_function = None


    def update_ready_engines(self):
        '''Updates the ready engines set and initializes new engines'''
        #If some engines have stoped remove them from ready_ids
        dead_engins = self.ready_ids.difference(self.ipp_client.ids)
        if dead_engins:
            self.ready_ids = self.ready_ids.difference(dead_engines)
        #find new engines that joined the cluster
        new_engines = set(self.ipp_client.ids).difference(self.ready_ids)
        # initialize new engines in parallel
        if new_engines:
            async_res = [self._initialize_engine(engine,block=False)\
                         for engine in new_engines]
            for a in async_res:
                a[0].wait()
                a[1].wait()
            success = np.array([(a[0].successful(),a[1].successful()) \
                                for a in async_res]\
                               ).all()
            #update ready_ids
            self.ready_ids.update(new_engines)
        else:
            success = True

        return len(new_engines),success

    def init_all_engines(self):
        '''Initialize all the engines in the ready engines set'''
        async_res = [self._initialize_engine(engine,block=False)\
                     for engine in self.ready_ids]
        for a in async_res:
            a[0].wait()
            a[1].wait()
        success = np.array([(a[0].successful(),a[1].successful()) \
                            for a in async_res]\
                          ).all()
        return success,async_res

    def _initialize_engine(self,engine,block=True):
        ''' Initialize a singe engine with id: engine
            If block is True will wait for async result to arive
            (to catch/rethrow remote exepctions)
            If block is False will return the async result objects of the
            initialization.
        '''
        if self.init_function == None:
            raise RuntimeError("init_finction not defined")

        dview = self.ipp_client.direct_view(engine)
        dview.block = False

        if self.pre_init_function == None:
            a0 = _dummy_async_res()
        else:
            a0 = dview.apply(self.pre_init_function)
        a1 = dview.apply(self.init_function,engine,self.init_dict)
        if block:
            a1.get()
        else:
            return a0,a1

    def get_ready_views(self):
        ''' get a load balanced view and a direct view with the current
            initialized engines'''
        return self.ipp_client.load_balanced_view(list(self.ready_ids)),\
               self.ipp_client.direct_view(list(self.ready_ids))
