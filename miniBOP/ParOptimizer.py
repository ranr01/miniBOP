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

import time

class ParOptimizer(object):
    def __init__(self,workmanager,\
                 cluster_work_manager,\
                 process_sim_results,\
                 save_results,\
                 run_job,\
                 select_point,\
                 pre_init = None,
                 post_optimization = None):
        '''
        A virtual class to run Parllel BO. Should be derived from to define
        parralel framework specific functions.

            workmanager - a BO_Manager like object

            cluster_work_manager - a manager object to submit jobs and recieve results.

            process_sim_results - a function to process sim results:
                 call signature:
                     process_sim_results(results,job_info)-->SimRes,value

                 where results are what is returned from run_job, job_info
                 is a dictionary with job metadata, SimRes is the result that
                 will be saved in save_results and value is the evaluation of
                 the optimization function.

            save_results - a function to save results
                call signature:
                    save_results(ListOfSimRes)
                 after a call to save_results the optimizer will not maintain
                 references to saved results.

            run_job - a function to run jobs (will run on engines)
                call signature:
                    run_job(params)-->results

                params is a vector of parameters in real parameter units (not hypercube)

            select_point - a function to select new points (will run on engines)
                call signature:
                    select_point(complete,value,pending)-->candidate

                where complete,value,pending are ndarrays of complete jobs params,
                values at these params, and params of pending jobs. Here, all
                params are in scaled hypercube units.
                candidate is the selected point in hypercube units.

            pre_init - a function to run before initial submition
                call signature:
                    pre_init()

            post_optimization - a function to run at the end of optimize
                call signature:
                    post_optimization()
        '''
        self.BOWorkManager = workmanager
        self._init_cluster_work_manager(cluster_work_manager)
        # AUX. FUNCTIONS
        self.process_sims_results = process_sim_results
        self.save_results = save_results
        self.run_job = run_job
        self.select_point = select_point
        self.pre_init = pre_init
        self.post_optimization = post_optimization
        self._init_cluster_work_manager(cluster_work_manager)
        # parameters
        self.min_n_save = 100
        self.t_sleep = 1. #time between initial submissions
        self.max_n_jobs = 1000 # max. number of jobs to submit
        # constants
        self._SIMULATION = 0
        self._BO = 1
        # job bookkeeping
        self.running_jobs = 0
        self.max_running_jobs = self._update_max_running_jobs()

    def submit_init(self):
        '''
        Initial start up of the cluster jobs.
        Can also be used to resume simulation on cluster,
        with the proper initialization of the AsyncManager (self.BOWorkManager).

        Fills the lbview with jobs from the pending list of HO.
        If there are more pending jobs than CPUs in lbview removes the remaining
        jobs from the pending list.
        Fills remaining slots in lbview with BO jobs

        Assumes the lbview is empty.
        '''
        if self.pre_init != None:
            self.pre_init()

        # init bookkeeping
        self.pending_work_dict = {}
        self.pending = set()
        self.res_list =[]
        self.n_jobs = len(self.BOWorkManager.complete) + len(self.BOWorkManager.pending)

        # if we have more pending jobs than cpus to run we simply remove the
        # first unsubmitted jobs since they are the most uninformative
        # (in a resume situation)
        n_remove = len(self.BOWorkManager.pending) - self.max_running_jobs
        if n_remove > 0:
            del self.BOWorkManager.pending[:n_remove]
            del self.BOWorkManager.pending_job_id[:n_remove]

        # submit/resubmit pending simulations
        for HCube_params,job_id in zip(self.BOWorkManager.pending,\
                                       self.BOWorkManager.pending_job_id):
            # submit job in self.BOWorkManager.pending params are in hypercube units
            params = self.BOWorkManager.gmap.unit_to_list(HCube_params)
            self._submit_sim_job(job_id,params)
            self.running_jobs += 1
            time.sleep(self.t_sleep)

        # filling remaining slots with BO jobs
        while self.n_jobs < self.max_n_jobs and \
              self.running_jobs < self.max_running_jobs:
            self._submit_BO_job()
            self.n_jobs += 1
            self.running_jobs += 1
            time.sleep(self.t_sleep)

        # finally report the submitted jobs
        print('\n'+time.strftime('%H:%M:%S')+\
        ' Submited {} jobs. n_jobs={}, N_sim={}, N_BO={}\n'.format(\
                        self.running_jobs,self.n_jobs,self._N_sim(),self._N_BO()))

    def optimize(self,wait_time=1e-1):
        print("Monitoring results")

        n_sim_submit = 0

        while self.pending:
            try:
                self._wait_for_distributed_job(wait_time)
            except self._TimeoutException:
                # ignore timeouterrors, since they only mean that at least one isn't done
                pass

            # finished is the set of msg_ids that are complete
            finished = self._get_finished_jobs()
            # update pending to exclude those that just finished
            self.pending = self.pending.difference(finished)

            # handle the results
            for cluster_job_id in finished:
                job_info = self.pending_work_dict.pop(cluster_job_id)
                result  = self._get_job_result(job_info)
                dur = self._get_job_duration(job_info)

                if job_info['type'] == self._SIMULATION:
                    SimRes,value = self.process_sims_results(result,job_info)
                    #tell the manager about the completed job
                    self.BOWorkManager.process_result(job_info['job_id'],value,dur)
                    #append the full simulation result to the list
                    self.res_list.append(SimRes)
                    #increase the number of CPUs availabe for BO
                    self.running_jobs -= 1
                    print("Recived job_id {}, val={:.3}, dur={:.4}".format(\
                            job_info['job_id'],value,dur))
                elif job_info['type'] == self._BO:
                    #submit the point for simulation
                    candidate = result
                    #tell the manager about the new point and get a job_id
                    # and the params in real units
                    job_id,params = self.BOWorkManager.process_next_point(candidate)
                    # submit job
                    self._submit_sim_job(job_id,params)
                    n_sim_submit += 1
                    print("Submited job_id {}. t_BO={:.3}".format(job_id,dur))
                else:
                    raise RuntimeError("Wrong job type")

            # find out if more/less nodes are availabe for us
            self.max_running_jobs = self._update_max_running_jobs()

            # submit new BO jobs to fill up the work queue
            n_submit = 0
            while self.n_jobs < self.max_n_jobs and \
                  self.running_jobs < self.max_running_jobs:
                self.n_jobs += 1
                n_submit += 1
                self.running_jobs += 1
                self._submit_BO_job()

            # If needed do some post processing after submiting the jobs
            if n_submit > 0:
                print('\n'+time.strftime('%H:%M:%S')+\
                ' Submited {} chooser jobs. n_jobs={}, N_sim={}, N_BO={}\n'.format(\
                                n_submit,self.n_jobs,self._N_sim(),self._N_BO()))

            #saving results
            if len(self.res_list)>self.min_n_save or n_sim_submit > 50:
                t_init = time.time()
                self.save_results(self.res_list)
                t_sav = time.time()-t_init
                print("\n"+time.strftime('%H:%M:%S')+\
                        " Saved {} results. t_save={:.4}\n".format(\
                                        len(self.res_list),t_sav))
                # empty the results list
                self.res_list =[]
                n_sim_submit = 0

        #saving final results
        t_init = time.time()
        self.save_results(self.res_list)
        t_sav = time.time()-t_init
        print("\nSaved {} results. t_save={:.4}".format(len(self.res_list),t_sav))
        self.res_list =[]

        if self.post_optimization != None:
            self.post_optimization()

    ######################### HELPER FUNCTIONS #################################
    # These should be defined in derived class.                                #
    ############################################################################
    def _init_cluster_work_manager(self,cluster_work_manager):
        pass

    def _wait_for_distributed_job(self,wait_time):
        pass

    def _get_finished_jobs(self):
        pass

    def _get_job_result(self, job_info):
        pass

    def _get_job_duration(self,job_info):
        pass

    def _N_BO(self):
        ''' returns the number of running BO jobs '''
        pass
    def _N_sim(self):
        ''' returns the number of running sim jobs '''
        pass

    def _submit_sim_job(self,job_id,params):
        pass

    def _submit_BO_job(self):
        ''' Submits BO job and updates pending DB '''
        pass

    def _update_max_running_jobs(self):
        pass
