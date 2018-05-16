##
# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
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

from .utils.ExperimentGrid  import GridMap
import numpy as np

class BO_Manager(object):
    def __init__(self,variables,init_grid_size = 100,sobol_seed=1):
        '''
        This is just a simple class to do the book-keeping of the BO.
        It colects results and pending jobs and provides mechanism to
        convert between hypercube units and real units, through its
        gmap GridMap object.

        The pending list is initialized with init_grid_size points taken from
        a quasi random grid.

        All parameter values in complete and pending are kept in hypercube units.
        '''
        self.gmap = GridMap(variables)

        # get initial points for initial submit
        self.pending = list(self.gmap.hypercube_grid(init_grid_size,sobol_seed))
        self.pending_job_id = list(range(init_grid_size))
        self.next_job_id = len(self.pending_job_id)
        self.pending_job_type = ['grid' for jid in self.pending_job_id]

        # initialize complete, values and durations
        self.values = []
        self.complete = []
        self.durations = []
        self.types = []

    def process_result(self,complete_job_id,val,dur):
        '''
        Registers the results of pending job with complete_job_id
        and updates the complete, values, durations and pending lists.
        '''
        #update pending, complete, value and durations
        idx = np.nonzero(np.array(self.pending_job_id) == complete_job_id)[0][0]
        self.complete.append(self.pending[idx])
        self.values.append(val)
        self.durations.append(dur)
        self.types.append(self.pending_job_type[idx])
        del self.pending[idx]
        del self.pending_job_id[idx]
        del self.pending_job_type[idx]

    def process_next_point(self,candidate):
        '''
        Adds candidate (expects candidate in hypercube units) to the list
        of pending jobs and assign a job_id to it.

        Returns the job_id and the job parameters in REAL units.
        '''
        #add job to pending
        x,job_type = candidate
        self.pending.append(x)
        self.pending_job_id.append(self.next_job_id)
        self.pending_job_type.append(job_type)
        self.next_job_id += 1

        #return job_id for next point pending, and return scaled parametes
        return self.pending_job_id[-1],self.gmap.unit_to_list(self.pending[-1])
