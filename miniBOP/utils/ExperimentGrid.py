##
# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
#
# This code is written for research and educational purposes only to
# supplement the paper entitled "Practical Bayesian Optimization of
# Machine Learning Algorithms" by Snoek, Larochelle and Adams Advances
# in Neural Information Processing Systems, 2012
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.
import numpy        as np
from .sobol_lib     import *

class Parameter:
    def __init__(self):
        self.type = []
        self.name = []
        self.type = []
        self.min = []
        self.max = []
        self.options = []
        self.int_val = []
        self.dbl_val = []
        self.str_val = []

class GridMap:

    def __init__(self, variables):
        self.variables   = []
        self.cardinality = 0

        # Count the total number of dimensions and roll into new format.
        for variable in variables:
            self.cardinality += variable['size']

            if variable['type'] == 'int':
                self.variables.append({ 'name' : variable['name'],
                                        'size' : variable['size'],
                                        'type' : 'int',
                                        'min'  : int(variable['min']),
                                        'max'  : int(variable['max'])})

            elif variable['type'] == 'float':
                self.variables.append({ 'name' : variable['name'],
                                        'size' : variable['size'],
                                        'type' : 'float',
                                        'min'  : float(variable['min']),
                                        'max'  : float(variable['max'])})

            elif variable['type'] == 'enum':
                self.variables.append({ 'name'    : variable['name'],
                                        'size'    : variable['size'],
                                        'type'    : 'enum',
                                        'options' : list(variable['options'])})
            else:
                raise Exception("Unknown parameter type.")

    # Get a list of candidate experiments generated from a sobol sequence
    def hypercube_grid(self, size, seed):
        # Generate from a sobol sequence
        sobol_grid = np.transpose(i4_sobol_generate(self.cardinality,size,seed))

        return sobol_grid

    # Convert a variable to the unit hypercube
    # Takes a single variable encoded as a list, assuming the ordering is
    # the same as specified in the configuration file
    def to_unit(self, v):
        unit = np.zeros(self.cardinality)
        index  = 0

        for variable in self.variables:
            #param.name = variable['name']
            if variable['type'] == 'int':
                for dd in range(variable['size']):
                    unit[index] = self._index_unmap(float(v.pop(0)) - variable['min'], (variable['max']-variable['min'])+1)
                    index += 1

            elif variable['type'] == 'float':
                for dd in range(variable['size']):
                    unit[index] = (float(v.pop(0)) - variable['min'])/(variable['max']-variable['min'])
                    index += 1

            elif variable['type'] == 'enum':
                for dd in range(variable['size']):
                    unit[index] = variable['options'].index(v.pop(0))
                    index += 1

            else:
                raise Exception("Unknown parameter type.")

        if (len(v) > 0):
            raise Exception("Too many variables passed to parser")
        return unit

    def unit_to_list(self, u):
        params = self.get_params(u)
        paramlist = []
        for p in params:
            if p.type == 'int':
                for v in p.int_val:
                    paramlist.append(v)
            if p.type == 'float':
                for v in p.dbl_val:
                    paramlist.append(v)
            if p.type == 'enum':
                for v in p.str_val:
                    paramlist.append(v)
        return paramlist

    def get_params(self, u):
        if u.shape[0] != self.cardinality:
            raise Exception("Hypercube dimensionality is incorrect.")

        params = []
        index  = 0
        for variable in self.variables:
            param = Parameter()

            param.name = variable['name']
            if variable['type'] == 'int':
                param.type = 'int'
                for dd in range(variable['size']):
                    param.int_val.append(variable['min'] + self._index_map(u[index], variable['max']-variable['min']+1))
                    index += 1

            elif variable['type'] == 'float':
                param.type = 'float'
                for dd in range(variable['size']):
                    val = variable['min'] + u[index]*(variable['max']-variable['min'])
                    val = variable['min'] if val < variable['min'] else val
                    val = variable['max'] if val > variable['max'] else val
                    param.dbl_val.append(val)
                    index += 1

            elif variable['type'] == 'enum':
                param.type = 'enum'
                for dd in range(variable['size']):
                    ii = self._index_map(u[index], len(variable['options']))
                    index += 1
                    param.str_val.append(variable['options'][ii])

            else:
                raise Exception("Unknown parameter type.")

            params.append(param)

        return params

    def card(self):
        return self.cardinality

    def _index_map(self, u, items):
        return int(np.floor((1-np.finfo(float).eps) * u * float(items)))

    def _index_unmap(self, u, items):
        return float(float(u) / float(items))
