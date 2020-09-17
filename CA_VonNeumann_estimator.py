#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:38:41 2019

@author: txuslopez
""" 

from sklearn.base import BaseEstimator
from copy import deepcopy
from sklearn.base import clone
from collections import deque
from functools import reduce

import numpy as np
import operator
import random

############################## FUNCTIONS ##############################
def empties(b):
    invB = np.flip(b, axis=0)
    empty = []
    for b in invB:
        build = deepcopy(empty)
        empty = []
        for i in range(0,b):
            empty.append(build)

    return np.array(empty).tolist()

# Increment index counter function
# Needed because number of dimensions is unknown
def increment(cntr, dims):

    cntr[cntr.shape[0]-1] += 1
    zeros = False
    if np.where(dims-cntr == 0)[0] != 0: zeros = True
    while zeros:
        idx = np.where(dims-cntr == 0)[0][0]
        cntr[idx] = 0
        cntr[idx-1] += 1
        if np.where(dims-cntr == 0)[0] != 0: zeros = True
        else: zeros = False
        
    return cntr

############################## CLASSES ##############################

class cell():
    
    def __init__(self, s):
        self.species = s
        
class CA_VonNeumann_Classifier(BaseEstimator):
    
    def __init__(self, bins_margin=0.1, bins = [], dimensions=[3,3], cells=empties([3,3])):
        self.bins = bins
        self.dimensions = dimensions
        self.cells = cells
        self.bins_margin=bins_margin

    def fit(self, data, classes):#,bins_margin

        # dimension variables
        dims = np.array(self.dimensions)
        n = len(self.dimensions)
        limits=[]

        # Creation of bins
        for j in range(0,n):
            
            min_dat = np.min(data[:, j]) - self.bins_margin*(np.max(data[:, j])-np.min(data[:, j]))
            max_dat = np.max(data[:, j]) + self.bins_margin*(np.max(data[:, j])-np.min(data[:, j]))
            delta = (max_dat-min_dat)/dims[j]
                                    
            self.bins.append(np.arange(min_dat, max_dat, delta)+delta)            
            limits.append([np.min(data[:, j]),np.max(data[:, j])])
            
        # Sorting of data into bins
        for i, r in enumerate(data):            
            idxs = []
            for j, c in enumerate(r):
                idxs.append(np.argmax(c <= self.bins[j]))
                
            self.plant(idxs, cell(classes[i]))
                        
        # Competition step needed to ensure there is only one cell per bin
        # Species with max cells in neighborhood is given control of the bin
        self.contest()
        
        #Se comprueba que después de evolucionar no queden celdas vacias. Si es así se evoluciona de nuevo hasta que no queden vacías
        empties_VN=True
        while empties_VN:
            self,abun_VN,empties_VN = self.evolve()            

        return self,limits

    def partial_fit(self, data, classes,s,limits_VN):#,bins_margin
        
        dims = np.array(self.dimensions)
        n = len(self.dimensions)
        new_bin=[]
        
        #Bins updating
        for j in range(0,n):
            if data[0][j]<limits_VN[j][0]:
                minim=data[0][j]
                min_dat = minim - self.bins_margin*(limits_VN[j][1]-minim)
            else:
                minim=limits_VN[j][0]
                min_dat = minim - self.bins_margin*(limits_VN[j][1]-minim)

            limits_VN[j][0]=minim

            if limits_VN[j][1]<data[0][j]:
                maxim=data[0][j]
                max_dat = maxim + self.bins_margin*(maxim-limits_VN[j][0])
            else:
                maxim=limits_VN[j][1]
                max_dat = maxim + self.bins_margin*(maxim-limits_VN[j][0])

            limits_VN[j][1]=maxim

            delta = (max_dat-min_dat)/dims[j]
            new_bin.append(np.arange(min_dat, max_dat, delta)+delta)
                        
        '''            
        if len(new_bin[0])>n:
            print('len(new_bin[0]):',len(new_bin[0]))
            print('n:',n)
            print('Entra 1')
            new_bin[0]=new_bin[0][:-1]
        if len(new_bin[1])>n:
            print('len(new_bin[1]):',len(new_bin[1]))
            print('n:',n)
            print('Entra 2')
            new_bin[1]=new_bin[1][:-1]
        '''
        
        self.bins=new_bin
        
        muta=False
        # Sorting of data into bins
        for i, r in enumerate(data):

            idxs = []
            for j, c in enumerate(r):
                idxs.append(np.argmax(c <= self.bins[j]))  
                                                
            cel = self.get_cell(self.cells, idxs)

            #Se considera que no hay evolucion al hacer fit, y entonces pueden quedar celdas vacias
            if len(cel)==0:
                self.plant(idxs, cell(classes[i]))                                
            else:
                if cel[0].species!=classes[i]:      
                    muta=True 
#                    print('Muta en :',s)
                    '''
                    #SOLO SI EL 50p DE LOS VECINOS TIENEN UNA ETIQUETA DIFERENTE SE SUBSTITUYE
                    d=self.getSimpleCA(self)
                    vecindad=self.get_VNneighbourhood(d,idxs,1)
                    vecindad_vote=sum(vecindad)
                    
                    if vecindad_vote>=2:                    
                        self.substitute(idxs, cell(classes[i]))
                    '''
                    
                    self.substitute(idxs, cell(classes[i]))
#                    
#                    #Después de actualizar la celda evolucionamos segun VN                    
#                    self,abun_VN,empties_VN = self.evolve()                                
                    
                    #Evolucion local
#                    coords_vecindad_Moore=self.get_neighboursMoore(np.array(idxs), exclude_p=True, shape=tuple(dims))
#                    for k,c in enumerate(coords_vecindad_Moore):
#                        principal_cel = self.get_cell(self.cells, c)
#                        coords_vecindadlocal_Moore=self.get_neighboursMoore(np.array(c), exclude_p=False, shape=tuple(dims))
#                        st=0
#                        for h,nc in enumerate(coords_vecindadlocal_Moore):                                                    
#                            ad_cel = self.get_cell(self.cells, nc)
#                            st+=ad_cel[0].species
#                            
#                        new_state=0.0
#                        if st>len(coords_vecindadlocal_Moore): 
#                            new_state=1.0
#                        
#                        if principal_cel[0].species!=new_state:
#                            self.substitute(np.array(c), cell(new_state))
                        
                    
                                   
        return self,limits_VN,muta,idxs

    def predict(self, data):
        
        ness = []
        for i, r in enumerate(data):
            idxs = []
            for j, c in enumerate(r):
                idxs.append(np.argmax(c <= self.bins[j]))

            paula = self.get_cell(self.cells, idxs)
            if paula:
                ness.append(paula[0].species)
            else:
                print("ERROR: VN Grid is not full, some observations have no mapping.")
                ness.append(-1)
                #COMO HEMOS HECHO QUE PUEDAN QUEDAR CELDAS VACIAS AL INICIALIZAR, HABRA SAMPLES SIN PREDICCION
#                ness.append(np.nan)
            
        return np.array(ness)
    
    def evolve(self):
                
#        print('VN evolving')
        # Create dictionary to track cell totals
        abundance = dict()
        
        # Iteration variables
        n = len(self.dimensions)
        c0 = deepcopy(self.cells)
        dims = np.array(self.dimensions)
        frmnt = np.zeros(dims.shape).astype(int)
                
        # Loop through every cell
        done = False
        empties=False
        
        while not done:
            
            # Get cell array for the current index - check if empty
            cel = self.get_cell(c0, frmnt)
            
            if cel:
            
                # Update the abundances
                if cel[0].species in abundance:
                    abundance[cel[0].species] += 1
                else:
                    abundance[cel[0].species] = 1
                                                    
                # Add cells to von Neumann neighbors
                for d in range(0,dims.shape[0]):
                    if frmnt[d] > 0:
                        
                        instance = deepcopy(frmnt)
                        instance[d] -= 1
                        
                        if not self.get_cell(c0, instance):
                            self.plant(instance, cell(cel[0].species))
                                                                                
                    if frmnt[d] < dims[d]-1:
                        instance = deepcopy(frmnt)
                        instance[d] += 1
                        
                        if not self.get_cell(c0, instance):
                            self.plant(instance, cell(cel[0].species))
                    
            else:
                empties=True
                        
            # Increment cell index
            frmnt = increment(frmnt, dims)
                        
            if 0 in np.where(dims-frmnt == 0)[0]: 
                done = True
        
        
        # Competition step needed to ensure there is only one cell per bin
        # Species with max cells in neighborhood is given control of the bin
        self.contest()
        
        # Return new cell totals
        return self,abundance,empties

    # Competition function - ensure one cell per space on the grid
    def contest(self):
        
        # Iteration variables
        dims = np.array(self.dimensions)
        cmpt = np.zeros(dims.shape).astype(int)
        
        # Loop through all cells
        done = False
        while not done:
            # Check if cell array empty
            cel = self.get_cell(self.cells, cmpt)
            if cel:
                
                # Calculate species with maximum cell count
                competitors = dict()
                for c in cel:
                    if c.species in competitors:
                        competitors[c.species] += 1
                    else:
                        competitors[c.species] = 1
                                        
                cel[:] = []
                
                # Cell becomes the resulting species
                spec = max(competitors.items(), key=operator.itemgetter(1))[0]        
                cel.append(cell(spec))
            
            # Increment the index
            cmpt = increment(cmpt, dims)
            if 0 in np.where(dims-cmpt == 0)[0]: done = True
        
        return self
    
    # Function to add cells to cell array at a given index
    # Catch common location error
    def plant(self, pos, daughter):
    
        location = self.cells
        for p in pos:
            try:
                location = location[p]
            except:
                print("Location Error - Please Restart Simulation")
                quit()
        
        location.append(daughter)
        
#        self.cells[pos[0]][pos[1]].append(daughter)

    def substitute(self, pos, new_candidate):
#        self.cells[pos[0]][pos[1]]=[new_candidate]

        location = self.cells
        for p in pos:
            try:
                location = location[p]
            except:
                print("Location Error - Please Restart Simulation")
                quit()

        location[0]=new_candidate
        
    # Function to get cell array at a given index
    def get_cell(self, c0, pos):

        c = c0
        for i in pos:
            c = c[i]
    
        return c        
    
    def score(self, X, y=None):
            # counts number of values bigger than mean
            return print('Scoring should be implemented in case of need')
        
    def get_params(self, deep=True):
        return {'dimensions': self.dimensions, 'cells': self.cells, 'bins': self.bins, 'bins_margin': self.bins_margin}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self        
    
    def get_abundance(self):
        
        abundance = dict()
        dims = np.array(self.dimensions)
        c0 = deepcopy(self.cells)
        frmnt = np.zeros(dims.shape).astype(int)
        done = False
        
        while not done:
            
            # Get cell array for the current index - check if empty
            cel = self.get_cell(c0, frmnt)
            
            if cel:
                # Update the abundances
                if cel[0].species in abundance:
                    abundance[cel[0].species] += 1
                else:
                    abundance[cel[0].species] = 1
        
            # Increment cell index
            frmnt = increment(frmnt, dims)
                    
            if 0 in np.where(dims-frmnt == 0)[0]: 
                done = True
            
        return abundance
    
    def get_VNneighbourhood(self,matrix, coordinates, distance):
        '''
        Se obtienen los estados de las celdas adyacentes
        '''
        dimensions = len(coordinates)
        neigh = []
        app = neigh.append
    
        def recc_von_neumann(arr, curr_dim=0, remaining_distance=distance, isCenter=True):
            #the breaking statement of the recursion
            if curr_dim == dimensions:
                if not isCenter:
                    app(arr)
                return
    
            dimensions_coordinate = coordinates[curr_dim]
                        
            if not (0 <= dimensions_coordinate < len(arr)):
                return 
    
            dimesion_span = range(dimensions_coordinate - remaining_distance, dimensions_coordinate + remaining_distance + 1)
                        
            for c in dimesion_span:

                if 0 <= c < len(arr):
                    recc_von_neumann(arr[c], curr_dim + 1, remaining_distance - abs(dimensions_coordinate - c), isCenter and dimensions_coordinate == c)
            return
    
        recc_von_neumann(matrix)
        
        return neigh
    
    def get_neighboursMoore(self,p, exclude_p=True, shape=None):
        '''
        Se obtienen las coordenadas de las celdas adyacentes.
        https://stackoverflow.com/questions/34905274/how-to-find-the-neighbors-of-a-cell-in-an-ndarray/34908879#34908879
        '''

        ndim = len(p)
    
        # generate an (m, ndims) array containing all strings over the alphabet {0, 1, 2}:
        offset_idx = np.indices((3,) * ndim).reshape(ndim, -1).T
    
        # use these to index into np.array([-1, 0, 1]) to get offsets
        offsets = np.r_[-1, 0, 1].take(offset_idx)
    
        # optional: exclude offsets of 0, 0, ..., 0 (i.e. p itself)
        if exclude_p:
            offsets = offsets[np.any(offsets, 1)]
    
        neighbours = p + offsets    # apply offsets to p
    
        # optional: exclude out-of-bounds indices
        if shape is not None:
            valid = np.all((neighbours < np.array(shape)) & (neighbours >= 0), axis=1)
            neighbours = neighbours[valid]

        return neighbours
    
    def getSimpleCA(self,cellular_aut):
    
        dim=cellular_aut.dimensions
        # Create image arrays
        arr = deepcopy(empties(dim))
        # Set variables to model results
        cells = cellular_aut.cells
            
        if len(dim)==2:
        
            for i in range(0, len(cells)):
                for j in range(0, len(cells)):
                    
                    if len(cells[i][j])==0:
                        arr[i][j] = np.nan            
                    else:
                        s = cells[i][j][0].species
                        arr[i][j] = s

        if len(dim)==3:
        
            for i in range(0, len(cells)):
                for j in range(0, len(cells)):
                    for k in range(0, len(cells)):
                    
                        if len(cells[i][j][k])==0:
                            arr[i][j][k] = np.nan            
                        else:
                            s = cells[i][j][k][0].species
                            arr[i][j][k] = s

        if len(dim)==4:
        
            for i in range(0, len(cells)):
                for j in range(0, len(cells)):
                    for k in range(0, len(cells)):
                        for l in range(0, len(cells)):
                    
                            if len(cells[i][j][k][l])==0:
                                arr[i][j][k][l] = np.nan            
                            else:
                                s = cells[i][j][k][l][0].species
                                arr[i][j][k][l] = s
                                
        #AMPLIAR SI HAY DATASETS DE MAS DE 4 DIMENSIONES

    
        return arr    
    