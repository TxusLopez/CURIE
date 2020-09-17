
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:45:29 2019

@author: txuslopez
"""

'''
This Script is a RUN function which uses the cellular automation defined in 'biosystem.py' to classify data from the popular Iris Flower dataset.  Error between predicted results is then calculated and compared to other models.
'''
#import os
#os.system("%matplotlib inline")

from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.lazy.knn import KNN
from copy import deepcopy
from skmultiflow.drift_detection.adwin import ADWIN
#from collections import deque
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.ddm import DDM
from skmultiflow.bayes import NaiveBayes
from skmultiflow.drift_detection.page_hinkley import PageHinkley
#from sklearn import preprocessing
from timeit import default_timer as timer
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
#from skmultiflow.data import DataStream
#from pylatex import Document, LongTable, MultiColumn
from CA_VonNeumann_estimator import CA_VonNeumann_Classifier
from sklearn import preprocessing

#import matplotlib.animation as animat; animat.writers.list()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import warnings
import pickle
import psutil
import sys
import traceback
import logging

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 12})
#mpl.rcParams['lines.linewidth'] = 2.0

#style.use('seaborn-dark') #sets the size of the charts
#style.use('ggplot')

#==============================================================================
# CLASSES
#==============================================================================
        
#==============================================================================
# FUNCTIONS
#==============================================================================

def empty_mutant(b):
    invB = np.flip(b, axis=0)
    empty = [0]
    for b in invB:
        build = deepcopy(empty)
        empty = []
        for i in range(0,b):
            empty.append(build)

    return np.array(empty).tolist()      
  
def empties(b):
    invB = np.flip(b, axis=0)
    empty = []
    for b in invB:
        build = deepcopy(empty)
        empty = []
        for i in range(0,b):
            empty.append(build)

    return np.array(empty).tolist()    

def plot_CA_boundaries_allCAs(cellular_aut,ca_names,punto,num_automs,buch_X,buch_y,X_columns,y_columns,mutant_cs,mutants_t,mutants_d):#mutants_w

    images=[]
    
    for ca in range(num_automs):
                                
        dim=cellular_aut[ca].dimensions
        # Create image arrays
        img = deepcopy(empties(dim))
        # Set variables to model results
        cells = cellular_aut[ca].cells
            
        for i in range(0, len(cells)):
            for j in range(0, len(cells)):
                                
                if cells[i][j]:                                              
                    s = cells[i][j][0].species

                    if int(s)==0:
                        rgb = [255, 157, 137]#254,232,138-99,194,131
                    else:
                        rgb = [255, 82, 115]#196,121,0-99,100,139                      

                    img[i][j] = rgb
                else:
                    img[i][j] = [255,255,255]
    
        # Convert image arrays to appropriate data types
        rotated_img= np.rot90(img, 1)
        img = np.array(rotated_img, dtype='uint8')

        images.append(img)
    
    # Show the results
#    fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(14, 7))
    
    fig = plt.figure(figsize=(30, 15))
#    ax1 = fig.add_subplot(1,5,1, aspect=1.0)
    
    buch_pd_X=pd.DataFrame(buch_X)
    buch_pd_X.columns=X_columns
    buch_pd_y=pd.DataFrame(buch_y)
    buch_pd_y.columns=[y_columns]
    
    todo=pd.concat([buch_pd_X,buch_pd_y],axis=1)
    
    X1=todo[todo[y_columns]==0]
    X2=todo[todo[y_columns]==1]
#    X3=todo[todo['class']==2]
    
    # Data Subplot
    ax1 = fig.add_subplot(1,5,1,aspect=0.8)
#    ax1.set_xlim([0.0,1.0])
#    ax1.set_ylim([0.0,1.0])
    ax1.set_xlabel('$x_1$',fontsize=22)
    ax1.set_ylabel('$x_2$',fontsize=22)    
    ax1.title.set_text('Learned instances')
    ax1.scatter(X1.iloc[:,0], X1.iloc[:,1], color='#ff9d89', marker='.',edgecolors='k',linewidths=0.0, s=200)#FEE88A-#63c283
    ax1.scatter(X2.iloc[:,0], X2.iloc[:,1], color='#ff5273', marker='.',edgecolors='k',linewidths=0.0, s=200)#C47900-#63648b
    
    
    if num_automs==1:
        ax2_t = fig.add_subplot(1,5,2)
    elif num_automs==2:
        ax2_t = fig.add_subplot(1,5,2)        
        ax3_t = fig.add_subplot(1,5,3)
    
    if num_automs==1:                

        ax2_t.set_xticks([], [])
        ax2_t.set_yticks([], [])
        ax2_t.title.set_text('CURIE 2x10')
        ax2_t.imshow(images[0])
        
        flipped_mutants_t=np.flip(mutants_t[0],0)
        rotated_mutant_t= np.rot90(flipped_mutants_t, 2)

    elif num_automs==2:

        ax2_t.set_xticks([], [])
        ax2_t.set_yticks([], [])
        ax2_t.title.set_text('CURIE 2x10')
        ax2_t.imshow(images[0])
        
        flipped_mutants_t=np.flip(mutants_t[0],0)
        rotated_mutant_t= np.rot90(flipped_mutants_t, 2)

        for i in range(0, len(rotated_mutant_t)):
            for j in range(0, len(rotated_mutant_t)):                
                ax2_t.text(i,j,rotated_mutant_t[i][j][0],ha='center',va='center')
                

        ax3_t.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')    
        ax3_t.title.set_text(ca_names[1]+': t last mut')
        ax3_t.imshow(images[1])
        
        flipped_mutants_t=np.flip(mutants_t[1],0)
        rotated_mutant_t= np.rot90(flipped_mutants_t, 2)

        for i in range(0, len(rotated_mutant_t)):
            for j in range(0, len(rotated_mutant_t)):                
                ax3_t.text(i,j,rotated_mutant_t[i][j][0],ha='center',va='center')
                    
    
    fig.tight_layout()    

    plt.savefig('current_image_'+str(punto)+'.svg')
    
    plt.show() 

def prequential_acc(predicted_class,Y_tst,PREQ_ACCS,t,f):

    #Prequential accuracy
    pred=0
    if predicted_class==Y_tst:    
        pred=1
    else:
        pred=0

    if t==0:
        preqAcc=1
    else:   
        preqAcc=(PREQ_ACCS[-1]+float((pred-PREQ_ACCS[-1])/(t-f+1)))

    return preqAcc

#def cellular_automatas_naming(cellular_automatas):
#    
#    ca_names=[str()]*len(cellular_automatas)
#    for ca in range(len(cellular_automatas)):
#        ca_names[ca]=r'\texttt{sCA}$'
#
#    return ca_names

    
#def automatas_Texttable(cellular_automatas,automatas_results_mean,automatas_results_std,bd,ad,drift_position,measure_position_after_drift,t_automatas,title,names):    
#    
#    bd_automatas_mean=[[]]*len(cellular_automatas)
#    bd_automatas_std=[[]]*len(cellular_automatas)
#    for h in range(len(cellular_automatas)):
#        bd_automatas_mean[h]=np.round((automatas_results_mean[h][bd]),3)
#        bd_automatas_std[h]=np.round((automatas_results_std[h][bd]),3)
#
#    d_automatas_mean=[[]]*len(cellular_automatas)
#    d_automatas_std=[[]]*len(cellular_automatas)
#    for h in range(len(cellular_automatas)):
#        d_automatas_mean[h]=np.round((automatas_results_mean[h][measure_position_after_drift]),3)
#        d_automatas_std[h]=np.round((automatas_results_std[h][measure_position_after_drift]),3)
#
#    ad_automatas_mean=[[]]*len(cellular_automatas)
#    ad_automatas_std=[[]]*len(cellular_automatas)
#    for h in range(len(cellular_automatas)):
#        ad_automatas_mean[h]=np.round((automatas_results_mean[h][ad]),3)
#        ad_automatas_std[h]=np.round((automatas_results_std[h][ad]),3)
#
#    for h in range(len(cellular_automatas)):
#        t_automatas.add_rows([['AUTOMATAS_'+title, 'Accuracy BD','Accuracy D','Accuracy AD'],[str(names[h]),str(bd_automatas_mean[h])+str('+-')+str(bd_automatas_std[h]),str(d_automatas_mean[h])+str('+-')+str(d_automatas_std[h]),str(ad_automatas_mean[h])+str('+-')+str(ad_automatas_std[h])]])
#    
#    print (t_automatas.draw())    


def plot_automatas(size_X,size_Y,color,font_size,title,ca_name,no_scores,drift_pos,mean_scores):
    
    fig, axes = plt.subplots(1,1,figsize=(size_X,size_Y))
    
    plt.title(title,size=font_size)
    axes.set_xlabel(r't',size=font_size)
    axes.set_ylabel(r'Prequential accuracy',size=font_size)
    plt.ylim(0.0,1.0)
    axes.set_xlim(0,len(mean_scores))
            
    axes.plot(mean_scores,color='b',label=ca_name,linestyle='-')
                              
    axes.axvspan(0, no_scores, alpha=0.5, color='#C47900')  
                 
    for ps in range(len(drift_pos)):
        axes.axvline(x=drift_pos[ps],color='k', linestyle='-')

    plt.show()  
    

#def plot_learners(size_X,size_Y,color,font_size,title,learner_name,no_scores,drift_pos,mean_scores,stds_scores):
#
#    fig, axes = plt.subplots(1,1,figsize=(size_X,size_Y))
#    
#    plt.title(title,size=font_size)
#    axes.set_xlabel(r't',size=font_size)
#    axes.set_ylabel(r'Prequential accuracy',size=font_size)
#    plt.ylim(0.0,1.0)
#    axes.set_xlim(0,len(mean_scores))
#            
#    axes.plot(mean_scores,color='b',label=learner_name,linestyle='-')
#    axes.fill_between(range(len(mean_scores)), mean_scores-stds_scores, mean_scores+stds_scores,facecolor='#C47900', alpha=0.1)    
#
#    axes.axvspan(0, no_scores, alpha=0.5, color='#C47900')  
#                 
#    for ps in range(len(drift_pos)):
#        axes.axvline(x=drift_pos[ps],color='k', linestyle='-')
#    
#    plt.show()  
    
def get_neighbourhood(matrix, coordinates, distance):
    
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

        dimesion_span = range(dimensions_coordinate - remaining_distance, 
                              dimensions_coordinate + remaining_distance + 1)
        
        for c in dimesion_span:
            if 0 <= c < len(arr):
                recc_von_neumann(arr[c], 
                                 curr_dim + 1, 
                                 remaining_distance - abs(dimensions_coordinate - c), 
                                 isCenter and dimensions_coordinate == c)
        return

    recc_von_neumann(matrix)
    return neigh    

#def prequential_mut_calc(m,alpha,t,prev_fading_sum,prev_fading_increment):
#
#    f_sum=m+(alpha*prev_fading_sum)
#    f_increment=1+(alpha*prev_fading_increment)
#    preq_mut=f_sum/f_increment
#        
#    return preq_mut

def hyperparametertuning_classifiers(learn,X,y,knn_max_w_size):
            
    cl_name=learn.__class__.__name__                                                
#    print (cl_name)

    scor='balanced_accuracy'
    cv=10

    if cl_name=='KNN':

        KNN_grid = {'n_neighbors': [3,5,7,10,15],
                      'leaf_size': [3,5,7,10,15],
                      'algorithm':['kd_tree']
                      }        

        grid_cv_KNN = GridSearchCV(estimator=KNeighborsClassifier(), cv=cv,scoring=scor,param_grid=KNN_grid)
#        grid_cv_KNN = RandomizedSearchCV(estimator=KNeighborsClassifier(), cv=cv,scoring=scor,param_distributions=KNN_grid)
        grid_cv_KNN.fit(X.as_matrix(),y.as_matrix().ravel())                
#        print('grid_cv_KNN.best_params_: ',grid_cv_KNN.best_params_)
        n_neighbors=grid_cv_KNN.best_params_['n_neighbors']
        leaf_size=grid_cv_KNN.best_params_['leaf_size']
        
        tuned_params = {'n_neighbors': n_neighbors,'leaf_size': leaf_size,'max_window_size':knn_max_w_size} 
        
        tuned_learn=KNN()            
        tuned_learn.set_params(**tuned_params)
        tuned_learn.fit(X.as_matrix(), y.as_matrix().ravel())
        
    elif cl_name=='HoeffdingTree':
        
        grace_period_range=np.array([25,75,150,300])
        tie_threshold_range=np.linspace(0.001,1.0,5)
        split_confidence_range=np.linspace(0.000000001,0.1,5)    
        split_criterion_range=['gini','info_gain', 'hellinger']
        leaf_prediction_range=['mc','nb', 'nba']
    
        HT_grid = {
            'grace_period': grace_period_range,
            'tie_threshold': tie_threshold_range,
            'split_confidence': split_confidence_range,
            'split_criterion':split_criterion_range,
            'leaf_prediction':leaf_prediction_range
       }        
        
        grid_cv_HT=GridSearchCV(estimator=learn,scoring=scor,cv=cv,param_grid=HT_grid)
#        grid_cv_HT=RandomizedSearchCV(estimator=learn,scoring=scor,cv=cv,param_distributions=HT_grid)
        grid_cv_HT.fit(X.as_matrix(), y.as_matrix().ravel())                
#        print('grid_cv_HT.best_params_: ',grid_cv_HT.best_params_)

        tuned_params=grid_cv_HT.best_params_
        tuned_learn=grid_cv_HT.best_estimator_

    elif cl_name=='NaiveBayes':
        
        tuned_params = {'nominal_attributes': None} 
        tuned_learn=NaiveBayes()
        tuned_learn.set_params(**tuned_params)        
        tuned_learn.fit(X.as_matrix(), y.as_matrix().ravel())

#    print('Final tuned algorithm: ',tuned_learn)
                            
    return tuned_learn,tuned_params
        
def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()
    
#def genenerate_LatexTable():
#    geometry_options = {
#        "margin": "2.54cm",
#        "includeheadfoot": True
#    }
#    doc = Document(page_numbers=True, geometry_options=geometry_options)
#
#    
#    # Generate data table
#    with doc.create(LongTable("l l l")) as data_table:
#            
#            data_table.add_hline()
#            data_table.add_row(["header 1", "header 2", "header 3"])
#            data_table.add_hline()
#            data_table.end_table_header()
#            data_table.add_hline()
#            data_table.add_row((MultiColumn(3, align='r',data='Continued on Next Page'),))
#            data_table.add_hline()
#            data_table.end_table_footer()
#            data_table.add_hline()
#            data_table.add_row((MultiColumn(3, align='r',data='Not Continued on Next Page'),))
#            data_table.add_hline()
#            data_table.end_table_last_footer()
#            
#            row = ["Content1", "9", "Longer String"]
#            for i in range(3):
#                data_table.add_row(row)
#
#    doc.generate_pdf('ejemplo', clean_tex=False)
#    
#    doc.generate_pdf('synteticos', clean_tex=False)

#==============================================================================
# DATASETS REALES
#==============================================================================


#==============================================================================
# DATASETS SINTETICOS
#==============================================================================


#TXUS

#name_data='txus'    
datasets=['sine','rt','mixed','sea','stagger']#['noaa']#['gmsc']#['poker']
tipos=['abrupto','gradual']#['real']
#noise=0.0

#==============================================================================
# VARIABLES
#==============================================================================

#CA
bins_margin=0.001
mutation_period=10#5,10,20,50
num_mutantneighs_fordetection=2#2-synthetics,4-real
preparatory_size=50#50. Para real=500
sliding_window_size=preparatory_size#50
radius=2#2
    
#row_ref=2
#column_ref=3

#==============================================================================
# MAIN
#==============================================================================
# Ignore warnings
warnings.simplefilter("ignore")

path_saving_results='//home//txuslopez//Insync//txusbarrell@gmail.com//Google Drive//Dropbox//jlopezlobo//Publicaciones//ECML_2021//Results//F2//'

DAT_SCORES=[]
DAT_TIMES=[]
DAT_RAMS=[]
DAT_DETECTIONS=[]

for dats in datasets:

    TIPO_SCORES=[]
    TIPO_TIMES=[]
    TIPO_RAMS=[]
    TIPO_DETECTIONS=[]
    
    for tipo in tipos:

        if dats=='sine':
            functions_order=[3,2,1,0]
            functions_name_file=[3,2,1,0]
            columns=['X1','X2','class']
            file_name=str(dats)+'_'+str(functions_name_file[0])+str(functions_name_file[1])+str(functions_name_file[2])+str(functions_name_file[3])+'_'+str(tipo)
            n_bins=20#divisiones por feature
        elif dats=='rt':
            functions_order=[2563,7896,9856,8873]
            functions_name_file=[2563,7896,9856,8873]
            columns=['X1','X2','class']
            file_name=str(dats)+'_'+str(functions_name_file[0])+str(functions_name_file[1])+str(functions_name_file[2])+str(functions_name_file[3])+'_'+str(tipo)
            n_bins=20#divisiones por feature
        elif dats=='mixed':
            functions_order=[1,0,1,0]
            functions_name_file=[1,0,1,0]
            columns=['X1','X2','X3','X4','class']
            file_name=str(dats)+'_'+str(functions_name_file[0])+str(functions_name_file[1])+str(functions_name_file[2])+str(functions_name_file[3])+'_'+str(tipo)
            n_bins=10#divisiones por feature
        elif dats=='sea':
            functions_order=[3,2,1,0]
            functions_name_file=[3,2,1,0]
            columns=['X1','X2','X3','class']
            noise=0.2#0.0,0.2
            file_name=str(dats)+'_'+str(functions_name_file[0])+str(functions_name_file[1])+str(functions_name_file[2])+str(functions_name_file[3])+'_'+str(tipo)+'_noise_'+str(noise)                
            n_bins=10#divisiones por feature
        elif dats=='stagger':
            functions_order=[2,1,0,2]
            functions_name_file=[2,1,0,2]
            columns=['X1','X2','X3','class']
            n_bins=10#divisiones por feature
#        elif dats=='noaa':
#            columns=['X1','X2','X3','X4','X5','X6','X7','X8','class']
#            n_bins=3#3#divisiones por feature
#        elif dats=='gmsc':
#            columns=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','class']
#            n_bins=3#3#divisiones por feature                            
#        elif dats=='poker':
#            columns=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','class']
#            n_bins=3#3#divisiones por feature
        
        
        
        if tipo=='gradual':
            drift_positions=[9500,20000,30500]
            anch=1000    
        elif tipo=='abrupto':
            drift_positions=[10000,20000,30000]
            lengt_concept=9500        
            anch=1
    
#        if dats=='noaa':
#            path='/home/txuslopez/Insync/txusbarrell@gmail.com/Google Drive/Dropbox/jlopezlobo/PY/multiflow_txus/scikit-multiflow-master/src/skmultiflow/data/datasets/weather.csv'
#            raw_data= pd.read_csv(path, sep=',',header=0)
#            
#            x = raw_data.values
#            min_max_scaler = preprocessing.MinMaxScaler()
#            x_scaled = min_max_scaler.fit_transform(x)
#            raw_data = pd.DataFrame(x_scaled)
#
#        elif dats=='gmsc':
#            path='/home/txuslopez/Insync/txusbarrell@gmail.com/Google Drive/Dropbox/jlopezlobo/Data sets/Non stationary environments/GMSC/cs-training_Amazon_def.csv'
#            raw_data = pd.read_csv(path, sep=',', header=0)
#            
#            raw_data = raw_data.drop('Unnamed: 0', 1)#Quitamos la primera columna
#            raw_data=raw_data.dropna(how='any')#Se quitan las filas con Nan
#            raw_data=raw_data[0:20000]#Limitar datos a 20k samples    
#            raw_data.columns=['RevolvingUtilizationOfUnsecuredLines', 'age',
#                   'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
#                   'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
#                   'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
#                   'NumberOfDependents', 'class']
#            
#            
#            x = raw_data.values
#            min_max_scaler = preprocessing.MinMaxScaler()
#            x_scaled = min_max_scaler.fit_transform(x)
#            raw_data = pd.DataFrame(x_scaled)
#
#        elif dats=='poker':
#            path='/home/txuslopez/Insync/txusbarrell@gmail.com/Google Drive/Dropbox/jlopezlobo/Data sets/Non stationary environments/Poker_hand/norm.csv'
#            raw_data = pd.read_csv(path, sep=',', header=None)
#            raw_data=raw_data.iloc[np.random.permutation(len(raw_data))]
#            raw_data=raw_data.iloc[:20000]
#                                            
#        else:
        path='//home//txuslopez//Insync//txusbarrell@gmail.com//Google Drive//Dropbox//jlopezlobo//Publicaciones//ECML_2021//Data//F2//'
        raw_data= pd.read_csv(path +file_name+'.csv', sep=',',header=True)
            #print(path +file_name+'.csv')                
            
        if dats=='sine' or dats=='rt':
            caso=raw_data[raw_data.columns[0:3]]
            XT=caso.iloc[:,0:2]
            YT=caso.iloc[:,2]
        elif dats=='mixed':
            caso=raw_data[raw_data.columns[0:5]]
            XT=caso.iloc[:,0:4]
            YT=caso.iloc[:,4]
        elif dats=='sea':
            caso=raw_data[raw_data.columns[0:4]]
            XT=caso.iloc[:,0:3]
            YT=caso.iloc[:,3]
        elif dats=='stagger':
            caso=raw_data[raw_data.columns[0:4]]
            XT=caso.iloc[:,0:3]
            YT=caso.iloc[:,3]
        elif dats=='noaa':
            caso=raw_data[raw_data.columns[0:9]]
            XT=caso.iloc[:,0:8]
            YT=caso.iloc[:,8]
        elif dats=='gmsc':
            caso=raw_data[raw_data.columns[0:11]]
            XT=caso.iloc[:,0:10]
            YT=caso.iloc[:,10]                        
        elif dats=='poker':
            caso=raw_data[raw_data.columns[0:11]]
            XT=caso.iloc[:,0:10]
            YT=caso.iloc[:,10]

        caso.columns=columns        
        
        columns=columns[:-1]#le quitamos el class a partir de ahora            
        n_feats=len(columns)        
                            
        #Data
        features=pd.DataFrame(XT)
        labels=pd.DataFrame(YT)
        features.columns=columns            
        labels.columns=['class']
        n_samples=XT.shape[0]-preparatory_size
       
        
        ######################## CURIE ###################
        
        lst_dim=[n_bins]*n_feats
        curie=CA_VonNeumann_Classifier(bins=[],bins_margin=bins_margin,dimensions=lst_dim, cells=empties(lst_dim))
        limits_automata=list(np.zeros(1))
        #ca_names=['CURIE']
        mutants_time=empty_mutant(curie.dimensions)
        
        ######################## LEARNERS ###################
        learners_ref=[HoeffdingTree(),KNN(),NaiveBayes()]
        ######################## DETECTORS ###################
        detectores_ref=[DDM(),EDDM(),ADWIN(),PageHinkley(),curie]
        
        n_pasos=len(datasets)*len(tipos)*len(learners_ref)*len(detectores_ref)
        
        SCORES_LER=[]
        TIMES_LER=[]
        RAMS_LER=[]
        DETECTIONS_LER=[]
                
        for ler in range(len(learners_ref)):    

            learner=deepcopy(learners_ref[ler])
            
            SCORES_DET=[]
            TIMES_DET=[]
            RAMS_DET=[]
            DETECTIONS_DET=[]
            
            for det in range(len(detectores_ref)):    
            
                scores_ler=[]
                time_ler=0
                ram_ler=0
                f_ler=1
                detections=[]
                detector=deepcopy(detectores_ref[det])
                            
                for s in range(features.shape[0]):     
                               
                    sample=np.array(features.iloc[s,:]).reshape(1, -1)
                    lab=np.array(labels.iloc[s,:])
                    
                    if s<preparatory_size-1:
                        scores_ler.append(np.nan)   
#                        time_ler.append(np.nan)   
#                        ram_ler.append(np.nan)   
            
                    elif s==preparatory_size:

#                            print ('PREPARATORY PROCESS ...')
                                        
                        X_init=features.iloc[0:preparatory_size,:]
                        y_init=labels.iloc[0:preparatory_size,:]

                        #Hyperparameter tuning for learners
                        tuned_learner,tuned_params=hyperparametertuning_classifiers(learner,X_init,y_init,sliding_window_size)                                                
                        learner=deepcopy(tuned_learner)                            
            
                        start_time = timer()#time.clock()                                
                        start_ram = psutil.virtual_memory().used#measured in bytes
                                        
                        learner.fit(X_init.as_matrix(), y_init.as_matrix().ravel())                        

                        #CURIE
                        if detector.__class__.__name__=='CA_VonNeumann_Classifier':                             
                            detector,lim_automat=detector.fit(X_init.as_matrix(), y_init.as_matrix().ravel())
                        
                        process_time=timer()-start_time
                        process_ram=psutil.virtual_memory().used-start_ram
                        if process_ram<0:
                            process_ram=0
                        
                        scores_ler.append(np.nan)   
                        time_ler+=process_time
                        ram_ler+=process_ram
                        
                    elif s>preparatory_size:
                                  
#                            print ('TEST-THEN-TRAIN PROCESS ...')
                        
                        #Testing                                           

                        start_time = timer()#time.clock()                                
                        start_ram = psutil.virtual_memory().used#measured in bytes
                        
                        pred=learner.predict(sample)    

                        process_time=timer()-start_time
                        process_ram=psutil.virtual_memory().used-start_ram
                        if process_ram<0:
                            process_ram=0
                        time_ler+=process_time
                        ram_ler+=process_ram
                            
                        #Scoring
                        if str(scores_ler[-1])=='nan':
                            if pred==lab:
                                scores_ler.append(1.0)
                            else:
                                scores_ler.append(0.0)
                                
                        else:
                            preqAcc=prequential_acc(pred,lab,scores_ler,s,f_ler)
                            scores_ler.append(preqAcc)   
                            
                        #Training   

                        start_time = timer()#time.clock()                                
                        start_ram = psutil.virtual_memory().used#measured in bytes

                        learner.partial_fit(sample,lab)

                        process_time=timer()-start_time
                        process_ram=psutil.virtual_memory().used-start_ram
                        if process_ram<0:
                            process_ram=0
                        time_ler+=process_time
                        ram_ler+=process_ram
        
                        ############
                        #DETECTION
                        ############
                        change=False
                        
                        start_time = timer()#time.clock()                                
                        start_ram = psutil.virtual_memory().used#measured in bytes
                        
                        if detector.__class__.__name__=='CA_VonNeumann_Classifier':                             
                            
                            #Train 
                            detector,lim_automat,muta,indxs=detector.partial_fit(sample,lab,s,lim_automat)      
                                                                                                                                                                                          
                            if muta:   
                
                                if dats=='sine' or dats=='rt':                                                                            
                                    mutants_time[indxs[0]][indxs[1]][0]=s
                                elif dats=='mixed':                                                                            
                                    mutants_time[indxs[0]][indxs[1]][indxs[2]][indxs[3]][0]=s
                                elif dats=='sea' or dats=='stagger':                                                                            
                                    mutants_time[indxs[0]][indxs[1]][indxs[2]][0]=s
                                                            
                                #Buscamos drift
                                vecinos_mutantes_drift=get_neighbourhood(mutants_time, indxs, radius)
                
                                num_mutantes_drift=0         
                                ms=[]
                                for v in range(len(vecinos_mutantes_drift)):
                                    if vecinos_mutantes_drift[v][0]>s-mutation_period and vecinos_mutantes_drift[v][0]<=s:
                                        num_mutantes_drift+=1
                                        ms.append(vecinos_mutantes_drift[v][0])
                                                                                               
                                #Si hay drift:
                                if num_mutantes_drift>=num_mutantneighs_fordetection:
                                    change=True
                                                                                    
                                    #Adaptacion                                
                                    mutants_time=empty_mutant(detector.dimensions)
                                        
                                    X_init=features.iloc[s-preparatory_size:s,:]
                                    y_init=labels.iloc[s-preparatory_size:s,:]                                    
                                        
                                    detector=deepcopy(curie)
                
                                    detector,lim_automat=detector.fit(X_init.as_matrix(), y_init.as_matrix().ravel())
                                            
                            
                            
                        else:
        
                            if pred==lab:
                                detector.add_element(1)
                            else:
                                detector.add_element(0)
                                
                            if detector.detected_change():
                                change=True

                        if change:

                            ############
                            #ADAPTATION
                            ############                                
                            f_ler=s                        
                            detections.append(s)
                            #Se reinicia el detector
                            detector=deepcopy(detectores_ref[det])

                            X_init=features.iloc[s-preparatory_size:s,:]
                            y_init=labels.iloc[s-preparatory_size:s,:]
            
                            learner=deepcopy(learners_ref[ler])
                            learner.set_params(**tuned_params)                            
                            learner.fit(X_init.as_matrix(), y_init.as_matrix().ravel())
                            
                        process_time=timer()-start_time
                        process_ram=psutil.virtual_memory().used-start_ram
                        if process_ram<0:
                            process_ram=0
                        time_ler+=process_time
                        ram_ler+=process_ram
                        
                SCORES_DET.append(scores_ler)
                TIMES_DET.append(time_ler)
                RAMS_DET.append(ram_ler)
                DETECTIONS_DET.append(detections)
                
            SCORES_LER.append(SCORES_DET)
            TIMES_LER.append(TIMES_DET)
            RAMS_LER.append(RAMS_DET)
            DETECTIONS_LER.append(DETECTIONS_DET)
            
        TIPO_SCORES.append(SCORES_LER)
        TIPO_TIMES.append(TIMES_LER)
        TIPO_RAMS.append(RAMS_LER)
        TIPO_DETECTIONS.append(DETECTIONS_LER)
    
    DAT_SCORES.append(TIPO_SCORES)
    DAT_TIMES.append(TIPO_TIMES)
    DAT_RAMS.append(TIPO_RAMS)
    DAT_DETECTIONS.append(TIPO_DETECTIONS)
  
    ######################## SAVING ########################    
    
    output = open(path_saving_results+'DAT_SCORES_'+dats+'.pkl', 'wb')
    pickle.dump(DAT_SCORES, output)
    output.close()
    
    output = open(path_saving_results+'DAT_TIMES_'+dats+'.pkl', 'wb')
    pickle.dump(DAT_TIMES, output)
    output.close()
    
    output = open(path_saving_results+'DAT_RAMS_'+dats+'.pkl', 'wb')
    pickle.dump(DAT_RAMS, output)
    output.close()
    
    output = open(path_saving_results+'DAT_DETECTIONS_'+dats+'.pkl', 'wb')
    pickle.dump(DAT_DETECTIONS, output)
    output.close()    
        
    
######################## RESUMEN ########################    
    
for ds in range(len(datasets)): 
    print('######## DATASET: ',datasets[ds])                                                                         
          
    ######################## LOADING RESULTS AND METRICS ########################    
    
    fil = open(path_saving_results+'DAT_SCORES_'+dats+'.pkl','rb')
    DAT_SCORES = pickle.load(fil)
    fil.close()
    
    fil = open(path_saving_results+'DAT_TIMES_'+dats+'.pkl','rb')
    DAT_TIMES = pickle.load(fil)
    fil.close()
    
    fil = open(path_saving_results+'DAT_RAMS_'+dats+'.pkl','rb')
    DAT_RAMS = pickle.load(fil)
    fil.close()
    
    fil = open(path_saving_results+'DAT_DETECTIONS_'+dats+'.pkl','rb')
    DAT_DETECTIONS = pickle.load(fil)
    fil.close()        
          
    dat_score=DAT_SCORES[ds]
    dat_time=DAT_TIMES[ds]
    dat_ram=DAT_RAMS[ds]
    dat_detections=DAT_DETECTIONS[ds]
    
    for tip in range(len(tipos)): 
        print('###### TIPO: ',tipos[tip])                                                                         
        tipo_score=dat_score[tip]
        tipo_times=dat_time[tip]
        tipo_rams=dat_ram[tip]
        tipo_detections=dat_detections[tip]
  
        for l in range(len(learners_ref)):
            print('#### LEARNER: ',learners_ref[l].__class__.__name__)                                                                         
            scores_ler=tipo_score[l]
            times_ler=tipo_times[l]
            rams_ler=tipo_rams[l]
            detections_ler=tipo_detections[l]

            for d in range(len(detectores_ref)):
                print('## DETECTOR: ',detectores_ref[d].__class__.__name__)                                                                         
                scores_det=scores_ler[d]
                times_det=times_ler[d]
                rams_det=rams_ler[d]
                detections_det=detections_ler[d]
                print('')
                print('-MEAN PREQ.ACC: ',np.nanmean(scores_det))
                print('-TIME: ',np.nanmean(times_det))
                print('-RAM: ',np.nanmean(rams_det))
                print('-RAM-Hour: ',(np.nanmean(rams_det)/1073741824)*(np.nanmean(times_det)/360))#bytes/secs to gigas/hours)                
                print('-DETECTIONS: ',detections_det)                        
                print('')


                

    
with open(path_saving_results+'temp.csv', mode='w') as results_synths:
    
    results_synths_writer = csv.writer(results_synths, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)            
    results_synths_writer.writerow(['Dataset','Type','Learner','Detector','pACC','RAM-Hours','TP','FP','TN','FN','UD','Precision','Recall','MCC'])

#    try:
    
    friedman_DDM_preqacc=[]
    friedman_DDM_ramhours=[]
    friedman_DDM_ud=[]
    friedman_DDM_mcc=[]

    friedman_EDDM_preqacc=[]
    friedman_EDDM_ramhours=[]
    friedman_EDDM_ud=[]
    friedman_EDDM_mcc=[]
    
    friedman_ADWIN_preqacc=[]
    friedman_ADWIN_ramhours=[]
    friedman_ADWIN_ud=[]
    friedman_ADWIN_mcc=[]
    
    friedman_PH_preqacc=[]
    friedman_PH_ramhours=[]
    friedman_PH_ud=[]
    friedman_PH_mcc=[]
    
    friedman_CURIE_preqacc=[]
    friedman_CURIE_ramhours=[]
    friedman_CURIE_ud=[]
    friedman_CURIE_mcc=[]
    
    for ds in range(len(datasets)):
        dat_score=DAT_SCORES[ds]
        dat_time=DAT_TIMES[ds]
        dat_ram=DAT_RAMS[ds]
        dat_detections=DAT_DETECTIONS[ds]
        
        for tip in range(len(tipos)):
            tipo_score=dat_score[tip]
            tipo_times=dat_time[tip]
            tipo_rams=dat_ram[tip]
            tipo_detections=dat_detections[tip]
            
            for l in range(len(learners_ref)):
                scores_ler=tipo_score[l]
                times_ler=tipo_times[l]
                rams_ler=tipo_rams[l]
                detections_ler=tipo_detections[l]
                
                for det in range(len(detectores_ref)):
                    scores_det=scores_ler[det]
                    times_det=times_ler[det]
                    rams_det=rams_ler[det]
                    detections_det=detections_ler[det]                                        
                    
                    if tipos[tip]=='abrupto' or tipos[tip]=='gradual':
                        if tipos[tip]=='abrupto':
                            detection_margin=0.02
                        elif tipos[tip]=='gradual':
                            detection_margin=0.1

                        lear_tp=0
                        lear_fp=0
                        lear_tn=0
                        lear_fn=0
                        lear_mcc=0
                        lear_udd=0
                        cont_udd=0
                
                        for d in detections_det:
                            #Checking BEFORE drift 1
                            if d<drift_positions[0]:
                                lear_fp+=1
                            #Checking drift 1
                            elif d>drift_positions[0] and d<drift_positions[1] and d-drift_positions[0]<=detection_margin*lengt_concept:
                                lear_tp+=1
                                lear_udd+=(d-drift_positions[0])
                                cont_udd+=1                                    
                            elif d>drift_positions[0] and d<drift_positions[1] and d-drift_positions[0]>detection_margin*lengt_concept:
                                lear_fp+=1
                            #Checking drift 2
                            elif d>drift_positions[1] and d<drift_positions[2] and d-drift_positions[1]<=detection_margin*lengt_concept:
                                lear_tp+=1
                                lear_udd+=(d-drift_positions[1])
                                cont_udd+=1
                            elif d>drift_positions[1] and d<drift_positions[2] and d-drift_positions[1]>detection_margin*lengt_concept:
                                lear_fp+=1
                            #Checking drift 3
                            elif d>drift_positions[2] and d-drift_positions[2]<=detection_margin*lengt_concept:
                                lear_tp+=1
                                lear_udd+=(d-drift_positions[2])
                                cont_udd+=1
                            elif d>drift_positions[2] and d-drift_positions[2]>detection_margin*lengt_concept:
                                lear_fp+=1

                        lear_tn=n_samples-len(detections_det)
                        
                        lear_fn=len(drift_positions)-lear_tp
                        if lear_fn<0:
                            lear_fn=0
                                                            
                        if cont_udd>0:
                            lear_udd=np.round(lear_udd/cont_udd,2)                                
                        else:
                            lear_udd=np.inf
                        
                        if (lear_tp+lear_fp)==0:
                            lear_precision=0.0                            
                        else:
                            lear_precision=lear_tp/(lear_tp+lear_fp)
                            
                        if (lear_tp+lear_fn)==0:     
                            lear_recall=0.0                            
                        else:                            
                            lear_recall=lear_tp/(lear_tp+lear_fn)

                        if np.sqrt((lear_tp+lear_fp)*(lear_tp+lear_fn)*(lear_tn+lear_fp)*(lear_tn+lear_fn))==0:                                 
                            lear_mcc=0.0
                        else:
                            lear_mcc=((lear_tp*lear_tn)-(lear_fp*lear_fn))/np.sqrt((lear_tp+lear_fp)*(lear_tp+lear_fn)*(lear_tn+lear_fp)*(lear_tn+lear_fn))
               
                        lear_ram_hours=(rams_det/1073741824)*(times_det/360)#bytes/secs to gigas/hours
                
                        results_synths_writer.writerow([datasets[ds],tipos[tip],learners_ref[l].__class__.__name__,detectores_ref[det].__class__.__name__,np.round(np.nanmean(scores_det),2),np.round(lear_ram_hours,6),lear_tp,lear_fp,lear_tn,lear_fn,np.round(lear_udd,2),np.round(lear_precision,2),np.round(lear_recall,2),np.round(lear_mcc,2)])
                        
                        ###### FRIEDMAN
                        if detectores_ref[det].__class__.__name__=='DDM':
                            friedman_DDM_preqacc.append(np.round(np.nanmean(scores_det),2))
                            friedman_DDM_ramhours.append(np.round(lear_ram_hours,6))
                            friedman_DDM_ud.append(np.round(lear_udd,2))
                            friedman_DDM_mcc.append(np.round(lear_mcc,2))
                        elif detectores_ref[det].__class__.__name__=='EDDM':
                            friedman_EDDM_preqacc.append(np.round(np.nanmean(scores_det),2))
                            friedman_EDDM_ramhours.append(np.round(lear_ram_hours,6))
                            friedman_EDDM_ud.append(np.round(lear_udd,2))
                            friedman_EDDM_mcc.append(np.round(lear_mcc,2))
                        elif detectores_ref[det].__class__.__name__=='ADWIN':
                            friedman_ADWIN_preqacc.append(np.round(np.nanmean(scores_det),2))
                            friedman_ADWIN_ramhours.append(np.round(lear_ram_hours,6))
                            friedman_ADWIN_ud.append(np.round(lear_udd,2))
                            friedman_ADWIN_mcc.append(np.round(lear_mcc,2))
                        elif detectores_ref[det].__class__.__name__=='PageHinkley':
                            friedman_PH_preqacc.append(np.round(np.nanmean(scores_det),2))
                            friedman_PH_ramhours.append(np.round(lear_ram_hours,6))
                            friedman_PH_ud.append(np.round(lear_udd,2))
                            friedman_PH_mcc.append(np.round(lear_mcc,2))
                        elif detectores_ref[det].__class__.__name__=='CA_VonNeumann_Classifier':
                            friedman_CURIE_preqacc.append(np.round(np.nanmean(scores_det),2))
                            friedman_CURIE_ramhours.append(np.round(lear_ram_hours,6))
                            friedman_CURIE_ud.append(np.round(lear_udd,2))
                            friedman_CURIE_mcc.append(np.round(lear_mcc,2))                            
                        
                    elif tipos[tip]=='real':

                        lear_ram_hours=(rams_det/1073741824)*(times_det/360)#bytes/secs to gigas/hours
                        results_synths_writer.writerow([datasets[ds],tipos[tip],learners_ref[l].__class__.__name__,detectores_ref[det].__class__.__name__,np.round(np.nanmean(scores_det),2),np.round(lear_ram_hours,6),np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
                                
        
#    except Exception as e:
#        print('En ',ds,'_',tip,'_',l,'_',det)
#        print (e.__doc__)
#        print (e.message)
       
    
######################## FRIEDMAN TESTS ########################    

from scipy.stats import friedmanchisquare    

alpha = 0.05

#FOR PREQ.ACC
stat_preqacc,p_preqacc = friedmanchisquare(friedman_DDM_preqacc,friedman_EDDM_preqacc,friedman_ADWIN_preqacc,friedman_PH_preqacc,friedman_CURIE_preqacc)
print('---- PREQ. ACC ----')
print('Statistics=%.3f, p=%.3f' % (stat_preqacc, p_preqacc))
if p_preqacc > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
print('')

#FOR RAM-HOURS
stat_ramhours,p_ramhours = friedmanchisquare(friedman_DDM_ramhours,friedman_EDDM_ramhours,friedman_ADWIN_ramhours,friedman_PH_ramhours,friedman_CURIE_ramhours)
print('---- RAM HOURS ----')
print('Statistics=%.3f, p=%.3f' % (stat_ramhours, p_ramhours))
if p_ramhours > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
print('')

#FOR UD
stat_ud,p_ud = friedmanchisquare(friedman_DDM_ud,friedman_EDDM_ud,friedman_ADWIN_ud,friedman_PH_ud,friedman_CURIE_ud)
print('---- UD ----')
print('Statistics=%.3f, p=%.3f' % (stat_ud, p_ud))
if p_ud > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
print('')

#FOR UD
stat_mcc,p_mcc= friedmanchisquare(friedman_DDM_mcc,friedman_EDDM_mcc,friedman_ADWIN_mcc,friedman_PH_mcc,friedman_CURIE_mcc)
print('---- MCC ----')
print('Statistics=%.3f, p=%.3f' % (stat_mcc, p_mcc))
if p_mcc > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
print('')

######################## NEMENYI TESTS AND GRAPHICS ########################    
  
import Orange
names = ["DDM", "EDDM", "ADWIN", "PH","CURIE" ]

print('---------PREQ.ACC:')
avranks_preqacc =  [2.72,4.00,2.18,3.24,2.81]#Cogido del excel res.xlsx
cd_preqacc = Orange.evaluation.compute_CD(avranks_preqacc, 20) #tested on 20 datasets
Orange.evaluation.graph_ranks(avranks_preqacc, names, cd=cd_preqacc, width=6, textspace=1.5)
plt.show()
print('CD=',cd_preqacc)
print('')

print('---------RAM-HOURS:')
avranks_ramhours =  [3.31,2.56,3.00,2.32,3.82]#Cogido del excel res.xlsx
cd_ramhours = Orange.evaluation.compute_CD(avranks_ramhours, 20) #tested on 20 datasets
Orange.evaluation.graph_ranks(avranks_ramhours, names, cd=cd_ramhours, width=6, textspace=1.5)
plt.show()
print('CD=',cd_ramhours)
print('')

print('---------UD:')
avranks_ud =  [3.81,2.85,2.40,3.54,1.90]#Cogido del excel res.xlsx
cd_ud = Orange.evaluation.compute_CD(avranks_ud, 20) #tested on 20 datasets
Orange.evaluation.graph_ranks(avranks_ud, names, cd=cd_ud, width=6, textspace=1.5)
plt.show()
print('CD=',cd_ud)
print('')

print('---------MCC:')
avranks_mcc =  [3.93,3.22,2.53,3.56,1.76]#Cogido del excel res.xlsx
cd_mcc = Orange.evaluation.compute_CD(avranks_mcc, 20) #tested on 20 datasets
Orange.evaluation.graph_ranks(avranks_mcc, names, cd=cd_mcc, width=6, textspace=1.5)
print('CD=',cd_mcc)
plt.show()
