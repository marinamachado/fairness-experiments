import sys
import numpy as np
import pandas as pd
from dataset_loader import datasets_loader
from settings import classifiers_settings_eniac_weights, classifiers_settings_eniac
from fairness_experiments import Experiment

from preprocessing_methods.massaging import Massaging
from preprocessing_methods.uniformSampling import UniformSampling
from preprocessing_methods.preferentialSampling import PreferentialSampling

from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover,LFR


def prepare_dataset(d,dataset,preprocessing = None, standard = False,weights = None):

    
    if(preprocessing != None):
        if(preprocessing == 'DispImpact'):
            DIR = DisparateImpactRemover()
            dataset = DIR.fit_transform(dataset)
            X = dataset.convert_to_dataframe()[0]
            y = X[d.label].replace(2.0,0.0)
        elif(preprocessing == 'RW'):
            RW = Reweighing(unprivileged_groups=d.unprivileged_groups,
               privileged_groups=d.privileged_groups)
            RW.fit(dataset)
            dataset = RW.transform(dataset) 
            weights = dataset.instance_weights
            X = dataset.convert_to_dataframe()[0]
            y = X[d.label].replace(2.0,0.0)
            
        elif(preprocessing == 'Massaging'):

            X_aux = dataset.convert_to_dataframe()[0]
            y_aux = X_aux[d.label].replace(2.0,0.0)
            y_aux = y_aux.ravel()
            X_aux = X_aux.drop(columns = [d.label],axis =1)
            MSS = Massaging(dataset)
            MSS = MSS.fit(X_aux,y_aux)
            dataset_mss = MSS.transform(X_aux,y_aux)

            X = dataset_mss
            y = X[d.label]
        
        elif(preprocessing == 'USample'):
            
            X_aux = dataset.convert_to_dataframe()[0]
            X_aux[d.label] = X_aux[d.label].replace(2.0,0.0)
            US = UniformSampling(d.attr,d.label)
            US = US.fit(X_aux)
            dataset_us,_ = US.transform(X_aux)

            X = dataset_us
            y = X[d.label]
            
        elif(preprocessing == 'PSample'):
            
            X_aux = dataset.convert_to_dataframe()[0]
            X_aux[d.label] = X_aux[d.label].replace(2.0,0.0)
            PS = PreferentialSampling(d.attr,d.label)
            PS = PS.fit(X_aux)
            dataset_ps = PS.transform(X_aux)

            X = dataset_ps
            y = X[d.label]
            
            
            
          
        elif(preprocessing == 'LFR'):
            Lfr = LFR(unprivileged_groups=d.unprivileged_groups,
               privileged_groups=d.privileged_groups)
            Lfr.fit(dataset)
            dataset = Lfr.transform(dataset)
            X = dataset.convert_to_dataframe()[0]
            y = X[d.label].replace(2.0,0.0)
    else:
        X = dataset.convert_to_dataframe()[0]
        y = X[d.label].replace(2.0,0.0)
            
    
    y = y.ravel()
    X = X.drop(columns = [d.label],axis =1)
    
    multindex = pd.MultiIndex.from_frame(X[dataset.protected_attribute_names])
    X = pd.DataFrame(X.to_numpy(),index = multindex,columns = X.columns)
    
    level_to_change = -1
    X.index = X.index.set_levels(X.index.levels[level_to_change].astype(str), level=level_to_change)
    
    X = X.drop(columns = [d.attr],axis =1)
    
    return X, y, weights

def execute(d,dataset,preprocess,data_name,preprocess_name):


    X, y, weights = prepare_dataset(d,dataset,preprocess,standard = False)

    print(y)

    
    
    if preprocess_name == 'RW':
        exp_teste = Experiment(classifiers_settings_eniac_weights, dataset_name=data_name, preprocessing_name=preprocess_name, 
                 privileged_group='1.0')
        exp_teste.execute(X, y, weights)
    else: 
        exp_teste = Experiment(classifiers_settings_eniac, dataset_name=data_name, preprocessing_name=preprocess_name, 
                 privileged_group='1.0')
        exp_teste.execute(X, y)
    
    return exp_teste.report



def german(df):
    d  = datasets_loader()
    d.load_german_dataset()
    german_orig = d.dataset

    non_preproc = execute(d,german_orig,None,'German','sem pré-processamento')

    df = df.append(non_preproc)
    
    preproc = execute(d,german_orig,'DispImpact','German','DispImpact')

    df = df.append(preproc)
    
    preproc = execute(d,german_orig,'RW','German','RW')

    df = df.append(preproc)
    
    preproc = execute(d,german_orig,'Massaging','German','Massaging')

    df = df.append(preproc)

    return df

def adult(df):
    d  = datasets_loader()
    d.load_adult_dataset()
    adult_orig = d.dataset

    #non_preproc = execute(d,adult_orig,None,'Adult','sem pré-processamento')

    #df = df.append(non_preproc)
    
    #preproc = execute(d,adult_orig,'DispImpact','Adult','DispImpact')

    #df = df.append(preproc)
    
    preproc = execute(d,german_orig,'RW','Adult','RW')

    df = df.append(preproc)
    
    preproc = execute(d,german_orig,'Massaging','Adult','Massaging')

    df = df.append(preproc)


    return df

def bank(df):
    d  = datasets_loader()
    d.load_bank_dataset()
    bank_orig = d.dataset

    #non_preproc = execute(d,bank_orig,None,'Bank','sem pré-processamento')

    #df = df.append(non_preproc)
    
    #preproc = execute(d,bank_orig,'DispImpact','Bank','DispImpact')

    #df = df.append(preproc)
    
    preproc = execute(d,bank_orig,'RW','Bank','RW')

    df = df.append(preproc)
    
    preproc = execute(d,bank_orig,'Massaging','Bank','Massaging')

    df = df.append(preproc)


    return df


def compas(df):
    d  = datasets_loader()
    d.load_compas_dataset()
    compas_orig = d.dataset
    
    l = []
    for f in compas_orig.feature_names:
        x = f 
        x = x.replace('[','')
        x = x.replace(']','')
        x = x.replace('<','')
        l.append(x)
    d.dataset.feature_names= l
        
     
    
#     try:
    non_preproc = execute(d,compas_orig,None,'Compas','sem pré-processamento')

    non_preproc.to_csv('compas_sem.csv')

    df = df.append(non_preproc)

#     except:
#         None
    
#     try:
    preproc = execute(d,compas_orig,'DispImpact','Compas','DispImpact')

    preproc.to_csv('compas_di.csv')

    df = df.append(preproc)
#     except:
#         None
        
#     try:
        
    preproc = execute(d,compas_orig,'RW','Compas','RW')

    preproc.to_csv('compas_rw.csv')

    df = df.append(preproc)
#     except:
#         None
        
#     try:

    preproc = execute(d,compas_orig,'Massaging','Compas','Massaging')

    df = df.append(preproc)

    preproc.to_csv('compas_mss.csv')
#     except:
#         None


    return df


def titanic(df):
    d  = datasets_loader()
    d.load_titanic_dataset()
    titanic_orig = d.dataset


    non_preproc = execute(d,titanic_orig,None,'Titanic','sem pré-processamento')

    df = df.append(non_preproc)

    
    preproc = execute(d,titanic_orig,'DispImpact','Titanic','DispImpact')

    df = df.append(preproc)
    
    preproc = execute(d,titanic_orig,'RW','Titanic','RW')

    df = df.append(preproc)
    
    preproc = execute(d,titanic_orig,'Massaging','Titanic','Massaging')

    df = df.append(preproc)

    return df




def main():

    df = pd.DataFrame()

    df1 = adult(df)
    df1.to_csv('resultados-por-metodo/parcial/adult_di.csv')


    
if __name__ == '__main__':
    main()