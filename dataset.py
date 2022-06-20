import pandas as pd
import aif360
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset, StandardDataset


class Dataset():
    """ Classe base representando a base de dados
    """

    def __init__(self, df, protected_att_name, label_name, drop, categorical, label_map, protected_att_map, privileged_class):
        self.df = df
        self.protected_att_name = protected_att_name
        self.label_name = label_name
        self.label_map = label_map
        self.protected_att_map = protected_att_map
        self.privileged_class = privileged_class
        self.dataset_converted = None

        self.y = self.df[self.label_name].ravel()
        X = self.df.drop(columns = [self.label_name],axis =1)
        
        multindex = pd.MultiIndex.from_frame(X[[self.protected_att_name]])
        X = pd.DataFrame(X.to_numpy(),index = multindex,columns = X.columns)
            
        level_to_change = -1
        X.index = X.index.set_levels(X.index.levels[level_to_change].astype(str), level=level_to_change)
        self.X = X

        
    def convert_to_aif_0(self):
        ''' Converte a base para o formato da aif360
        '''
        
        dataset_converted = StandardDataset(self.df,label_name=self.label_name,
                protected_attribute_names=[self.protected_att_name],
                favorable_classes = [1],
                privileged_classes = [[1]],   
                metadata={'label_maps': self.label_map, 'protected_attribute_maps': [self.protected_att_map]})
        return dataset_converted

    def convert_to_aif(self, X, y, label):
        ''' Converte X,Y para o formato da aif360
        '''
        new_df = X.copy()
        new_df[label] = y
        dataset_converted = StandardDataset(new_df,label_name=label,
                protected_attribute_names=[self.protected_att_name],
                favorable_classes = [1],
                privileged_classes = [[1]],   
                metadata={'label_maps': self.label_map, 'protected_attribute_maps': [self.protected_att_map]})
        return dataset_converted

class German(Dataset):
    '''Classe derivada da classe Dataset e representa German Dataset'''
 
    def __init__(self):
        
        self.dataset = None
        self.label_name = 'credit'
        self.protected_att_name = 'sex'
        self.privileged_class = 'male'
        self.drop = ['personal_status']
        self.categorical = ['status', 'credit_history', 'purpose',
                         'savings', 'employment', 'other_debtors', 'property',
                         'installment_plans', 'housing', 'skill_level', 'telephone',
                         'foreign_worker']
        self.label_map = {1.0: 'Good Credit', 2.0: 'Bad Credit'}

        self.protected_att_map = {1.0: 'Male', 0.0: 'Female'} 
    
        
        self.dataset = GermanDataset(label_name=self.label_name,
        protected_attribute_names=[self.protected_att_name],         
        privileged_classes=[[self.privileged_class]],     
        features_to_drop=self.drop,
        categorical_features=self.categorical,metadata={'label_maps': self.label_map, 'protected_attribute_maps': [self.protected_att_map]})
        df = self.dataset.convert_to_dataframe()[0].replace(2.0,0.0)

        super().__init__(df,self.protected_att_name,self.label_name,self.drop,self.categorical,self.label_map,self.protected_att_map,self.privileged_class)
        
class Adult(Dataset):
    '''Classe derivada da classe Dataset e representa Adult Dataset'''

    def __init__(self):
        
        self.dataset = None
        self.label_name = 'income-per-year'
        self.protected_att_name = 'race'
        self.privileged_class = 'White'
        self.drop = ['fnlwgt']
        self.categorical = ['sex','workclass', 'education',
                        'marital-status', 'occupation', 'relationship',
                        'native-country']
        self.label_map = {1.0: '>50K', 0.0: '<=50K'}

        self.protected_att_map = {1.0: 'White', 0.0: 'Non-white'}
        self.na_values=['?']
        self.favorable_classes=['>50K', '>50K.']
    
        
        self.dataset = AdultDataset(label_name=self.label_name,
        favorable_classes = self.favorable_classes,
        na_values = self.na_values,
        protected_attribute_names=[self.protected_att_name],         
        privileged_classes=[[self.privileged_class]],     
        features_to_drop=self.drop,
        categorical_features=self.categorical,metadata={'label_maps': self.label_map, 'protected_attribute_maps': [self.protected_att_map]})
        
        super().__init__(self.dataset.convert_to_dataframe()[0],self.protected_att_name,self.label_name,self.drop,self.categorical,self.label_map,self.protected_att_map,self.privileged_class)
        
class Bank(Dataset):
    '''Classe derivada da classe Dataset e representa Bank Dataset'''

    
    def __init__(self):
        
        self.dataset = None
        self.label_name = 'y'
        self.favorable_classes=['yes']
        self.protected_att_name = 'age'
        self.privileged_class = [lambda x: x >= 25]
        self.categorical = ['job', 'marital', 'education', 'default',
                         'housing', 'loan', 'contact', 'month', 'day_of_week',
                         'poutcome']
        
        self.na_values=['unknown']
        
        self.drop = None
        self.label_map = None
        self.protected_att_map = None
        
        self.dataset= BankDataset(
            label_name=self.label_name, favorable_classes=self.favorable_classes,
                         protected_attribute_names=[self.protected_att_name],
                         privileged_classes=self.privileged_class,
                         instance_weights_name=None,
                         categorical_features=self.categorical,
                         na_values=self.na_values)
        
        
        super().__init__(self.dataset.convert_to_dataframe()[0],self.protected_att_name,self.label_name,self.drop,self.categorical,self.label_map,self.protected_att_map,self.privileged_class)
        
class Compas(Dataset):
    '''Classe derivada da classe Dataset e representa Compas Dataset'''

    
    def __init__(self):
        
        self.dataset = None
        self.label_name = 'two_year_recid'
        self.favorable_classes=[0]
        self.protected_att_name = 'race'
        self.privileged_class = 'Caucasian'
        self.drop = None
        self.keep = ['sex', 'age', 'age_cat', 'race',
                     'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                     'priors_count', 'c_charge_degree', 'c_charge_desc',
                     'two_year_recid']
        self.categorical = ['age_cat', 'c_charge_degree',
                     'c_charge_desc', 'sex']
        self.label_map = {1.0: 'Did recid.', 0.0: 'No recid.'}

        self.protected_att_map = {1.0: 'Caucasian', 0.0: 'Not Caucasian'}
    
        
        self.dataset = CompasDataset(label_name=self.label_name,
        favorable_classes = self.favorable_classes,
        protected_attribute_names=[self.protected_att_name],         
        privileged_classes=[[self.privileged_class]],     
        features_to_keep=self.keep,
        categorical_features=self.categorical,metadata={'label_maps': self.label_map, 'protected_attribute_maps': [self.protected_att_map]})
        
        super().__init__(self.dataset.convert_to_dataframe()[0],self.protected_att_name,self.label_name,self.drop,self.categorical,self.label_map,self.protected_att_map,self.privileged_class)
        
class Titanic(Dataset):
    '''Classe derivada da classe Dataset e representa Titanic Dataset'''

    def __init__(self):
        
        self.dataset = None
        self.label_name = 'Survived'
        self.favorable_classes=[1.0]
        self.protected_att_name = 'Sex'
        self.privileged_class = 'female'
        self.drop = None
        self.keep = None
        self.categorical = None
        self.label_map = {1.0: 1, 0.0: 0}
        self.protected_att_map = {1.0: 'female', 0.0: 'male'}
        
        self.df = pd.read_csv('bases/titanic.csv', sep = ';')
        self.df = self.df.replace('male',0.0)
        self.df = self.df.replace('female',1.0)
        super().__init__(self.df,self.protected_att_name,self.label_name,self.drop,self.categorical,self.label_map,self.protected_att_map,self.privileged_class)

        
class Arrhythmia(Dataset):
    '''Classe derivada da classe Dataset e representa Arrhythmia Dataset'''

    
    def __init__(self):
        
        self.dataset = None
        self.label_name = 'Class'
        self.favorable_classes=[1.0]
        self.protected_att_name = 'Sex'
        
        self.privileged_class = 1.0
        self.drop = [13]
        self.keep = None
        self.categorical = None
        self.label_map = {1.0: 1, 0.0: 0}
        
        self.protected_att_map = {1.0:1, 0.0: 0}
        self.df = pd.read_csv('bases/arrhythmia.data',header = None )
        self.df = self.df.rename(columns = {1:'Sex',279:'Class'})
        self.df['Class'] = [1.0 if i > 1 else 0.0 for i in self.df['Class']]
        self.df['Sex'] = self.df['Sex'].astype(float) 
        
        self.df = self.df.replace('?',pd.NA)
        self.df = self.df.dropna()
        
       
        super().__init__(self.df,self.protected_att_name,self.label_name,self.drop,self.categorical,self.label_map,self.protected_att_map,self.privileged_class)
        
        
class Contraceptive(Dataset):
    '''Classe derivada da classe Dataset e representa Contraceptive Dataset'''

    def __init__(self):
        
        self.dataset = None
        self.label_name = 'Class'
        self.favorable_classes=[1.0]
        self.protected_att_name = 'Religion'
        
        self.privileged_class = 1.0
        self.drop = None
        self.keep = None
        self.categorical = None
        self.label_map = {1.0: 1, 0.0: 0}
        
        self.protected_att_map = {1.0:1.0, 0.0: 0.0}
        self.df = pd.read_csv('bases/cmc.data',header = None)
        self.df = self.df.rename(columns = {4:'Religion',9:'Class'})
        self.df['Class'] = [1 if i > 2 else 0 for i in self.df['Class']]
        self.df['Religion'] = self.df['Religion'].astype(float) 

        super().__init__(self.df,self.protected_att_name,self.label_name,self.drop,self.categorical,self.label_map,self.protected_att_map,self.privileged_class)

        
class Alcohol(Dataset):
    '''Classe derivada da classe Dataset e representa Alcohol Dataset'''
    
    def __init__(self):
        
        self.dataset = None
        self.label_name = 'Class'
        self.favorable_classes=[1.0]
        self.protected_att_name = 'Ethnicity'
        
        self.privileged_class = 1.0
        self.drop = None
        self.keep = None
        self.categorical = None
        self.label_map = {1.0: 1, 0.0: 0}
        
        self.protected_att_map = {1.0:1.0, 0.0: 0.0}
        self.df = pd.read_csv('bases/drug_consumption.data',header = None)
        self.df = self.df.iloc[:,0:14]      
        self.df = self.df.rename(columns = {5:'Ethnicity',13:'Class'})
        self.df['Class'] = [0 if i == 'CL0' else 1 for i in self.df['Class']]
        self.df['Ethnicity'] = [1.0 if i == -0.31685 else 0.0 for i in self.df['Ethnicity']]
        
        super().__init__(self.df,self.protected_att_name,self.label_name,self.drop,self.categorical,self.label_map,self.protected_att_map,self.privileged_class)

            
class Canabis(Dataset):
    '''Classe derivada da classe Dataset e representa Canabis Dataset'''

    
    def __init__(self):
        
        self.dataset = None
        self.label_name = 'Class'
        self.favorable_classes=[1.0]
        self.protected_att_name = 'Ethnicity'
        
        self.privileged_class = 1.0
        self.drop = None
        self.keep = None
        self.categorical = None
        self.label_map = {1.0: 1, 0.0: 0}
        
        self.protected_att_map = {1.0:1.0, 0.0: 0.0}
        df = pd.read_csv('bases/drug_consumption.data',header = None)
        self.df = df.iloc[:,0:13]
        self.df['Class'] = df[18]      
        self.df = self.df.rename(columns = {5:'Ethnicity'})
        self.df['Class'] = [0 if i == 'CL0' else 1 for i in self.df['Class']]
        self.df['Ethnicity'] = [1.0 if i == -0.31685 else 0.0 for i in self.df['Ethnicity']]
        super().__init__(self.df,self.protected_att_name,self.label_name,self.drop,self.categorical,self.label_map,self.protected_att_map,self.privileged_class)
            
class Heroin(Dataset):
    '''Classe derivada da classe Dataset e representa Heroin Dataset'''
    
    def __init__(self):
        
        self.dataset = None
        self.label_name = 'Class'
        self.favorable_classes=[1.0]
        self.protected_att_name = 'Ethnicity'
        
        self.privileged_class = 1.0
        self.drop = None
        self.keep = None
        self.categorical = None
        self.label_map = {1.0: 1, 0.0: 0}
        
        self.protected_att_map = {1.0:1.0, 0.0: 0.0}
        df = pd.read_csv('bases/drug_consumption.data',header = None)
        self.df = df.iloc[:,0:13]
        self.df['Class'] = df[23]      
        self.df = self.df.rename(columns = {5:'Ethnicity'})
        self.df['Class'] = [0 if i == 'CL0' else 1 for i in self.df['Class']]
        self.df['Ethnicity'] = [1.0 if i == -0.31685 else 0.0 for i in self.df['Ethnicity']]
        super().__init__(self.df,self.protected_att_name,self.label_name,self.drop,self.categorical,self.label_map,self.protected_att_map,self.privileged_class)
        