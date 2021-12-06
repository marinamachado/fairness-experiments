import aif360
import pandas as pd
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset, StandardDataset

class datasets_loader:

    label = None
    attr = None
    privileged_groups = None
    unprivileged_groups = None
    dataset = None
    
    
    def set_variables(self):
        
        self.label = self.dataset.label_names[0]
 
        self.attr = self.dataset.protected_attribute_names[0]
        idx = self.dataset.protected_attribute_names.index(self.attr)
        self.privileged_groups =  [{self.attr:self.dataset.privileged_protected_attributes[idx][0]}] 
        self.unprivileged_groups = [{self.attr:self.dataset.unprivileged_protected_attributes[idx][0]}] 

    #Genero
    def load_german_dataset(self):

        label_map = {1.0: 'Good Credit', 2.0: 'Bad Credit'}

        
        self.dataset = GermanDataset(label_name='credit',
        protected_attribute_names=['sex'],         
        privileged_classes=[['male']],     
        features_to_drop=['personal_status'],
        categorical_features=['status', 'credit_history', 'purpose',
                     'savings', 'employment', 'other_debtors', 'property',
                     'installment_plans', 'housing', 'skill_level', 'telephone',
                     'foreign_worker'],metadata={'label_maps': label_map, 'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}]}
        )
        self.set_variables()


    # idade
    def load_bank_dataset(self):
    
        self.dataset= BankDataset(
        label_name='y', favorable_classes=['yes'],
                     protected_attribute_names=['age'],
                     privileged_classes=[lambda x: x >= 25],
                     instance_weights_name=None,
                     categorical_features=['job', 'marital', 'education', 'default',
                         'housing', 'loan', 'contact', 'month', 'day_of_week',
                         'poutcome'],
                     features_to_keep=[], features_to_drop=[],
                     na_values=["unknown"], custom_preprocessing=None,
                     metadata=None)

        self.set_variables()

    # Gênero e Raça
    def load_adult_dataset(self):

    
        default_mappings = {
            'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
            'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'}]
            }
        self.dataset = AdultDataset(label_name='income-per-year',
                    favorable_classes=['>50K', '>50K.'],
                    protected_attribute_names=['race'],
                    privileged_classes=[['White']],
                    instance_weights_name=None,
                    categorical_features=['sex','workclass', 'education',
                        'marital-status', 'occupation', 'relationship',
                        'native-country'],
                    features_to_keep=[], features_to_drop=['fnlwgt'],
                    na_values=['?'], custom_preprocessing=None,
                    metadata=default_mappings)
        self.set_variables()

    # Gênero e Raça    
    def load_compas_dataset(self):
    

        default_mappings = {
            'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}],
            'protected_attribute_maps': [{1.0: 'Caucasian', 0.0: 'Not Caucasian'}]
        }
    
        self.dataset = CompasDataset(label_name='two_year_recid', favorable_classes=[0],
                 protected_attribute_names=['race'],
                 privileged_classes=[['Caucasian']],
                 instance_weights_name=None,
                 categorical_features=['age_cat', 'c_charge_degree',
                     'c_charge_desc', 'sex'],
                 features_to_keep=['sex', 'age', 'age_cat', 'race',
                     'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                     'priors_count', 'c_charge_degree', 'c_charge_desc',
                     'two_year_recid'],
                 features_to_drop=[], na_values=[],
                 custom_preprocessing=None,
                 metadata=default_mappings)
        self.set_variables()

    def load_titanic_dataset(self):
        
        

        df = pd.read_csv('bases/titanic.csv', sep = ';')
        default_mappings = {
        'label_maps': [{1.0: 1, 0.0: 0}],
        'protected_attribute_maps': [{1.0: 'female', 0.0: 'male'}]
        }
        
        self.dataset = StandardDataset(df,label_name='Survived',
                 favorable_classes=[1],
                 protected_attribute_names=['Sex'],
                 privileged_classes=[['female']],
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[], custom_preprocessing=None,
                 metadata=default_mappings)

        self.set_variables()

def main():
    d = datasets_loader()
    datasets = {'german':d.load_german_dataset(),
            'bank':d.load_bank_dataset(),
            'adult':d.load_adult_dataset(),
            'compas':d.load_compas_dataset(),
            'titanic':d.load_titanic_dataset()}

    for name, loader in datasets.items():
    
        dataset_orig = loader
        print(name)


if __name__ == "__main__":
    main()

    


    

    




    

    





