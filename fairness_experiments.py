import os
import pandas as pd
import numpy as np
import time

# seleção de modelos
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


# métodos de pré-processamento
from preprocessing_methods.massaging import Massaging
from preprocessing_methods.uniformSampling import UniformSampling
from preprocessing_methods.preferentialSampling import PreferentialSampling
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover,LFR


# medidas de desempenho
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
from aif360.sklearn.metrics import generalized_fpr, selection_rate

# medidas de fairness
from aif360.sklearn.metrics import difference, statistical_parity_difference, disparate_impact_ratio, average_odds_difference, equal_opportunity_difference

# tela
from IPython.display import clear_output

measures_columns = ['accuracy', 'dif_accuracy', 'balanced_accuracy', 'dif_balanced_accuracy', 'recall', 'dif_recall', 
                    'precision', 'dif_precision', 'fpr', 'dif_fpr', 'selection_rate', 'dif_selection_rate', 
                    'dif_statistical_parity', 'dif_equal_opp', 'dif_avg_odds', 'disparate_impacto_ratio']


def fpr_score(y, y_pred):
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return fp/(fp+tn)

performance_measures = {
    'accuracy' : accuracy_score,
    'balanced_accuracy' : balanced_accuracy_score,
    'recall' : recall_score,
    'precision' : precision_score,
    'fpr' : fpr_score,
    'selection_rate' : selection_rate 
}

fairness_measures = {
    'dif_accuracy' : [difference, accuracy_score],
    'dif_balanced_accuracy' : [difference, balanced_accuracy_score],
    'dif_recall' : [difference, recall_score],
    'dif_precision' : [difference, precision_score],
    'dif_fpr' : [difference, fpr_score],
    'dif_selection_rate' : [difference, selection_rate],
    'dif_statistical_parity' : statistical_parity_difference, 
    'dif_equal_opp' : equal_opportunity_difference, 
    'dif_avg_odds' : average_odds_difference, 
    'disparate_impacto_ratio' : disparate_impact_ratio
}

''' Métodos de Estratificação '''
class _StratifiedBy():
    """
    Classe com os métodos possíveis de estratificação
    """

    @staticmethod
    def target(x, y):
        """
        Estratificação pelo valor de y (classe)
        """
        return y['target'].to_numpy()

    @staticmethod
    def group(x, y):
        """
        Estratificação pelo grupo protegido
        """
        return y.index.to_numpy()

    @staticmethod
    def group_target(x, y):
        """
        Estratificação por grupo protegido e classe
        """
        groups = _StratifiedBy.group(x, y)
        targets = _StratifiedBy.target(x, y)
        group_target = [str(group) + str(target) for group, target in zip(groups, targets)]
        return LabelEncoder().fit_transform(group_target)

def concat_results(*, relative_dir='', file_format='.csv', sep=';'):
    ''' Função que concatena aqruivos do tipo .csv em um mesmo frame.
    '''
    
    # concatena o diretório atual com o relativo passado no parâmetro relative_dir
    dirname = os.path.join(os.getcwd(), relative_dir)
    
    # verifica se é um diretório válido
    if not os.path.exists(dirname):
        raise ValueError('Erro: O seguinte diretório não existe: %s' % (dirname))
    
    results = None
    # percorre a lista dos arquivos e sub-diretórios 
    for item in os.listdir(dirname):
        
        # se não for um arquivo passa para o próximo item da lista
        if not os.path.isfile(os.path.join(dirname, item)):
            continue
        
        # copia o dir do arquivo junto ao nome do arquivo
        file = os.path.join(dirname, item)                   
        # recupera a extensão do arquivo
        ext = os.path.splitext(file)[1]

        # se o arquivo for no formato desejado
        if ext == file_format:
            # concatena os arquivos
            results = pd.concat([results, pd.read_csv(file, sep=sep)])
    
    return results

def convert_index(l, privileged_group):
    ''' Função que converte o valor do index para numérico, necessário para utilizar o AIF360
    '''
    
    if l == privileged_group:
        return 1
    else:
        return 0

''' Vetoriza a função convert_index
'''
convert_index = np.vectorize(convert_index)

def apply_preprocess(preprocess_name, X, y, dataset):
    
    ''' Função que aplica os diferentes tipos de métodos de pré-processamento no dataset
    '''
    
    y_train = pd.DataFrame()
    x_train = pd.DataFrame()
    
    # Disparte Impact Remover - AIF360
    if (preprocess_name == "DIR"):
        
        df_aif = dataset.convert_to_aif(X,y,'target')
        DIR = DisparateImpactRemover()
        df_aif = DIR.fit_transform(df_aif)

        y_train['target'] = df_aif.convert_to_dataframe()[0]['target'].copy()
        x_train = df_aif.convert_to_dataframe()[0]
        x_train = x_train.drop(columns = ['target'],axis =1)

        return x_train,y_train, None
    
    # Reweighing - AIF360
    elif (preprocess_name == "RW"):
        df_aif = dataset.convert_to_aif(X,y,'target')


        privileged_groups =  [{dataset.protected_att_name:1.0}] 
        unprivileged_groups = [{dataset.protected_att_name:0.0}] 
        RW = Reweighing(unprivileged_groups=privileged_groups,
               privileged_groups=unprivileged_groups)
        RW.fit(df_aif)
        df_aif = RW.transform(df_aif) 
        weights = df_aif.instance_weights

        y_train['target'] = df_aif.convert_to_dataframe()[0]['target'].copy()
        x_train = df_aif.convert_to_dataframe()[0]
        x_train = x_train.drop(columns = ['target'],axis =1)


        return x_train,y_train, weights

    # Massaging
    elif (preprocess_name == "MSS"):

         
        MSS = Massaging('target',dataset.protected_att_name)
        MSS = MSS.fit(X,y['target'].ravel())
        
        dataset_mss = pd.DataFrame()
        dataset_mss = MSS.transform(X,y['target'].ravel())
        
        y_train['target'] = dataset_mss['target']
        x_train = dataset_mss.drop(columns = ['target']) 

        return x_train,y_train, None

                
    #Preferential Sampling
    elif (preprocess_name == "PS"):

        PS = PreferentialSampling(dataset.protected_att_name,'target')

        X['target'] = y
        PS = PS.fit(X)
        dataset_ps = pd.DataFrame()
        dataset_ps = PS.transform(X)
        
        y_train['target'] = dataset_ps['target']
        x_train = dataset_ps.drop(columns = ['target']) 

        return x_train,y_train, None

    #Uniform Sampling
    elif (preprocess_name == "US"):
        US = UniformSampling(dataset.protected_att_name,'target')

        X['target'] = y
        US = US.fit(X)
        dataset_us = pd.DataFrame()
        dataset_us = US.transform(X)
        
        y_train['target'] = dataset_us['target']
        x_train = dataset_us.drop(columns = ['target']) 

        return x_train,y_train, None

       

def apply_lfr(X, y, X_test,y_test, dataset):   
    ''' Função que aplica o método de pré-processamento LFR
    '''
     
    df_aif = dataset.convert_to_aif(X,y,'target')
    df_test = dataset.convert_to_aif(X_test,y_test,'target')
        
    privileged_groups =  [{dataset.protected_att_name:1.0}] 
    unprivileged_groups = [{dataset.protected_att_name:0.0}]
    
    settings = lfr_settings[dataset.__class__.__name__]
    
    lfr = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups,
        k = settings['k'], Ax=settings['Ax'], Ay=settings['Ay'], Az=settings['Az'], seed=42)
    lfr.fit(df_aif, maxiter=settings['maxiter'], maxfun=settings['maxfun'])
  
    df_aif = lfr.transform(df_aif) 
    df_test = lfr.transform(df_test)
        
    y_train = pd.DataFrame()
    x_train = pd.DataFrame()
    y_test = pd.DataFrame()
    x_test = pd.DataFrame()

    y_train['target'] = df_aif.convert_to_dataframe()[0]['target'].copy()
    x_train = df_aif.convert_to_dataframe()[0]
    x_train = x_train.drop(columns = ['target'],axis =1)

    y_test['target'] = df_test.convert_to_dataframe()[0]['target'].copy()
    x_test = df_test.convert_to_dataframe()[0]
    x_test = x_test.drop(columns = ['target'],axis =1)


    return x_train,y_train,x_test,y_test, None
    
    
def multindex(X,dataset):
    ''' Função que tranforma o dateframe em multindex, onde o index é o atributo protegido
    '''
    
    multindex = pd.MultiIndex.from_frame(X[[dataset.protected_att_name]])
    X = pd.DataFrame(X.to_numpy(),index = multindex,columns = X.columns)

    level_to_change = -1
    X.index = X.index.set_levels(X.index.levels[level_to_change].astype(str), level=level_to_change)
    
    X = X.drop(columns =[dataset.protected_att_name])
    
    return X




def kfold(clf, X, y, weights, preprocess_name, dataset, k=2, stratified_by='group_target'):
    ''' Função que realiza o kfold estratificado e retorna a média de medidas de desempenho
    '''
    
    # instancia o KFold estratificado (sem utilizar todos os parametros)
    kf = StratifiedKFold(n_splits=k)
    
    # verifica qual o tipo de estratificação (as opções são: target, group e group_target)
    by = getattr(_StratifiedBy, stratified_by)(X, y)
    
    # inicia um DataFrame para salvar os resultados 
    # (colunas são as medidas que irão no relatório (neste caso só tem acurácia), 
    # as linhas é o número de iterações)
    results = pd.DataFrame(index=np.arange(k), columns=measures_columns)

    
    for i, (train_index, test_index) in enumerate(kf.split(X, by)):
        
        # Separa os conjuntos
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        

        x_train = x_train.reset_index(drop=True).copy()
        y_train = y_train.reset_index(drop=True).copy()
        x_test = x_test.reset_index(drop=True).copy()
        y_test = y_test.reset_index(drop=True).copy()
         
        
        # Aplica os métodos de pré-processamento
        if(preprocess_name == "LFR"):
            x_train,y_train,x_test,y_test,weights = apply_lfr(x_train, y_train,x_test,y_test, dataset)

        elif(preprocess_name != "sem pré-processamento"):

            x_train,y_train,weights = apply_preprocess(preprocess_name, x_train, y_train, dataset)
            
        # Retira o atributo protegido do treino e adiciona como index 
        x_train = multindex(x_train,dataset)
        x_test = multindex(x_test,dataset)
        
        # Treina o classificador
        if weights is None:
            clf.fit(x_train, y_train['target'])
        else:
            # mudei os weigths aqui
            clf.fit(x_train, y_train['target'], weights)
        
        # Realiza as predições
        y_predict = clf.predict(x_test)
        
        y_predict = pd.DataFrame(y_predict, columns=['target'])
        y_test = pd.DataFrame(y_test, columns=['target'])
        
        y_predict.index = convert_index(list(map('/'.join, list(x_test.index))), '1.0')
        y_predict.index.names = ['group']   
                    
        y_test.index = convert_index(list(map('/'.join, list(x_test.index))), '1.0')
        y_test.index.names = ['group']           
        
        
        # Calcula as medidas de desempenho
        for name, measure in performance_measures.items():
            results.loc[i, name] = measure(y_test['target'], y_predict)
            
        # Calcula as medidas de fairness
        for name, measure in fairness_measures.items():
            # verifica se é uma lista (medidas que precisam de duas funções)
            if isinstance(measure, list):

                results.loc[i, name] =  abs(measure[0](measure[1], y_test, y_predict, prot_attr='group', priv_group=1))

            else:
                results.loc[i, name] = abs(measure(y_test, y_predict, prot_attr='group', priv_group=1))     
                    
            
    # retorna o resultado como sendo a média de todas iterações
    return results.mean()

class Experiment:
    
    def __init__(self, classifiers_settings, *, k=10, dataset_name, preprocessing_name, privileged_group):
        
        self.classifiers_settings = classifiers_settings
        self.k = k
        self.dataset_name = dataset_name
        self.preprocessing_name = preprocessing_name
        self.report = pd.DataFrame(columns=['dataset', 'preprocessing', 'clf_type', 'params'] + measures_columns)
        self.counter = 0
        self.privileged_group = privileged_group
        self.n_classifiers = self.__get_number_classifiers()
        
    
    def display(self, clf_type, n_clf_type, counter_by_clf):
        """ Função que mostra o progresso no jupyter

        Args:
            clf_name (str): nome do classificador
        """
        clear_output()
        print('(%s - %s) - Classificador %s (%d/%d) - Progresso Geral (%d/%d)' % 
              (self.dataset_name, self.preprocessing_name, clf_type, counter_by_clf, n_clf_type, self.counter,
               self.n_classifiers))
                  
    def __get_number_classifiers(self):
        """
        Recupera o número de classificadores no experimento para auxiliar no progresso do experimento
        """
        n_classifiers = 0
        for _, (_, param_grid) in self.classifiers_settings.items():
            grid = ParameterGrid(param_grid)
            for _ in grid:
                n_classifiers += 1
        return n_classifiers
    
    def export_report(self, relative_path='', complement_name=''):
        """
          Salva o report em um csv
        """
        filename = 'rep_' + self.dataset_name + '_' + self.preprocessing_name + '_' + complement_name + '.csv'
        self.report.to_csv(relative_path + filename, sep=';', index=False)
        
    def execute(self, X, y, dataset, weights=None):
        
        """
          Executa os experimentos
        """
        
        # transforma os indices do y para ficar compatível com o AIF360
        y = pd.DataFrame(y, columns=['target'])
        
        if not weights is None:
            weights = np.array(weights)
            
        
        # verifica se o grupo privilegiado está contido nos indices (se não estiver gera exceção)
        if isinstance(X.index, pd.MultiIndex):
            if not self.privileged_group in list(map('/'.join, list(X.index))):
                raise ValueError('Erro: privileged_group (%s) não está contido em X' % (self.privileged_group))
            else:
                y.index = convert_index(list(map('/'.join, list(X.index))), self.privileged_group)
                y.index.names = ['group']
                
        else:
            if not self.privileged_group in list(X.index):
                raise ValueError('Erro: privileged_group (%s) não está contido em X' % (self.privileged_group))
            else:
                y.index = convert_index(list(X.index), self.privileged_group)
                y.index.names = ['group']   
        
        
        count = 0

        for clf_type, (Classifier, param_grid) in self.classifiers_settings.items():
                        
            # função que executa todas as configurações de um determinado algoritmo de classificação
            grid = ParameterGrid(param_grid)
            # contador para ver o progresso no algoritmo de classificação em questão
            counter_by_clf = 0
            # imprime o status
            self.display(clf_type, len(grid), counter_by_clf)
            
            for params in grid: 
                
                # Adiciona o parametro random state em cada classificador que tenha esse parâmetro
                try:
                    params['random_state']=42                
                
                    # instancia o classificador com os novos parâmetros
                    clf = Classifier(**params)
                except:
                    params.pop('random_state')
                    clf = Classifier(**params)


                # realiza o kfold
                result = kfold(clf, X, y, weights, self.preprocessing_name, dataset)
                
                # salva o resultado no relatório
                self.report.loc[self.counter] = [self.dataset_name, self.preprocessing_name, clf_type, str(params)] + list(result)
                
                # incrementa os contadores e atualiza o status
                self.counter += 1
                counter_by_clf += 1
                self.display(clf_type, len(grid), counter_by_clf)

