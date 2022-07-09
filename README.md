# Experimentos: pré-processamento de _Fairness in Machine Learning_

## Resumo do projeto
Fairness in Machine Learning é uma área que tem como objetivo utilizar técnicas focadas em minimizar efeitos discriminatórios nas decisões executadas por Aprendizado de Máquina, preservando ao máximo a precisão da decisão. Nessa área há três possibilidades de intervenção com o propósito de produzir classificadores mais justos, que se distinguem entre si pela etapa que intervêm. São elas: pré-processamento, em-processamento e pós-processamento. As abordagens de pré-processamento possuem algumas vantagens, como: não ter a necessidade de modificar o classificador, nem de acessar as informações sensíveis (e.g. raça, sexo, religião, etc.) nas etapas de treinamento ou de teste e, ainda, por serem abordagens que podem ser utilizadas por outras tarefas de Aprendizado de Máquina. Por ser uma área recente, demanda trabalhos que analisem comparativamente os diversos algoritmos que estão sendo propostos na literatura. Motivado por esta lacuna e pelas vantagens supracitadas, este projeto de pesquisa pretende realizar uma análise comparativa dos métodos de pré-processamento propostos em Fairness in Machine Learning, com o objetivo de identificar, de entender e de relacionar os algoritmos mais adequados e eficientes para cada diferente conceito de justiça, medida de desempenho e tipo de dados.

## Algoritmo de pré-processamento _Fairness in Machine Learning_
 - Massaging
 - Disparate Impact Remover
 - Reweighing
 - Uniform Sampling
 - Preferential Sampling

## Bases de dados
 - German Credit
 - Adult
 - Bank Marketing
 - COMPAS
 - Titanic
 - Arrhythmia
 - Contraceptive
 - Drug Consumption 

## Divisão das Pastas

- **img**: Pasta que contém os gráficos gerados no experimento
- **results**: Pasta que contém as tabelas com os resultados para cada base de dados
- **preprocessing-methods**: Pasta que contém os algoritmos de pré-processamento que foram desenvolvidos
  * Massaging
  * Preferential Sampling
  * Uniform Sampling
- **data**: Pasta contendo arquivos de algumas das bases utilizadas
- **dataset.py**: Classe que representa os _datasets_ utilizados e suas funções
- **fairness-experiment.py**: O experimento em si, classes e funções necessárias
- **settings.py**: Arquivo com as configurações dos classificadores utilizados no experimento
- **run_experiments.py**: Notebook com um exemplo de como rodar os algoritmos
- **graphics.py**: Script que gera gráficos para visualização dos resultados

## Como executar os experimentos

1. Baixar todas as bibliotecas necessárias
```
pip install -r requirements.txt
```

2. Rodar o notebook run_experiments e definir quais bases e quais métodos de pré-processamento serão utilizados
```
jupyter noteboook run_experiments.ipynb
```
3. Rodar o notebook graphics caso queria uma visualização gráfica de cada método por base de dados

```
jupyter noteboook graphics.ipynb
```

  
