# Experimentos: pré-processamento de _Fairness in Machine Learning_


## Detalhes

- **img**: Pasta que contém os gráficos gerados no experimento
- **results**: Pasta que contém as tabelas com os resultados para cada base de dados
- **preprocessing-methods**: Pasta que contém os algoritmos de pré-processamento que foram desenvolvidos
  * Massaging
  * Preferential Sampling
  * Uniform Sampling
- **data**: Pasta contendo arquivos de algumas das bases utilizadas
- **dataset.py**: Classe que representa os _datasets_ utilizados e suas funções
- **fairness-experiment**: O experimentoe em si, classes e funções necessárias
- **settings**: Arquivo com as configurações dos classificadores utilizados no experimento
- **run_experiments**: Notebook com um exemplo de como rodar os algoritmos
- **results_views**: Script que gera gráficos para visualização dos resultados

## Como executar os experimentos

1. Baixar todas as bibliotecas necessárias
```
pip install -r requirements.txt
```

2. Rodar o notebook run_experiments e definir quais bases e quais métodos de pré-processamento serão utilizados
```
jupyter noteboook run_experiments.ipynb
```


  
