# ************************************* Machine Learning Library For Data Science (Versão 01) *************************************
# Neste arquivo estão definidas algumas funções de "Machine Learning" para serem utilizadas em Data Science.
#

# ************************************* Importando Pacotes e/ou Funções *************************************
#
# Importa o pacote "NumPy":
import numpy as np
# Importa o pacote "Pandas":
import pandas as pd

# Importa o pacote "os" (Operation System with its Packages and Functions)
import os

# O pacote "sys" permite manipulações com o sistema operacional:
import sys

# Imports para "Avaliação do Modelo":
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score

# Funções para balanceamento dos dados:
import imblearn
from imblearn.over_sampling import SMOTE
import collections

# Path: onde estão armazenadas as classes e funções que serão utilizadas neste módulo:
LIB_PATH = os.path.join(".")

# Adicionando o diretório ao 'path' do Sistema, para podermos importar classes e funções que serão
# utilizadas neste módulo:
sys.path.append(LIB_PATH)

# ************************************* Definindo Funções *************************************
#
# ***** Função para mostrar os resultados das pontuações (Scores) da validação cruzada (Cross Validation).
#
def display_scores(scores, decimals=4):
    '''
    Input:
        "scores": pontuações (scores) calculados de um processo de validação cruzada.
        "decimals": número de dígitos decimais para apresentação dos resultados.

    Output: None
    '''
    # Código da função:

    print("Scores:", len(scores))
    print(np.round(scores, decimals=decimals))
    print("Mean:", np.round(scores.mean(), decimals=decimals)) 
    print("Standard deviation:", np.round(scores.std(), decimals=decimals))

# ***** Fução para mostrar os melhores resultados encontrados para os hiperparâmetros do modelo:
#
def best_results_report (estimator, title):
    '''
    Input:
        "estimator": modelo de machine learning que já foi treinado (fit).
        "title": título do relatório.

    Output: None
    '''
    # Código da função:

    # Mostra o título do relatório:
    print(title)
    
    # Mostra a melhor seleção de hiperparâmetros:
    print('Best params:', estimator.best_params_)

    # Mostra o melhor estimador:
    print('Best estimator:', estimator.best_estimator_)

    # Mostra o melhor score:
    print('Best score:', np.round(estimator.best_score_, decimals=4))

# ***** Função para calcularmos as métricas de classificação de um modelo de "Machine Learning":
#
def binary_classif_metrics (y_actual, y_pred, y_scores, msg=None, verbose=False):
    '''
    Input:
        "y_actual": dados "reais" da nossa variável target.
        "y_pred": dados "previstos" da nossa variável target.
        "y_scores": scores para cálculo da métrica AUC.
        'msg': Imprime uma mensagem na tela antes de mostrar as métricas;
        "verbose": se "True" algumas informações serão mostradas na tela;

    Output: retorna um objeto (dict) com os valores calculados de cada métrica.
    '''
    # Código da função:

    # Criando um objeto (dicionário) vazio para armazenar as métricas:
    metrics = dict()

    # Calculando o AUC (Area Under Curve):
    metrics['auc'] = np.round(roc_auc_score(y_actual, y_scores), decimals=4)

    # Calculando a acurácia:
    metrics['accuracy'] = np.round(accuracy_score(y_actual, y_pred), decimals=4)
    
    # Calculando a precisão:
    metrics['precision'] = np.round(precision_score(y_actual, y_pred), decimals=4)
    
    # Calculando a revocação:
    metrics['recall'] = np.round(recall_score(y_actual, y_pred), decimals=4)

    # Calculando a pontuação F1:
    metrics['f1_score'] = np.round(f1_score(y_actual, y_pred), decimals=4)

    # Verifica se as métricas serão mostradas na tela:
    if(verbose):
        # Print dos resultados:
        if(msg != None):
            print(msg)
        print ("AUC      : {}".format(metrics['auc']))
        print ("Accuracy : {}".format(metrics['accuracy']))
        print ("Precision: {}".format(metrics['precision']))
        print ("Recall   : {}".format(metrics['recall']))
        print ("f1-score : {}".format(metrics['f1_score']))
    
    # Retorna as métricas calculadas:
    return metrics

# Função para treinar e avalidar um modelo preditivo de classificação binária:
#
def train_validate_binary_clf_model(classifier, X_train, y_train, X_valid, y_valid, probability=True, threshold=0.5, verbose=False):
    '''
    Input:
        "classifier": algoritmo de machine learning (classificação binária) que será treinado;
        "X_train": variáveis preditoras para treinamento do modelo de classificação binária;
        "y_train": variável target para treinamento do modelo de classificação binária;
        "X_valid": variáveis preditoras para avaliação do modelo de classificação binária;
        "y_valid": variável target para avaliação do modelo de classificação binária;
        "probability": se "True" utiliza o método "predict_proba" (com o parâmetro threshold) para fazer as previsões,
                       se "False" utiliza o método "decision_function" para calcular os "scores" e o método "predict"
                       (sem o parâmetro threshold) para fazer as previsões.
        "threshold": valor limite (default = 0.5) para rotular uma amostra prevista como positiva.
        "verbose": se "True" algumas informações serão mostradas na tela;

    Output:
        "model": retorna o modelo de classificação binária treinado;
        "train_metrics": retorna um objeto (dict) com os valores calculados de cada métrica nos dados de treino;
        "valid_metrics": retorna um objeto (dict) com os valores calculados de cada métrica nos dados de avaliação.
    '''
    # Código da função:
    
    # Criando uma instância do classificador:
    model = classifier

    # Treina (Fit) o modelo com os dados de treino:
    model.fit(X_train, y_train)

    # Utilizando o modelo treinado para fazer as previsões:
    # Verifica qual o método será utilizado para fazer as previsões:
    if(probability):
        # Dados de treino: obtendo as probabilidades da classe positiva (1):
        y_train_scores = model.predict_proba(X_train)[:,1]

        # Calcula as previsões de acordo com o threshold:
        y_train_preds = y_train_scores > threshold

        # Dados de avaliação: obtendo as probabilidades da classe positiva (1):
        y_valid_scores = model.predict_proba(X_valid)[:,1]

        # Calcula as previsões de acordo com o threshold:
        y_valid_preds = y_valid_scores > threshold
    else:
        # Dados de treino: obtendo os scores:
        y_train_scores = model.decision_function(X_train)

        # Dados de treino: obtendo as previsões:
        y_train_preds = model.predict(X_train)

        # Dados de avaliação: obtendo os scores:
        y_valid_scores = model.decision_function(X_valid)

        # Dados de avaliação: obtendo as previsões:
        y_valid_preds = model.predict(X_valid)        
    
    # Calculando as métricas nos dados de treino:
    train_metrics = binary_classif_metrics(
        y_actual=y_train, y_pred=y_train_preds, y_scores=y_train_scores, msg="Training Metrics:", verbose=verbose
        )
    
    # Calculando as métricas nos dados de avaliação:
    valid_metrics = binary_classif_metrics(
        y_actual=y_valid, y_pred=y_valid_preds, y_scores=y_valid_scores, msg="\nValidation Metrics:", verbose=verbose
        )
    
    # Retorna o modelo treinado (fit) e as métricas de treino e avaliação:
    return (model, train_metrics, valid_metrics)

# Função para preparar os dados de treino e teste que serão utilizados para criação dos modelos preditivos:
#
def prepare_train_test_data(
    train_set, test_set, target, num_features, cat_features=None, scaler=None, balance=False,
    random_state=None, verbose=False
):
    '''
    Input:
        "train_set": conjunto dos dados de treino que serão preparados;
        "test_set": conjunto dos dados de teste que serão preparados;
        "target": variável dependente;
        "cat_features": variáveis preditoras (atributos) categóricas;
        "num_features": variáveis preditoras (atributos) numéricas;
        "scaler": classe que será utilizada para padronizar as variáveis preditoras numéricas (num_features);
        "balance": se "True" os dados de treino serão balanceados de acordo com as classes da variável target (ver nota abaixo);
        "random_state": "seed""que poderá ser utilizada para termos a reprodutibilidade no balanceamento dos dados de treino.
        "verbose": se "True" algumas informações serão mostradas na tela;

    Output:
        "X_train": array com os dados de treino preparados (variáveis preditoras);
        "y_train": vetor com os dados de treino preparados (variável target);
        "X_test": array com os dados de teste preparados (variáveis preditoras);
        "y_test": vetor com os dados de teste preparados (variável target);
        "scaler": se scaler não for "None", retorna a instância da classe utilizada para a padronização das variáveis numéricas.
    
    Nota: utilizaremos a função "SMOTE()" (https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
    do pacote "imblearn.over_sampling" para fazermos o balancemaneto dos dados de treino. 
    Após o balanceamento a quantidade de registros para cada categoria será a mesma, ou seja, "50%" de registros para cada categoria.
    '''
    # Código da função:

    # Separando as variáveis categóricas e numéricas:
    if(cat_features != None):
        # Cria um dataframe dos dados de treino para as variáveis categóricas:
        train_cat_df = train_set[cat_features].copy(deep=True)
        # Cria um dataframe dos dados de teste para as variáveis categóricas:
        test_cat_df = test_set[cat_features].copy(deep=True)

    # Cria um dataframe dos dados de treino para as variáveis numéricas:
    train_num_df = train_set[num_features].copy(deep=True)
    
    # Cria um dataframe dos dados de teste para as variáveis numéricas:
    test_num_df = test_set[num_features].copy(deep=True)

    # Verificando se os dados serão padronizados:
    if (scaler != None):
        # Padronizando as variáveis numéricas nos dados de treino:
        train_num_tf = scaler.fit_transform(X=train_num_df)
        # Atualizando o dataframe com o resultado:
        train_num_df = pd.DataFrame(train_num_tf, columns=list(scaler.feature_names_in_))

        # Padronizando as variáveis numéricas nos dados de teste:
        test_num_tf = scaler.transform(X=test_num_df)
        # Atualizando o dataframe com o resultado:
        test_num_df = pd.DataFrame(test_num_tf, columns=list(scaler.feature_names_in_))

    if(cat_features != None):
        # Criando a matriz "X_train":
        X_train = pd.concat([train_cat_df, train_num_df], axis=1).values
        # Criando a matriz "X_test":
        X_test = pd.concat([test_cat_df, test_num_df], axis=1).values
    else:
        # Temos apenas variáveis numéricas:
        # Criando a matriz "X_train":
        X_train = train_num_df.values
        # Criando a matriz "X_test":
        X_test = test_num_df.values

    # Criando o vetor "y_train":
    y_train = train_set[target].values
    # Criando o vetor "y_test":
    y_test = test_set[target].values
    # Verifica se vamos mostrar alguma informação na tela:
    if (verbose):
        # Mostra os shapes dos dados de treino e teste preparados:
        print("Train data set:")
        print("X_train: {}".format(X_train.shape))
        print("y_train: {}".format(y_train.shape))
        print() # pula uma linha.
        print("Test data set:")
        print("X_test: {}".format(X_test.shape))
        print("y_test: {}".format(y_test.shape))
    
    # Verifica se os dados de treino serão balanceados:
    if (balance):
        # Balanceando os dados de treino com a função "SMOTE".
        # SMOTE é um metodo de oversampling.
        # Ele cria exemplos sintéticos da classe minoritaria ao invés de criar cópias.
        # Aplicando a função SMOTE:
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
        # Verifica se vamos mostrar alguma informação na tela:
        if (verbose):
            # Mostra o "shape" dos dados de treino após o balanceamento:
            print() # pula uma linha.
            print("Balanced training data set:")
            print("X_train: {}".format(X_train.shape))
            print("y_train: {}".format(y_train.shape))

        # Utilizando a função "Counter" do pacote "collections" para contarmos as quantidades de classes da nossa variável
        # "target" após o balanceamento:
        # Verifica se vamos mostrar alguma informação na tela:
        if (verbose):
            # Mostra a contagem de classes na tela:
            print() # pula uma linha.
            print("Counting classes for target variable: {}". format(collections.Counter(y_train)))
    
    # Retorna os resultados:
    return X_train, y_train, X_test, y_test, scaler

