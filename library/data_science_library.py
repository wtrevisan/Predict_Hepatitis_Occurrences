# ************************************* Data Science Library *************************************
# Neste arquivo estão definidas algumas funções (uso geral) para serem utilizadas neste projeto.
#

# ************************************* Importando Pacotes ou Funções *************************************
#
# Importa o pacote "os" Operation System (Packages and Functions):
import os

# Importa o pacote  "numpy":
import numpy as np

# Importa o pacote  "pandas":
import pandas as pd

# Importa o pacote "pickle" para salvar/carregar objetos:
import pickle as pck

# Importa o pacote "time":
import time

# Importa o pacote "stats":
from scipy import stats

# Importa função para verificar se uma variável numérica
# segue uma distribuição normal:
from scipy.stats import shapiro

import statsmodels.api as sm

# Função VIF para verificar a multicolinearidade:
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ************************************* Definindo Classes *************************************
#
#  ***** Classe para calcular o tempo decorrido de um processo/atividade qualquer:
#
class ElapsedTime():
    
    # Este método vai inicializar cada objeto criado a partir desta classe
    # O nome deste método é __init__, e ele é chamado de "Construtor", porque é ele que inicializa os objetos desta classe.
    # (self) é uma referência a cada atributo de um objeto criado a partir desta classe
    def __init__(self, builder_msg=True, display=True):
        
        # Atributos de cada objeto criado a partir desta classe. 
        # O self indica que estes são atributos dos objetos.
        self.display = display # Define se queremos mostrar a mensagem do tempo decorrido.
        self.start_time = 0 # Começa a contagem do tempo.
        self.end_time = 0 # Termina a contagem do tempo.
        self.elapsed_time = 0 # Calcula o tempo decorrido.
        if (builder_msg):
            print("Builder called to create an object of class ElapsedTime!")
        
    # Métodos são funções, que recebem como parâmetro atributos do objeto criado.
    # Método para iniciar a contagem do tempo:
    def start (self, msg=None):
        if (msg != None):
            print(msg) # Print message!
        
        self.start_time = time.perf_counter()

    # Método para terminar a contagem do tempo e imprimir o tempo decorrido se for desejado:
    def end (self, msg=None):
        # Calcula o tempo decorrido:
        self.end_time = time.perf_counter()
        self.elapsed_time = np.round((self.end_time - self.start_time), decimals=2)

        if (msg != None):
            msg_str = msg
        else:
            msg_str = "Elapsed time:"
                
        if (self.display == True):
            if (self.elapsed_time == 1):
                print("%s 1 second." % (msg_str))
            elif (self.elapsed_time == 60):
                print("%s 1 minute." % (msg_str))
            elif (self.elapsed_time == 60*60):
                print("%s 1 hour." % (msg_str))
            elif (self.elapsed_time == 60*60*24):
                print("%s 1 day." % (msg_str))
            elif (self.elapsed_time < 60):
                print("%s %.2f seconds." % (msg_str, self.elapsed_time))
            elif (self.elapsed_time < 60*60):
                print("%s %.2f minutes." % (msg_str, self.elapsed_time/60))
            elif (self.elapsed_time < 60*60*24):
                print("%s %.2f hours." % (msg_str, self.elapsed_time/(60*60)))
            else:
                print("%s %.2f days." % (msg_str, self.elapsed_time/(60*60*24)))

    # Método para obter a contagem do tempo:
    def get (self):
        return self.elapsed_time

# ************************************* Definindo Funções *************************************
#
# ***** Função para carregar (load) um objeto python qualquer, armazenado em um arquivo "pickle":
#
def pickle_object_load (path=".", file="None", msg=None):
    '''
    Input:
        "path": diretório (path) do arquivo que será carregado.
        "file": nome do arquivo que será carregado.
        "msg": mensagem que será impressa na tela (default é None, ou seja,
               não será impressa uma mensagem).

    Output:
        "obj": retorna os dados do "objeto" armazenado no arquivo.
    '''
    # Código da função:

    # Prepara o nome completo do arquivo piclke que será carregado (load):
    object_file = os.path.join(path, file)
    
    try:
        # Abre o arquivo para o modo leitura (read):
        pickle_in = open(object_file,"rb")
        
        # Faz a leitura do arquivo e carrega os dados no objeto ("obj"):
        obj = pck.load(pickle_in)

        # Fecha o arquivo "pickle":
        pickle_in.close()

        # Verifica se existe uma mensagem para ser impressa na tela:
        if (msg != None):
            # Imprime na tela a mensagem:
            print(msg)

        # Retorna os dados carregados:
        return obj
    
    except FileNotFoundError as error:
        # Erro encontrado na abertura do arquivo:
        print(error)
        # Retorna um valor nulo (None):
        return None

    except ValueError:
        # Erro encontrado na leitura do arquivo:
        print("I can not upload the '{}' file".format(file))
        
        # Fecha o arquivo "pickle":
        pickle_in.close()
        
        # Retorna um valor nulo (None):
        return None

# ***** Função para salvar (save) um objeto python qualquer em um arquivo "pickle":
#
def pickle_object_save (path=".", file="None", object_name=None, msg=None):
    '''
    Input:
        "path": diretório (path) onde o arquivo será criado.
        "file": nome do arquivo que será armazenado (salvo) o objeto python.
        "object_name": nome do objeto python que será armazenado (salvo) no arquivo.
        "msg": mensagem que será impressa na tela (default é None, ou seja,
               não será impressa uma mensagem).

    Output: None
    '''
    # Código da função:

    # Prepara o nome completo do arquivo piclke que será criado:
    object_file = os.path.join(path, file)
    
    try:
        # Abre o arquivo para o modo escrita (write):
        pickle_out = open(object_file,"wb")
        
        # Faz o 'dump' do objeto ("object_name") e salva os dados no arquivo ("file"):
        pck.dump(object_name, pickle_out)

        # Fecha o arquivo "pickle":
        pickle_out.close()

        # Verifica se existe uma mensagem para ser impressa na tela:
        if (msg != None):
            # Imprime na tela a mensagem:
            print(msg)

    except FileNotFoundError as error:
        # Erro encontrado na abertura do arquivo:
        print(error)

    except:
        # Erro encontrado ao fazer o "dump" do objeto:
        print("I can not save the '{}' object".format(object_name))
        
        # Fecha o arquivo "pickle":
        pickle_out.close()

# ***** Função para calcular os "missing values" em um DataFrame:
#
def missing_values(data, axis=0, threshold=0.0, target=None):
    '''
    Input:
        "data": Data Frame com as variáveis.
        "axis": indica se os valores serão contabilizados pelas colunas (axis=0) ou pelos índices (axis=1).
        "threshold": define o limite (valores entre 0 e 1) para a porcentagem de valores ausentes que serão retornados.
                     Por default, todas as colunas ou índices serão retornados.
        "target": informa se a variável target está presente no dataframe.

    Output:
        "df_mv": objeto do tipo "Dataframe" do Pandas com os resultados.
    '''
    # Código da função:

    # Faz uma cópia do dataframe original:
    df = data.copy(deep=True)

    # Verifica se a variável target está presente no dataframe:
    if(target != None):
        # Elimina a variável "target":
        df.drop(columns=target, inplace=True)

    # Calcula o total de missing values para cada atributo (feature) ou observação (index) do dataframe:
    total = df.isnull().sum(axis=axis).sort_values(ascending=False)
    
    # Calcula a porcentagem de missing values para cada atributo (feature) ou observação (index) do dataframe:
    percent = (df.isnull().sum(axis=axis)/df.isnull().count(axis=axis)).sort_values(ascending=False)
    
    # Define o nome para os ínices do dataframe com os resultados:
    if(axis == 0):
        names = 'Attributes (Columns)'
    else:
        names = 'Observations (Index)'

    # Cria o dataframe com os resultados:
    df_mv = pd.concat([total, percent], axis=1, names=names, keys=['Total', 'Percent'])
    
    # Retorna um dataframe com os resultados de acordo com o threshold:
    index = df_mv['Percent'] >= threshold
    return (df_mv[index])

# ***** Função para calcular a contagem ou proporção de cada categoria em cada variável,os valores atribuidos para uma variável qualquer em um DataFrame:
#
def groupby_count(data, variables, percent=False):

    '''
    Entrada:
        "data": dataframe com os dados;
        "variables": nomes das variáveis no dataframe "data";
        "percent": se True, calculará as contagens em porcentagens;

    Saída:
        temp_df: Objeto do tipo "Dataframe" do Pandas, com os resultados calculados.
    '''
    # Código da função:

    # Define um dicionário onde serão calculadas as contagens de cada variável:
    vars_dict = {}
    # Loop para realizar as contagens
    # Se "percent = True" as contagens serão normalizadas (%):
    for var in variables:
        vars_dict[var] = data[var].value_counts(normalize=percent)

    # Cria um dataframe para armazenar os resultados:
    df = pd.DataFrame(vars_dict)
    # Substitui os valores "NaN" por "0":
    df.fillna(value=0, inplace=True)
    # Arredonda os valores se "percent = True":
    if(percent == True):
        df = pd.DataFrame(data=np.round(df.values, decimals=4), columns=df.columns, index = df.index)
    else:
        df = pd.DataFrame(data=np.int64(df.values), columns=df.columns, index = df.index)
    
    # Retorna um dataframe com os resultados:
    return (df)

# ***** Função para calcular a proporção dos valores atribuidos para uma variável qualquer em um DataFrame:
#
def percent_count_feature(df, feature, lines_drop=False):

    '''
    Entrada:
        "df": Data Frame;
        "feature": atributo (variável) do Dataframe.
        "lines_drop":
            "True": eliminar as linhas cujo "Total" seja igual a zero ("0").
            "False": não eliminar as linhas cujo "Total" seja igual a zero ("0"). Este é o valor default.

    Saída:
        temp_df: Objeto do tipo "Dataframe" do Pandas, com os resultados calculados.
    '''
    # Código da função:

    # Cria um DataFrame temporário calculando a contagem de cada valor atribuido a variável desejada ('feature'):
    temp_df = pd.DataFrame(df[feature].groupby(df[feature]).count())
    temp_df.rename(columns={feature:'Total'}, inplace=True)
    temp_df.sort_values(by='Total', ascending=False, inplace=True)

    # Verifica se devemos eliminar as linhas cujo "Total" seja igual a zero ("0"), ou seja, "lines_drop=True":
    if(lines_drop):
        # Eliminando as linhas:
        lines = list(temp_df[temp_df['Total'] == 0].index) # Retorna uma lista com os índices das linhas.
        temp_df.drop(list(lines), inplace=True) # Elimina do DataFrame as linhas selecionadas.
    
    # Calcula a soma total (geral) dos valore para o atributo (feature) do DataFrame "df":
    TotalGeral = temp_df.Total.sum()
    
    # Cria uma nova coluna ('Percent'), calculando a porcentagem de cada valor do atributo (feature) do DataFrame "df":
    temp_df['Percent'] = round(number=temp_df.Total / TotalGeral, ndigits=4)

    # Retorna um dataframe com os resultados:
    return (temp_df)

# ***** Função para obter os "percentis" das features de um dataset qualquer.
#
def get_features_percentiles(data, features, quantiles):
    '''
    Input:
        "data": Data Frame com os dados.
        "features": Variáveis do dataframe que desejamos calcular os "Percentis".
        "quantiles": Lista dos percentis que desejamos calcular, no formato de 0 a 100.
    
    Output:
        "df": Data Frame com os percentis calculados para cada variável.
    '''
    # Código da função:

    # Cria um array dos "percentis" desejados:
    #percentiles = np.array(percent_range)
    
    # Cria um data frame onde serão armazenados os percentis de cada feature:
    percentiles_str = [str(v)+'%' for v in quantiles] # prepara os índices do data frame.
    
    # Cria o dataframe:
    df = pd.DataFrame(index=percentiles_str)
    df.index.name = 'Percentile' # nomeia o índice do data frame.
    
    # Verifica se temos apenas "1" variável em "features":
    if (type(features) == str):
        # Calcula os percentis e salva no dataframe:
        for index, q in enumerate(quantiles):
            df.at[percentiles_str[index], features] = data[features].quantile(q/100)
    else:
        # Neste caso, "features" representa uma lista de variáveis.
        # Loop para calcular os percentis de cada variável em 'features':
        for feat in features:
            # Calcula os percentis:
            for index, q in enumerate(quantiles):
                df.at[percentiles_str[index], feat] = data[feat].quantile(q/100)

    # Retorna o dataframe com os resultados:
    return df

# ***** Função para calcular algumas medidas estatísticas de tendência central (moda, média e mediana),
# de dispersão (desvio padrão), de forma (assimetria e curtose), e também o coeficiente de variação (CV).
#
def statistical_measures(df, feature, decimals=3):

    '''
    Entrada:
        "df": Data Frame que contém os atributos (features);
        "feature": atributos (variáveis) do Dataframe.

    Saída:
        results_df: Objeto do tipo "Dataframe" do Pandas, com os resultados calculados.
    '''
    # Código da função:

    # Cria um dicionário, calculando as medidas estatísticas para cada atributo (feature) do DataFrame (df):
    measures = {'count': df[feature].count(), # contagem.
                'mean': df[feature].mean(), # média.
                'median': df[feature].median(), # mediana.
                'mode': df[feature].mode(axis=0).iloc[0], # moda.
                'std': df[feature].std(), # desvio padrão.
                'var_coff': (df[feature].std()/df[feature].mean())*100, # CV = (std/mean)*100 (Coeficiente de variação).
                'min': df[feature].min(), # valor mínimo.
                '25%': df[feature].quantile(0.25), # 25% percentil.
                '50%': df[feature].quantile(0.50), # 50% percentil = mediana.
                '75%': df[feature].quantile(0.75), # 75% percentil.
                'max': df[feature].max(), # valor máximo.
                'skewness': df[feature].skew(axis=0), # Assimetria
                'kurtosis': df[feature].kurtosis(axis=0) # Curtose
               }
    
    # Criando um DataFrame para armazenar os resultados:
    results_df = pd.DataFrame(data=measures).T

    # Verifica se os resultados serão arredondados:
    if (decimals != None):
        # Retorna o resultado com arredondamento dos resultados:
        return results_df.round(decimals=decimals)
    else:
        # Retorna os resultados sem arredondamento:
        return (results_df)

# ***** Função para verificar (check) "features" x "parâmetros" passados em uma função qualquer:
#
def check_params(features, params):
    '''
    Input:
        "features": lista de variáveis (features) de um dataframe qualquer.
        "params": valor único ou valores (lista) definidos para cada feature.
    
    Output:
        "None": Se o tamanho da lista de features for diferente do tamanho da lista de parâmetros.
        "params": retorna a própria lista de parâmetros.
        "default" Retorna uma lista de parâmetros com valores iguais (default).
    '''
    # Código da função:
    
    # Verifica se "params" é uma lista:
    if (type(params) == list):
            # Compara "params x features":
            if (len(params) != len(features)):
                # Erro: Size of 'params' and 'features' are different!"
                return None
            else:
                # Retorna a própria lista de parâmetros:
                return params
    else:
        # Retorna uma lista de parâmetros com valores iguais (default):
        return [params for i in range(0, len(features))]

# ***** Função para calcular os valores "Inner fences" e/ou "Outer fences" de uma variável utilizando o método de Tukey:
# Fonte: https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-3-dcb54abaf7b0
#
def fences(df, variable_name, threshold_inner=1.5, threshold_outer=3, kind="both"):
    '''
    Input:
        "df": dataframe com os dados que serão analisados.
        "variable_name": variável de interesse em formato texto (string).
        "threshold_inner": limite para calcular os valores para a cerca interna (inner fence).
        "threshold_outer": limite para calcular os valores para a cerca externa (outer fence).
        "kind": tipo cerca (fence) que será retornada: "both", "inner" ou "outer"
    
    Output:
        Se "kind="both":
            "inner_fence_le": retorna o valor do limite inferior da cerca interna para a variável de interesse.
            "inner_fence_ue": retorna o valor do limite superior da cerca interna para a variável de interesse.
            "outer_fence_le": retorna o valor do limite inferior da cerca externa para a variável de interesse.
            "outer_fence_ue": retorna o valor do limite superior da cerca externa para a variável de interesse.
        Se "kind="inner":
            "inner_fence_le": retorna o valor do limite inferior da cerca interna para a variável de interesse.
            "inner_fence_ue": retorna o valor do limite superior da cerca interna para a variável de interesse.
        Se "kind="outer":
            "outer_fence_le": retorna o valor do limite inferior da cerca externa para a variável de interesse.
            "outer_fence_ue": retorna o valor do limite superior da cerca externa para a variável de interesse.
    '''
    # Código da função:
    
    # Calcula o quartil Q1:
    q1 = df[variable_name].quantile(0.25)
    # Calcula o quartil Q3:
    q3 = df[variable_name].quantile(0.75)
    # Calcula o intervalo interquartílico (IQR):
    iqr = q3-q1
    
    # Calcula os dois extremos de "inner fence" (inferior e superior):
    inner_fence = threshold_inner*iqr
    inner_fence_le = q1-inner_fence
    inner_fence_ue = q3+inner_fence
    
    # Calcula os dois extremos de "outer fence" (inferior e superior):
    outer_fence = threshold_outer*iqr
    outer_fence_le = q1-outer_fence
    outer_fence_ue = q3+outer_fence
    
    # Retorna os limites inferior e superior de acordo com o parâmetro "type":
    if(kind == 'both'):
        return inner_fence_le, inner_fence_ue, outer_fence_le, outer_fence_ue
    else:
        if(kind == 'inner'):
            return inner_fence_le, inner_fence_ue
        else:
            if(kind == 'outer'):
                return outer_fence_le, outer_fence_ue
            else:
                print("Error: Need to set the 'kind' parameter correctly: 'both', 'inner' or 'outer'!")
                return np.nan, np.nan

# ***** Função para detectar "outliers univaridos" utilizando o método de Tukey:
# Fonte: https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-1-4ece5098b755
#
def tukeys_method(df, variable_name, threshold_inner=1.5, threshold_outer=3):
    '''
    Input:
        "df": dataframe com os dados que serão analisados.
        "variable_name": variável de interesse em formato texto (string).
        "threshold_inner": limite para calcular os valores para a cerca interna (inner fence).
        "threshold_outer": limite para calcular os valores para a cerca externa (outer fence).
    
    Output:
        "outliers_prob": prováveis outliers detectados para a variável de interesse.
        "outliers_poss": possíveis outliers detectados para a variável de interesse.
    '''
    # Código da função:
    
    # Calcula o quartil Q1:
    q1 = df[variable_name].quantile(0.25)
    # Calcula o quartil Q3:
    q3 = df[variable_name].quantile(0.75)
    # Calcula o intervalo interquartílico (IQR):
    iqr = q3-q1
    # Calcula os valores para a cerca interna (inner fence) e externa (outer fence): 
    inner_fence = threshold_inner*iqr
    outer_fence = threshold_outer*iqr
    
    # Calcula os limites internos:
    inner_fence_le = q1-inner_fence
    inner_fence_ue = q3+inner_fence
    
    # Calcula os limites externos:
    outer_fence_le = q1-outer_fence
    outer_fence_ue = q3+outer_fence
    
    # Lista para salvar os outliers prováveis:
    outliers_prob = []
    # Lista para salvar os outliers possíveis:
    outliers_poss = []
    # Verificando e salvando os outliers prováveis:
    for index, x in enumerate(df[variable_name]):
        if x <= outer_fence_le or x >= outer_fence_ue:
            outliers_prob.append(index)

    # Verificando e salvando os outliers possíveis:
    for index, x in enumerate(df[variable_name]):
        if x <= inner_fence_le or x >= inner_fence_ue:
            outliers_poss.append(index)
    
    # Retorna os outliers prováveis e os outliers possíveis:
    return outliers_prob, outliers_poss

# ***** Função para detectar "outliers univariados" utilizando o método z-score:
# Fonte: https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-1-4ece5098b755
#
def z_score_method(df, variable_name, threshold=3):
    '''
    Input:
        "df": dataframe com os dados que serão analisados.
        "variable_name": variável de interesse em formato texto (string).
        "threshold": Limite para definir se a informação é um outlier.
    
    Output:
        "outliers": outliers detectados para a variável de interesse.
    '''
    # Código da função:
    
    # Calcula o z-score da variável de interesse em "df":
    z = np.abs(stats.zscore(df[[variable_name]]))
    
    # Lista para salvar os outliers:
    outliers = []

    # Para cada z-score verifico se é um outlier.
    # Se for, salvo o índice "i" na lista de outliers:
    for i, v in enumerate(z.loc[:, variable_name]):
        if v > threshold:
            outliers.append(i)
        else:
            continue    
    
    # Retorno a lista de outliers:
    return outliers

# ***** Função para detectar "outliers univariados" utilizando o método MAD (Desvio Absoluto Mediano):
# Fonte: https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-1-4ece5098b755
#
def mad_method(df, variable_name, threshold=3):
    '''
    Input:
        "df": dataframe com os dados que serão analisados.
        "variable_name": variável de interesse em formato texto (string).
        "threshold": Limite para definir se a informação é um outlier.
    
    Output:
        "outliers": outliers detectados para a variável de interesse.
    '''
    # Código da função:
    
    # Calcula a mediana da variável de interesse:
    med = np.median(df.loc[:,variable_name], axis = 0)
    # Calcula o valor MAD da variável de interesse:
    mad = np.abs(stats.median_abs_deviation(df.loc[:,variable_name], scale='normal'))
    # Cria a lista de outilers:
    outliers = []
    
    # Loop para detectar e salvar os outliers:
    for i, v in enumerate(df.loc[:,variable_name]):
        t = (v-med)/mad
        if t > threshold:
            outliers.append(i)
        else:
            continue
    # Retorna a lista dos outliers detectados:
    return outliers

# ***** Função para detectar "outliers multivariados" utilizando o método MD (Mahalonibis Distance):
# Fonte: https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-2-3a3319ec2c33
#
def mahalanobis_method(df):
    '''
    Input:
        "df": dataframe com as variáveis que serão analisadas.
    
    Output:
        "outliers": outliers detectados para as variáveis de interesse.
        "md": distância de Mahalanobis.
    '''
    # Código da função:
    
    # M-Distance
    x_minus_mu = df - np.mean(df)
    cov = np.cov(df.values.T)                           # Covariance
    inv_covmat = sp.linalg.inv(cov)                     # Inverse covariance
    left_term = np.dot(x_minus_mu, inv_covmat) 
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())
    
    # Cria lista para salvar os outliers:
    outliers = []

    # Cut-off point (Ponto de corte):
    C = np.sqrt(chi2.ppf((1-0.001), df=df.shape[1]))    # degrees of freedom = number of variables
    
    # Loop para detectar e salvar os outliers:
    for index, value in enumerate(md):
        if value > C:
            outliers.append(index)
        else:
            continue
    
    # Retorna os outliers e os valores de MD:
    return outliers, md

# ***** Função para detectar "outliers multivariados" utilizando o método RMD (Robust Mahalonibis Distance):
# Fonte: https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-2-3a3319ec2c33
#
def robust_mahalanobis_method(df):
    '''
    Input:
        "df": dataframe com as variáveis que serão analisadas.
    
    Output:
        "outliers": outliers detectados para as variáveis de interesse.
        "md": distância robusta de Mahalanobis.
    '''
    # Código da função:
    
    # Minimum covariance determinant
    rng = np.random.RandomState(0)
    real_cov = np.cov(df.values.T)
    X = rng.multivariate_normal(mean=np.mean(df, axis=0), cov=real_cov, size=506)
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_ # robust covariance metric
    robust_mean = cov.location_  # robust mean
    inv_covmat = sp.linalg.inv(mcd) # inverse covariance metric
    
    # Robust M-Distance:
    x_minus_mu = df - robust_mean
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())
    
    # Cria lista para salvar os outliers:
    outliers = []

    # Cut-off point (Ponto de corte):
    C = np.sqrt(chi2.ppf((1-0.001), df=df.shape[1]))           # degrees of freedom = number of variables
    
   # Loop para detectar e salvar os outliers:
    for index, value in enumerate(md):
        if value > C:
            outliers.append(index)
        else:
            continue
    
    # Retorna os outliers e os valores robustos de MD:
    return outliers, md

# ***** Função para substituir os prováveis outliers por valores ausentes (missing values) para imputação:
# Fonte: https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-2-3a3319ec2c33
#
def outliers_to_nan(df, variable_name, probable_outliers):
    '''
    Input:
        "df": dataframe com os dados que serão analisadas.
        "variable_name": variável de interesse em formato texto (string).
        "probable_outliers": prováveis outliers detectados através de um método univariado.
    
    Output:
        "df_copy": dataframe com os outliers substituidos por valores ausentes (NaN).
    '''
    # Código da função:
    
    # Faz uma cópia do data frame:
    df_copy = df.copy(deep=True)
    
    # Loop para substituir os prováveis outliers por "NaN":
    for i in probable_outliers:
        df_copy.at[i, variable_name] = None
    
    # Retorna o data frame modificado:
    return df_copy

# ***** Função para realizar teste estatístico, verificando se os dados de cada variável seguem (ou não) uma distribuição normal.
#
def shapiro_test(data, features, alpha=0.05):
    '''
    Input:
        "data": dataframe com os dados.
        "features": variáveis do dataframe que desejamos realizar o teste.
        "alpha": nível de significância utilizado no teste.
    
    Output:
        "df": Data Frame com as informações sobre os testes.
    '''
    # Código da função:

    # Cria o dataframe:
    df = pd.DataFrame(index=features, columns=['confidence level', 'p-value', 'test'])
    df.index.name = 'Variable' # nomeia o índice do data frame.
    
    # Loop para realizar o teste de hipóteses em cada variável:
    for feat in features:
        # Salva o nível de confiança:
        df.at[feat, 'confidence level'] = 1 - alpha
        
        # Realiza o teste de hipóteses:
        test_shapiro = shapiro(data[feat].values)
        # Salva o valor do teste (p-value):
        df.at[feat, 'p-value'] = str(test_shapiro.pvalue)
        # Verifica o resultado do teste:
        if(test_shapiro.pvalue > alpha):
            # Os dados da variável 'feat' SEGUEM uma distribuição normal:
            df.at[feat, 'test'] = 'yes'
        else:
            # Os dados da variável 'feat' NÃO SEGUEM uma distribuição normal:
            df.at[feat, 'test'] = 'no'
    
    # Retorna o dataframe com os resultados dos testes:
    return df

# ***** Função para fazer a transformação "logarítmica", "raiz quadrada" e "raiz cúbica" de uma variável numérica.
#
def transf_num(data, variables, neg_or_zero_values=None, scalar_sum=0):
    '''
    Input:
        "data": dataframe com os dados.
        "variable": variável que será utilizada para fazer as transformações.
        "neg_or_zero_values": informa as variáveis que possuem valores negativos ou iguais a "0".
        "scalar_sum": valor escalar a ser somado em variáveis que possuem valores negativos ou iguais a zero,
                      antes de aplicarmos as transformações "logarítmica" e "raiz quadrada".
    
    Output:
        "df": dataframe com as variáveis transformadas.
    '''
    # Código da função:
    
    # Cria um dataframe para armazenar as variáveis:
    df = pd.DataFrame()
    
    # Loop para fazer as transformações em cada variável:
    for var in variables:
        # Salva a variável original:
        df[var] = data[var].values

        # Verifica se a variável possui valores negativos ou iguais a "0":
        if(neg_or_zero_values == None):
            # Transformação "logarítmica":
            df[str(var + "_log")] = np.log(data[var].values)
            # Transformação "raiz quadrada":
            df[str(var + "_r2")] = np.sqrt(data[var].values)
        else:
            if(var in neg_or_zero_values):
                # Transformação "logarítmica":
                df[str(var + "_log")] = np.log(data[var].values+scalar_sum)
                # Transformação "raiz quadrada":
                df[str(var + "_r2")] = np.sqrt(data[var].values+scalar_sum)
            else:
                # Transformação "logarítmica":
                df[str(var + "_log")] = np.log(data[var].values)
                # Transformação "raiz quadrada":
                df[str(var + "_r2")] = np.sqrt(data[var].values)

        # Transformação "raiz cúbica":
        df[str(var + "_r3")] = np.cbrt(data[var].values)
    
    # Retorna o dataframe com os resultados:
    return df

# ***** Função para fazer a conversão do tipo das variáveis categóricas: tipo "object" para o tipo "categorical".
#
def conv_object_to_categorical(data, variables, categories=None, ordered=False):
    '''
    Input:
        "data": dataframe com os dados.
        "variables": variáveis categóricas que serão convertidas.
        "categories": dicionário com a lista de categorias de cada variável. Se valor igual a "None" (default)
                      serão obtidos os valores únicos de cada variável.
        "ordered": se "True", informa que as categorias são ordinais.
    
    Output:
        "df": dataframe com as variáveis convertidas para "categorical".
    '''
    # Código da função:
    
    # Faz uma cópia do dataframe:
    df = data.copy(deep=True)

    # Loop para analisarmos cada variável
    for var in variables:
        # Verifica se as categorias foram fornecidas:
        if(categories == None):
            cats = list(np.unique(df[var]))
        else:
            cats = categories[var]

        # Converte as categorias no objeto "cats" para o objeto "categorical" do Pandas:
        df[var] = pd.Categorical(values=df[var], categories=cats, ordered=ordered)

    # Retorna o dataframe com as variáveis convertidas:
    return df

# ***** Função para realizar o teste do qui-quadrado entre 2 variáveis categóricas:
# Para este teste temos as seguintes hipóteses:
# H₀: "var1" e "var2" são independentes, ou seja, não possuem relação.
# H₁: "var1" e "var2" não são independentes, ou seja, existe um relacionamento (são dependentes).
# αlpha = 0.05 (nível de significância) por default.
#
def chi_square_test(data, var1, var2, alpha=0.05, threshold=0.2):
    '''
    Input:
        "data": dataframe com as variáveis que serão utilizadas no teste.
        "var1": uma das variáveis do dataframe "data".
        "var2": uma das variáveis do dataframe "data".
        "alpha": nível de significância utilizado no teste (default = 5%)
        "threshold": valor limite (%) para a quantidade de células com frequências esperadas "< 5"
    
    Output: None
    Notas:
        1- Uma diretriz frequentemente citada na literatura para a validade deste cálculo é que
        o teste deve ser usado somente se as freqüências esperadas em cada célula da tabela
        de contingência são pelo menos ">=5" em pelo menos 20% das células.
        2-Este é um teste para a independência de diferentes categorias de uma população.
        O teste só é significativo quando a dimensão de "Observado" são dois ou mais, ou seja,
        se as variáveis "var1" e "var2" possuem duas ou mais categorias. Aplicando o teste a
        uma dimensão sempre resultará em uma tabela "esperada" igual a "observada" e um estatística
        qui-quadrado igual a 0.
    '''
    # Código da função:
    
    # Verificando se as variáveis possuem duas ou mais categorias:
    cat_var1 = data[var1].unique()
    cat_var2 = data[var2].unique()
    if((len(cat_var1) == 1) or (len(cat_var2) == 1)):
        # Print variables:
        print("Chi-Square Test: '{}' vs '{}'".format(var1, var2))
        # Mensagem de atenção:
        print("Attention: the test will only be significant if the variables have 2 or more categories!")
        return
    
    # Tabela de contingência dos dados observados:
    data_observed = pd.crosstab(data[var1], data[var2], margins=False)
    
    # Verificando se podemos aplicar o teste, ou seja, a quantidade de frequências esperadas, maior ou igual a 5,
    # na tabela de contingência, deverá ser igual ou maior que 20% (0.02): 
    # Tabela de contingência dos dados esperados:
    data_expected = stats.contingency.expected_freq(observed=data_observed)
    percent_expected = (data_expected < 5).sum()/(data_expected.size)
    # Verificando se as frequências esperadas são "< 5" em mais de 20% das células (default):
    if(percent_expected > threshold):
        # Print variables:
        print("Chi-Square Test: '{}' vs '{}'".format(var1, var2))
        # Mensagem de atenção:
        print("Attention: {}% of the cells have expected frequencies 'less than 5' in the contingency table!"
              .format(np.round(percent_expected*100, decimals=1)))
        return
    
    # Cálculo do teste "Chi-Square":
    chi2_test, p_value, dof, expected = stats.chi2_contingency(
        observed=data_observed, correction=True, lambda_=None
    )
    
    # Print variables:
    print("Chi-Square Test: '{}' vs '{}'".format(var1, var2))
    
    # Avaliando o resultado do teste:
    conclusion = "Failed to reject the null hypothesis. Variables are independent."
    if p_value <= alpha:
        conclusion = "Null Hypothesis is rejected. Variables are not independent."

    print("chisquare-score is:", chi2_test, " and p-value is:", p_value)
    print(conclusion)

    return

# ***** Função para realizar o teste do "Fisher’s exact" entre 2 variáveis categóricas:
# Para este teste temos as seguintes hipóteses:
# H₀: "var1" e "var2" são independentes, ou seja, não possuem relação.
# H₁: "var1" e "var2" não são independentes, ou seja, existe um relacionamento (são dependentes).
# αlpha = 0.05 (nível de significância) por default.
#
def fisher_exact_test(data, var1, var2, alpha=0.05, alternative='two-sided'):
    '''
    Input:
        "data": dataframe com as variáveis que serão utilizadas no teste.
        "var1": uma das variáveis do dataframe "data".
        "var2": uma das variáveis do dataframe "data".
        "alpha": nível de significância utilizado no teste (default = 5%)
        "threshold": valor limite (%) para a quantidade de células com frequências esperadas "< 5"
    
    Output: None
    Notas:
        Este teste só poderá ser aplicado para variáveis categórias que tenham "2" categorias,
        ou seja, a tabela de contingência deverá ser no formato (shape) "2 x 2".
    '''
    # Código da função:
    
    # Verificando se as variáveis possuem apenas duas categorias, ou seja, o shape "2 x 2":
    cat_var1 = data[var1].unique()
    cat_var2 = data[var2].unique()
    if((len(cat_var1) != 2) or (len(cat_var2) != 2)):
        # Print variables:
        print("Fisher’s exact test: '{}' vs '{}'".format(var1, var2))
        # Mensagem de atenção:
        print("Attention: The input 'contingency table' must be of shape (2, 2)!")
        return
    
    # Tabela de contingência dos dados observados:
    data_observed = pd.crosstab(data[var1], data[var2], margins=False)
    
    # Cálculo do teste "Fisher’s exact":
    oddsr, p_value = stats.fisher_exact(table=data_observed, alternative=alternative)
    
    # Print variables:
    print("Fisher’s exact test: '{}' vs '{}'".format(var1, var2))
    
    # Avaliando o resultado do teste:
    conclusion = "Failed to reject the null hypothesis. Variables are independent."
    if p_value <= alpha:
        conclusion = "Null Hypothesis is rejected. Variables are not independent."

    print("p-value is:", p_value)
    print(conclusion)

    return

# ***** Função para calcular um coeficiente de correlação "Spearman" com o p-value associado entre 2 variáveis numéricas:
# Documentação no site: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
#
def spearmanr_test(a, b=None, alpha=0.05, axis=0, nan_policy='propagate', alternative='two-sided'):
    '''
    Input:
        "a, b, axis, nan_policy": veja a documentação no site oficial.
        "alpha": nível de significância utilizado no teste (default = 5%)
        "alternative": define a hipótese alternativa. O padrão é "two-sided".
        As seguintes opções estão disponíveis:
            'two-sided': a correlação é diferente de zero.
            'less': a correlação é negativa (menor que zero).
            'greater': a correlação é positiva (maior que zero).
    
    Output: None
    '''
    # Código da função:
    
    # Cálculo do teste de correlação de "Spearman":
    # Será calculado o 'p-value' para um teste de hipóteses cuja hipótese nula é que dois conjuntos de dados não estão correlacionados
    # (alternative='two-sided'), ou que não possuem uma correlação negativa (alternative='less'),
    # ou que não possuem uma correlação positiva (alternative='greater').
    correlation, p_value = stats.spearmanr(a=a, b=b, axis=axis, nan_policy=nan_policy, alternative=alternative)
    
    # Avaliando o resultado do teste:
    # Teste se a correlação é diferente de zero (H1):
    if(alternative == 'two-sided'):
        conclusion = "Failed to reject the null hypothesis. Two sets of data are uncorrelated."
        if p_value <= alpha:
            conclusion = "Null Hypothesis is rejected. Two sets of data are correlated."
    else:
        # Teste se a correlação é menor que zero (H1):
        if(alternative == 'less'):
            conclusion = "Failed to reject the null hypothesis. Correlation between the two data sets is not negative (less than zero)."
            if p_value <= alpha:
                conclusion = "Null Hypothesis is rejected. Correlation between the two data sets is negative (less than zero)."
        else:
            # Teste se a correlação é maior que zero (H1):
            if(alternative == 'greater'):
                conclusion = "Failed to reject the null hypothesis. Correlation between the two data sets is not positive (greater than zero)."
                if p_value <= alpha:
                    conclusion = "Null Hypothesis is rejected. Correlation between the two data sets is positive (greater than zero)."

    # Mostra os valores calculados e a conclusão do teste:
    print("Spearman correlation coefficient is:{} and 'p-value' is:{}".format(
        np.round(correlation, decimals=3), str(p_value)))
    print(conclusion)

    return

# ***** Função para calcular um coeficiente de correlação "Pearson" com o p-value associado entre 2 variáveis numéricas:
# Documentação no site: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
#
def pearsonr_test(x, y, alpha=0.05):
    '''
    Input:
        "x, y": veja a documentação no site oficial.
        "alpha": nível de significância utilizado no teste (default = 5%)
    
    Output: None
    '''
    # Código da função:
    
    # Cálculo do teste de correlação de "Pearson":
    # Será calculado o 'p-value' para um teste de hipóteses cuja hipótese nula é que dois conjuntos de dados
    # não estão correlacionados (H1='two-sided').
    correlation, p_value = stats.pearsonr(x=x, y=y)
    
    # Avaliando o resultado do teste:
    # Testa se a correlação é diferente de zero (H1), ou seja, as variáveis "x e y" estão correlacionadas:
    conclusion = "Failed to reject the null hypothesis. Two sets of data are uncorrelated."
    if p_value <= alpha:
        conclusion = "Null Hypothesis is rejected. Two sets of data are correlated."
    
    # Mostra os valores calculados e a conclusão do teste:
    print("Pearson correlation coefficient is:{} and 'p-value' is:{}".format(
        np.round(correlation, decimals=3), str(p_value)))
    print(conclusion)

    return

# ***** Função para realizar o teste "t de students" entre uma variável categórica (com 2 classes) e uma variável numérica:
# Documentação no site: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
#
def t_test_independent(
    a, b, axis=0, equal_var=True, nan_policy='propagate', permutations=None, random_state=None,
    alternative='two-sided', trim=0 , alpha=0.05
    ):
    '''
    Input:
        "a, b, axis, equal_var, nan_policy, permutations, random_state, alternative, trim": veja a documentação no site oficial.
        "alpha": nível de significância utilizado no teste (default = 5%)
    
    Output: None
    '''
    # Código da função:
    
    # Calcula o teste T para as médias de duas amostras independentes de scores.
    # Será calculado o 'p-value' para um teste de hipóteses cuja hipótese nula é que as "médias" dos dois conjuntos de dados (a e b) são iguais
    # (alternative='two-sided'), ou que a "média" do conjunto "a" é "menor" que a "média" do conjunto "b" (alternative='less'),
    # ou que a "média" do conjunto "a" é "maior" que a "média" do conjunto "b" (alternative='greater').
    statistic, p_value = stats.ttest_ind(
        a=a, b=b, axis=axis, equal_var=equal_var, nan_policy=nan_policy, permutations=permutations,
        random_state=random_state, alternative=alternative, trim=trim
        )
    
    # Avaliando o resultado do teste:
    # Testa se as médias de "a" e "b" não são iguais (H1):
    if(alternative == 'two-sided'):
        conclusion = "Failed to reject the null hypothesis. Averages of sets 'a' and 'b' are EQUAL."
        if p_value <= alpha:
            conclusion = "Null Hypothesis is rejected. Averages of sets 'a' and 'b' are NOT EQUAL."
    else:
        # Testa se a média de "b" é menor que a média de "a" (H1):
        if(alternative == 'less'):
            conclusion = "Failed to reject the null hypothesis. Average of set 'b' is LESS than average of set 'a'."
            if p_value <= alpha:
                conclusion = "Null Hypothesis is rejected. Average of set 'b' is NOT LESS than average of set 'a'."
        else:
            # Testa se a média de "b" é maior que a média de "a" (H1):
            if(alternative == 'greater'):
                conclusion = "Failed to reject the null hypothesis. Average of set 'b' is GREATER than average of 'a'."
                if p_value <= alpha:
                    conclusion = "Null Hypothesis is rejected. Average of set 'b' is NOT GREATER than average of 'a'."

    # Mostra os valores calculados e a conclusão do teste:
    print("The calculated t-statistic is:{} and 'p-value' is:{}".format(
        np.round(statistic, decimals=3), str(p_value)))
    print(conclusion)

    return

# ***** Função para calcular o VIF (Variance Inflation Factor) entre as variáveis numéricas:
#
def vif(data, variables, const=False):
    '''
    Input:
        "data": dataframe com os dados que serão analisados.
        "variables": lista das variáveis que serão utilizadas nos cálculos.
        "const": se "False", incluímos uma constante nos dados para calcularmos o VIF.
    
    Output:
        vif_df: dataframe com os valores VIF de cada variável.
    '''
    # Código da função:

    if(const == False):
        # Criando uma cópia do dataframe com as variáveis preditoras:
        X = data[variables].copy(deep=True)

        # Adiciona uma contante:
        X_constant = sm.add_constant(X, prepend=False)
    else:
        # Criando uma cópia do dataframe com as variáveis preditoras:
        X_constant = data[variables].copy(deep=True)
    
    # Criando um dataframe para calularmos o VIF:
    vif_df = pd.DataFrame()

    # Coluna com os nomes das variáveis:
    vif_df["variables"] = X_constant.columns

    # Calculando os valores VIF:
    vif_df["VIF"] = [variance_inflation_factor(X_constant.values, i) for i in range(X_constant.shape[1])]

    # Retorna os resultados:
    return vif_df
