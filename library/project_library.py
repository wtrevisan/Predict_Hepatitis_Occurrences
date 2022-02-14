# ************************************* Project Library *************************************
# Neste arquivo estão definidas algumas funções específicas utilizadas neste projeto.
#

# ************************************* Importando Pacotes ou Funções *************************************
#
# Importa o pacote  "numpy":
import numpy as np

# Importa o pacote  "pandas":
import pandas as pd

# ************************************* Definindo Classes *************************************
#

# ************************************* Definindo Funções *************************************
#
# ***** Função para preparar os "dados" de acordo com o contexto do projeto. Etapa 1:
#
def data_prep_step_1(df, target=False):
    '''
    Input:
        "df": data frame com as variáveis do projeto;
        "target": se "True" informa que a variável "target" está presente no data frame "df".

    Output:
        "data_df": retorna um data frame com as variáveis preparadas.
    '''
    # Código da função:
    
    # Faz uma cópia do data frame "df":
    data_df = df
    
    # Preparando a variável target "Class" se estiver presente no data frame.
    # Verifica se a variável target está presente:
    if(target):
        # Variável "Class" está presente no data frame "df".
        # Substituindo os valores:
        data_df.replace({'Class': {1:'DIE', '1':'DIE', 2:'LIVE', '2':'LIVE'}}, inplace=True)
       
    # Preparando as variáveis categóricas preditoras.
    # Substituindo os valores:
    data_df.replace({'Gender'        : {1:'male', '1':'male', 2:'female', '2':'female', '?':np.nan},
                     'Steroid'       : {1:'no', '1':'no', 2:'yes', '2':'yes', '?':np.nan},
                     'Antivirals'    : {1:'no', '1':'no', 2:'yes', '2':'yes', '?':np.nan},
                     'Fatigue'       : {1:'no', '1':'no', 2:'yes', '2':'yes', '?':np.nan},
                     'Malaise'       : {1:'no', '1':'no', 2:'yes', '2':'yes', '?':np.nan},
                     'Anorexia'      : {1:'no', '1':'no', 2:'yes', '2':'yes', '?':np.nan},
                     'LiverBig'      : {1:'no', '1':'no', 2:'yes', '2':'yes', '?':np.nan},
                     'LiverFirm'     : {1:'no', '1':'no', 2:'yes', '2':'yes', '?':np.nan},
                     'SpleenPalpable': {1:'no', '1':'no', 2:'yes', '2':'yes', '?':np.nan},
                     'Spiders'       : {1:'no', '1':'no', 2:'yes', '2':'yes', '?':np.nan},
                     'Ascites'       : {1:'no', '1':'no', 2:'yes', '2':'yes', '?':np.nan},
                     'Varices'       : {1:'no', '1':'no', 2:'yes', '2':'yes', '?':np.nan},
                     'Histology'     : {1:'no', '1':'no', 2:'yes', '2':'yes', '?':np.nan}
                    }, inplace=True)

    # Preparando as variáveis numéricas preditoras.
    # Substituindo os valores ausentes "?" por "NaN":
    data_df.replace({'Age'         : {'?':np.nan},
                     'Bilirubin'   : {'?':np.nan},
                     'AlkPhosphate': {'?':np.nan},
                     'SGOT'        : {'?':np.nan},
                     'Albumin'     : {'?':np.nan},
                     'Protime'     : {'?':np.nan}
                    }, inplace=True)
    
    # Alterando os tipos dos dados para "float64":
    data_df['Age']          = data_df['Age'].astype(dtype='float64')
    data_df['Bilirubin']    = data_df['Bilirubin'].astype(dtype='float64')
    data_df['AlkPhosphate'] = data_df['AlkPhosphate'].astype(dtype='float64')
    data_df['SGOT']         = data_df['SGOT'].astype(dtype='float64')
    data_df['Albumin']      = data_df['Albumin'].astype(dtype='float64')
    data_df['Protime']      = data_df['Protime'].astype(dtype='float64')
    
    # Bilirubin: alterando a unidade dos valores de "mg/dL" para "mg/L":
    data_df['Bilirubin'] = data_df['Bilirubin']*10

    # Albumin: alterando a unidade dos valores de "mg/dL" para "mg/L":
    data_df['Albumin'] = data_df['Albumin']*10
    
    # Retorna o data frame preparado:
    return data_df

# ***** Função para aplicar o método "encoding" nas variáveis categóricas:
#
def method_encoding_categorical(df, target=False):
    '''
    Input:
        "df": data frame com as variáveis do projeto;
        "target": se "True" informa que a variável "target" está presente no data frame "df".

    Output:
        "data_df": retorna um data frame com as variáveis categóricas codificadas.
    '''
    # Código da função:
    
    # Faz uma cópia do data frame "df":
    data_df = df.copy(deep=True)
    
    # Preparando a variável target "Class" se estiver presente no data frame.
    # Verifica se a variável target está presente:
    if(target):
        # Variável "Class" está presente no data frame "df".
        # Substituindo os valores:
        data_df.replace({'Class': {'DIE':0, 'LIVE':1}}, inplace=True)

    # Preparando as variáveis categóricas preditoras.
    # Substituindo os valores:
    data_df.replace({'Gender'        : {'male':0, 'female':1},
                     'Steroid'       : {'no':0, 'yes':1},
                     'Antivirals'    : {'no':0, 'yes':1},
                     'Fatigue'       : {'no':0, 'yes':1},
                     'Malaise'       : {'no':0, 'yes':1},
                     'Anorexia'      : {'no':0, 'yes':1},
                     'LiverBig'      : {'no':0, 'yes':1},
                     'LiverFirm'     : {'no':0, 'yes':1},
                     'SpleenPalpable': {'no':0, 'yes':1},
                     'Spiders'       : {'no':0, 'yes':1},
                     'Ascites'       : {'no':0, 'yes':1},
                     'Varices'       : {'no':0, 'yes':1},
                     'Histology'     : {'no':0, 'yes':1},
                    }, inplace=True)

    # Retorna o data frame preparado:
    return data_df

# ***** Função para realizar o diagnóstico do exame de Bilirrubina de acordo com os valores de referência
# obtidos neste site: https://www.ebmconsult.com/articles/lab-test-bilirubin-total-level
#
def Bilirubin_test_diagnosis(data, age, results):
    '''
    Input:
        "data": dataframe com os dados.
        "age": variável com a idade dos pacientes que realizaram o exame de Bilirrubina.
        "results": resultados dos exames de Bilirrubina de cada paciente.
    
    Output:
        "diagnosis": diagnóstico dos resultados dos exames de cada paciente. Categorias: 'low', 'normal' ou 'high'.
    '''
    # Código da função:

    # Cria uma lista para armazenar o diagnóstico de cada exame:
    diagnosis = []
    # Obtem as idades dos pacientes:
    ages = data[age].values
    # Obtem os resultados dos exames de Bilirrubina dos pacientes:
    bilirubin = data[results].values
    
    # Loop para realizar os diagnósticos de cada exame:
    for index in range(len(data)):
        # Verifica se a idade é <= 18 anos:
        if(ages[index] <= 18):
            # Faz o diagnóstico do exame:
            if(bilirubin[index] < 3):
                # Bilirrubina baixa:
                diagnosis.append('low')
            else:
                if(bilirubin[index] > 12):
                    # Bilirrubina alta:
                    diagnosis.append('high')
                else:
                    # Bilirrubina normal:
                    diagnosis.append('normal')
        else:
            # Então, a idade está acima de 18 anos:
            # Faz o diagnóstico do exame:
            if(bilirubin[index] < 3):
                # Bilirrubina baixa:
                diagnosis.append('low')
            else:
                if(bilirubin[index] > 10):
                    # Bilirrubina alta:
                    diagnosis.append('high')
                else:
                    # Bilirrubina normal:
                    diagnosis.append('normal')

    # Converte a lista para um objeto "categorical" do pandas:
    diagnosis = pd.Categorical(values=diagnosis, categories=['low', 'normal', 'high'], ordered=True)

    # Retorna os diagnósticos dos exames de Bilirrubina:
    return diagnosis

# ***** Função para realizar o diagnóstico do exame de fosfatase alcalina (ALP) de acordo com os valores de referência
# obtidos neste site: https://www.tuasaude.com/exame-fosfatase-alcalina/
#
def AP_test_diagnosis(data, age, results):
    '''
    Input:
        "data": dataframe com os dados.
        "age": variável com a idade dos pacientes que realizaram o exame ALP.
        "results": resultados dos exames ALP de cada paciente.
    
    Output:
        "diagnosis": diagnóstico dos resultados dos exames de cada paciente. Categorias: 'low', 'normal' ou 'high'.
    '''
    # Código da função:

    # Cria uma lista para armazenar o diagnóstico de cada exame:
    diagnosis = []
    # Obtem as idades dos pacientes:
    ages = data[age].values
    # Obtem os resultados dos exames ALP dos pacientes:
    ALP = data[results].values
    
    # Loop para realizar os diagnósticos de cada exame:
    for index in range(len(data)):
        # Verifica se a idade é < 2 anos:
        if(ages[index] < 2):
            # Faz o diagnóstico do exame:
            if(ALP[index] < 85):
                # ALP baixo:
                diagnosis.append('low')
            else:
                if(ALP[index] > 235):
                    # ALP alto:
                    diagnosis.append('high')
                else:
                    # ALP normal:
                    diagnosis.append('normal')
        else:
            # Verifica se a idade está entre 2 a 8 anos:
            if(ages[index] <= 8):
                # Faz o diagnóstico do exame:
                if(ALP[index] < 65):
                    # ALP baixo:
                    diagnosis.append('low')
                else:
                    if(ALP[index] > 210):
                        # ALP alto:
                        diagnosis.append('high')
                    else:
                        # ALP normal:
                        diagnosis.append('normal')
            else:
                # Verifica se a idade está entre 9 a 15 anos:
                if(ages[index] <= 15):
                    # Faz o diagnóstico do exame:
                    if(ALP[index] < 60):
                        # ALP baixo:
                        diagnosis.append('low')
                    else:
                        if(ALP[index] > 300):
                            # ALP alto:
                            diagnosis.append('high')
                        else:
                            # ALP normal:
                            diagnosis.append('normal')         
                else:
                    # Verifica se a idade está entre 16 a 21 anos:
                    if(ages[index] <= 21):
                        # Faz o diagnóstico do exame:
                        if(ALP[index] < 30):
                            # ALP baixo:
                            diagnosis.append('low')
                        else:
                            if(ALP[index] > 200):
                                # ALP alto:
                                diagnosis.append('high')
                            else:
                                # ALP normal:
                                diagnosis.append('normal')         
                    else:
                        # Então, a idade está acima de 21 anos (Adulto):
                        # Faz o diagnóstico do exame:
                        if(ALP[index] < 46):
                            # ALP baixo:
                            diagnosis.append('low')
                        else:
                            if(ALP[index] > 120):
                                # ALP alto:
                                diagnosis.append('high')
                            else:
                                # ALP normal:
                                diagnosis.append('normal')         
    
    # Converte a lista para um objeto "categorical" do pandas:
    diagnosis = pd.Categorical(values=diagnosis, categories=['low', 'normal', 'high'], ordered=True)

    # Retorna os diagnósticos dos exames ALP:
    return diagnosis
      
# ***** Função para realizar o diagnóstico do exame AST ou TGO de acordo com os valores de referência
# obtidos neste site: https://www.tuasaude.com/exame-aspartato-aminotransferase/
#
def SGOT_test_diagnosis(data, results):
    '''
    Input:
        "data": dataframe com os dados.
        "results": resultados dos exames AST ou TGO de cada paciente.
    
    Output:
        "diagnosis": diagnóstico dos resultados dos exames de cada paciente. Categorias: 'low', 'normal' ou 'high'.
    '''
    # Código da função:

    # Cria uma lista para armazenar o diagnóstico de cada exame:
    diagnosis = []
    # Obtem os resultados dos exames ALP dos pacientes:
    AST = data[results].values
    
    # Loop para realizar os diagnósticos de cada exame:
    for index in range(len(data)):
        # Faz o diagnóstico do exame:
        if(AST[index] < 5):
            # AST baixo:
            diagnosis.append('low')
        else:
            if(AST[index] > 40):
                # AST alto:
                diagnosis.append('high')
            else:
                # AST normal:
                diagnosis.append('normal')

    # Converte a lista para um objeto "categorical" do pandas:
    diagnosis = pd.Categorical(values=diagnosis, categories=['low', 'normal', 'high'], ordered=True)

    # Retorna os diagnósticos dos exames AST:
    return diagnosis

# ***** Função para realizar o diagnóstico do exame de "Albumina" de acordo com os valores de referência
# obtidos neste site: https://labs.selfdecode.com/blog/albumin/
#
def Albumin_test_diagnosis(data, results):
    '''
    Input:
        "data": dataframe com os dados.
        "results": resultados dos exames de Albumina de cada paciente.
    
    Output:
        "diagnosis": diagnóstico dos resultados dos exames de cada paciente. Categorias: 'low', 'normal' ou 'high'.
    '''
    # Código da função:

    # Cria uma lista para armazenar o diagnóstico de cada exame:
    diagnosis = []
    # Obtem os resultados dos exames de Albumina dos pacientes:
    Albumin = data[results].values
    
    # Loop para realizar os diagnósticos de cada exame:
    for index in range(len(data)):
        # Faz o diagnóstico do exame:
        if(Albumin[index] < 35):
            # Albumina baixa:
            diagnosis.append('low')
        else:
            if(Albumin[index] > 56):
                # Albumina alta:
                diagnosis.append('high')
            else:
                # Albumina normal:
                diagnosis.append('normal')

    # Converte a lista para um objeto "categorical" do pandas:
    diagnosis = pd.Categorical(values=diagnosis, categories=['low', 'normal', 'high'], ordered=True)

    # Retorna os diagnósticos dos exames de Albumina:
    return diagnosis

# ***** Função para realizar o diagnóstico do exame do "Tempo de Protrombina (TP)" de acordo com os valores de referência
# obtidos neste site: https://www.tuasaude.com/tempo-de-protrombina/
#
def PT_test_diagnosis(data, results):
    '''
    Input:
        "data": dataframe com os dados.
        "results": resultados dos exames do Tempo de Protrombina (TP) de cada paciente.
    
    Output:
        "diagnosis": diagnóstico dos resultados dos exames de cada paciente. Categorias: 'low', 'normal' ou 'high'.
    '''
    # Código da função:

    # Cria uma lista para armazenar o diagnóstico de cada exame:
    diagnosis = []
    # Obtem os resultados dos exames do Tempo de Protrombina (TP) dos pacientes:
    TP = data[results].values
    
    # Loop para realizar os diagnósticos de cada exame:
    for index in range(len(data)):
        # Faz o diagnóstico do exame:
        if(TP[index] < 10):
            # TP baixo:
            diagnosis.append('low')
        else:
            if(TP[index] > 14):
                # TP alto:
                diagnosis.append('high')
            else:
                # TP normal:
                diagnosis.append('normal')

    # Converte a lista para um objeto "categorical" do pandas:
    diagnosis = pd.Categorical(values=diagnosis, categories=['low', 'normal', 'high'], ordered=True)

    # Retorna os diagnósticos dos exames de Albumina:
    return diagnosis

# ***** Função para realizar o diagnóstico dos exames indicados no parâmetro "results"
#
def test_diagnosis(data, results):
    '''
    Input:
        "data": dataframe com os dados.
        "results": lista ou variável (string) com os resultados dos exames de cada paciente.
    
    Output:
        "diagnosis": dataframe com os diagnósticos dos resultados dos exames de cada paciente.
    '''
    # Código da função:

    # Cria um dataframe "vazio" para armazenar o diagnóstico de cada exame:
    diagnosis = pd.DataFrame()

    # Cria um dicionário com as funções de diagnósticos de cada exame:
    diag_func = {
        "Bilirubin"   : Bilirubin_test_diagnosis(data=data, age='Age', results='Bilirubin'),
        "AlkPhosphate": AP_test_diagnosis(data=data, age='Age', results='AlkPhosphate'),
        "SGOT": SGOT_test_diagnosis(data=data, results='SGOT'),
        "Albumin": Albumin_test_diagnosis(data=data, results='Albumin'),
        "Protime": PT_test_diagnosis(data=data, results='Protime')
        }

    # Verifica se "results" é uma lista:
    if (type(results) == list):
        # Loop para realizar os diagnósticos de cada exame:
        for var in results:
            # Faz o diagnóstico do exame:
            diagnosis[var] = diag_func[var]
    else:
        # Faz o diagnóstico do exame:
        diagnosis[results] = diag_func[results]

    # Retorna os diagnósticos dos exames:
    return diagnosis

# ***** Função para converter os diagnósticos dos exames ('low', 'normal', 'high') para valores numéricos.
#
def diagnosis_to_num(data, diagnosis, num_values=[1, 3, 7]):
    '''
    Input:
        "data": dataframe com os dados.
        "diagnosis": diagnósticos dos resultados dos exames de cada paciente.
        "num_values": lista com os 3 valores numéricos que irão substituir as categorias: "low"; "normal"; e "high", nesta
                      ordem respectivamente. Default: "low = 1", "normal = 3" e "high = 7".
    
    Output:
        "diagnosis_num": valores numéricos para o diagnóstico dos resultados dos exames de cada paciente.
    '''
    # Código da função:

    # Cria uma lista para armazenar o diagnóstico numérico de cada exame:
    diagnosis_num = []
    
    # Obtem os diagnósticos dos pacientes:
    diags = data[diagnosis].values

    # Loop para realizar as substituições das categorias pelos valores numéricos:
    for index in range(len(data)):
        # Verifica se o diagnóstico é "low":
        if(diags[index] == 'low'):
            # Faz a substituição para o primeiro valor numérico da lista: num_values[0]:
            diagnosis_num.append(num_values[0])
        else:
            # Verifica se o diagnóstico é "normal":
            if(diags[index] == 'normal'):
                # Faz a substituição para o segundo valor numérico da lista: num_values[1]:
                diagnosis_num.append(num_values[1])
            else:
                # Verifica se o diagnóstico é "high":
                if(diags[index] == 'high'):
                    # Faz a substituição para o terceiro valor numérico da lista: num_values[2]:
                    diagnosis_num.append(num_values[2])
                else:
                    # Erro: o diagnóstico informado (categoria) não é correto.
                    print('Error: the diagnosis "{}" is not correct!'.format(diags[index]))
    
    # Retorna os diagnósticos numéricos dos resultados dos exames ALP:
    return diagnosis_num
      


