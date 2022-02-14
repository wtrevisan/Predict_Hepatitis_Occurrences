# ************************************* Plots Library For Data Science (Versão 1.0) *************************************
# Neste arquivo estão definidas algumas funções gráficas para serem utilizadas em Data Science.
#

# ************************************* Importando Pacotes e/ou Funções *************************************
#
# Importa o pacote "numpy":
import numpy as np

# Importa o pacote "pandas":
import pandas as pd

# Importa modelos estatísticos:
import statsmodels.api as sm

# Importa o pacote "itertools":
import itertools

# Importa a função "time" do pacote "time":
from time import time

# Importa o pacote "os" (Operation System with its Packages and Functions)
import os

# Importa a função "Image" do pacote "IPython.display":
from IPython.display import Image

# Importa o pacote "matplotlib":
import matplotlib as mpl

# Importa o pacote "pyplot" do "matplotlib":
import matplotlib.pyplot as plt
# Lista de cores: https://matplotlib.org/stable/gallery/color/named_colors.html
# Palettes: https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette

# Importa o pacote "seaborn"
import seaborn as sns

# Importa a função "confusion_matriz"
from sklearn.metrics import confusion_matrix

# O pacote "sys" permite manipulações com o sistema operacional:
import sys

# Path: onde estão armazenadas as classes e funções que serão utilizadas neste módulo:
LIB_PATH = os.path.join(".")

# Adicionando o diretório ao 'path' do Sistema, para podermos importar classes e funções que serão
# utilizadas neste módulo:
sys.path.append(LIB_PATH)

# Importando para este notebook, as classes e funções definidas no módulo "DataScience_Library_v1_0":
import data_science_library as dslib

# ************************************* Definindo Funções *************************************
#  ***** Function to save any figures (e.g., graphics, dashboards, etc.)
#
def save_figure (figure_id, figure_path, figure_file_format="png", figure_file_extension=".png",
                 tight_layout=True, dpi=300):
    '''
    Input:
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
    
    # Define o path (diretório) onde será salvo a figura:
    path = os.path.join(figure_path, figure_id + figure_file_extension)

    # Imprime na tela uma mensagem informando o nome do arquivo:
    print("Saving figure:", figure_id)

    # Tight layout?
    if tight_layout:
        plt.tight_layout()

    # Saving the figure:
    plt.savefig(path, format=figure_file_format, dpi=dpi)

# ***** Função para plotar as contagens de observações de cada classe de uma variável categórica.
# É como se fosse um "histograma" para variáveis qualitativas (categóricas):
#
def plot_count_feature(df=None, feature=None, order=True,
                       plot_style='darkgrid',
                       title=None, title_fontsize=16,
                       xlabel=None, xlabel_fontsize=13,
                       ylabel_fontsize=13,
                       xtick_labelsize=None, ytick_labelsize=None,
                       width=8, height=6,
                       figure_id=None, figure_path=None,
                       figure_file_format="png", figure_file_extension=".png",
                       tight_layout=True, dpi=300
                       ):
    '''
    Input:
        "df": DataFrame com a base de dados do dataset que será utilizado.
        "feature": Variável categórica que será utilizada para construir o gráfico (Plot).
        "order":
            "True": Os valores das classes serão apresentados no gráfico em ordem decrescente. Este é o valor "default".
            "False": Os valores das classes serão apresentados no gráfico em ordem normal, ou seja, de acordo com tipo definido para a variável.
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "title": Define o título do gráfico.
        "title_fontsize": Define o tamanho da fonte do título para o gráfico.
        "xlabel": Define o label do eixo "x" para o gráfico.
        "xlabel_fontsize": Define tamanho da fonte do label do eixo "x" para o gráfico.
        "ylabel_fontsize": Define tamanho da fonte do label do eixo "y" para o gráfico.
        "xtick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "x".
        "ytick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "y".
        "width": Define a largura da figura onde será desenhado (plot) o gráfico.
        "heigth": Define a altura da figura onde será desenhado (plot) o gráfico.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:

    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))

    # Order
    if(order):
        value_counts = df[feature].value_counts().index
    else:
        value_counts = None
    
    # Plot
    sns.countplot(x = feature,
                  data = df,
                  order = value_counts
                 )

    # Definindo o título:
    plt.title(label=title, size=title_fontsize)

    # Definindo o label do eixo "x":
    plt.xlabel(xlabel=xlabel, size=xlabel_fontsize)

    # Definindo o label do eixo "y":
    plt.ylabel(ylabel="Count", size=ylabel_fontsize)

    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y" do gráfico:
    plt.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    plt.tick_params(axis='y', which='major', labelsize=ytick_labelsize)
    
    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)
    
    # Mostrando o gráfico:
    plt.show()

    # Fechando as instâncias da figura (Figure) criada:
    plt.close()
    return

# ***** Função para plotar as contagens e proporções de observações de cada categoria de uma variável categórica qualquer.
# É como se fossemos criar um "histograma" para variáveis qualitativas (categóricas):
#
def plot_percent_count_feature(df=None, feature=None,
                               plot_style='darkgrid',
                               title=None, title_fontsize=16, title_share=False,
                               xlabel1=None, xlabel1_fontsize=13,
                               xlabel2=None, xlabel2_fontsize=13,
                               ylabel_fontsize=13,
                               columns_nr=2, rows_nr=1,
                               share_x=False, share_y=False,
                               width=17, height=6,
                               xtick_labelsize=None, ytick_labelsize=None,
                               wspace=None, hspace=None,
                               figure_id=None, figure_path=None,
                               figure_file_format="png", figure_file_extension=".png",
                               tight_layout=True, dpi=300
                               ):
    '''
    Input:
        "df": DataFrame com a base de dados do dataset que será utilizado.
        "feature": Variável categórica que será utilizada para construir o gráfico (Plot).
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "title": Define o título dos gráficos.
        "title_fontsize": Define o tamanho da fonte do título para os gráficos.
        "title_share": Define se o título do gráfico 1 deverá ser compartilhado com o gráfico 2.
        "xlabel1": Define o label do eixo "x" para o gráfico 1.
        "xlabel1_fontsize": Define tamanho da fonte do label do eixo "x" para o gráfico 1.
        "xlabel2": Define o label do eixo "x" para o gráfico 2.
        "xlabel2_fontsize": Define tamanho da fonte do label do eixo "x" para o gráfico 2.
        "ylabel_fontsize": Define tamanho da fonte do label do eixo "y" para os gráficos 1 e 2.
        "colums_nr": Define se os gráficos serão colocados na horixontal (2), ou na vertical (1). 
        "rows_nr": Define se os gráficos serão colocados na horixontal (1), ou na vertical (2).
        "share_x": Define se os dois gráficos compartilharão o eixo "x".
        "share_y": Define se os dois gráficos compartilharão o eixo "y".
        "width": Define a largura da figura onde serão desenhados (plot) os dois gráficos.
        "heigth": Define a altura da figura onde serão desenhados (plot) os dois gráficos.
        "xtick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "x".
        "ytick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "y".
        "wspace": Define o espaçamento (largura) entre os gráficos nos subplots.
        "hspace": Define o espaçamento (altura) entre os gráficos nos subplots.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output:
        "Temp_df": objeto DataFrame com os valores das contagens e proporções da variável categórica "feature".
    '''
    # Código da função:
    
    # Criando um DataFrame temporário, calculando as contagens e proporções de valores da nossa variável (feature),
    # e em seguida faz o "reset" do índice, ou seja, o nosso índice passa a ser uma nova coluna em nosso
    # DataFrame temporário:
    Temp_df = dslib.percent_count_feature(df, feature).sort_index().reset_index()
       
    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura e os dois eixos onde serão plotados os gráficos:
    fig, (ax1, ax2) = plt.subplots(nrows=rows_nr, ncols=columns_nr,
                                   sharex=share_x, sharey=share_y,
                                   figsize=(width,height))
    
    # Desenhando o gráfico no primeiro eixo (ax1), com as contagens das classes da nossa variável categórica "feature":
    sns.barplot(x = feature,
                y = 'Total',
                data = Temp_df,
                ax=ax1
                )
    
    # Definindo o título do primeiro gráfico:
    ax1.set_title(label = title, fontdict = {'fontsize': title_fontsize})
    
    # Definindo o label para o eixo "x" do primeiro gráfico:
    ax1.set_xlabel(xlabel = xlabel1, fontdict = {'fontsize': xlabel1_fontsize})
    
    # Definindo o label para o eixo "y" do primeiro gráfico:
    ax1.set_ylabel(ylabel = 'Count', fontdict = {'fontsize': ylabel_fontsize})

    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y" do primeiro gráfico:
    ax1.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    ax1.tick_params(axis='y', which='major', labelsize=ytick_labelsize)
    
    # Desenhando o gráfico no segundo eixo (ax2), com as proporções das classes da nossa variável categórica "feature":
    sns.barplot(x = feature,
                y = 'Percent',
                data = Temp_df,
                ax=ax2
                )
        
    # Definindo o título do segundo gráfico.
    # Verifica se o segundo gráfico será desenhado na mesma coluna, e se o eixo "x" será compartilhado com o
    # primeiro gráfico:
    if((columns_nr==1) and (share_x==True)):
        # Não coloca o título no segundo gráfico:
        ax2.set_title(label = None)
    else:
        if(title_share):
            # Compartilha o título do gráfico 1 com o gráfico 2:
            ax2.set_title(label = None)
            
        else: # Não compartilha o título do gráfico 1 com o gráfico 2.
            # Coloca o título no segundo gráfico:
            ax2.set_title(label = title, fontdict = {'fontsize': title_fontsize})
    
    # Definindo o label para o eixo "x" do primeiro gráfico:
    ax2.set_xlabel(xlabel = xlabel2, fontdict = {'fontsize': xlabel2_fontsize})
    
    # Definindo o label para o eixo "y" do segundo gráfico:
    ax2.set_ylabel(ylabel = 'Percent', fontdict = {'fontsize': ylabel_fontsize})
       
    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y" do segundo gráfico:
    ax2.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    ax2.tick_params(axis='y', which='major', labelsize=ytick_labelsize)

    # Ajustando os espaçamentos (largura e altura) entre os gráficos nos subplots:
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura (Figure) criada:
    plt.close(fig)
    
    # Retorna um objeto DataFrame com os valores das contagens e proporções da variável categórica "feature":
    return (Temp_df)

# ***** Função para plotar as contagens e proporções de observações de cada categoria entre duas variáveis categóricas qualquer.
# Estamos criando uma tabela de referência cruzada para fazermos uma análise "bi-variada" entre duas variáveis qualitativas (categóricas):
#
def plot_crosstab_feature(x, y, plot_style='darkgrid', ticks_fontsize=12,
                          title=None, title_fontsize=16, title_share=False,
                          xlabel1=None, xlabel1_fontsize=13,
                          xlabel2=None, xlabel2_fontsize=13,
                          ylabel_fontsize=13,
                          columns_nr=2, rows_nr=1,
                          share_x=False, share_y=False,
                          width=17, height=6,
                          bar_width=0.5,
                          xtick_labelsize=None, ytick_labelsize=None,
                          wspace=None, hspace=None,
                          figure_id=None, figure_path=None,
                          figure_file_format="png", figure_file_extension=".png",
                          tight_layout=True, dpi=300
                          ):
    '''
    Input:
        "x": Variável categórica que será representada no eixo "x" dos gráficos.
        "y": Variável categórica que será representada no eixo "y" dos gráficos.
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "ticks_fontsize": Define o tamanho da fonte dos ticks para os gráficos.
        "title": Define o título dos gráficos.
        "title_fontsize": Define o tamanho da fonte do título para os gráficos.
        "title_share": Define se o título do gráfico 1 deverá ser compartilhado com o gráfico 2.
        "xlabel1": Define o label do eixo "x" para o gráfico 1.
        "xlabel1_fontsize": Define tamanho da fonte do label do eixo "x" para o gráfico 1.
        "xlabel2": Define o label do eixo "x" para o gráfico 2.
        "xlabel2_fontsize": Define tamanho da fonte do label do eixo "x" para o gráfico 2.
        "ylabel_fontsize": Define tamanho da fonte do label do eixo "y" para os gráficos 1 e 2.
        "colums_nr": Define se os gráficos serão colocados na horixontal (2), ou na vertical (1). 
        "rows_nr": Define se os gráficos serão colocados na horixontal (1), ou na vertical (2).
        "share_x": Define se os dois gráficos compartilharão o eixo "x".
        "share_y": Define se os dois gráficos compartilharão o eixo "y".
        "width": Define a largura da figura onde serão desenhados (plot) os dois gráficos.
        "heigth": Define a altura da figura onde serão desenhados (plot) os dois gráficos.
        "bar_width": Define a largura das barras dos gráficos.
        "xtick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "x".
        "ytick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "y".
        "wspace": Define o espaçamento (largura) entre os gráficos nos subplots.
        "hspace": Define o espaçamento (altura) entre os gráficos nos subplots.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output:
        Objeto DataFrame contendo a tabela de referência cruzada das "contagens" e "Proporções".
    '''
    # Código da função:
    
    # Cria uma tabela de referência cruzada para as variáveis "x" e "y":
    crosstab_counts = pd.crosstab(index=x, columns=y)
    
    # Normalizando para que a soma de cada linha seja igual a "1", ou seja, calculando as proporções de cada linha
    # da tabela:
    crosstab_pcts = crosstab_counts.div(crosstab_counts.sum(1), axis=0)
    
    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura e os dois eixos onde serão plotados os gráficos:
    fig, (ax1, ax2) = plt.subplots(nrows=rows_nr, ncols=columns_nr,
                                   sharex=share_x, sharey=share_y,
                                   figsize=(width,height))
    
    # Desenhando o gráfico das proporções no primeiro eixo (ax1):
    crosstab_pcts.plot.bar(fontsize=ticks_fontsize, width=bar_width, ax=ax1)
    
    # Definindo o título do primeiro gráfico:
    ax1.set_title(label = title, fontdict = {'fontsize': title_fontsize})
    
    # Definindo o label para o eixo "x" do primeiro gráfico:
    ax1.set_xlabel(xlabel = xlabel1, fontdict = {'fontsize': xlabel1_fontsize})
    
    # Definindo o label para o eixo "y" do primeiro gráfico:
    ax1.set_ylabel(ylabel = 'Percent', fontdict = {'fontsize': ylabel_fontsize})

    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y" do primeiro gráfico:
    ax1.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    ax1.tick_params(axis='y', which='major', labelsize=ytick_labelsize)
    
    # Desenhando o gráfico das contagens no segundo eixo (ax2):
    crosstab_counts.plot.bar(fontsize=ticks_fontsize, width=bar_width, ax=ax2)
    
    # Definindo o título do segundo gráfico.
    # Verifica se o segundo gráfico será desenhado na mesma coluna, e se o eixo "x" será compartilhado com o
    # primeiro gráfico:
    if((columns_nr==1) and (share_x==True)):
        # Não coloca o título no segundo gráfico:
        ax2.set_title(label = None)
    else:
        if(title_share):
            # Compartilha o título do gráfico 1 com o gráfico 2:
            ax2.set_title(label = None)
            
        else: # Não compartilha o título do gráfico 1 com o gráfico 2.
            # Coloca o título no segundo gráfico:
            ax2.set_title(label = title, fontdict = {'fontsize': title_fontsize})

    # Definindo o label para o eixo "x" do segundo gráfico:
    ax2.set_xlabel(xlabel = xlabel2, fontdict = {'fontsize': xlabel2_fontsize})
    
    # Definindo o label para o eixo "y" do segundo gráfico:
    ax2.set_ylabel(ylabel = 'Count', fontdict = {'fontsize': ylabel_fontsize})
    
    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y" do segundo gráfico:
    ax2.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    ax2.tick_params(axis='y', which='major', labelsize=ytick_labelsize)

    # Ajustando os espaçamentos (largura e altura) entre os gráficos nos subplots:
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura (Figure) criada:
    plt.close(fig)
    
    # Retorna um objeto DataFrame contendo a tabela de referência cruzada das "contagens" e "Proporções":
    return (pd.concat({'Count':crosstab_counts, 'Percent': round(number=crosstab_pcts, ndigits=2)}, axis=1))

# ***** Função para plotar distribuição de frequências de uma variável qualquer:
#
def plot_distribution(
    data=None, x=None, y=None, hue=None, row=None, col=None, weights=None, kind='hist', rug=False, rug_kws=None,
    log_scale=None, legend=True, palette=None, hue_order=None, hue_norm=None, color=None, col_wrap=None, row_order=None,
    col_order=None, height=5, aspect=1, facet_kws=None, kde=False, plot_style=None, title=None, title_fontsize=16,
    xlabel=None, xlabel_fontsize=13, ylabel=None, ylabel_fontsize=13, xtick_labelsize=None, ytick_labelsize=None,
    figure_id=None, figure_path=None, figure_file_format="png", figure_file_extension=".png", tight_layout=True, dpi=300
    ):
    '''
    Input:
        Lista de cores no site: https://matplotlib.org/stable/gallery/color/named_colors.html
        "data", "x", "y", "hue", "row", "col", "weights", "kind", "rug", "rug_kws", "log_scale", "legend", "palette",
        "hue_order", "hue_norm", "color", "col_wrap", "row_order", "col_order", "height", "aspect",
        "facet_kws": informações no site oficial em https://seaborn.pydata.org/generated/seaborn.displot.html#seaborn.displot.
        "kde": Adiciona ao histograma a curva kde.
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "title": Define o título do gráfico.
        "title_fontsize": Define o tamanho da fonte do título para o gráfico.
        "xlabel": Define o label do eixo "x" para o gráfico.
        "xlabel_fontsize": Define tamanho da fonte do label do eixo "x" para o gráfico.
        "ylabel": Define o label do eixo "y" para o gráfico.
        "ylabel_fontsize": Define tamanho da fonte do label do eixo "y" para o gráfico.
        "xtick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "x".
        "ytick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "y".
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
        
    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Plot:
    p = sns.displot(
        data=data, x=x, y=y, hue=hue, row=row, col=col, weights=weights, kind=kind, rug=rug, rug_kws=rug_kws, kde=kde,
        log_scale=log_scale, legend=legend, palette=palette, hue_order=hue_order, hue_norm=hue_norm, color=color,
        col_wrap=col_wrap, row_order=row_order, col_order=col_order, height=height, aspect=aspect, facet_kws=facet_kws
    )

    # Definindo o título:
    plt.title(label=title, fontdict={'fontsize': title_fontsize})

    # Definindo o label do eixo "x":
    p.set_xlabels(label=xlabel, fontdict={'fontsize': xlabel_fontsize})

    # Definindo o label do eixo "y":
    p.set_ylabels(label=ylabel, fontdict={'fontsize': ylabel_fontsize})

    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
    p.set_xticklabels(fontdict={'fontsize': xtick_labelsize})
    p.set_yticklabels(fontdict={'fontsize': xtick_labelsize})

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure(
            figure_id=figure_id, figure_path=figure_path,
            figure_file_format=figure_file_format,
            figure_file_extension=figure_file_extension,
            tight_layout=tight_layout, dpi=dpi
        )

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias do gráfico:
    plt.close()
    
    # Fim da função:
    return

# ***** Função para plotar um histograma e um gráfico de densidade duas variáveis numéricas.
# Está função é útil para verificarmos as distribuições de frequências de uma variável numérica, de acordo com as
# classes de uma variável categórica. Neste caso estamos considerando apenas "duas" classes para a variável categórica.
#
def plot_hist_kde_feature(data1, data2, bins=None,
                          plot_style=None,
                          title_hist=None, title_kde=None,title_fontsize=16,
                          legend_label1=None, legend_label2=None,
                          legend_fontsize=10,
                          xlabel=None, xlabel_fontsize=13,
                          color1=None, color2=None,
                          xtick_labelsize=None, ytick_labelsize=None,
                          width=17, height=6,
                          wspace=None, hspace=None,
                          figure_id=None, figure_path=None,
                          figure_file_format="png", figure_file_extension=".png",
                          tight_layout=True, dpi=300
                         ):
    '''
    Input:
        Lista de cores no site: https://matplotlib.org/stable/gallery/color/named_colors.html
        "data1": Variável numérica que será representada no eixo "x" do gráfico para a primeira classe (categoria).
        "data2": Variável numérica que será representada no eixo "x" do gráfico para a segunda classe (categoria).
        "bins": Define o número de bins para os histogramas.
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "title_hist": Define o título do gráfico de histograma.
        "title_kde": Define o título do gráfico de densidade (KDE).
        "title_fontsize": Define o tamanho da fonte do título para os dois gráficos.
        "legend_label1": Define o label da legenda para a primeira classe (categoria).
        "legend_label2": Define o label da legenda para a segunda classe (categoria).
        "legend_fontsize": Define o tamanho da fonte para a legenda nos dois gráficos.
        "xlabel": Define o label do eixo "x" para os dois gráficos.
        "xlabel_fontsize": Define tamanho da fonte do label do eixo "x" para os dois gráficos.
        "color1": Define a cor do gráfico para a primeira classe (categoria).
        "color2": Define a cor do gráfico para a segunda classe (categoria).
        "xtick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "x".
        "ytick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "y".
        "width": Define a largura da figura onde serão desenhados (plot) os dois gráficos.
        "heigth": Define a altura da figura onde serão desenhados (plot) os dois gráficos.
        "wspace": Define o espaçamento (largura) entre os gráficos nos subplots.
        "hspace": Define o espaçamento (altura) entre os gráficos nos subplots.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
    
    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura e os dois eixos onde serão plotados os gráficos:
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(width,height))
    
    # Desenhando os histogramas no primeiro eixo (ax1):
    sns.distplot(a=data1, bins=bins, color=color1, kde=False, label=legend_label1, ax=ax1)
    sns.distplot(a=data2,  bins=bins, color=color2, kde=False, label=legend_label2, ax=ax1)
    
    # Definindo o título do primeiro gráfico (Histogramas):
    ax1.set_title(label = title_hist, fontdict = {'fontsize': title_fontsize})
    
    # Definindo o label para o eixo "x" do primeiro gráfico (Histogramas):
    ax1.set_xlabel(xlabel = xlabel, fontdict = {'fontsize': xlabel_fontsize})
    
    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y" do primeiro gráfico (Histogramas):
    ax1.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    ax1.tick_params(axis='y', which='major', labelsize=ytick_labelsize)

    # Colocando a legenda no primeiro gráfico (Histogramas):
    ax1.legend(fontsize=legend_fontsize)
    
    # Desenhando os gráficos de densidade (KDE) no segundo eixo (ax2):
    sns.kdeplot(data=data1, label=legend_label1, color=color1, shade=True, ax=ax2)
    sns.kdeplot(data=data2, label=legend_label2, color=color2, shade=True, ax=ax2)
    
    # Definindo o título do segundo gráfico (KDE's):
    ax2.set_title(label = title_kde, fontdict = {'fontsize': title_fontsize})
    
    # Definindo o label para o eixo "x" do segundo gráfico (KDE's):
    ax2.set_xlabel(xlabel = xlabel, fontdict = {'fontsize': xlabel_fontsize})
    
    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y" do segundo gráfico (KDE's):
    ax2.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    ax2.tick_params(axis='y', which='major', labelsize=ytick_labelsize)

    # Colocando a legenda no segundo gráfico (KDE's):
    ax2.legend(fontsize=legend_fontsize)
    
    # Ajustando os espaçamentos (largura e altura) entre os gráficos nos subplots:
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura (Figure) criada:
    plt.close(fig)
    
    return

# ***** Função para plotar um "histograma" de uma variável numérica:
#
def plot_histogram(
    data=None, x=None, y=None, hue=None, weights=None, stat='count', bins='auto', binwidth=None, binrange=None,
    discrete=None, cumulative=False, common_bins=True, common_norm=True, multiple='layer', element='bars',
    fill=True, shrink=1, kde=False, kde_kws=None, line_kws=None, thresh=0, pthresh=None, pmax=None, cbar=False,
    cbar_ax=None, cbar_kws=None, palette=None, hue_order=None, hue_norm=None, log_scale=None, legend=True,
    plot_style=None, color=None, title=None, title_fontsize=16, xlabel=None, xlabel_fontsize=13,
    ylabel=None, ylabel_fontsize=13, xtick_labelsize=None, ytick_labelsize=None, width=8, height=6,
    figure_id=None, figure_path=None, figure_file_format="png", figure_file_extension=".png", tight_layout=True, dpi=300
    ):
    '''
    Input:
        Parâmetros da função "sns.histplot" no site: https://seaborn.pydata.org/generated/seaborn.histplot.html
        Lista de cores no site: https://matplotlib.org/stable/gallery/color/named_colors.html
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "title": título do gráfico.
        "title_fontsize": tamanho da fonte do título do gráfico.
        "xlabel": label do eixo "x".
        "xlabel_fontsize": tamanho da fonte do label do eixo "x".
        "ylabel": label do eixo "y".
        "ylabel_fontsize": tamanho da fonte do label do eixo "y".
        "xtick_labelsize": tamanho da fonte do label dos "ticks" do eixo "x".
        "ytick_labelsize": tamanho da fonte do label dos "ticks" do eixo "y".
        "width": largura da figura onde será desenhado (plot) o gráfico.
        "heigth": altura da figura onde será desenhado (plot) o gráfico.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
    
    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))

    # Plot do histograma:
    sns.histplot(
        data=data, x=x, y=feature, hue=hue, weights=weights, stat=stat, bins=bins, binwidth=binwidth,
        binrange=binrange, discrete=discrete, cumulative=cumulative, common_bins=common_bins, common_norm=common_norm,
        multiple=multiple, element=element, fill=fill, shrink=shrink, kde=kde, kde_kws=kde_kws, line_kws=line_kws,
        thresh=thresh, pthresh=pthresh, pmax=pmax, cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws, palette=palette,
        hue_order=hue_order, hue_norm=hue_norm, color=color, log_scale=log_scale, legend=legend
        )

    # Definindo o título:
    plt.title(label=title, size=title_fontsize)

    # Definindo o label do eixo "x":
    plt.xlabel(xlabel=xlabel, size=xlabel_fontsize)

    # Definindo o label do eixo "y":
    plt.ylabel(ylabel=ylabel, size=ylabel_fontsize)

    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
    plt.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    plt.tick_params(axis='y', which='major', labelsize=ytick_labelsize)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    # Fim da função:
    return

# ***** Função para plotar um "boxplot" de uma variável numérica:
#
def plot_boxplot(
    x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient=None, palette=None, saturation=0.75,
    width_boxplot=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5, color=None, plot_style=None, title=None,
    title_fontsize=16, xlabel=None, xlabel_fontsize=13, ylabel=None, ylabel_fontsize=13, xtick_labelsize=None,
    ytick_labelsize=None, width=8, height=6, figure_id=None, figure_path=None, figure_file_format="png",
    figure_file_extension=".png", tight_layout=True, dpi=300
    ):
    '''
    Input:
        Parâmetros da função "sns.boxplot" no site: https://seaborn.pydata.org/generated/seaborn.boxplot.html
        Lista de cores no site: https://matplotlib.org/stable/gallery/color/named_colors.html
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "title": título do gráfico.
        "title_fontsize": tamanho da fonte do título do gráfico.
        "xlabel": label do eixo "x".
        "xlabel_fontsize": tamanho da fonte do label do eixo "x".
        "ylabel": label do eixo "y".
        "ylabel_fontsize": tamanho da fonte do label do eixo "y".
        "xtick_labelsize": tamanho da fonte do label dos "ticks" do eixo "x".
        "ytick_labelsize": tamanho da fonte do label dos "ticks" do eixo "y".
        "width": largura da figura onde será desenhado (plot) o gráfico.
        "heigth": altura da figura onde será desenhado (plot) o gráfico.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
    
    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))

    # Plot
    sns.boxplot(
        x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order, orient=orient,
        color=color, palette=palette, saturation=saturation, width=width_boxplot,
        dodge=dodge, fliersize=fliersize, linewidth=linewidth, whis=whis
        )

    # Definindo o título:
    plt.title(label=title, size=title_fontsize)

    # Definindo o label do eixo "x":
    plt.xlabel(xlabel=xlabel, size=xlabel_fontsize)

    # Definindo o label do eixo "y":
    plt.ylabel(ylabel=ylabel, size=ylabel_fontsize)

    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
    plt.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    plt.tick_params(axis='y', which='major', labelsize=ytick_labelsize)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    # Fim da função:
    return

# ***** Função para construir na mesma figura um "histograma" e um "boxplot" de um variável numérica.
#
def plot_histogram_and_boxplot(
    data=None, x=None, y=None, hue=None, weights=None, stat='count', bins='auto', binwidth=None, binrange=None,
    discrete=None, cumulative=False, common_bins=True, common_norm=True, multiple='layer', element='bars',
    fill=True, shrink=1, kde=False, kde_kws=None, line_kws=None, thresh=0, pthresh=None, pmax=None, cbar=False,
    cbar_ax=None, cbar_kws=None, palette=None, hue_order=None, hue_norm=None, color=None, log_scale=None, legend=True,
    order=None, orient=None, saturation=0.75, width_boxplot=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5,
    plot_style=None, plot_orient='v', titles=None, titles_fontsize=16, xlabels=None, xlabels_fontsize=13, ylabels=None,
    ylabels_fontsize=13, xtick_labelsize=None, ytick_labelsize=None, width=8, height=6, wspace=None, hspace=None,
    figure_id=None, figure_path=None, figure_file_format="png", figure_file_extension=".png", tight_layout=True, dpi=300
    ):
    '''
    Input:
        Parâmetros da função "sns.histplot" no site: https://seaborn.pydata.org/generated/seaborn.histplot.html
        Parâmetros da função "sns.boxplot" no site: https://seaborn.pydata.org/generated/seaborn.boxplot.html
        Lista de cores no site: https://matplotlib.org/stable/gallery/color/named_colors.html
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "cols_nr": Define o número de colunas dos subplots.
        "rows_nr": Define o número de linhas dos subplots.
        "titles": lista para definir os títulos para os dois gráfico.
        "titles_fontsize": tamanho da fonte do título para os dois gráficos.
        "xlabels": lista para definir os labels do eixo "x" para os dois gráficos.
        "xlabels_fontsize": tamanho da fonte dos labels do eixo "x" para os dois gráficos.
        "ylabels": lista para definir os labels do eixo "y" para os dois gráficos.
        "ylabels_fontsize": tamanho da fonte dos labels do eixo "y" para os dois gráficos.
        "xtick_labelsize": tamanho da fonte do label do "tick" do eixo "x" para os dois gráficos.
        "ytick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "y" para os dois gráficos.
        "width": largura da figura onde serão desenhados (plot) os dois gráficos.
        "heigth": altura da figura onde serão desenhados (plot) os dois gráficos.
        "wspace": Define o espaçamento (largura) entre os gráficos nos subplots.
        "hspace": Define o espaçamento (altura) entre os gráficos nos subplots.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:

    # Verificando se os dois gráficos serão construidos na vertical ou na horizontal:
    if(plot_orient == 'v'):
        # Vertical:
        cols_nr = 1
        rows_nr = 2
    else:
        if(plot_orient == 'h'):
            # Horizontal:
            cols_nr = 2
            rows_nr = 1
        else:
            # Erro na definição do parâmetro "plot_orient":
            print("Error: need to set the 'plot_orient' parameter correctly: 'v' or 'h'!")
            return

    # Verificando (check) e definindo os valores para alguns parâmetros:
    #
    # titles:
    titles_names = dslib.check_params(features=features, params=titles)
    # Verifica se o tamanho (size) de "titles" x "features" são diferentes:
    if (titles_names == None):
        # Erro:
        print("Error: size of 'titles' and 'features' are different!")
        return

    # xlabels:
    xlabels_names = dslib.check_params(features=features, params=xlabels)
    # Verifica se o tamanho (size) de "xlabels" x "features" são diferentes:
    if (xlabels_names == None):
        # Erro:
        print("Error: size of 'xlabels' and 'features' are different!")
        return
    
    # ylabels:
    ylabels_names = dslib.check_params(features=features, params=ylabels)
    # Verifica se o tamanho (size) de "ylabels" x "features" são diferentes:
    if (ylabels_names == None):
        # Erro:
        print("Error: size of 'ylabels' and 'features' are different!")
        return

    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')
    
    # Definindo a figura e os dois eixos onde serão plotados os gráficos:
    fig, (ax1, ax2) = plt.subplots(nrows=rows_nr, ncols=cols_nr, figsize=(width,height))
    
    # Plot do "histograma" no primeiro eixo (ax1):
    sns.histplot(
        data=data, x=x, y=y, hue=hue, weights=weights, stat=stat, bins=bins, binwidth=binwidth,
        binrange=binrange, discrete=discrete, cumulative=cumulative, common_bins=common_bins, common_norm=common_norm,
        multiple=multiple, element=element, fill=fill, shrink=shrink, kde=kde, kde_kws=kde_kws, line_kws=line_kws,
        thresh=thresh, pthresh=pthresh, pmax=pmax, cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws, palette=palette,
        hue_order=hue_order, hue_norm=hue_norm, color=color, log_scale=log_scale, legend=legend, ax=ax1
        )
    
    # Definindo o título do primeiro gráfico (Histograma):
    ax1.set_title(label = titles_names[0], fontdict = {'fontsize': titles_fontsize})
    
    # Definindo o label para o eixo "x" do primeiro gráfico (Histograma):
    ax1.set_xlabel(xlabel = xlabels_names[0], fontdict = {'fontsize': xlabels_fontsize})

    # Definindo o label para o eixo "y" do primeiro gráfico (Histograma):
    ax1.set_ylabel(ylabel = ylabels_names[0], fontdict = {'fontsize': ylabels_fontsize})
    
    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y" do primeiro gráfico (Histograma):
    ax1.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    ax1.tick_params(axis='y', which='major', labelsize=ytick_labelsize)

    # Plot do "boxplot" no segundo eixo (ax2):
    sns.boxplot(
        x=feature, y=y, hue=hue, data=data, order=order, hue_order=hue_order, orient=orient, color=color,
        palette=palette, saturation=saturation, width=width_boxplot, dodge=dodge, fliersize=fliersize,
        linewidth=linewidth, whis=whis, ax=ax2
        )

    # Definindo o título do segundo gráfico (BoxPlot):
    ax2.set_title(label = titles_names[1], fontdict = {'fontsize': titles_fontsize})
    
    # Definindo o label para o eixo "x" do segundo gráfico (BoxPlot):
    ax2.set_xlabel(xlabel = xlabels_names[1], fontdict = {'fontsize': xlabels_fontsize})
    
    # Definindo o label para o eixo "y" do segundo gráfico (BoxPlot):
    ax2.set_ylabel(ylabel = ylabels_names[1], fontdict = {'fontsize': ylabels_fontsize})
    
    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y" do segundo gráfico (BoxPlot):
    ax2.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    ax2.tick_params(axis='y', which='major', labelsize=ytick_labelsize)

    # Ajustando os espaçamentos (largura e altura) entre os gráficos nos subplots:
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura (Figure) criada:
    plt.close(fig)
    
    # Fim da função:
    return

# ***** Função para plotar dois ou mais "histogramas" de acordo com as variáveis definidas em "x" ou "y".
#
def plot_histograms(
    data=None, x=None, y=None, hue=None, weights=None, stat='count', bins='auto', binwidth=None, binrange=None,
    discrete=None, cumulative=False, common_bins=True, common_norm=True, multiple='layer', element='bars',
    fill=True, shrink=1, kde=False, kde_kws=None, line_kws=None, thresh=0, pthresh=None, pmax=None, cbar=False,
    cbar_ax=None, cbar_kws=None, palette=None, hue_order=None, hue_norm=None, log_scale=None, legend=True,
    plot_style=None, color=None, titles=None, titles_fontsize=16, xlabels=None, xlabels_fontsize=13,
    ylabels=None, ylabels_fontsize=13, xtick_labelsize=None, ytick_labelsize=None, cols_nr=2, rows_nr=2,
    width=8, height=6, wspace=None, hspace=None, figure_id=None, figure_path=None, figure_file_format="png",
    figure_file_extension=".png", tight_layout=True, dpi=300
    ):
    '''
    Input:
        Informações dos principais parâmetros no site: https://seaborn.pydata.org/generated/seaborn.histplot.html
        Lista de cores no site: https://matplotlib.org/stable/gallery/color/named_colors.html
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "titles": Lista com o título de cada histograma.
        "titles_fontsize": Lista com os tamanhos das fontes dos títulos para cada histograma.
        "xlabels": Lista com os labels do eixo "x" para cada histograma.
        "xlabels_fontsize": Define tamanho da fonte dos labels do eixo "x" para todos os histogramas.
        "ylabels": Lista com os labels do eixo "y" para cada histograma.
        "ylabels_fontsize": Define tamanho da fonte dos labels do eixo "y" para todos os histogramas.
        "xtick_labelsize": Define o tamanho da fonte dos labels dos "ticks" do eixo "x" para todos os histogramas.
        "ytick_labelsize": Define o tamanho da fonte dos labels dos "ticks" do eixo "y" para todos os histogramas.
        "colums_nr": Define o número de colunas para a plotagem dos histogramas.
        "rows_nr": Define o número de linhas para a plotagem dos histogramas.
        "width": Define a largura da figura onde serão desenhados (plot) todos os histogramas.
        "heigth": Define a altura da figura onde serão desenhados (plot) todos os histogramas.
        "wspace": Define o espaçamento (largura) entre os gráficos nos subplots.
        "hspace": Define o espaçamento (altura) entre os gráficos nos subplots.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:

    # Verificando se as variáveis foram definidas em "x" ou "y":
    if (x != None):
        features = x
    else:
        if (y != None):
            features = y
        else:
            # Erro: as variáveis não foram definidas.
            print("Error: the features have not been defined!")
            return

    # Verificando "Total de Subplots" x "Total de Variáveis":
    if(len(features) > (cols_nr * rows_nr)):
        # A quantidade de variáveis que deveremos plotar é maior que o número total de subplots definido.
        # Sendo assim, enviamos uma mensagem de erro e terminamos a função.
        print("***** Erro: a quantidade total de variáveis é MAIOR que o total de subplots (cols_nr * rows_nr) *****")
        print("Redimensionar os parâmetros 'cols_nr' e 'rows_nr', ou diminuir a quantidade de variáveis!")
        return
    
    # Verificando (check) e definindo os valores para alguns parâmetros:
    #
    # bins
    bins_values = dslib.check_params(features=features, params=bins)
    # Verifica se o tamanho (size) de "bins" x "features" são diferentes:
    if (bins_values == None):
        # Erro:
        print("Error: size of 'bins' and 'features' are different!")
        return

    # titles:
    titles_names = dslib.check_params(features=features, params=titles)
    # Verifica se o tamanho (size) de "titles" x "features" são diferentes:
    if (titles_names == None):
        # Erro:
        print("Error: size of 'titles' and 'features' are different!")
        return

    # xlabels:
    xlabels_names = dslib.check_params(features=features, params=xlabels)
    # Verifica se o tamanho (size) de "xlabels" x "features" são diferentes:
    if (xlabels_names == None):
        # Erro:
        print("Error: size of 'xlabels' and 'features' are different!")
        return
    
    # ylabels:
    ylabels_names = dslib.check_params(features=features, params=ylabels)
    # Verifica se o tamanho (size) de "ylabels" x "features" são diferentes:
    if (ylabels_names == None):
        # Erro:
        print("Error: size of 'ylabels' and 'features' are different!")
        return

    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))

    # "Loop for" para plotarmos os gráficos:
    for i, feature in enumerate(features):
        # Cria o subplot:
        ax = plt.subplot(rows_nr, cols_nr, i+1)
        
        # Verifica se "x" foi definido:
        if(x != None):
            # A lista de variáveis está em "x":
            xi = feature
            yi = y
        else:
            # A lista de variáveis está em "y":
            xi = x
            yi = feature
        
        # Plot do gráfico:
        sns.histplot(
            data=data, x=xi, y=yi, hue=hue, weights=weights, stat=stat, bins=bins_values[i],
            binwidth=binwidth, binrange=binrange, discrete=discrete, cumulative=cumulative,
            common_bins=common_bins, common_norm=common_norm, multiple=multiple, element=element,
            fill=fill, shrink=shrink, kde=kde, kde_kws=kde_kws, line_kws=line_kws,
            thresh=thresh, pthresh=pthresh, pmax=pmax, cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws,
            palette=palette, hue_order=hue_order, hue_norm=hue_norm, color=color,
            log_scale=log_scale, legend=legend, ax=ax
            )

        # Definindo o título:
        ax.set_title(label = titles_names[i], fontdict = {'fontsize': titles_fontsize})

        # Definindo o label do eixo "x":
        ax.set_xlabel(xlabel = xlabels_names[i], fontdict = {'fontsize': xlabels_fontsize})

        # Definindo o label do eixo "y":
        ax.set_ylabel(ylabel = ylabels_names[i], fontdict = {'fontsize': ylabels_fontsize})
        
        # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
        plt.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
        plt.tick_params(axis='y', which='major', labelsize=ytick_labelsize)
   
    # Ajustando os espaçamentos entre os gráficos nos subplots:
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando os gráficos:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    return

# ***** Função para ajustar (Fit) e plotar (Plot) uma "Estimativa da Densidade de kernel" (KDE) univariada ou bivariada.
#
def plot_kde_features(df, features, data2=None, shade=True, vertical=False, kernel='gau', bw='scott', gridsize=100,
                      cut=3, clip=None, legend=False, cumulative=False, shade_lowest=True, cbar=False, cbar_ax=None,
                      cbar_kws=None,
                      plot_style=None, colors=None, titles=None, titles_fontsize=16, xlabels=None, xlabels_fontsize=13,
                      ylabels=None, ylabels_fontsize=13, xtick_labelsize=None, ytick_labelsize=None, cols_nr=2, rows_nr=2,
                      width=8, height=6, wspace=None, hspace=None, figure_id=None, figure_path=None, figure_file_format="png",
                      figure_file_extension=".png", tight_layout=True, dpi=300
                     ):
    '''
    Input:
        "df": DataFrame com as variáveis que serão utilizadas.
        "features": Lista das variáveis numéricas que serão utilizadas nos plots.
        "data2": Second input data. If present, a bivariate KDE will be estimated.
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "colors": Lista com as cores de cada KDE de cada variável (https://matplotlib.org/stable/gallery/color/named_colors.html).
        "titles": Lista com o título de cada KDE.
        "titles_fontsize": Lista com os tamanhos das fontes dos títulos para cada KDE.
        "xlabels": Lista com os labels do eixo "x" para cada KDE.
        "xlabels_fontsize": Define tamanho da fonte dos labels do eixo "x" para todos os KDEs.
        "ylabels": Lista com os labels do eixo "y" para cada KDE.
        "ylabels_fontsize": Define tamanho da fonte dos labels do eixo "y" para todos os KDEs.
        "xtick_labelsize": Define o tamanho da fonte dos labels dos "ticks" do eixo "x" para todos os KDEs.
        "ytick_labelsize": Define o tamanho da fonte dos labels dos "ticks" do eixo "y" para todos os KDEs.
        "colums_nr": Define o número de colunas para a plotagem dos KDEs.
        "rows_nr": Define o número de linhas para a plotagem dos KDEs.
        "width": Define a largura da figura onde serão desenhados (plot) todos os KDEs.
        "heigth": Define a altura da figura onde serão desenhados (plot) todos os KDEs.
        "wspace": Define o espaçamento (largura) entre os gráficos nos subplots.
        "hspace": Define o espaçamento (altura) entre os gráficos nos subplots.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
        
    # Verificando "Total de Subplots" x "Total de Variáveis":
    if(len(features) > (cols_nr * rows_nr)):
        # A quantidade de variáveis que deveremos plotar é maior que o número total de subplots definido.
        # Sendo assim, enviamos uma mensagem de erro e terminamos a função.
        print("***** ERROR: Quantidade total de variáveis é MAIOR que o total de subplots (cols_nr * rows_nr) *****")
        print("Redimensionar os parâmetros 'cols_nr' e 'rows_nr', ou diminuir a quantidade de variáveis!")
        return
    
    # Verificando (check) e definindo os valores para alguns parâmetros:
    #
    # gridsize
    gridsize_values = dslib.check_params(features=features, params=gridsize)
    # Verifica se o tamanho (size) de "bins" x "features" são diferentes:
    if (gridsize_values == None):
        # Erro:
        print("Error: Size of 'gridsize' and 'features' are different!")
        return

    # colors:
    colors_values = dslib.check_params(features=features, params=colors)
    # Verifica se o tamanho (size) de "colors" x "features" são diferentes:
    if (colors_values == None):
        # Erro:
        print("Error: Size of 'colors' and 'features' are different!")
        return

    # titles:
    titles_names = dslib.check_params(features=features, params=titles)
    # Verifica se o tamanho (size) de "titles" x "features" são diferentes:
    if (titles_names == None):
        # Erro:
        print("Error: Size of 'titles' and 'features' are different!")
        return

    # xlabels:
    xlabels_names = dslib.check_params(features=features, params=xlabels)
    # Verifica se o tamanho (size) de "xlabels" x "features" são diferentes:
    if (xlabels_names == None):
        # Erro:
        print("Error: Size of 'xlabels' and 'features' are different!")
        return
    
    # xlabels:
    ylabels_names = dslib.check_params(features=features, params=ylabels)
    # Verifica se o tamanho (size) de "ylabels" x "features" são diferentes:
    if (ylabels_names == None):
        # Erro:
        print("Error: Size of 'ylabels' and 'features' are different!")
        return

    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))

    # "Loop for" para plotarmos os gráficos:
    for i, feature in enumerate(features):
        # Cria o subplot:
        ax = plt.subplot(rows_nr, cols_nr, i+1)
        
        # Plot do gráfico:
        sns.kdeplot(data=df[feature], gridsize=gridsize_values[i], color=colors_values[i], shade=shade, vertical=vertical,
                    kernel=kernel, bw=bw, cut=cut, clip=clip, legend=legend, cumulative=cumulative, shade_lowest=shade_lowest,
                    cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws, ax=ax)

        # Definindo o título:
        ax.set_title(label = titles_names[i], fontdict = {'fontsize': titles_fontsize})

        # Definindo o label do eixo "x":
        ax.set_xlabel(xlabel = xlabels_names[i], fontdict = {'fontsize': xlabels_fontsize})

        # Definindo o label do eixo "y":
        ax.set_ylabel(ylabel = ylabels_names[i], fontdict = {'fontsize': ylabels_fontsize})
        
        # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
        plt.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
        plt.tick_params(axis='y', which='major', labelsize=ytick_labelsize)
   
    # Ajustando os espaçamentos entre os gráficos nos subplots:
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando os gráficos:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    return

# ***** Função para plotar dois ou mais "boxplots" de acordo com as variáveis definidas em "x" ou "y".
#
def plot_boxplots(
    x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient=None, palette=None, saturation=0.75,
    width_boxplot=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5, colors=None, plot_style=None, titles=None,
    titles_fontsize=16, xlabels=None, xlabels_fontsize=13, ylabels=None, ylabels_fontsize=13, xtick_labelsize=None,
    ytick_labelsize=None, cols_nr=2, rows_nr=2, width=8, height=6, wspace=None, hspace=None, figure_id=None,
    figure_path=None, figure_file_format="png", figure_file_extension=".png", tight_layout=True, dpi=300
    ):
    '''
    Input:
        Informações dos principais parâmetros no site: https://seaborn.pydata.org/generated/seaborn.boxplot.html
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "colors": Lista com as cores de cada boxplot de cada variável (https://matplotlib.org/stable/gallery/color/named_colors.html).
        "titles": Lista com o título de cada boxplot.
        "titles_fontsize": Lista com os tamanhos das fontes dos títulos para cada boxplot.
        "xlabels": Lista com os labels do eixo "x" para cada boxplot.
        "xlabels_fontsize": Define tamanho da fonte dos labels do eixo "x" para todos os boxplots.
        "ylabels": Lista com os labels do eixo "y" para cada boxplot.
        "ylabels_fontsize": Define tamanho da fonte dos labels do eixo "y" para todos os boxplots.
        "xtick_labelsize": Define o tamanho da fonte dos labels dos "ticks" do eixo "x" para todos os boxplots.
        "ytick_labelsize": Define o tamanho da fonte dos labels dos "ticks" do eixo "y" para todos os boxplots.
        "colums_nr": Define o número de colunas para a plotagem dos boxplots.
        "rows_nr": Define o número de linhas para a plotagem dos boxplots.
        "width": Define a largura da figura onde serão desenhados (plot) todos os boxplots.
        "heigth": Define a altura da figura onde serão desenhados (plot) todos os boxplots.
        "wspace": Define o espaçamento (largura) entre os gráficos nos subplots.
        "hspace": Define o espaçamento (altura) entre os gráficos nos subplots.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
    
    # Verificando se as variáveis foram definidas em "x" ou "y":
    if ((x == None) and (y == None)):
        # Erro: as variáveis não foram definidas.
        print("Error: the features have not been defined!")
        return

    # Verificando se "x" foi definido:
    if (x != None):
        # Verificando se a lista de variáveis está em "x":
        if(type(x) == list):
            # A lista de variáveis está em "x":
            features = x
            # Seta o flag como "True":
            feats_in_x = True
        else:
            # A lista não está em "x".
            # Então, seta o flag como "False":
            feats_in_x = False
    
    # Verificando se "y" foi definido:
    if (y != None):
        # Verificando se a lista de variáveis está em "y":
        if(type(y) == list):
            # A lista de variáveis está em "y":
            features = y
            # Seta o flag como "True":
            feats_in_y = True
        else:
            # A lista não está em "y".
            # Então, seta o flag como "False":
            feats_in_y = False

    # Verificando "Total de Subplots" x "Total de Variáveis":
    if(len(features) > (cols_nr * rows_nr)):
        # A quantidade de variáveis que deveremos plotar é maior que o número total de subplots definido.
        # Sendo assim, enviamos uma mensagem de erro e terminamos a função.
        print("***** Erro: a quantidade total de variáveis é MAIOR que o total de subplots (cols_nr * rows_nr) *****")
        print("Redimensionar os parâmetros 'cols_nr' e 'rows_nr', ou diminuir a quantidade de variáveis!")
        return
    
    # Verificando (check) e definindo os valores para alguns parâmetros:
    #
    # colors:
    colors_values = dslib.check_params(features=features, params=colors)
    # Verifica se o tamanho (size) de "colors" x "features" são diferentes:
    if (colors_values == None):
        # Erro:
        print("Error: size of 'colors' and 'features' are different!")
        return

    # titles:
    titles_names = dslib.check_params(features=features, params=titles)
    # Verifica se o tamanho (size) de "titles" x "features" são diferentes:
    if (titles_names == None):
        # Erro:
        print("Error: size of 'titles' and 'features' are different!")
        return

    # xlabels:
    xlabels_names = dslib.check_params(features=features, params=xlabels)
    # Verifica se o tamanho (size) de "xlabels" x "features" são diferentes:
    if (xlabels_names == None):
        # Erro:
        print("Error: size of 'xlabels' and 'features' are different!")
        return
    
    # ylabels:
    ylabels_names = dslib.check_params(features=features, params=ylabels)
    # Verifica se o tamanho (size) de "ylabels" x "features" são diferentes:
    if (ylabels_names == None):
        # Erro:
        print("Error: size of 'ylabels' and 'features' are different!")
        return

    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))

    # "Loop for" para plotarmos os gráficos:
    for i, feature in enumerate(features):
        # Cria o subplot:
        ax = plt.subplot(rows_nr, cols_nr, i+1)
        
        # Verifica se "x" e "y" foram definidos:
        if((x != None) and (y != None)):
            # Verifica se a lista de features está em "x":
            if(feats_in_x):
                xi = feature
            else:
                xi = x

            # Verifica se a lista de features está em "y":
            if(feats_in_y):
                yi = feature
            else:
                yi = y

        else:
            # Verifica se "x" foi definido:
            if(x != None):
                # A lista de variáveis está em "x":
                xi = feature
                yi = y
            else:
                # A lista de variáveis está em "y":
                xi = x
                yi = feature

        # Plot do gráfico:
        sns.boxplot(
            x=xi, y=yi, hue=hue, data=data, order=order, hue_order=hue_order, orient=orient,
            color=colors_values[i], palette=palette, saturation=saturation, width=width_boxplot,
            dodge=dodge, fliersize=fliersize, linewidth=linewidth, whis=whis, ax=ax
            )

        # Definindo o título:
        ax.set_title(label = titles_names[i], fontdict = {'fontsize': titles_fontsize})

        # Definindo o label do eixo "x":
        ax.set_xlabel(xlabel = xlabels_names[i], fontdict = {'fontsize': xlabels_fontsize})

        # Definindo o label do eixo "y":
        ax.set_ylabel(ylabel = ylabels_names[i], fontdict = {'fontsize': ylabels_fontsize})
        
        # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
        plt.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
        plt.tick_params(axis='y', which='major', labelsize=ytick_labelsize)
   
    # Ajustando os espaçamentos entre os gráficos nos subplots:
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando os gráficos:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    return

# ***** Função para plotar a contagem ou proporção (%) de dois ou mais gráficos de acordo com as variáveis definidas em "x" ou "y".
#
def plot_countplots(
    x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient=None, color=None, palette=None,
    saturation=0.75, dodge=True, percent=False, output=True, plot_style=None, titles=None, titles_fontsize=16,
    xlabels=None, xlabels_fontsize=13, ylabels=None, ylabels_fontsize=13, xtick_labelsize=None, ytick_labelsize=None,
    cols_nr=2, rows_nr=2, width=8, height=6, wspace=None, hspace=None, figure_id=None, figure_path=None,
    figure_file_format="png", figure_file_extension=".png", tight_layout=True, dpi=300
    ):
    '''
    Input:
        Informações dos principais parâmetros no site: https://seaborn.pydata.org/generated/seaborn.countplot.html
        "percent": se True, calculará as contagens em porcentagens;
        "output": se True, retornará os valores calculados;
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "titles": Lista com o título de cada gráfico.
        "titles_fontsize": Lista com os tamanhos das fontes dos títulos para cada gráfico.
        "xlabels": Lista com os labels do eixo "x" para cada gráfico.
        "xlabels_fontsize": Define tamanho da fonte dos labels do eixo "x" para todos os gráficos.
        "ylabels": Lista com os labels do eixo "y" para cada gráfico.
        "ylabels_fontsize": Define tamanho da fonte dos labels do eixo "y" para todos os gráficos.
        "xtick_labelsize": Define o tamanho da fonte dos labels dos "ticks" do eixo "x" para todos os gráficos.
        "ytick_labelsize": Define o tamanho da fonte dos labels dos "ticks" do eixo "y" para todos os gráficos.
        "colums_nr": Define o número de colunas para a plotagem dos gráficos.
        "rows_nr": Define o número de linhas para a plotagem dos gráficos.
        "width": Define a largura da figura onde serão desenhados (plot) todos os gráficos.
        "heigth": Define a altura da figura onde serão desenhados (plot) todos os gráficos.
        "wspace": Define o espaçamento (largura) entre os gráficos nos subplots.
        "hspace": Define o espaçamento (altura) entre os gráficos nos subplots.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: se o parâmetro "output=True" (default), retornará o dataframe com os valores calculados.
    '''
    # Código da função:

    # Verificando se as variáveis foram definidas em "x" ou "y":
    if (x != None):
        features = x
    else:
        if (y != None):
            features = y
        else:
            # Erro: as variáveis não foram definidas.
            print("Error: the features have not been defined!")
            return

    # Verificando "Total de Subplots" x "Total de Variáveis":
    if(len(features) > (cols_nr * rows_nr)):
        # A quantidade de variáveis que deveremos plotar é maior que o número total de subplots definido.
        # Sendo assim, enviamos uma mensagem de erro e terminamos a função.
        print("***** Erro: a quantidade total de variáveis é MAIOR que o total de subplots (cols_nr * rows_nr) *****")
        print("Redimensionar os parâmetros 'cols_nr' e 'rows_nr', ou diminuir a quantidade de variáveis!")
        return
    
    # Verificando (check) e definindo os valores para alguns parâmetros:
    #
    # titles:
    titles_names = dslib.check_params(features=features, params=titles)
    # Verifica se o tamanho (size) de "titles" x "features" são diferentes:
    if (titles_names == None):
        # Erro:
        print("Error: size of 'titles' and 'features' are different!")
        return

    # xlabels:
    xlabels_names = dslib.check_params(features=features, params=xlabels)
    # Verifica se o tamanho (size) de "xlabels" x "features" são diferentes:
    if (xlabels_names == None):
        # Erro:
        print("Error: size of 'xlabels' and 'features' are different!")
        return
    
    # ylabels:
    ylabels_names = dslib.check_params(features=features, params=ylabels)
    # Verifica se o tamanho (size) de "ylabels" x "features" são diferentes:
    if (ylabels_names == None):
        # Erro:
        print("Error: size of 'ylabels' and 'features' are different!")
        return

    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))

    # Calcula as contagens, armazenando os resultados em um dataframe:
    df = dslib.groupby_count(data=data, variables=features, percent=percent)
    
    # "Loop for" para plotarmos os gráficos:
    for i, feature in enumerate(features):
        # Cria o subplot:
        ax = plt.subplot(rows_nr, cols_nr, i+1)
        
        # Plot do gráfico:
        if(x != None):
            sns.barplot(
                x=list(df.index), y=feature, hue=hue, data=df, order=order, hue_order=hue_order, orient=orient,
                color=color, palette=palette, saturation=saturation, dodge=dodge, ax=ax
                )
        else:
            sns.barplot(
                x=feature, y=list(df.index), hue=hue, data=df, order=order, hue_order=hue_order, orient=orient,
                color=color, palette=palette, saturation=saturation, dodge=dodge, ax=ax
                )

        # Definindo o título:
        ax.set_title(label = titles_names[i], fontdict = {'fontsize': titles_fontsize})

        # Definindo o label do eixo "x":
        ax.set_xlabel(xlabel = xlabels_names[i], fontdict = {'fontsize': xlabels_fontsize})

        # Definindo o label do eixo "y":
        ax.set_ylabel(ylabel = ylabels_names[i], fontdict = {'fontsize': ylabels_fontsize})
        
        # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
        plt.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
        plt.tick_params(axis='y', which='major', labelsize=ytick_labelsize)
   
    # Ajustando os espaçamentos entre os gráficos nos subplots:
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando os gráficos:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    # Se "output=True", retorna os resultados:
    if(output):
        return df
    else:
        return

# ***** Função para plotar dois ou mais gráficos de barras de acordo com as variáveis definidas em "x", "y" e "hue".
#
def plot_barplot(
    x=None, y=None, hue=None, data=None, order=None, hue_order=None, estimator=None, ci=95, n_boot=1000, units=None,
    seed=None, orient=None, color=None, palette=None, saturation=0.75, errcolor='.26', errwidth=None, capsize=None,
    dodge=True, plot_style=None, title=None, title_fontsize=16, xlabel=None, xlabel_fontsize=13, ylabel=None,
    ylabel_fontsize=13, xtick_labelsize=None, ytick_labelsize=None, width=8, height=6, figure_id=None, figure_path=None,
    figure_file_format="png", figure_file_extension=".png", tight_layout=True, dpi=300
    ):
    '''
    Input:
        Informações dos principais parâmetros no site: https://seaborn.pydata.org/generated/seaborn.barplot.html
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "title": título do gráfico.
        "title_fontsize": tamanho da fonte do título do gráfico.
        "xlabel": label do eixo "x" do gráfico.
        "xlabel_fontsize": tamanho da fonte do label do eixo "x".
        "ylabel": label do eixo "y" do gráfico.
        "ylabel_fontsize": tamanho da fonte do label do eixo "y".
        "xtick_labelsize": tamanho da fonte dos labels dos "ticks" do eixo "x".
        "ytick_labelsize": tamanho da fonte dos labels dos "ticks" do eixo "y".
        "width": Define a largura da figura onde serão desenhados (plot) todos os gráficos.
        "heigth": Define a altura da figura onde serão desenhados (plot) todos os gráficos.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None.
    '''
    # Código da função:

    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))
    
    # Verifica se foi definido algum estimador:
    if(estimator != None):
        # Foi definido um estimador:
        est = estimator
    else:
        # Não foi definido um estimador. Neste caso, utilizaremos
        # o estimador padrão: "mean" (média).
        est = np.mean

    # Plot do gráfico:
    sns.barplot(
        x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order, estimator=est, ci=ci,
        n_boot=n_boot, units=units, seed=seed, orient=orient, color=color, palette=palette,
        saturation=saturation, errcolor=errcolor, errwidth=errwidth, capsize=capsize, dodge=dodge
        )

    # Definindo o título:
    plt.title(label=title, size=title_fontsize)

    # Definindo o label do eixo "x":
    plt.xlabel(xlabel=xlabel, size=xlabel_fontsize)

    # Definindo o label do eixo "y":
    plt.ylabel(ylabel=ylabel, size=ylabel_fontsize)

    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
    plt.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    plt.tick_params(axis='y', which='major', labelsize=ytick_labelsize)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    # Fim da função:
    return

# ***** Função para plotar dois ou mais "barplots" de acordo com as variáveis definidas em "x" ou "y".
#
def plot_barplots(
    x=None, y=None, hue=None, data=None, order=None, hue_order=None, estimator=None, ci=95, n_boot=1000, units=None,
    seed=None, orient=None, colors=None, palette=None, saturation=0.75, errcolor='.26', errwidth=None, capsize=None,
    dodge=True, plot_style=None, titles=None, titles_fontsize=16, xlabels=None, xlabels_fontsize=13, ylabels=None,
    ylabels_fontsize=13, xtick_labelsize=None, ytick_labelsize=None, cols_nr=1, rows_nr=1, width=8, height=6, wspace=None, hspace=None,
    figure_id=None, figure_path=None, figure_file_format="png", figure_file_extension=".png", tight_layout=True, dpi=300
    ):
    '''
    Input:
        Informações dos principais parâmetros no site: https://seaborn.pydata.org/generated/seaborn.barplot.html
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "colors": Lista com as cores de cada barplot de cada variável (https://matplotlib.org/stable/gallery/color/named_colors.html).
        "titles": Lista com o título de cada barplot.
        "titles_fontsize": Lista com os tamanhos das fontes dos títulos para cada barplot.
        "xlabels": Lista com os labels do eixo "x" para cada barplot.
        "xlabels_fontsize": Define tamanho da fonte dos labels do eixo "x" para todos os barplots.
        "ylabels": Lista com os labels do eixo "y" para cada barplot.
        "ylabels_fontsize": Define tamanho da fonte dos labels do eixo "y" para todos os barplots.
        "xtick_labelsize": Define o tamanho da fonte dos labels dos "ticks" do eixo "x" para todos os barplots.
        "ytick_labelsize": Define o tamanho da fonte dos labels dos "ticks" do eixo "y" para todos os barplots.
        "colums_nr": Define o número de colunas para a plotagem dos barplots.
        "rows_nr": Define o número de linhas para a plotagem dos barplots.
        "width": Define a largura da figura onde serão desenhados (plot) todos os barplots.
        "heigth": Define a altura da figura onde serão desenhados (plot) todos os barplots.
        "wspace": Define o espaçamento (largura) entre os gráficos nos subplots.
        "hspace": Define o espaçamento (altura) entre os gráficos nos subplots.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
    
    # Verificando se as variáveis foram definidas em "x" ou "y":
    if ((x == None) or (y == None)):
        # Erro: as variáveis não foram definidas.
        print("Error: the feature ('x' or 'y') has not been defined!")
        return

    # Verificando se a lista de variáveis está em "x":
    if(type(x) == list):
        # A lista de variáveis está em "x":
        features = x
        # Seta o flag como "True":
        feats_in_x = True
    else:
        # A lista não está em "x".
        # Então, seta o flag como "False":
        feats_in_x = False
    
    # Verificando se a lista de variáveis está em "y":
    if(type(y) == list):
        # A lista de variáveis está em "y":
        features = y
        # Seta o flag como "True":
        feats_in_y = True
    else:
        # A lista não está em "y".
        # Então, seta o flag como "False":
        feats_in_y = False

    # Verifica se foram definidas listas de variáveis em 'x' e 'y':
    if(feats_in_x and feats_in_y):
        # Erro: apenas 'x' ou 'y' deve conter a lista de variáveis.
        print("Erro: apenas 'x' ou 'y' deve conter a lista de variáveis!!!")
        return

    # Verifica a quantidade de variáveis (features):
    if(feats_in_x or feats_in_y):
        qty_feats = len(features)
    else:
        qty_feats = 1
        print("Atenção: para '1' variável em 'x' ou 'y', utilize a função 'plot_barplot'!!!")
        return

    # Verificando "Total de Subplots" x "Total de Variáveis":
    if(qty_feats > (cols_nr * rows_nr)):
        # A quantidade de variáveis que deveremos plotar é maior que o número total de subplots definido.
        # Sendo assim, enviamos uma mensagem de erro e terminamos a função.
        print("***** Erro: a quantidade total de variáveis é MAIOR que o total de subplots (cols_nr * rows_nr) *****")
        print("Redimensionar os parâmetros 'cols_nr' e 'rows_nr', ou diminuir a quantidade de variáveis!")
        return
    
    # Verificando (check) e definindo os valores para alguns parâmetros:
    #
    # colors:
    colors_values = dslib.check_params(features=features, params=colors)
    # Verifica se o tamanho (size) de "colors" x "features" são diferentes:
    if (colors_values == None):
        # Erro:
        print("Error: size of 'colors' and 'features' are different!")
        return

    # titles:
    titles_names = dslib.check_params(features=features, params=titles)
    # Verifica se o tamanho (size) de "titles" x "features" são diferentes:
    if (titles_names == None):
        # Erro:
        print("Error: size of 'titles' and 'features' are different!")
        return

    # xlabels:
    xlabels_names = dslib.check_params(features=features, params=xlabels)
    # Verifica se o tamanho (size) de "xlabels" x "features" são diferentes:
    if (xlabels_names == None):
        # Erro:
        print("Error: size of 'xlabels' and 'features' are different!")
        return
    
    # ylabels:
    ylabels_names = dslib.check_params(features=features, params=ylabels)
    # Verifica se o tamanho (size) de "ylabels" x "features" são diferentes:
    if (ylabels_names == None):
        # Erro:
        print("Error: size of 'ylabels' and 'features' are different!")
        return

    # Verifica se foi definido algum estimador:
    if(estimator != None):
        # Foi definido um estimador:
        est = estimator
    else:
        # Não foi definido um estimador. Neste caso, utilizaremos
        # o estimador padrão: "mean" (média).
        est = np.mean

    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))

    # "Loop for" para plotarmos os gráficos:
    for i, feature in enumerate(features):
        # Cria o subplot:
        ax = plt.subplot(rows_nr, cols_nr, i+1)
        
        # Verifica se a lista de features está em "x":
        if(feats_in_x):
            # A lista de variáveis está em "x":
            xi = feature
            yi = y
        else:
            # A lista de variáveis está em "y":
            xi = x
            yi = feature

        # Plot do gráfico:
        sns.barplot(
            x=xi, y=yi, hue=hue, data=data, order=order, hue_order=hue_order, estimator=est, ci=ci,
            n_boot=n_boot, units=units, seed=seed, orient=orient, color=colors_values[i], palette=palette,
            saturation=saturation, errcolor=errcolor, errwidth=errwidth, capsize=capsize, dodge=dodge, ax=ax
            )

        # Definindo o título:
        ax.set_title(label = titles_names[i], fontdict = {'fontsize': titles_fontsize})

        # Definindo o label do eixo "x":
        ax.set_xlabel(xlabel = xlabels_names[i], fontdict = {'fontsize': xlabels_fontsize})

        # Definindo o label do eixo "y":
        ax.set_ylabel(ylabel = ylabels_names[i], fontdict = {'fontsize': ylabels_fontsize})
        
        # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
        plt.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
        plt.tick_params(axis='y', which='major', labelsize=ytick_labelsize)
   
    # Ajustando os espaçamentos entre os gráficos nos subplots:
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando os gráficos:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    return

# ***** Função para plotar um gráfico de dispersão (scatter plot) entre 2 variáveis numéricas.
#
def plot_scatterplot(
    x=None, y=None, hue=None, style=None, size=None, data=None, palette=None, hue_order=None, hue_norm=None, sizes=None,
    size_order=None, size_norm=None, markers=True, style_order=None, x_bins=None, y_bins=None, units=None, estimator=None,
    ci=95, n_boot=1000, alpha=None, x_jitter=None, y_jitter=None, legend='auto', plot_style=None, title=None, title_fontsize=16,
    xlabel=None, xlabel_fontsize=13, ylabel=None, ylabel_fontsize=13, xtick_labelsize=None, ytick_labelsize=None, width=8, height=6,
    figure_id=None, figure_path=None, figure_file_format="png", figure_file_extension=".png", tight_layout=True, dpi=300
    ):
    '''
    Input:
        Informações dos principais parâmetros neste link: https://seaborn.pydata.org/generated/seaborn.scatterplot.html
        Lista de cores disponíveis neste link: https://matplotlib.org/stable/gallery/color/named_colors.html.
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "title": título do gráfico.
        "title_fontsize": tamanho da fonte do título do gráfico.
        "xlabel": label do eixo "x" do gráfico.
        "xlabel_fontsize": tamanho da fonte do label do eixo "x".
        "ylabel": label do eixo "y" do gráfico.
        "ylabel_fontsize": tamanho da fonte do label do eixo "y".
        "xtick_labelsize": tamanho da fonte dos labels dos "ticks" do eixo "x".
        "ytick_labelsize": tamanho da fonte dos labels dos "ticks" do eixo "y".
        "width": define a largura da figura onde será desenhado o gráfico.
        "heigth": define a altura da figura onde será desenhado o gráfico.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None.
    '''
    # Código da função:

    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))

    # Plot do gráfico:
    sns.scatterplot(
        x=x, y=y, hue=hue, style=style, size=size, data=data, palette=palette, hue_order=hue_order,
        hue_norm=hue_norm, sizes=sizes, size_order=size_order, size_norm=size_norm, markers=markers,
        style_order=style_order, x_bins=x_bins, y_bins=y_bins, units=units, estimator=estimator, ci=ci,
        n_boot=n_boot, alpha=alpha, x_jitter=x_jitter, y_jitter=y_jitter, legend=legend, ax=None)

    # Definindo o título:
    plt.title(label=title, size=title_fontsize)

    # Definindo o label do eixo "x":
    plt.xlabel(xlabel=xlabel, size=xlabel_fontsize)

    # Definindo o label do eixo "y":
    plt.ylabel(ylabel=ylabel, size=ylabel_fontsize)

    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
    plt.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    plt.tick_params(axis='y', which='major', labelsize=ytick_labelsize)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    # Fim da função:
    return

# ***** Função para plotar dois ou mais gráficos de dispersão (scatter plot) entre 2 variáveis numéricas.
#
def plot_scatterplots(
    features, hue=None, style=None, size=None, data=None, palette=None, hue_order=None, hue_norm=None, sizes=None,
    size_order=None, size_norm=None, markers=True, style_order=None, x_bins=None, y_bins=None, units=None, estimator=None,
    ci=95, n_boot=1000, alpha=None, x_jitter=None, y_jitter=None, legend='auto', plot_style=None, titles=None, titles_fontsize=16,
    xylabels=None, xlabels_fontsize=13, ylabels_fontsize=13, xtick_labelsize=None, ytick_labelsize=None, cols_nr=1, rows_nr=1,
    width=8, height=6, wspace=None, hspace=None, figure_id=None, figure_path=None, figure_file_format="png",
    figure_file_extension=".png", tight_layout=True, dpi=300
    ):
    '''
    Input:
        Informações dos principais parâmetros neste link: https://seaborn.pydata.org/generated/seaborn.scatterplot.html
        Lista de cores disponíveis neste link: https://matplotlib.org/stable/gallery/color/named_colors.html.
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "titles": títulos de cada gráfico de dispersão.
        "titles_fontsize": tamanho da fonte do título de cada gráfico de dispersão.
        "xylabels": labels do eixo "x" e do eixo "y" de cada gráfico de dispersão.
        "xlabels_fontsize": tamanho da fonte do label do eixo "x" de cada gráfico de dispersão.
        "ylabels_fontsize": tamanho da fonte do label do eixo "y" de cada gráfico de dispersão.
        "xtick_labelsize": tamanho da fonte dos labels dos "ticks" do eixo "x".
        "ytick_labelsize": tamanho da fonte dos labels dos "ticks" do eixo "y".
        "colums_nr": define o número de colunas para a plotagem dos gráficos de dispersão.
        "rows_nr": define o número de linhas para a plotagem dos gráficos de dispersão.
        "width": define a largura da figura onde serão desenhados os gráficos de dispersão.
        "heigth": define a altura da figura onde serão desenhados os gráficos de dispersão.
        "wspace": define o espaçamento (largura) entre os gráficos de dispersão nos subplots.
        "hspace": Define o espaçamento (altura) entre os gráficos de dispersão nos subplots.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None.
    '''
    # Código da função:

    # Verificando "Total de Subplots" x "Total de Variáveis":
    if(len(features) > (cols_nr * rows_nr)):
        # A quantidade de variáveis que deveremos plotar é maior que o número total de subplots definido.
        # Sendo assim, enviamos uma mensagem de erro e terminamos a função.
        print("***** Erro: a quantidade total de variáveis é MAIOR que o total de subplots (cols_nr * rows_nr) *****")
        print("Redimensionar os parâmetros 'cols_nr' e 'rows_nr', ou diminuir a quantidade de variáveis!")
        return
    
    # Verificando (check) e definindo os valores para alguns parâmetros:
    #
    # Verifica se os títulos foram definidos:
    if(titles != None):
        # Verifica se o tamanho (size) de "titles" x "features" são diferentes:
        if(len(titles) != len(features)):
            # Erro:
            print("Error: size of 'titles' and 'features' are different!")
            return

    # Verifica se os lables (x e y) foram definidos:
    if(xylabels != None):
        # Verifica se o tamanho (size) de "xylabels" x "features" são diferentes:
        if(len(xylabels) != len(features)):
            # Erro:
            print("Error: size of 'xylabels' and 'features' are different!")
            return

    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))

    # "Loop for" para plotarmos os gráficos de dispersão:
    for i, xy in enumerate(features):
        # Cria o subplot:
        ax = plt.subplot(rows_nr, cols_nr, i+1)
        
        # Define as variáveis "x" e "y":
        xi = xy[0]
        yi = xy[1]

        # Verifica se os títulos foram definidos:
        if(titles != None):
            # Pega o título:
            title = titles[i]
        else:
            title = None
        
        # Verifica se os labels "x" e "y" foram definidos:
        if(xylabels != None):
            # Pega os labels "x" e "y":
            xlabel = xylabels[i][0]
            ylabel = xylabels[i][1]
        else:
            xlabel = None
            ylabel = None

        # Plot do gráfico:
        sns.scatterplot(
            x=xi, y=yi, hue=hue, style=style, size=size, data=data, palette=palette, hue_order=hue_order,
            hue_norm=hue_norm, sizes=sizes, size_order=size_order, size_norm=size_norm, markers=markers,
            style_order=style_order, x_bins=x_bins, y_bins=y_bins, units=units, estimator=estimator, ci=ci,
            n_boot=n_boot, alpha=alpha, x_jitter=x_jitter, y_jitter=y_jitter, legend=legend, ax=ax
            )

        # Definindo o título:
        ax.set_title(label = title, fontdict = {'fontsize': titles_fontsize})

        # Definindo o label do eixo "x":
        ax.set_xlabel(xlabel = xlabel, fontdict = {'fontsize': xlabels_fontsize})

        # Definindo o label do eixo "y":
        ax.set_ylabel(ylabel = ylabel, fontdict = {'fontsize': ylabels_fontsize})
        
        # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
        plt.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
        plt.tick_params(axis='y', which='major', labelsize=ytick_labelsize)
   
    # Ajustando os espaçamentos entre os gráficos nos subplots:
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando os gráficos:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    return

# ***** Função para plotar as correlações entre as variáveis preditoras ("Numéricas") de um dataset qualquer:
#
def plot_corr_features(
    df, features, method='pearson', digits=3, title=None, title_fontsize=None, xtick_labelsize=None, ytick_labelsize=None,
    width=8, height=6, figure_id=None, figure_path=None, figure_file_format="png", figure_file_extension=".png",
    tight_layout=True, dpi=300
    ):
    '''
    Input:
        "df": DataFrame onde estão armazenadas as informações das variáveis preditoras.
        "features": Lista com as variáveis preditoras.
        "title": Define o título do gráfico.
        "title_fontsize": Define o tamanho da fonte do título do gráfico.
        "xtick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "x".
        "ytick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "y".
        "width": Define a largura da figura onde será desenhado (plot) o gráfico.
        "heigth": Define a altura da figura onde será desenhado (plot) o gráfico.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
    
    # Definindo a figura e o eixo onde será plotado o gráfico:
    fig, ax = plt.subplots(figsize=(width,height))
    
    # Plot:
    #sns.heatmap(data=df[features].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
    sns.heatmap(data=np.around(df[features].corr(method=method), decimals=digits), annot=True, linewidths=.5, ax=ax)
    
    # Definindo o título:
    plt.title(label=title, size=title_fontsize)
    
    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
    plt.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    plt.tick_params(axis='y', which='major', labelsize=ytick_labelsize)
  
    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando a instância da figura criada:
    plt.close(fig)
    return

# ***** Função para plotar o número de features versus as pontuações (scores) da validação cruzada:
# ***** Este plot pode ser utilizado após o treinamento de um algorítmo utilizando a função "RFECV".
#
def plot_nb_feature_vs_cv_scores(nb_feat, cv_scores, plot_style=None, linewidth=2.0, color=None,
                                 scales=None, title=None, title_fontsize=18,
                                 xlabel=None, xlabel_fontsize=15, ylabel=None, ylabel_fontsize=15,
                                 xtick_labelsize=13, ytick_labelsize=13, width=8, height=7,
                                 figure_id=None, figure_path=None,
                                 figure_file_format="png", figure_file_extension=".png",
                                 tight_layout=True, dpi=300
                                ):
    '''
    Input:
        "nb_feat": Quantidade de features que serão representado no eixo "x" do gráfico.
        "cv_scores": Pontuações (Scores) da Validação Cruzada que serão representados no eixo "y" do gráfico.
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "color": Define a cor do gráfico.
        "scales": define as escalas dos eixos "x" e "y".
        "title": Define o título do gráfico.
        "title_fontsize": Define o tamanho da fonte do título para o gráfico.
        "xlabel": Define o label do eixo "x" para o gráfico.
        "xlabel_fontsize": Define tamanho da fonte do label do eixo "x" para o gráfico.
        "ylabel": Define o label do eixo "y" para o gráfico.
        "ylabel_fontsize": Define tamanho da fonte do label do eixo "y" para o gráfico.
        "xtick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "x".
        "ytick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "y".
        "width": Define a largura da figura onde será desenhado (plot) o gráfico.
        "heigth": Define a altura da figura onde será desenhado (plot) o gráfico.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
        
    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))

    # Plot
    plt.plot(nb_feat, cv_scores, color=color, linewidth=linewidth, linestyle='-')

    # Definindo as escalas dos eixos "x" e "y":
    if (scales != None):
        # Define as escalas:
        plt.axis(scales)

    # Definindo o título:
    plt.title(label=title, size=title_fontsize)

    # Definindo o label do eixo "x":
    plt.xlabel(xlabel=xlabel, size=xlabel_fontsize)

    # Definindo o label do eixo "y":
    plt.ylabel(ylabel=ylabel, size=ylabel_fontsize)

    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
    plt.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    plt.tick_params(axis='y', which='major', labelsize=ytick_labelsize)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    return

# ***** Função para plotar as features selecionadas (importantes) com as suas pontuações (scores):
# ***** Este plot pode ser utilizado após o treinamento de um algorítmo utilizando a função "RFE".
#
def plot_importances_features(feats, scores, plot_style=None, bar='h', bardim=0.8, align='center',
                              color=None, title=None, title_fontsize=20,
                              xlabel=None, xlabel_fontsize=15, ylabel=None, ylabel_fontsize=15,
                              xtick_labelsize=13, ytick_labelsize=13, width=12, height=10,
                              figure_id=None, figure_path=None,
                              figure_file_format="png", figure_file_extension=".png",
                              tight_layout=True, dpi=300
                             ):
    '''
    Input:
        "feats": Nomes das features que serão representadas no eixo "x" ou "y" do gráfico.
        "scores": Pontuações (Scores) das features que serão representados no eixo "x" ou y" do gráfico.
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "bar": Define se as barras serão desenhadas, no gráfico, na vertical ('v') ou na horizontal ('h').
        "bardim": Define o tamanho da largura ou altura das barras que serão desenhadas no gráfico.
        "align": Define o tipo de alinhamento das barras que serão desenhadas no gráfico.
        "color": Define a cor das barras do gráfico.
        "title": Define o título do gráfico.
        "title_fontsize": Define o tamanho da fonte do título para o gráfico.
        "xlabel": Define o label do eixo "x" para o gráfico.
        "xlabel_fontsize": Define tamanho da fonte do label do eixo "x" para o gráfico.
        "ylabel": Define o label do eixo "y" para o gráfico.
        "ylabel_fontsize": Define tamanho da fonte do label do eixo "y" para o gráfico.
        "xtick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "x".
        "ytick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "y".
        "width": Define a largura da figura onde será desenhado (plot) o gráfico.
        "heigth": Define a altura da figura onde será desenhado (plot) o gráfico.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
        
    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))
    
    # Verifica o tipo de gráfico que será desenhado:
    if (bar == 'v'):
        # Plot das barras na vertical:
        plt.bar(x=feats, height=scores, width=bardim, align=align, color=color)
    else:
        plt.barh(y=feats, width=scores, height=bardim, align=align, color=color)

    # Definindo o título:
    plt.title(label=title, size=title_fontsize)

    # Definindo o label do eixo "x":
    plt.xlabel(xlabel=xlabel, size=xlabel_fontsize)

    # Definindo o label do eixo "y":
    plt.ylabel(ylabel=ylabel, size=ylabel_fontsize)

    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
    plt.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    plt.tick_params(axis='y', which='major', labelsize=ytick_labelsize)

    # Verifica se os "ticks" do eixo "x" serão escritos na vertical:
    if (bar == 'v'):
        # Escreve os nomes das features na vertical:
        plt.xticks(rotation='vertical')

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    return

# ***** Função para plotar os modelos de machine learning com as suas pontuações (scores):
#
def plot_models_vs_scores(scores, plot_style=None, bar_color=None, alpha=0.8,
                          score_color=None, score_fontsize=15,
                          title=None, title_fontsize=20,
                          xlabel=None, xlabel_fontsize=18, ylabel=None, ylabel_fontsize=18,
                          ticks_fontsize=15, width=10, height=8,
                          figure_id=None, figure_path=None,
                          figure_file_format="png", figure_file_extension=".png",
                          tight_layout=True, dpi=300
                         ):
    '''
    Input:
        "scores": objeto Series do pandas com os modelos (index) e as suas pontuações (Scores).
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "bar_color": Define a cor das barras no gráfico.
        "alpha": Set the alpha value used for blending - not supported on all backends.
        "score_color": Define a cor dos valores (scores) que serão escritos ao lado das barras no gráfico.
        "score_fontsize": Define o tamanho da fonte dos valores (scores) que serão escritos ao lado das barras no gráfico.
        "title": Define o título do gráfico.
        "title_fontsize": Define o tamanho da fonte do título para o gráfico.
        "xlabel": Define o label do eixo "x" para o gráfico.
        "xlabel_fontsize": Define tamanho da fonte do label do eixo "x" para o gráfico.
        "ylabel": Define o label do eixo "y" para o gráfico.
        "ylabel_fontsize": Define tamanho da fonte do label do eixo "y" para o gráfico.
        "ticks_fontsize": Define o tamanho da fonte dos labels dos "tick" dos eixos "x" e "y".
        "width": Define a largura da figura onde será desenhado (plot) o gráfico.
        "heigth": Define a altura da figura onde será desenhado (plot) o gráfico.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
        
    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Plot:
    ax = scores.plot(kind='barh', figsize=(width,height), color=bar_color, fontsize=ticks_fontsize);

    # Definindo o valor de alpha:
    ax.set_alpha(0.8)

    # Definindo o título:
    ax.set_title(title, fontsize=title_fontsize)

    # Definindo o label do eixo "x":
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize);
    
    # Definindo o label do eixo "y":
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize);
    
    # Definindo os ticks do eixo "x":
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])

    # set individual bar lables:
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_width()+.01, i.get_y()+.38, \
                str(round(i.get_width()*100, 2))+'%', fontsize=score_fontsize,
                color=score_color)

    # invert for largest on top 
    ax.invert_yaxis()

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    return

# ***** Função para plotar a "Confusion Matrix" dos modelos de machine learning com as suas previsões:
#
def plot_confusion_matrix(truth, pred, display_labels, normalize=False, plot_style=None,
                          colors=plt.cm.Greens, title=None, title_fontsize=18,
                          labels_fontsize=15, ticks_fontsize=13, values_fontsize=13,
                          rotation=0, width=6, height=6,
                          figure_id=None, figure_path=None,
                          figure_file_format="png", figure_file_extension=".png",
                          tight_layout=True, dpi=300
                         ):
    '''
    Input:
        "truth": valores reais das classes.
        "pred": valores previstos das classes.
        "display_labels": lista com os nomes (labels) das classes.
        "normalize": se 'True' os valores serão normalizados.
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "colors": Define as cores do gráfico.
        "title": Define o título do gráfico.
        "title_fontsize": Define o tamanho da fonte do título para o gráfico.
        "labels_fontsize": Define tamanho da fonte dos labels dos eixos "x" e "y" para o gráfico.
        "ticks_fontsize": Define o tamanho da fonte dos labels dos "tick" dos eixos "x" e "y".
        "values_fontsize": Define o tamanho da fonte dos valores (números) que serão apresentados no gráfico.
        "rotation": Define a rotação das classes no eixo "x".
        "width": Define a largura da figura onde será desenhado (plot) o gráfico.
        "heigth": Define a altura da figura onde será desenhado (plot) o gráfico.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
        
    # Calculando os dados da matriz de confusão:
    cm = confusion_matrix(truth, pred)
    
    # Se "True", normaliza os dados:
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')
 
    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))

    plt.imshow(cm, interpolation='nearest', cmap=colors)
    
    # Definindo o título:
    if (title == None):
        plt.title('Confusion Matrix', size=title_fontsize)
    else:
        plt.title(title, size=title_fontsize)
    
    # Definindo a barra de cores (vertical):
    cb = plt.colorbar(fraction=0.046, pad=0.05, orientation="vertical")
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(values_fontsize)
    
    # Definindo os ticks nos eixos "x" e "y":
    plt.tick_params(direction='out', length=6, width=1, axis='both', which='major')
    tick_marks = np.arange(len(display_labels))
    plt.xticks(tick_marks, display_labels, rotation=rotation, size=ticks_fontsize)
    plt.yticks(tick_marks, display_labels, size=ticks_fontsize)

    # Formatando os valores do gráfico:
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=values_fontsize,
                 color="white" if cm[i, j] > thresh else "black")

    # Define os labels dos eixos "x" e "y":
    plt.ylabel('Real Classes', size=labels_fontsize)
    plt.xlabel('Predicted Classes', size=labels_fontsize)

    # Desliga o 'grid':
    plt.grid(False)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    return

# ***** Função para plotar a "ROC Curve" dos modelos de machine learning:
#
def plot_roc_curve(fpr, tpr, auc, auc_decimals=4, plot_style=None, color=None, line_width=2,
                   title=None, title_fontsize=18, labels_fontsize=15,
                   ticks_fontsize=13, legendtitle=None, legendtitle_fontsize=13,
                   legend_label=None, legend_fontsize=13,
                   width=6, height=6,
                   figure_id=None, figure_path=None,
                   figure_file_format="png", figure_file_extension=".png",
                   tight_layout=True, dpi=300
                  ):
    '''
    Input:
        "fpr": São as taxas de falsos positivos (False Positive Rate).
        "tpr": São as taxas de verdadeiros positivos (True Positive Rate).
        "auc": É o valor da "AUC" (Area Uder Curve).
        "auc_decimals": Define o número de casas decimais do valor da "AUC".
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "color": Define a cor do gráfico.
        "line_width": define a largura da linha.
        "title": Define o título do gráfico.
        "title_fontsize": Define o tamanho da fonte do título para o gráfico.
        "labels_fontsize": Define tamanho da fonte dos labels dos eixos "x" e "y" para o gráfico.
        "ticks_fontsize": Define o tamanho da fonte dos labels dos "tick" dos eixos "x" e "y".
        "legendtitle": Define o título da legenda.
        "legendtitle_fontsize": Define o tamanho da fonte do título da legenda.
        "legend_label": Define o label da legenda.
        "legend_fontsize": Define o tamanho da fonte da legenda.
        "width": Define a largura da figura onde será desenhado (plot) o gráfico.
        "heigth": Define a altura da figura onde será desenhado (plot) o gráfico.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
        
    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))

    # Definindo o "label" da legenda:
    label = legend_label + "(" + str(np.round(auc, decimals=auc_decimals)) + ")"

    # Plot da curva ROC:
    plt.plot(fpr, tpr, linewidth=line_width, color=color, label=label)
    
    # Plot da linha diagonal tracejada:
    plt.plot([0, 1], [0, 1], 'k--')
    
    # Definindo as escalas dos eixos "x" e "y":
    plt.axis([0, 1, 0, 1])

    # Definindo o título:
    plt.title(title, size=title_fontsize)

    # Definindo os labels dos eixos "x" e "y":
    plt.xlabel('False Positive Rate', fontsize=labels_fontsize)
    plt.ylabel('True Positive Rate', fontsize=labels_fontsize)

    # Definindo a "legenda":
    plt.legend(title=legendtitle, title_fontsize=legendtitle_fontsize,
               fontsize=legend_fontsize, loc="lower right")

    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
    plt.tick_params(axis='x', which='major', labelsize=ticks_fontsize)
    plt.tick_params(axis='y', which='major', labelsize=ticks_fontsize)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    return

# ***** Função para plotar duas ou mais "ROC Curves" dos modelos de machine learning:
#
def plot_roc_curves(fpr, tpr, auc, auc_decimals=4, plot_style=None, color=None, line_width=2, scales=None,
                    title=None, title_fontsize=18, labels_fontsize=15,
                    ticks_fontsize=13, legendtitle=None, legendtitle_fontsize=13,
                    legend_label=None, legend_fontsize=13,
                    width=6, height=6,
                    figure_id=None, figure_path=None,
                    figure_file_format="png", figure_file_extension=".png",
                    tight_layout=True, dpi=300
                   ):
    '''
    Input:
        "fpr": "Dicionário" com as taxas de falsos positivos (False Positive Rate) de cada modelo.
        "tpr": "Dicionário" com  as taxas de verdadeiros positivos (True Positive Rate) de cada modelo.
        "auc": "Data frame" com os valores da "AUC" (Area Uder Curve) de cada modelo.
        "auc_decimals": Define o número de casas decimais do valor da "AUC".
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "color": "Lista" com as cores do gráfico.
        "line_width": define a largura das linhas do gráfico.
        "scales": define as escalas dos eixos "x" e "y".
        "title": Define o título do gráfico.
        "title_fontsize": Define o tamanho da fonte do título para o gráfico.
        "labels_fontsize": Define tamanho da fonte dos labels dos eixos "x" e "y" para o gráfico.
        "ticks_fontsize": Define o tamanho da fonte dos labels dos "tick" dos eixos "x" e "y".
        "legendtitle": Define o título da legenda.
        "legendtitle_fontsize": Define o tamanho da fonte do título da legenda.
        "legend_label": "Lista" com os labels da legenda para cada modelo.
        "legend_fontsize": Define o tamanho da fonte da legenda.
        "width": Define a largura da figura onde será desenhado (plot) o gráfico.
        "heigth": Define a altura da figura onde será desenhado (plot) o gráfico.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
        
    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))

    # Loop for para plotar cada modelo:
    for i in range(0, len(legend_label)):
        # Definindo o "label" da legenda:
        label = legend_label[i] + "(" + str(np.round(auc.iloc[i,0], decimals=auc_decimals)) + ")"
        
        # Plot da curva ROC:
        plt.plot(fpr[legend_label[i]], tpr[legend_label[i]], linewidth=line_width, color=color[i], label=label)
    
    # Plot da linha diagonal tracejada:
    plt.plot([0, 1], [0, 1], 'k--')
    
    # Definindo as escalas dos eixos "x" e "y":
    if (scales == None):
        # Define as escalas "default":
        plt.axis([0, 1, 0, 1])
    else:
        plt.axis(scales)

    # Definindo o título:
    plt.title(title, size=title_fontsize)

    # Definindo os labels dos eixos "x" e "y":
    plt.xlabel('False Positive Rate', fontsize=labels_fontsize)
    plt.ylabel('True Positive Rate', fontsize=labels_fontsize)

    # Definindo a "legenda":
    plt.legend(title=legendtitle, title_fontsize=legendtitle_fontsize,
               fontsize=legend_fontsize, loc="lower right")

    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
    plt.tick_params(axis='x', which='major', labelsize=ticks_fontsize)
    plt.tick_params(axis='y', which='major', labelsize=ticks_fontsize)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    return

# ***** Função para comparar as métricas dos modelos de machine learning:
#
def plot_ml_metrics(
    data, x, y, hue, plot_style=None, title=None, title_fontsize=20,
    xlabel=None, xlabel_fontsize=18, ylabel=None, ylabel_fontsize=18,
    ticks_fontsize=15, legendtitle=None, legendtitle_fontsize=13,
    legend_label=None, legend_fontsize=13, legend_loc='lower right',
    legend_borderaxespad=0., legend_bbox_to_anchor=(1.05, 1),
    width=10, height=8, figure_id=None, figure_path=None,
    figure_file_format="png", figure_file_extension=".png", tight_layout=True, dpi=300
):
    '''
    Input:
        "data": data frame com os dados.
        "x": variável do eixo "x";
        "y": variável do eixo "y";
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "title": Define o título do gráfico.
        "title_fontsize": Define o tamanho da fonte do título para o gráfico.
        "xlabel": Define o label do eixo "x" para o gráfico.
        "xlabel_fontsize": Define tamanho da fonte do label do eixo "x" para o gráfico.
        "ylabel": Define o label do eixo "y" para o gráfico.
        "ylabel_fontsize": Define tamanho da fonte do label do eixo "y" para o gráfico.
        "ticks_fontsize": Define o tamanho da fonte dos labels dos "tick" dos eixos "x" e "y".
        "legendtitle": Define o título da legenda.
        "legendtitle_fontsize": Define o tamanho da fonte do título da legenda.
        "legend_label": Define o label da legenda.
        "legend_fontsize": Define o tamanho da fonte da legenda.
        "legend_loc": Define a localização da legenda no gráfico.
        "legend_borderaxespad": Define a localização da legenda no gráfico.
        "legend_bbox_to_anchor": Define a localização da legenda no gráfico.
        "width": Define a largura da figura onde será desenhado (plot) o gráfico.
        "heigth": Define a altura da figura onde será desenhado (plot) o gráfico.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
        
    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura onde serão plotados os gráficos:
    if ((width != None) and (height != None)):
        plt.figure(figsize=(width,height))

    # Plot:
    sns.barplot(data=data, x=x, y=y, hue=hue)

    # Definindo o título:
    plt.title(label=title, size=title_fontsize)

    # Definindo o label do eixo "x":
    plt.xlabel(xlabel=xlabel, size=xlabel_fontsize)

    # Definindo o label do eixo "y":
    plt.ylabel(ylabel=ylabel, size=ylabel_fontsize)

    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
    plt.tick_params(axis='x', which='major', labelsize=ticks_fontsize)
    plt.tick_params(axis='y', which='major', labelsize=ticks_fontsize)

    # Definindo a "legenda":
    plt.legend(
        title=legendtitle, title_fontsize=legendtitle_fontsize,
        fontsize=legend_fontsize, borderaxespad=legend_borderaxespad,
        bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc
        )

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    return

# ***** Função para plotar um boxplot de uma variável preditora ("Numérica"), de acordo com as classes de uma outra variável
# ("Categórica"):
def plot_qqplot_feature(
    df, feature, line=None, plot_style=None, title=None, title_fontsize=16, xlabel='Theoretical Quantiles',
    xlabel_fontsize=13, ylabel='Sample Quantiles', ylabel_fontsize=13, xtick_labelsize=None, ytick_labelsize=None,
    width=8, height=6, figure_id=None, figure_path=None, figure_file_format="png", figure_file_extension=".png",
    tight_layout=True, dpi=300
):
    '''
    Input:
        "df": DataFrame onde estão armazenadas as informações da variável.
        "feature": label do nome da variável a ser analisada.
        "line": opções para a linha de referência com a qual os dados são comparados: veja a documentação da função.
        "plot_style": Define o estilo do gráfico (ver os estilos na função "matplotlib.style.available").
        "title": Define o título do gráfico.
        "title_fontsize": Define o tamanho da fonte do título do gráfico.
        "xlabel": Define o label do eixo "x" para o gráfico.
        "xlabel_fontsize": Define tamanho da fonte do label do eixo "x" para o gráfico.
        "ylabel": Define o label do eixo "y" para o gráfico.
        "ylabel_fontsize": Define tamanho da fonte do label do eixo "y" para o gráfico.
        "xtick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "x".
        "ytick_labelsize": Define o tamanho da fonte do label do "tick" do eixo "y".
        "width": Define a largura da figura onde será desenhado (plot) o gráfico.
        "heigth": Define a altura da figura onde será desenhado (plot) o gráfico.
        "figure_id": nome do arquivo onde será salvo a figura.
        "figure_path": nome do diretório (path) onde será salvo a figura.
        "figure_file_format": formato da figura que será salvo no arquivo.
        "figure_file_extension": extansão do arquivo onde será salvo a figura.
        "tight_layout": Se "True" chama o método "tight_layout()".
        "dpi": define a resolução da figura que será salva.

    Output: None
    '''
    # Código da função:
    
    # Definindo o estilo do gráfico:
    if (plot_style != None):
        plt.style.use(style=plot_style)
    else:
        plt.style.use(style='default')

    # Definindo a figura e o eixo onde será plotado o gráfico:
    fig, ax = plt.subplots(figsize=(width,height))

    # Plot
    sm.qqplot(data=df[feature].values, line=line, ax=ax)

    # Definindo o título:
    plt.title(label=title, size=title_fontsize)

    # Definindo o label do eixo "x":
    plt.xlabel(xlabel=xlabel, size=xlabel_fontsize)

    # Definindo o label do eixo "y":
    plt.ylabel(ylabel=ylabel, size=ylabel_fontsize)

    # Definindo o tamanho dos labels dos ticks nos eixos "x" e "y":
    plt.tick_params(axis='x', which='major', labelsize=xtick_labelsize)
    plt.tick_params(axis='y', which='major', labelsize=ytick_labelsize)

    # Verificando se o gráfico deverá se salvo:
    if (figure_id != None):
        save_figure (figure_id=figure_id, figure_path=figure_path,
                     figure_file_format=figure_file_format,
                     figure_file_extension=figure_file_extension,
                     tight_layout=tight_layout, dpi=dpi)

    # Mostrando o gráfico:
    plt.show()
    
    # Fechando as instâncias da figura criada:
    plt.close()
    
    return
