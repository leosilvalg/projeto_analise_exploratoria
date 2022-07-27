# %% [markdown]
# Trabalho sobre análise exploratória de dados utilizando Python, Linguagem SQL e Banco de Dados SQLite. 
# Os dados utilizados nesse projeto são dados reais, disponíveis a partir do IMDB.

# %% [markdown]
# Realizando a instalação e carregamento dos pacotes a serem utilizados

# %%
# Instalando o pacote
%pip install -q imdb-sqlite

# %%
# instalando o pacote pycountry (nomes dos paises, etc...)
%pip install -q pycountry

# %%
# Importando pacotes para realização das análises numéricas

import re
import time
import sqlite3
import pycountry
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore")
sns.set_theme(style = "whitegrid")

# %% [markdown]
# Carregamento dos Dados utilizados

# %%
# Data size: 10.6 GB !!!!
%%time
!imdb-sqlite

# %%
# abrindo conexão ao banco de dados
conn = sqlite3.connect("imdb.db")

# %%
# Realizando a extração das listas de tabelas
tabelas = pd.read_sql_query("SELECT NAME AS 'Table_Name' FROM sqlite_master WHERE type = 'table'", conn)

# %%
# type(tabelas)
tabelas.head()

# %%
# Convertendo o dataframe em uma lista
tabelas = tabelas["Table_Name"].values.tolist()

# %%
# Verificando o esquema de cada tabela
for tabela in tabelas:
    consulta = "PRAGMA TABLE_INFO({})".format(tabela)
    resultado = pd.read_sql_query(consulta, conn)
    print("Esquema da tabela:", tabela)
    display(resultado)
    print("-"*100)
    print("\n")

# %% [markdown]
# Primeira parte - Quais as categorias de filmes mais comuns no IMDB ?

# %%
# Criando a primeira consulta 
consulta1 = '''SELECT type, COUNT(*) AS COUNT FROM titles GROUP BY type'''

# %%
resultado1 = pd.read_sql_query(consulta1, conn)

# %%
# Total com base na categoria
display(resultado1)

# %%
# Além de calcular o percentual, é necessário criar uma coluna nova
resultado1['percentual'] = (resultado1['COUNT'] / resultado1['COUNT'].sum()) * 100

# %%
display(resultado1)

# %%
# Filtrando apenas 4 categorias. 3 categorias com mais títulos e 1 categorias com o restante
others = {}

others['COUNT'] = resultado1[resultado1['percentual'] < 5]['COUNT'].sum()

others['percentual'] = resultado1[resultado1['percentual'] < 5]['percentual'].sum()

others['type'] = 'others'

# %%
others

# %%
resultado1 = resultado1[resultado1['percentual'] > 5]

# %%
resultado1 = resultado1.append(others, ignore_index=True)

# %%
resultado1 = resultado1.sort_values(by = 'COUNT', ascending=False)

# %%
resultado1.head()

# %%
# List comprehension, para ajustar o label a ser construido no gráfico de rosca
labels = [str(resultado1['type'][i])+' '+'['+str(round(resultado1['percentual'][i],2)) +'%'+']' for i in resultado1.index]

# %%
#Plot

cs = cm.Set3(np.arange(100))

f = plt.figure()

plt.pie(resultado1['COUNT'], labeldistance=1, radius=3, colors=cs, wedgeprops= dict(width = 0.8))
plt.legend(labels = labels, loc = 'center', prop = {'size':12})
plt.title("Distribuição de Títulos", loc = 'Center', fontdict={'fontsize': 20, 'fontweight': 20})
plt.show()

# %% [markdown]
# Segunda Parte - Qual o número de títulos por gêneros ???

# %%
consulta2 = '''SELECT genres, COUNT(*) FROM titles WHERE type = 'movie' GROUP BY genres'''

# %%
resultado2 = pd.read_sql_query(consulta2, conn)

# %%
display(resultado2)

# %%
# O '\N' acima pode indicar que temos filmes que não possuem um genero definido
resultado2['genres'] = resultado2['genres'].str.lower().values

# %%
temp = resultado2['genres'].dropna()

# %%
# Neste ponto foi apresentado o conceito do one-hot encoded
padrao = '(?u)\\b[\\w-]+\\b'
vetor = CountVectorizer(token_pattern= padrao, analyzer='word').fit(temp)

# %%
type(vetor)

# %%
bag_generos = vetor.transform(temp)

# %%
type(bag_generos)

# %%
generos_unicos = vetor.get_feature_names()

# %%
generos = pd.DataFrame(bag_generos.todense(), columns=generos_unicos, index=temp.index)

# %%
generos.info()

# %%
generos = generos.drop(columns = 'n', axis=0)

# %%
generos_percentual = 100 * pd.Series(generos.sum()).sort_values(ascending=False) / generos.shape[0]

# %%
generos_percentual.head(10)

# %%
plt.figure(figsize = (16,8))
sns.barplot(x=generos_percentual.values, y=generos_percentual.index, orient="h", palette="terrain")
plt.ylabel('Genero')
plt.xlabel("\nPercentual de Filmes (%)")
plt.title("\nNúmero de Títulos por Gêneros")
plt.show

# %% [markdown]
# Terceira parte - Mediana de avaliação dos filmes por gênero

# %%
consulta3 = '''SELECT rating, genres FROM ratings JOIN titles ON ratings.title_id = titles.title_id WHERE premiered <= 2022 AND type = 'movie' '''

# %%
resultado3 = pd.read_sql_query(consulta3, conn)

# %%
display(resultado3)

# %%
# Diferente do passo anterior, nessa situação foi criada uma função para separar os diferentes generos.
def return_genres(df):
    df['genres'] = df['genres'].str.lower().values
    temp = df['genres'].dropna()
    padrao = '(?u)\\b[\\w-]+\\b'
    vetor = CountVectorizer(token_pattern= padrao, analyzer='word').fit(temp)
    generos_unicos = vetor.get_feature_names()
    generos_unicos = [genre for genre in generos_unicos if len(genre) > 1]
    return generos_unicos

# %%
generos_unicos = return_genres(resultado3)

# %%
generos_unicos

# %%
generos_count = []
generos_ratings = []

# %%
for item in generos_unicos:

    # contagem de filmes por generos
    consulta = 'SELECT COUNT(rating) FROM ratings JOIN titles ON ratings.title_id = titles.title_id WHERE genres LIKE '+ '\''+'%'+item+'%'+'\' AND type=\'movie\''
    resultado = pd.read_sql_query(consulta, conn)
    # print(resultado.values[0][0])
    generos_count.append(resultado.values[0][0])

    #  # Avaliação de filmes por generos
    # consulta = 'SELECT rating FROM ratings JOIN titles ON ratings.title_id = titles.title_id WHERE genres LIKE '+ '\''+'%'+item+'%'+'\' AND type=\'movie\''
    # resultado = pd.read_sql_query(consulta, conn)
    # generos_ratings.append(np.median(resultado['rating']))

# %%
for item in generos_unicos:

    #  contagem de filmes por generos
    # consulta = 'SELECT COUNT(rating) FROM ratings JOIN titles ON ratings.title_id = titles.title_id WHERE genres LIKE '+ '\''+'%'+item+'%'+'\' AND type=\'movie\''
    # resultado = pd.read_sql_query(consulta, conn)
    # print(resultado.values[0][0])
    # generos_count.append(resultado.values[0][0])

    # Avaliação de filmes por generos
    consulta = 'SELECT rating FROM ratings JOIN titles ON ratings.title_id = titles.title_id WHERE genres LIKE '+ '\''+'%'+item+'%'+'\' AND type=\'movie\''
    resultado = pd.read_sql_query(consulta, conn)
    generos_ratings.append(np.median(resultado['rating']))

# %%
df_generos_ratings = pd.DataFrame()
df_generos_ratings['genres'] = generos_unicos
df_generos_ratings['count'] = generos_count
df_generos_ratings['rating'] = generos_ratings

# %%
df_generos_ratings.head(20)

# %%
df_generos_ratings = df_generos_ratings.drop(index = 18)

# %%
from sympy import false


df_generos_ratings = df_generos_ratings.sort_values(by = 'rating', ascending = False)

# %%
plt.figure(figsize = (16,10))

sns.barplot(y = df_generos_ratings.genres, x = df_generos_ratings.rating, orient = "h")

for i in range(len(df_generos_ratings.index)):
    plt.text(4.0,
             i + 0.25,
             str(df_generos_ratings['count'][df_generos_ratings.index[i]]) + " filmes")

    plt.text(df_generos_ratings.rating[df_generos_ratings.index[i]],
            i + 0.25,
            round(df_generos_ratings["rating"][df_generos_ratings.index[i]],2))

plt.ylabel('Gênero')
plt.xlabel('Mediana de Avaliação')
plt.title('\nMediana de Avaliação por Gênero\n')
plt.show()


