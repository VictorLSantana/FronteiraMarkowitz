
# Importando as Bibliotecas.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Definir as ações que vão ser trabalhadas.
acoes = [
         'PETR3.SA',
         'VALE3.SA',
         'LIGT3.SA',
         'BBDC3.SA',
         'BBAS3.SA'
]

# Criar o DataFrame da carteira.
portfolio = pd.DataFrame()

# Pegando os dados do yfinance a respeito
# do fechamento ajustado de cada ação e adicionando ao DataFrame.
for acao in acoes:
  portfolio[acao] = yf.download(acao, period='max')['Adj Close']

# Plotando o gráfico da carteira ao londo do período estipulado de 1 ano.
sns.set_theme()
portfolio.plot(figsize=(21,9))
plt.xlabel('Data', fontsize=16)
plt.ylabel('Fechamento Ajustado', fontsize=16)
plt.title('Ações do Portfolio', fontsize=22)

# Calculando o retorno das ações.
retorno = portfolio.pct_change()

# Calculando medidas estatísticas dos retornos de cada ação.

# Retorno médio anual.
retorno_anual = retorno.mean()*252

# Covariância.
cov = retorno.cov()*252

# Criando portfolios aleatórios.
qnt_portfolios = 100000
qnt_acoes = len(acoes)

# Função para gerar pesos aleatórios.
def weight(qnt_acoes):
    pesos = []
    while sum(pesos) != 1.0:
            pesos = []
            aleatorio = np.random.random(qnt_acoes)
            for i in range(qnt_acoes):
                pesos.append(round(aleatorio[i]/sum(aleatorio), 2))
    return pesos

# Criando os portfolios baseados nos pesos aleatórios gerados.
pesos = []
retorno_esperado = []
volatilidade = []
sharpe = []

# Iterando cada peso aleatório para gerar o retorno esperado, volatilidade e o sharpe.
for i in range(qnt_portfolios):
    pesos.append(np.array(weight(qnt_acoes=qnt_acoes)))       
    retorno_esperado.append(np.dot(pesos[i], retorno_anual))
    volatilidade.append(np.sqrt(np.dot(pesos[i].T, np.dot(cov, pesos[i]))))
    sharpe.append(retorno_esperado[i]/volatilidade[i])


# Criando o DataFrame com os portfolios.
dic_portfolio = {
    'Retorno': retorno_esperado,
    'Volatilidade': volatilidade,
    'Índice Sharpe': sharpe
}
Portfolio = pd.DataFrame(dic_portfolio)

# Valores de destaque.

# Maior Retorno esperado.
maior_retorno = Portfolio[
    max(Portfolio['Retorno']) == Portfolio['Retorno']
]

# Menor Volatilidade (risco).
menor_volatilidade = Portfolio[
    min(Portfolio['Volatilidade']) == Portfolio['Volatilidade']
]

# Maior Sharpe
maior_sharpe = Portfolio[
    max(Portfolio['Índice Sharpe']) == Portfolio['Índice Sharpe']
]

# Função para retornar o peso de cada ação dado uma linha do DataFrame.
def get_weight(valor):
    lista_pesos = list(pesos[valor.iloc()[0].name])
    df = pd.DataFrame(columns=acoes, data=[lista_pesos])
    return print(df.to_string(index=False))


# Plotando os portfolios no gráfico de dispersão.
plt.figure(figsize=(21,9))
plt.scatter(
    Portfolio['Volatilidade'],
    Portfolio['Retorno'],
    c=Portfolio['Índice Sharpe'],
    cmap='viridis',
)

plt.xlabel('Volatilidade', fontsize=16)
plt.ylabel('Retorno Esperado', fontsize=16)
plt.title('Fronteira Eficiente de Markowitz', fontsize=22)
plt.colorbar(label='Índice Sharpe')

# Ponto maior retorno.
plt.scatter(
    maior_retorno['Volatilidade'],
    maior_retorno['Retorno'],
    c='orange',
    label='Maior Retorno Esperado',
     marker='*',
     s=500
)

# Ponto menor volatilidade.
plt.scatter(
    menor_volatilidade['Volatilidade'],
    menor_volatilidade['Retorno'],
    c='pink',
    label='Menor Volatilidade',
    marker="*",
    s=500
)

# Maior índice sharpe.
plt.scatter(
    maior_sharpe['Volatilidade'],
    maior_sharpe['Retorno'],
    c='red',
    label='Maior Índice Sharpe',
    marker='*',
    s=500
)
plt.legend()
plt.show()

# Pontos Críticos.
print('Carteira com maior Retorno Esperado:')
get_weight(maior_retorno)
print('\n')

print('Carteira com menor Volatilidade:')
get_weight(menor_volatilidade)
print('\n')

print('Carteira com maior índice Sharpe:')
get_weight(maior_sharpe)
