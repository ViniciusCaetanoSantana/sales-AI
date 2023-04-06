import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

tabela = pd.read_csv('advertising.csv')

x = tabela[['TV', 'Radio', 'Jornal']] 
y = tabela['Vendas'] 
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y) 

modelo_regressaolinear = LinearRegression() 
modelo_arvoredecisao = RandomForestRegressor()

modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste) 
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste) 


#to see witch one can predict better:
# print(r2_score(y_teste, previsao_regressaolinear)) 
# print(r2_score(y_teste, previsao_arvoredecisao)) # this one won


tabela_prever = pd.read_csv('novos.csv')
previsao = modelo_arvoredecisao.predict(tabela_prever)
print(previsao)