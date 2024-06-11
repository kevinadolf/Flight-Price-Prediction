import pandas as pd

df = pd.read_csv('Clean_Dataset.csv')

df.columns.values.tolist()

df.airline.value_counts()

df.source_city.value_counts()

df.destination_city.value_counts()

df.departure_time.value_counts()

df.arrival_time.value_counts()

df.stops.value_counts()

df['class'].value_counts()

df['duration'].min()

df['duration'].max()

df['duration'].median()


## Pre-processamento dos dados

#aqui estamos apagando colunas inuteis dos dados
df = df.drop('Unnamed: 0', axis=1)
df = df.drop('flight', axis=1)

#aqui estamos 'Binary encoding', o que é basicamente transformando informações em 0 ou 1, 
# facilitando nossa manipulação e entendimento. 
df['class'] = df['class'].apply(lambda x: 1 if x=='Business' else 0 )
df.stops = pd.factorize(df['stops'])[0]

df = df.join(pd.get_dummies(df.airline, prefix='airline')).drop('airline', axis = 1)
df = df.join(pd.get_dummies(df.source_city, prefix='source')).drop('source_city', axis = 1)
df = df.join(pd.get_dummies(df.destination_city, prefix='dest')).drop('destination_city', axis = 1)
df = df.join(pd.get_dummies(df.arrival_time, prefix='arrival')).drop('arrival_time', axis = 1)
df = df.join(pd.get_dummies(df.departure_time, prefix='departure')).drop('departure_time', axis = 1)


## Modelo de Treinamento de Regressão


#Separar a base em train_data e test_data para treinar e validar o modelo
from sklearn.model_selection import train_test_split
#Tipo de modelo que usaremos
from sklearn.ensemble import RandomForestRegressor 

#retirando o preco do df e guardando-o em uma variavel especifica 
# para que possamos utilizar para treino e validação
x, y = df.drop('price', axis=1), df.price

#test_size qr dizer que o valor assinalado (0.2=20%) vai ser utilizado como
# base para teste, e o restante (0.8=80%) vai ser utilizado para treinar o modelo
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#para acelerar o processo podemos usar processamento paralelo,
# que é basicamente utilizar mais de um CORE da CPU. Só é possível 
# pq esse método (randomforestregressor) suporta proces.paralelo

#assinalando n_jobs=-1 usamos todos os CORES da cpu
reg = RandomForestRegressor(n_jobs=-1)

reg.fit(x_train, y_train)

from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Mean_absolute_error serve para saber em media quanto desviamos/erramos quando comparamos ao valor real 
#Mean_squared_error serve para 

y_pred = reg.predict(x_test)

#Checando os resultados do nosso modelo
print('-----------------Modelo Normal-----------------\n')
print('R2: ',r2_score(y_test, y_pred))
print('MAE: ',mean_absolute_error(y_test, y_pred)) 
print('MSE: ',mean_squared_error(y_test, y_pred))
print('RMSE: ',sqrt(mean_squared_error(y_test, y_pred)))


## Podemos verificar graficamente quão próximos estamos dos preços reais

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel('Preço Real do Voo')
plt.ylabel('Preço Previsto do Voo')
plt.title('Preço Previsto vs Real - Modelo sem alteração de HiperParâmetros')

#Listando e ordenando os atributos mais importante em relação aos preços das passagens
importances = dict(zip(reg.feature_names_in_, reg.feature_importances_))
sorted_importances = sorted(importances.items(), key = lambda x: x[1], reverse=True)

#plotando esses atributos para vermos graficamente quao impactante eles são
plt.figure(figsize=(10,6))
plt.bar([x[0] for x in sorted_importances[:5]], [x[1] for x in sorted_importances[:5]])


#Para acharmos a combinação ideal dos hiperparâmetros do modelo usaremos GridSearch
#GridSearch ajuda a otimizar a performance do modelo e também a reduzir o overfitting
#Overfitting é quando um modelo nao consegue generalizar, pq acaba se adequando
# estritamente a base de treinamento 

from sklearn.model_selection import GridSearchCV
#GridSearch é basicamente uma busca exaustiva pela combinação mais adequada para o modelo

#lista de possíveis hiperparâmetros para o gridsearch combiná-los e testar cada combinação
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

#cv = CrossValidationé uma técnica para avaliar a capacidade de generalização de um modelo, 
# a partir de um conjunto de dados

## Fazendo a GridSearch propriamente dita

grid_search = GridSearchCV(reg, param_grid, cv=5)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_

y_pred = best_params.predict(x_test)

#Checando os resultados do nosso modelo
print('\n-----------------Grid Search-----------------\n')

print('R2: ',r2_score(y_test, y_pred))
print('MAE: ',mean_absolute_error(y_test, y_pred)) 
print('MSE: ',mean_squared_error(y_test, y_pred))
print('RMSE: ',sqrt(mean_squared_error(y_test, y_pred)))

plt.scatter(y_test, y_pred)
plt.xlabel('Preço Real do Voo')
plt.ylabel('Preço Previsto do Voo')
plt.title('Preço Previsto vs Real com GridSearch')


'''
O método acima é um método que leva bastante tempo para rodar, pois matematicamente temos
mais de 200 (3x4x3x3x2 = 216) combinações possíveis para o GridSearch avaliar. Por isso,
usaremos outro método, um mais rapido porém com muito menos combinações, potencialmente menos preciso
'''


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': [1.0, 'auto', 'sqrt']
}

random_search = RandomizedSearchCV(estimator= reg, param_distributions=param_dist, n_iter=2, cv=3,
                                   scoring='neg_mean_squared_error', verbose=2, random_state=10, n_jobs=-1)

random_search.fit(x_train, y_train)

best_regressor = random_search.best_estimator_

y_pred = best_regressor.predict(x_test)

#Checando os resultados do nosso modelo
print('\n-----------------Random Search-----------------\n')
print('R2: ',r2_score(y_test, y_pred))
print('MAE: ',mean_absolute_error(y_test, y_pred)) 
print('MSE: ',mean_squared_error(y_test, y_pred))
print('RMSE: ',sqrt(mean_squared_error(y_test, y_pred)))
plt.title('Preço Previsto vs Real com RandomSearch')