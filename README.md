# Flight-Price-Prediction
Este projeto utiliza aprendizado de máquina para prever os preços de passagens aéreas com base em diversos fatores, como companhia aérea, cidade de origem, cidade de destino, horário de partida e chegada, número de escalas e classe do voo. Utilizei técnicas de pré-processamento de dados e modelos de regressão.

# Predição de Preços de Passagens Aéreas

Este projeto utiliza aprendizado de máquina para prever os preços de passagens aéreas com base em vários fatores. O objetivo é fornecer previsões precisas para ajudar viajantes e empresas a tomar decisões informadas sobre a compra de passagens.

## Descrição do Projeto

O projeto segue as seguintes etapas principais:

1. **Carregamento e exploração dos dados**: Os dados são carregados de um arquivo CSV e explorados para entender melhor as características das passagens aéreas.
2. **Pré-processamento dos dados**: As colunas irrelevantes são removidas, e as colunas categóricas são codificadas em valores binários ou fatoriais.
3. **Treinamento do modelo de regressão**: Um modelo de Regressão de Floresta Aleatória é treinado usando os dados pré-processados.
4. **Avaliação do modelo**: O modelo é avaliado utilizando métricas como R², MAE, MSE e RMSE.
5. **Otimização de hiperparâmetros**: Técnicas como GridSearch e RandomizedSearch são utilizadas para encontrar a melhor combinação de hiperparâmetros e melhorar a performance do modelo.

## Dependências

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- scipy

## Instalação

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
pip install -r requirements.txt
```

# Airfare Price Prediction

This project uses machine learning to predict airfare prices based on various factors. The goal is to provide accurate predictions to help travelers and businesses make informed decisions about purchasing tickets.

## Project Description

The project follows these main steps:

1. **Data Loading and Exploration**: Load data from a CSV file and explore it to understand the characteristics of the airfare data.
2. **Data Preprocessing**: Remove irrelevant columns and encode categorical columns into binary or factor values.
3. **Training the Regression Model**: Train a Random Forest Regression model using the preprocessed data.
4. **Model Evaluation**: Evaluate the model using metrics such as R², MAE, MSE, and RMSE.
5. **Hyperparameter Optimization**: Use techniques like GridSearch and RandomizedSearch to find the best combination of hyperparameters and improve model performance.

## Dependencies

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- scipy

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
pip install -r requirements.txt
