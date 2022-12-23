# FSI - API

Para utilizar essa API, siga os seguintes passos.

```pip install -U poetry```

O `poetry` será a ferramenta utilizada para o controle das bibliotecas utilizadas no projeto.

```poetry install```

Para instalar as dependências.

```uvicorn app:app --reload```

Esse comando irá iniciar a API localmente. Assim que o comando executar com sucesso a inicialização, siga para `http://127.0.0.1:8000/docs`. Nesse enderenço, você terá disponível uma interface gráfica para interagir com a API manualmente.

## Endpoints

A API possui três endpoints:
- `/features`: A API retornará as features utilizadas pelo modelo, além de um exemplo de valores para serem inputados.
- `/predict`: A API receberá os valores das features, e retornará a predição, a probabilidade da predição e o log da probabilidade.
- `/retrain`: A API irá retreinar o modelo com os dados disponíveis, e irá retornar uma pequena amostra das predições e de suas probabilidades, além de duas métricas (F1 Score e ROC AUC Score).