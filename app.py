# Load the libraries
from fastapi import FastAPI, HTTPException
from pickle import load, dump
import pandas as pd
from models.ml.classifier import lr
from models.schemas.lr_clf import FSI, FSIPredictionResponse, FSIModelResponse
from dotenv import load_dotenv
import training.data_processing as dp
import training.model_training as mt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os

load_dotenv(".env")
url = os.environ.get("FILE")

# Initialize an instance of FastAPI
app = FastAPI(title="MRV - Ferramenta de Score de Inadimplência (FSI)")

lr = load(open('models/logistic_regression_hyper_tun.sav','rb'))

@app.get("/features")
def get_features_and_model():
    feature_names = lr.feature_names_in_
    return {
                #"feature_names": feature_names,
                "renda_cliente": 2674.0,
                "parcelas_entradas": 60.0,
                "renda_parcela_entrada": 0.08,
                "fgts_value": 2993.16,
                "sinal_value": 0.0,
                "mrv_value": 13469.22,
                "banco_value": 131699.04,
                "valor_parcelas": 224.487
            }

@app.post('/predict',
            tags=["Predictions"],
            response_model=FSIPredictionResponse)
async def predict(inad: FSI):
    data = dict(inad)['data']

    prediction = lr.predict(data).tolist()
    probability = lr.predict_proba(data).tolist()
    log_probability = lr.predict_log_proba(data).tolist()

    return {"prediction": prediction,
            "probability": probability,
            "log_probability": log_probability}

@app.post('/retrain',
            tags=["Retraining"],
            response_model=FSIModelResponse)
async def model_retraining(save_model: bool = False, samples: int = 10):

    df_url ='https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(df_url)
    preprocess = dp.TranformData(df)

    preprocess.change_column_name([
                                    "ano_venda",
                                    "estado_civil",
                                    'cidade',
                                    'renda_cliente',
                                    'fgts_imovel',
                                    'sinal_imovel',
                                    'mrv_imovel',
                                    'banco_imovel',
                                    'parcelas_entradas',
                                    'renda_parcela_entrada',
                                    'valor_imovel',
                                    'inadimplencia'
                                    ])

    preprocess.transform_columns_in_num(["renda_cliente", "valor_imovel"])

    preprocess.transform_percentage_in_number(columns_list=["fgts_imovel", "sinal_imovel", 
                                                "mrv_imovel", "banco_imovel"],
                                column_times="valor_imovel",
                                new_columns_list=["fgts_value", "sinal_value", "mrv_value", "banco_value"])

    preprocess.divide_two_columns("valor_parcelas", "mrv_value", "parcelas_entradas")

    preprocess.turn_percentage_in_decimal("renda_parcela_entrada")

    preprocess.removing_nan_inf("valor_parcelas")

    preprocess.transforming_string_into_category("inadimplencia", "inadimplencia_cat")

    df = preprocess.remove_object_column()

    X_train, X_test, y_train, y_test = dp.train_test_using_year(df, 
                                                                "inadimplencia_cat",
                                                                "ano_venda",
                                                                2017)


    lr = LogisticRegression(random_state=42)
    model, y_pred, f1_score_metric, roc_score_metric, y_proba = mt.run_model_training(lr, X_train, X_test, y_train, y_test)

    if save_model:
        dump(model, './models/lr_new_model.sav')

    return {
        "prediction": y_pred.tolist(),
        "f1_score": [f1_score_metric],
        "roc_auc_score": [roc_score_metric],
        "y_proba": y_proba[:,1].tolist()
    }