from sklearn.metrics import f1_score, roc_auc_score

def run_model_training(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    f1_score_metric = f1_score(y_test, y_pred)
    roc_score_metric = roc_auc_score(y_test, y_proba[:,1])

    return model, y_pred, f1_score_metric, roc_score_metric, y_proba
