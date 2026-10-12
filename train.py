# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MARK: Importar librerias

import os
import argparse

import pandas
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

import warnings

import mlflow

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MARK: Definicion de funciones

def arg_parse():
    print('\nParsing arguments ...')

    default_dict = {
        'RandomSeed':1,
        'InvReg':1.0,
        'Solver':'lbfgs',
        'MaxIter':100
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--RandomSeed', help='Random seed to pass onto functions that use it, to ensure reproducibility of results.')
    parser.add_argument('-c', '--InvReg', help='For the logistic regression, the value for the inverse of the regularization strength.')
    parser.add_argument('-s', '--Solver', help='Algorithm to use in the logistic regression optimization. Must use one of the options provided in the scikit-learn documentation for LogisticRegression.')
    parser.add_argument('-m', '--MaxIter', help='Maximum number of iterations taken for the solver to converge.')

    args = parser.parse_args()

    # Verify that there are arguments
    args_dict = vars(args)
    for k, v in args_dict.items():
        if v:
            if k not in ['Solver', 'InvReg']:
                args_dict[k] = int(v)
            elif k=='InvReg':
                args_dict[k] = float(v)
            print(f'Value detected for {k}: {v}')
        else:
            default_val = default_dict[k]
            args_dict[k] = default_val
            print(f'- No value detected for {k}. Using default value: {default_val}')

    return args_dict

def load_data(folder):
    print('\nLoading data ...')
    filenames = [f for f in os.listdir(folder) if f.endswith('.csv')]

    df_list = []
    for filename in filenames:
        print('- Reading file:', filename)

        filepath = f'./{folder}/{filename}'
        color = filename.split('-')[-1][:-4]

        df = pandas.read_csv(filepath, delimiter=';')
        df.columns = [col.replace(' ', '_') for col in df.columns]
        df['color'] = color
        print('  DataFrame shape:', df.shape)

        df_list.append(df)

    df_concat = pandas.concat(df_list, ignore_index=True)
    print('- DataFrame shape:', df_concat.shape)

    return df_concat

def clean_data(df, replace_values={}):
    print('\nChecking for common data issues ...')
    series_missing = df.isna().sum()
    series_negatives = ( df.drop(columns=['quality','color']) < 0 ).sum()
    series_ph_high = ( df[['pH']] > 14 ).sum()

    print('- Missing values:', series_missing.sum())
    print('- Unexpected negative values:', series_negatives.sum())
    print('- Out of range values (pH):', series_ph_high.sum())

    if (replace_values):
        print('- Replacing column values:')
    for k in replace_values:
        print(f'  - {k}')
        df[k] = df[k].replace(replace_values[k])

    if (series_missing.sum() + series_negatives.sum() + series_ph_high.sum()) == 0:
        print('Data clean')
    else:
        print('Need further data cleaning')

    return df

def preprocess_input(random_state, drop_cols=[], pca_n_components=None):
    print('\nMaking preprocessing step ...')
    print('- Removing columns:', ', '.join(drop_cols))
    preprocess_col_transf = ColumnTransformer([
        ('Remove unused columns', 'drop', drop_cols)
    ], remainder='passthrough')

    if (pca_n_components):
        print('- Adding PCA')
        preprocessing = make_pipeline(
            preprocess_col_transf,
            PCA(n_components=pca_n_components, svd_solver='full', random_state=random_state)
        )
    else:
        print('- Skipping PCA')
        preprocessing = preprocess_col_transf

    print('Done')
    return preprocessing

def split_data(df, target_col, test_size, random_state):
    print('\nSplitting data into train and test sets ...')
    X = df.drop(columns=[target_col]).copy()
    y = df[[target_col]].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print('Done')
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, preprocessing, random_state, c, solver, max_iter):
    print('\nTraining model ...')
    print('- Make pipeline')
    pipeline = make_pipeline(
        preprocessing,
        LogisticRegression(random_state=random_state, C=c, solver=solver, max_iter=max_iter)
    )

    print('- Fit training data')
    pipeline.fit(X_train, y_train)

    print('Done')
    return pipeline

def evaluate_model(y_true, y_pred):
    print('\nModel Evaluation:')
    eval_fn_dict = {
        'Recall': recall_score,
        'Precision': precision_score,
        'Accuracy': accuracy_score
    }

    eval_score_dict = {}
    for metric, fn in eval_fn_dict.items():
        score_value = fn(y_true, y_pred)
        eval_score_dict[metric] = score_value
        print(f'- {metric}:\t{score_value}')
    
    return eval_score_dict

def graph_confusion_matrix(y_true, y_pred, display_labels):
    print('\nGraphing confusion matrix ...')
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm_disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)

    print('- Exporting as png')
    cm_disp.plot().figure_.savefig('./confusion_matrix.png')

    print('Done')
    return cm

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MARK: Funcion principal

def main(random_state=1, c=1.0, solver='lbfgs', max_iter=100):
    df = load_data('./wine_quality')
    df = clean_data(df, replace_values={'color':{'red':0, 'white':1}})

    X_train, X_test, y_train, y_test = split_data(df, target_col='color', test_size=0.2, random_state=1)

    preprocessing = preprocess_input(random_state=random_state, drop_cols=['quality'])
    model = train_model(
        X_train, y_train, preprocessing=preprocessing,
        random_state=random_state, c=c, solver=solver, max_iter=max_iter
    )

    y_pred = model.predict(X_test)
    eval_score_dict = evaluate_model(y_test, y_pred)
    cm = graph_confusion_matrix(y_test, y_pred, display_labels=['red','white'])

    print('\nMLFlow Tracking ...')
    with mlflow.start_run() as run:
        # Log random seed
        mlflow.log_param('random_state', random_state)

        # Log Hyperparameters
        mlflow.log_param('c', c)
        mlflow.log_param('solver', solver)
        mlflow.log_param('max_iter', max_iter)

        # Log Metrics
        for k, v in eval_score_dict.items():
            mlflow.log_metric(k, v)

        # Log artifact
        mlflow.log_artifact('./confusion_matrix.png')

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            name='wine_color_classifier'
        )
    print('Done')

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MARK: Ejecutar

# Configuracion
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('Script_Base')

print('Tracking URI:', mlflow.get_tracking_uri())
print('Experiment:', mlflow.get_experiment_by_name('Script_Base'))

# Funcion principal
if __name__=='__main__':
    arg_dict = arg_parse()
    main(
        random_state=arg_dict['RandomSeed'],
        c=arg_dict['InvReg'],
        solver=arg_dict['Solver'],
        max_iter=arg_dict['MaxIter']
    )