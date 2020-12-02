"""
Model training code
"""
import os
import argparse
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score, precision_score, recall_score
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")

#Setup logger
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)

#Define all directory paths
curr_dir = os.path.dirname(__file__)
data_dir = os.path.join(curr_dir, 'data')
mlflow_tracking_dir = os.path.join(curr_dir, 'mlruns')

#Data Loading & Cleaning
data = pd.read_csv(os.path.join(data_dir, "final.csv"))
try:
    data = data.drop(['v0093'], axis=1)
    data = data.drop(['v0106'], axis=1)
    data = data.drop(['v0095'], axis=1)
    data = data.drop(['v4920'], axis=1)
    data = data.drop(['v0105'], axis=1)
except KeyError:
    logger.warning("Some or all columns to be dropped do not exist")


# Set features and target variables
features = data.iloc[:, data.columns != 'v6392']
target = data.iloc[:, data.columns == 'v6392']

# Train test split
def split_test_data(train_test_split_size, random_state=1):
    """
    Splits data into train and test samples
    :param train_test_split_size: Size of test data, example .3 for 30%
    :param random_state: Radom state so it always split the same way
    :return: x_train, x_test, y_train, y_test
    """
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=train_test_split_size, random_state=random_state)
    return x_train, x_test, y_train, y_test


def evaluate_model(y_true, y_pred, pred_score):
    """
    Evaluates classification models and calculate multiple metrics
    :param y_true:
    :param y_pred:
    :param pred_score:
    :return: accuracy score, precision score, recall_score, roc_auc_score
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, pred_score)
    return acc, precision, recall, roc_auc

if __name__ == "__main__":

    # MLFLOW parameters setting
    mlflow.set_tracking_uri(f"file:{mlflow_tracking_dir}")
    exp_name = "RandomForestClassification"
    mlflow.set_experiment(exp_name)

    parser = argparse.ArgumentParser(description='Get Input Values')

    #Parameter setting
    parser.add_argument('--split-size', dest="train_test_split_size", default=0.3, type=float, help='Test Train Split Size')
    parser.add_argument('--random-state', dest="random_state", default=1, type=int, help='Random State')

    # Model hyperparameters setting
    parser.add_argument('--max-depth', dest="max_depth", default=8, type=int)
    parser.add_argument('--criterion', dest="criterion", default='gini', type=str)
    parser.add_argument('--min-impurity-decrease', dest="min_impurity_decrease", default=0.0, type=float)
    parser.add_argument('--min-samples-leaf', dest="min_samples_leaf", default=4, type=int)
    parser.add_argument('--min-samples-split', dest="min_samples_split", default=10, type=int)
    parser.add_argument('--min-weight-fraction-leaf', dest="min_weight_fraction_leaf", default=0.0, type=float)
    parser.add_argument('--n-estimators', dest="n_estimators", default=6, type=int)
    parser.add_argument('--oob-score', dest="oob_score", default=False, type=bool)


    args = parser.parse_args()

    #Test Train Split
    x_train, x_test, y_train, y_test = split_test_data(args.train_test_split_size, args.random_state)

    #Class Weight calc
    class_weight = dict({0: 1, 1: len(y_train[y_train.v6392 == 0]) / len(y_train[y_train.v6392 == 1])})

    # Model Training
    with mlflow.start_run():
        # Model definition
        RF = RandomForestClassifier(bootstrap=True,
                                    class_weight=class_weight,
                                    criterion=args.criterion,
                                    max_depth=args.max_depth, max_features='auto', max_leaf_nodes=None,
                                    min_impurity_decrease=args.min_impurity_decrease, min_impurity_split=None,
                                    min_samples_leaf=args.min_samples_leaf, min_samples_split=args.min_samples_split,
                                    min_weight_fraction_leaf=args.min_weight_fraction_leaf, n_estimators=args.n_estimators,
                                    oob_score=args.oob_score,
                                    random_state=args.random_state,
                                    verbose=0, warm_start=False)
        # Model fitting
        RF.fit(x_train, y_train)

        #Model inference
        y_pred_score = RF.predict_proba(x_test)  # Predicting probability
        y_pred_score = pd.DataFrame(y_pred_score[:, 1])
        y_pred = RF.predict(x_test)

        #Model evaluation
        acc, precision, recall, roc_auc = evaluate_model(y_test, y_pred, y_pred_score)

        # Logging metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", roc_auc)

        #logging model parameters
        mlflow.log_param("test_size", args.train_test_split_size)
        mlflow.log_param('random_state', args.random_state)
        mlflow.log_param('max_depth', args.max_depth)
        mlflow.log_param('criterion', args.criterion)
        mlflow.log_param('min_impurity_decrease', args.min_impurity_decrease)
        mlflow.log_param('min_samples_leaf', args.min_samples_leaf)
        mlflow.log_param('min_samples_split', args.min_samples_split)
        mlflow.log_param('min_weight_fraction_leaf', args.min_weight_fraction_leaf)
        mlflow.log_param('n_estimators', args.n_estimators)
        mlflow.log_param('oob_score', args.oob_score)
        mlflow.sklearn.log_model(RF, "random_forest_model")

        #Logging model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(RF, "random_forest_model", registered_model_name="RandomForestModel")
        else:
            mlflow.sklearn.log_model(RF, "random_forest_model")
