Copy data file names final.csv to data folder

mlflow server --backend-store-uri file:///home/arora/work/mlflow-sklearn-classification/mlruns/ --default-artifact-root file:///home/arora/work/mlflow-sklearn-classification/mlruns --host 0.0.0.0 --port 5000



##### To run directly from github
```mlflow run https://github.com/mlflow/mlflow-example.git -P train_test_split_size=0.33 random_state=2```
where all parameters can be passed after -P


mlflow models serve -m "/home/arora/work/mlflow-sklearn-classification/mlruns/0/d76ff64363f2492b8788bd46f4dc7a13/artifacts/random_forest_model" -h 0.0.0.0 -p 2125