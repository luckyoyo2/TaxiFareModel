# imports
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ("dist_trans", DistanceTransformer()),
            ("std_scaler", StandardScaler())
        ])

        time_pipe = Pipeline([("time_enc",
                               TimeFeaturesEncoder('pickup_datetime')),
                              ("robust_scaler", RobustScaler())])

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
            ], remainder="drop")

        pipe = Pipeline([
            ("preproc_pipe", preproc_pipe),
            ("lin_model", LinearRegression())
        ])

        return pipe

    def run(self):
        """set and train the pipeline"""
        pipeline = self.set_pipeline()
        pipeline.fit(self.X, self.y)
        return pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        pipeline = self.run()
        y_pred = pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse



if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df["fare_amount"]
    X  = df.drop("fare_amount", axis=1)
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    print(trainer.evaluate(X_test, y_test))
