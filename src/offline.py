import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error


def batching(df, outputs, test_frac):
    model = Batching()
    model.train(df, outputs, test_frac)
    model.predict()

    return model


class Batching:
    def train(self, df, outputs, test_frac):
        self.data = df.copy().sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle the data
        self.outputs = outputs
        self.test_frac = test_frac
        train = self.data.copy().head(int(len(self.data)*(1 - self.test_frac)))

        self.model = dict()

        for out in self.outputs:
            X = train.copy().drop(columns=self.outputs)
            Y = train.copy()[[out]]

            model = XGBRegressor(
                booster="gbtree",
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=7,
                min_child_weight=1,
                colsample_bytree=0.8,
                subsample=0.8,
                random_state=42,
            )
            model.fit(X, Y)

            self.model[out] = model

    def predict(self):
        test = self.data.copy().tail(int(len(self.data)*self.test_frac))
        
        self.metric = dict()
        self.predictions = dict()

        for out in self.outputs:
            X = test.copy().drop(columns=self.outputs)
            Y = test.copy()[[out]]

            model = self.model[out]
            y_pred = model.predict(X)
            y_true = Y.to_numpy().ravel()

            metric = mean_squared_error(
                y_true=y_true, 
                y_pred=y_pred, 
                squared=False,
            )
            metric = f"RMSE: {round(metric, 6)}"

            predictions = pd.DataFrame({
                "Actual": y_true,
                "Predicted": y_pred,
            })

            self.metric[out] = metric
            self.predictions[out] = predictions
