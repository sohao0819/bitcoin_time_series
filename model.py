from statsmodels.api import tsa


class BitcoinModel:

    def __init__(self):

        self.model = None

    def preprocess_training_data(self, df):

        bitcoin_value_diff = df['Value'].diff(7).dropna()
        X = bitcoin_value_diff
        y = None
        return X, y

    def fit(self, X, y):

        arma = tsa.ARMA(X, order=(8, 1))
        self.model = arma.fit()
        return self.model

    def preprocess_unseen_data(self, df):

        X = df['Date']
        return X

    def predict(self, X):

        return self.model.predict(start=X[0])
