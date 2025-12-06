from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

class LinearModel:
    """A wrapper for Linear Regression model used in the Any% predictor."""

    def __init__(self):
        # Initialize the scikit-learn Linear Regression model
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        """Train the Linear Regression model."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test data."""
        y_pred = self.model.predict(X_test)
        return {
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": root_mean_squared_error(y_test, y_pred)  # ← Remove squared=False
        }