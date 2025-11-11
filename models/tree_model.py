from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class DecisionTreeModel:
    """Decision Tree model for Any% prediction."""

    def __init__(self, max_depth=None, random_state=42):
        # same setup as your notebook
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state
        )

    def train(self, X_train, y_train):
        """Train the Decision Tree model."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance and return metrics."""
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        return {
            "r2": r2,
            "mae": mae,
            "rmse": rmse
        }
