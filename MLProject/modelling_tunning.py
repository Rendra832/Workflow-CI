import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    mlflow.autolog()

    df = pd.read_csv("titanic_clean.csv")

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model awal
    model = LogisticRegression(max_iter=1000)

    # Parameter tuning
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear"],
        "penalty": ["l1", "l2"]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="accuracy"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Best Params:", grid.best_params_)
    print("Accuracy after tuning:", acc)

if __name__ == "__main__":
    main()
