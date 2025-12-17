import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    
    # Load dataset
    df = pd.read_csv("MLProject/titanic_clean.csv")
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        # Buat dan train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Prediksi dan hitung akurasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Manual logging parameter dan metric
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", acc)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log artefak tambahan (misal prediksi)
        predictions = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
        predictions.to_csv("predictions.csv", index=False)
        mlflow.log_artifact("predictions.csv")

        print("Accuracy:", acc)

if __name__ == "__main__":
    main()
