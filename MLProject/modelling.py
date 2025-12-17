import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    # 1. Set nama eksperimen agar mudah dicari
    mlflow.set_experiment("Titanic_Prediction")
    
    # 2. Aktifkan autolog sebelum memulai training
    mlflow.autolog()

    df = pd.read_csv("titanic_clean.csv")

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. WAJIB: Gunakan start_run() agar data benar-benar tersimpan ke mlruns
    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("Accuracy:", acc)
        # Metrik tambahan akan otomatis dicatat oleh autolog()

if __name__ == "__main__":
    main()
