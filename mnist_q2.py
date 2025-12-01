import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load MNIST CSV (28x28 flattened)
def load_mnist_csv(path):
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    X = data[:, 1:] / 255.0
    y = data[:, 0].astype(int)
    return X, y

# ---------------------------------------------------------
# Method 1: Logistic Regression
# ---------------------------------------------------------
def run_logistic_regression(X_train, y_train, X_test, y_test):
    print("\nRunning Logistic Regression...")

    model = LogisticRegression(
        max_iter=200,
        solver="saga",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Test Accuracy (Logistic Regression): {acc:.4f}")

    joblib.dump(model, "mnist_logistic.z")
    print("Logistic Regression model saved as 'mnist_logistic.z'.")

# ---------------------------------------------------------
# Method 2: K-Nearest Neighbors (KNN)
# ---------------------------------------------------------
def run_knn(X_train, y_train, X_test, y_test):
    print("\nRunning K-Nearest Neighbors (KNN)...")

    model = KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Test Accuracy (KNN, k=5): {acc:.4f}")

    joblib.dump(model, "mnist_knn.z")
    print("KNN model saved as 'mnist_knn.z'.")

def main():
    X_train, y_train = load_mnist_csv("mnist_train.csv")
    X_test, y_test = load_mnist_csv("mnist_test.csv")

    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)

    # run_logistic_regression(X_train, y_train, X_test, y_test)
    run_knn(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
