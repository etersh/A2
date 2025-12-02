import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
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

# ---------------------------------------------------------
# Method 3: Support Vector Machine (SVM, RBF Kernel)
# ---------------------------------------------------------
def run_svm(X_train, y_train, X_test, y_test):
    print("\nRunning Support Vector Machine (SVM, RBF kernel)...")

    model = SVC(
        kernel="rbf",
        gamma="scale"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Test Accuracy (SVM, RBF): {acc:.4f}")

    joblib.dump(model, "mnist_svm.z")
    print("SVM model saved as 'mnist_svm.z'.")

# ---------------------------------------------------------
# Method 4: Neural Network (MLP: 784 → 128 → 10)
# ---------------------------------------------------------
def run_mlp(X_train, y_train, X_test, y_test):
    print("\nRunning Neural Network (MLP: 784 → 128 → 10)...")

    # one-hot encoding
    y_train_oh = to_categorical(y_train, num_classes=10)
    y_test_oh = to_categorical(y_test, num_classes=10)

    # model
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(784,)))
    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train, y_train_oh,
        epochs=10,
        batch_size=128,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)
    print(f"Test Accuracy (MLP): {acc:.4f}")

    model.save("mnist_mlp.h5")
    print("Neural Network model saved as 'mnist_mlp.h5'.")

def main():
    X_train, y_train = load_mnist_csv("mnist_train.csv")
    X_test, y_test = load_mnist_csv("mnist_test.csv")

    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)

    # run_logistic_regression(X_train, y_train, X_test, y_test)
    # run_knn(X_train, y_train, X_test, y_test)
    # run_svm(X_train, y_train, X_test, y_test)
    run_mlp(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
