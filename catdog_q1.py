import os
import cv2
import glob
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import joblib

# Data loading

IMG_SIZE = (32, 32)
CLASSES = ["Cat", "Dog"]

def load_data(path):
    dataList = []
    labelList = []

    for cls in CLASSES:
        files = glob.glob(os.path.join(path, cls, "*"))
        for address in files:
            image = cv2.imread(address)
            if image is None:
                continue

            # resize
            image = cv2.resize(image, IMG_SIZE)

            # normalization
            image = image / 255.0

            # flatten
            image = image.flatten()

            dataList.append(image)
            labelList.append(cls)

    X = np.array(dataList, dtype=np.float32)
    y = np.array(labelList)
    return X, y

# Main

def main():
    # Load data
    print("Loading training and testing data...")
    X_train, y_train = load_data("train")
    X_test, y_test = load_data("test")

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # Label encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # One-hot encoding
    y_train_onehot = to_categorical(y_train_enc, num_classes=2)
    y_test_onehot = to_categorical(y_test_enc, num_classes=2)

    # ============================================================
    # Neural Network (MLP)
    # ============================================================
    print("\nBuilding Neural Network (MLP: 20 → 8 → 2)...")

    model = Sequential()
    model.add(Dense(20, activation="sigmoid", input_dim=X_train.shape[1]))
    model.add(Dense(8, activation="sigmoid"))
    model.add(Dense(2, activation="softmax"))

    optimizer = SGD(learning_rate=0.01)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nTraining Neural Network (epochs=10)...")
    model.fit(X_train, y_train_onehot, epochs=10, batch_size=32, verbose=1)

    nn_loss, nn_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"\nTest Accuracy (Neural Network): {nn_acc:.4f}")

    # Save NN model
    model.save("catdog_best_model.h5")
    joblib.dump(le, "catdog_label_encoder.z")

    print("\nNeural Network model saved as 'catdog_best_model.h5'")
    print("LabelEncoder saved as 'catdog_label_encoder.z'")

    # Test NN on internet images
    internet_folder = "internet_test"

    if os.path.isdir(internet_folder):
        print("\nTesting internet images (Neural Network)...")
        for path in glob.glob(os.path.join(internet_folder, "*")):
            img = cv2.imread(path)
            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0
            feat = img.flatten().reshape(1, -1)

            pred_raw = model.predict(feat)[0]
            pred = np.argmax(pred_raw)
            label = le.inverse_transform([pred])[0]

            print(f"{os.path.basename(path)} → predicted: {label}")
    else:
        print("\nNo 'internet_test/' folder found.")

    # ============================================================
    # Logistic Regression
    # ============================================================
    print("\nRunning Logistic Regression...")

    log_model = LogisticRegression(max_iter=2000)
    log_model.fit(X_train, y_train_enc)

    log_preds = log_model.predict(X_test)
    log_acc = accuracy_score(y_test_enc, log_preds)

    print(f"Test Accuracy (Logistic Regression): {log_acc:.4f}")

    joblib.dump({"model": log_model, "label_encoder": le}, "logistic_model.z")
    print("Logistic Regression model saved as 'logistic_model.z'.")


    # ============================================================
    # K-Nearest Neighbors (KNN)
    # ============================================================
    print("\nRunning K-Nearest Neighbors (KNN)...")

    from sklearn.neighbors import KNeighborsClassifier

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train_enc)

    knn_preds = knn_model.predict(X_test)
    knn_acc = accuracy_score(y_test_enc, knn_preds)

    print(f"Test Accuracy (KNN, k=5): {knn_acc:.4f}")

    joblib.dump({"model": knn_model, "label_encoder": le}, "knn_model.z")
    print("KNN model saved as 'knn_model.z'.")

    # ============================================================
    # Support Vector Machine (SVM)
    # ============================================================
    print("\nRunning Support Vector Machine (SVM)...")

    from sklearn.svm import SVC

    svm_model = SVC(kernel="rbf", C=1.0)
    svm_model.fit(X_train, y_train_enc)

    svm_preds = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test_enc, svm_preds)

    print(f"Test Accuracy (SVM, RBF kernel): {svm_acc:.4f}")

    joblib.dump({"model": svm_model, "label_encoder": le}, "svm_model.z")
    print("SVM model saved as 'svm_model.z'.")


if __name__ == "__main__":
    main()
