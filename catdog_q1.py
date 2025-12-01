import os
import cv2
import glob
import numpy as np

from sklearn.preprocessing import LabelEncoder
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
    # 1. Load data (train & test)
    print("Loading training and testing data...")
    X_train, y_train = load_data("train")
    X_test, y_test = load_data("test")

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # 2. Label Encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # 3. One-hot encoding
    y_train_onehot = to_categorical(y_train_enc, num_classes=2)
    y_test_onehot = to_categorical(y_test_enc, num_classes=2)

    # 4. Build Model
    print("\nBuilding Neural Network (20 → 8 → 2)...")

    model = Sequential()
    model.add(Dense(20, activation="sigmoid", input_dim=X_train.shape[1]))
    model.add(Dense(8, activation="sigmoid"))
    model.add(Dense(2, activation="softmax"))

    # optimizer = pure SGD (수업에서 사용)
    optimizer = SGD(learning_rate=0.01)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. Training
    print("\nTraining model (epochs=10)...")
    model.fit(X_train, y_train_onehot, epochs=10, batch_size=32, verbose=1)

    # 5. Evaluation
    loss, acc = model.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"\nTest Accuracy: {acc:.4f}")

    # 6. Save Model
    model.save("catdog_best_model.h5")
    joblib.dump(le, "catdog_label_encoder.z")

    print("\nModel saved as 'catdog_best_model.h5'")
    print("LabelEncoder saved as 'catdog_label_encoder.z'")

    # 7. Test on internet images
    internet_folder = "internet_test"

    if os.path.isdir(internet_folder):
        print("\nTesting internet images...")
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


if __name__ == "__main__":
    main()
