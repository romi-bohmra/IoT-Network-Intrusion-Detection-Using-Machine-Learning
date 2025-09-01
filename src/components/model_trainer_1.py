import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, accuracy_score
)
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

X_train = pd.read_csv("X_train.csv").values  # load as NumPy array
X_test = pd.read_csv("X_test.csv").values

y_train = pd.read_csv("y_train.csv").squeeze().values  # Series → NumPy array
y_test = pd.read_csv("y_test.csv").squeeze().values

df_train = pd.read_csv("df_train.csv")
df_test = pd.read_csv("df_test.csv")

def build_regularized_autoencoder(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.GaussianNoise(0.01)(inputs)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    bottleneck = layers.Dense(16, activation="relu", name="bottleneck")(x)
    x = layers.Dense(64, activation="relu")(bottleneck)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

def train_regularized_autoencoder(X_train, y_train, epochs=50, batch_size=512):
    X_train_benign = X_train[y_train == "Benign"]
    autoencoder = build_regularized_autoencoder(X_train.shape[1])
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    history = autoencoder.fit(
        X_train_benign, X_train_benign,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[es, rl],
        verbose=1
    )
    return autoencoder, history


def test_autoencoder(autoencoder, X_test, y_test, threshold=None):
    reconstructions = autoencoder.predict(X_test, verbose=0)
    mse = np.mean(np.square(X_test - reconstructions), axis=1)
    if threshold is None:
        benign_mask = (y_test == "Benign")
        threshold = np.percentile(mse[benign_mask], 95)
    preds = (mse > threshold).astype(int)
    y_binary = (y_test != "Benign").astype(int)
    return preds, mse, threshold, y_binary


def evaluate_detector(y_binary, preds, mse, threshold):
    precision, recall, f1, _ = precision_recall_fscore_support(y_binary, preds, average="binary")
    auc = roc_auc_score(y_binary, mse)
    acc = accuracy_score(y_binary, preds)
    print(f"Threshold: {threshold:.5f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    cm = confusion_matrix(y_binary, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Attack"],
                yticklabels=["Normal","Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Packet-Level Autoencoder)")
    plt.show()
    return acc, precision, recall, f1, auc


def save_packet_data_with_predictions(packet_df_test, preds, filename="packet_test_predictions.csv"):
    packet_df_test = packet_df_test.copy()
    packet_df_test['Flagged'] = preds
    packet_df_test.to_csv(filename, index=False)
    print(f"✅ Packet data with predictions saved to {filename}")
    return packet_df_test

if __name__ == "__main__":
    

    # Train autoencoder
    autoencoder, history = train_regularized_autoencoder(X_train, y_train, epochs=10, batch_size=256)

    # Test autoencoder
    preds, mse, threshold, y_binary = test_autoencoder(autoencoder, X_test, y_test)

    with open("preds_data.pkl", "wb") as f:
        pickle.dump({"preds": preds, "mse": mse, "threshold": threshold, "y_binary": y_binary}, f)
    
    # Evaluate
    evaluate_detector(y_binary, preds, mse, threshold)

    # Save packet-level test data with predictions
    packet_df_test = save_packet_data_with_predictions(df_test, preds)