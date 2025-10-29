import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

CLASS_NAMES = {0: 'Fallen', 1: 'Standing', 2: 'Sitting/Other'}

def load_and_prepare_data():
    print("Loading datasets...")
    train_df = pd.read_csv('fall_train_labels_clean.csv')
    test_df = pd.read_csv('fall_test_labels_clean.csv')

    def extract_features(label_str):
        try:
            label = ast.literal_eval(label_str)
            class_label = int(label[0])
            features = label[1:5] if len(label) >= 5 else [0, 0, 0, 0]
            if len(features) >= 4 and features[3] > 0:
                aspect_ratio = features[2] / features[3]
            else:
                aspect_ratio = 0
            features.append(aspect_ratio)
            return features, class_label
        except:
            return [0, 0, 0, 0, 0], 0

    train_features, train_labels = [], []
    for label_str in train_df['label']:
        feat, lbl = extract_features(label_str)
        train_features.append(feat)
        train_labels.append(lbl)

    test_features, test_labels = [], []
    for label_str in test_df['label']:
        feat, lbl = extract_features(label_str)
        test_features.append(feat)
        test_labels.append(lbl)

    X_train = np.array(train_features, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.int32)
    X_test = np.array(test_features, dtype=np.float32)
    y_test = np.array(test_labels, dtype=np.int32)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features per sample: {X_train.shape[1]}")
    print(f"\nClass distribution (Training):")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {CLASS_NAMES[cls]}: {count}")

    return X_train, y_train, X_test, y_test

def build_keras_model(input_dim, num_classes=3):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    print("=" * 70)
    print("FALL DETECTION MODEL TRAINING")
    print("=" * 70)
    print()

    X_train, y_train, X_test, y_test = load_and_prepare_data()
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    print(f"\nValidation samples: {len(X_val)}")
    print("\nBuilding model...")
    try:
        model = build_keras_model(X_train.shape[1])
        print("Model architecture:")
        model.summary()

        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]

        print("\n" + "=" * 70)
        print("TRAINING STARTED")
        print("=" * 70)
        history = model.fit(
            X_train_split, y_train_split,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )

        print("\n" + "=" * 70)
        print("MODEL EVALUATION ON TEST SET")
        print("=" * 70)
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(
            y_test, y_pred, 
            target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())],
            digits=3
        ))

        print("\n" + "=" * 70)
        print("CONFUSION MATRIX")
        print("=" * 70)
        cm = confusion_matrix(y_test, y_pred)
        print("\nRows = Actual, Columns = Predicted")
        print(f"{'':15s} {'Fallen':>12s} {'Standing':>12s} {'Sitting':>12s}")
        class_labels = ['Fallen', 'Standing', 'Sitting/Other']
        for i, label in enumerate(class_labels):
            print(f"{label:15s}", end="")
            for j in range(len(class_labels)):
                print(f"{cm[i][j]:12d}", end="")
            print()

        model.save('fall_detection_model.h5')
        print("\n" + "=" * 70)
        print("Model saved as: fall_detection_model.h5")
        print("=" * 70)

        import json
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }

        with open('training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)
        print("Training history saved as: training_history.json")
        return model, history, test_accuracy

    except ImportError:
        print("\nERROR: TensorFlow not found!")
        print("Please install TensorFlow: pip install tensorflow")
        return None, None, None

if __name__ == "__main__":
    model, history, accuracy = train_model()

    if model is not None:
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
        print("\nOutput files:")
        print("  - fall_detection_model.h5 (trained model)")
        print("  - training_history.json (training metrics)")
