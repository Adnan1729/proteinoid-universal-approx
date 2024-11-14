# train_classifier.py (scripts/train_classifier.py)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import load_spike_data
from src.models.neural_network import engineer_features, create_model, get_callbacks
from src.utils.metrics import calculate_classification_metrics
import argparse

def main(args):
    # Load data
    dataframes = load_spike_data(args.data_path)
    
    # Prepare training and test data
    train_data = pd.concat(dataframes[:4], ignore_index=True)
    test_data = dataframes[4]
    
    # Engineer features
    train_data = engineer_features(train_data)
    test_data = engineer_features(test_data)
    
    # Prepare features
    feature_columns = ['Time', 'Time_Diff', 'ISI', 'CV_ISI'] + \
                     [col for col in train_data.columns if col.startswith(('Rolling_', 'Sin_', 'Cos_'))]
    
    X_train = train_data[feature_columns].values
    y_train = train_data['Spike'].values
    X_test = test_data[feature_columns].values
    y_test = test_data['Spike'].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = create_model(X_train_scaled.shape[1])
    history = model.fit(
        X_train_scaled, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=get_callbacks(),
        verbose=1
    )
    
    # Evaluate model
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()
    metrics = calculate_classification_metrics(y_test, y_pred)
    
    print("\nClassification Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train spike classification model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to spike data CSV file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    args = parser.parse_args()
    main(args)
