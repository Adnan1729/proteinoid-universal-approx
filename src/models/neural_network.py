# neural_network.py (src/models/neural_network.py)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """
    Engineer features for the neural network model.
    
    Args:
        df (pd.DataFrame): Input dataframe with Time and Spike columns
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    df = df.sort_values('Time')
    df['Time_Diff'] = df['Time'].diff().fillna(0)
    df['ISI'] = df['Time_Diff']
    
    window_size = 10
    df['CV_ISI'] = df['ISI'].rolling(window=window_size).std() / df['ISI'].rolling(window=window_size).mean()
    
    window_sizes = [3, 5, 10]
    for size in window_sizes:
        df[f'Rolling_Mean_{size}'] = df['Spike'].rolling(window=size).mean()
        df[f'Rolling_Std_{size}'] = df['Spike'].rolling(window=size).std()
    
    for period in [5, 10, 20]:
        df[f'Sin_{period}'] = np.sin(2 * np.pi * df['Time'] / period)
        df[f'Cos_{period}'] = np.cos(2 * np.pi * df['Time'] / period)
    
    return df.fillna(0)

def create_model(input_shape):
    """
    Create the neural network model.
    
    Args:
        input_shape (int): Number of input features
        
    Returns:
        tf.keras.Model: Compiled neural network model
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_callbacks():
    """
    Get callbacks for model training.
    
    Returns:
        list: List of Keras callbacks
    """
    return [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]
