# Create the model
def create_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def engineer_features(df):
    df = df.sort_values('Time')
    df['Time_Diff'] = df['Time'].diff().fillna(0)
    
    # Inter-spike intervals
    df['ISI'] = df['Time_Diff']
    
    # Coefficient of variation of ISIs
    window_size = 10
    df['CV_ISI'] = df['ISI'].rolling(window=window_size).std() / df['ISI'].rolling(window=window_size).mean()
    
    # Rolling window features
    window_sizes = [3, 5, 10]
    for size in window_sizes:
        df[f'Rolling_Mean_{size}'] = df['Spike'].rolling(window=size).mean()
        df[f'Rolling_Std_{size}'] = df['Spike'].rolling(window=size).std()
    
    # Fourier features
    for period in [5, 10, 20]:
        df[f'Sin_{period}'] = np.sin(2 * np.pi * df['Time'] / period)
        df[f'Cos_{period}'] = np.cos(2 * np.pi * df['Time'] / period)
    
    return df.fillna(0)
