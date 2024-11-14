# File path
file_path = r"C:\...relu\meta_data.csv"
spike_data = pd.read_csv(file_path)

# Function to process dataset
def process_dataset(df, column_name):
    data = df[[column_name, df.columns[df.columns.get_loc(column_name) + 1]]].copy()
    data.columns = ['Time', 'Spike']
    data = data.iloc[1:]  # Remove the header row
    data['Time'] = pd.to_numeric(data['Time'], errors='coerce')
    data['Spike'] = pd.to_numeric(data['Spike'], errors='coerce')
    data = data.dropna()
    data['Dataset'] = column_name
    return data

dataframes = []
for column in ['Data1', 'Data2', 'Data3', 'Data4', 'Data5']:
    if column in spike_data.columns:
        df = process_dataset(spike_data, column)
        dataframes.append(df)
    else:
        print(f"Column {column} not found in the dataset")
