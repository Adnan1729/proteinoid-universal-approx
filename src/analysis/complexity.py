# Function F1
def F1(t):
    return 10 + (10 - 2*t) * np.cos(t * np.pi)

# Function F2 (vectorized)
def F2(x):
    frac_part = x - np.floor(x)
    n = np.ceil(np.log10(1 / frac_part)).astype(int)
    return np.floor((10**n) * frac_part)

# Function to find the point in the dataset just less than x(t)
def find_point_less_than(df, value):
    return df[df['Time'] < value]['Time'].max()

# Function to find points with the same first significant digit
def find_points_with_same_digit(df, x, d):
    return df[np.isclose(F2(df['Time']), d)]['Time'].tolist()

