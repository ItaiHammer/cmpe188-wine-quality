
# Functions for preparing wine-quality data before model training.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def prepare_data():
    # Load and check the data
    df = pd.read_csv("data/10_wine_quality_train.csv")
    print(df.head())
    print(df.info())
    print(df.describe())

    # Check and remove duplicate rows
    print("Duplicate rows:", df.duplicated().sum())
    df = df.drop_duplicates()
    print("Rows after removing duplicates:", len(df))

    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Check data types
    print("\nData types:")
    print(df.dtypes)

    # Define the target variable and features
    target = "good_quality"
    x = df.drop(columns=[target])
    y = df[target]

  
    # First split: 80% temporary data, 20% test
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Second split: 60% train, 20% validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_temp
    )

    print("Training rows:", len(x_train))
    print("Validation rows:", len(x_val))
    print("Testing rows:", len(x_test))

    # Scale features using training data only
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    print("Scaling completed!")
    print("Scaled training shape:", x_train_scaled.shape)
    print("Scaled validation shape:", x_val_scaled.shape)
    print("Scaled testing shape:", x_test_scaled.shape)

    # Save feature names
    feature_names = x.columns.tolist()

    return (
        x_train_scaled, x_val_scaled, x_test_scaled,
        y_train, y_val, y_test,
        scaler, feature_names
    )

# Run preprocessing when this file is executed directly
if __name__ == "__main__":
    prepare_data()
