import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Extract (Read data)
def extract_data(file_path):
    print("📥 Reading data from:", file_path)
    return pd.read_csv(file_path)

# Step 2: Transform (Clean + Encode + Scale)
def transform_data(df):
    print("🔄 Transforming data...")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Numeric: fill missing + scale
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical: fill missing + encode
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])

    transformed_data = preprocessor.fit_transform(df)
    return transformed_data

# Step 3: Load (Save cleaned data)
def load_data(data, output_file):
    print("💾 Saving processed data to:", output_file)
    # Convert back to DataFrame if encoded
    if hasattr(data, "toarray"):
        data = data.toarray()
    pd.DataFrame(data).to_csv(output_file, index=False)

# Main pipeline run
if __name__ == "__main__":
    input_file = "sample_data.csv"
    output_file = "processed_data.csv"

    try:
        raw_data = extract_data(input_file)
        processed_data = transform_data(raw_data)
        load_data(processed_data, output_file)
        print("✅ ETL process completed successfully.")
    except Exception as e:
        print("❌ Error:", e)
import pandas as pd  # Add this only if it's not already at the top

