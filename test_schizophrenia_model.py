import GEOparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

#Function to fetch GEO sample data and save to CSV
def fetch_geo_sample_to_csv(accession_number, output_file):
    try: 
        #Fetch the GEO sample data
        geo_sample = GEOparse.get_GEO(geo=accession_number, silent=False)
        print(f"Data for {accession_number} fetched successfully.")

        #Extract the data into a DataFrame
        data = geo_sample.table
        print("Data extracted successfully.")
        print(data.head())

        #Save the DataFrame to a CSV file
        data.to_csv(output_file)
        print(f"Data for {accession_number} saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

#Example usage
fetch_geo_sample_to_csv("GSM528851", "GSM528851_data.csv")    

# Function to test the schizophrenia model using the output file of fetch_geo_sample_to_csv function
def test_schizophrenia_model(model_file, test_file, scaler, imputer):
    #Load the trained model
    model = joblib.load(model_file)
    scaler = joblib.load(scaler)
    imputer = joblib.load(imputer)

    #Load the test data from CSV file
    test_data = pd.read_csv(test_file, header=0, index_col=0, usecols=['ID_REF', 'VALUE'])

    #Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(test_data)

    #Convert the imputed data back to a DataFrame
    data_imputed_df = pd.DataFrame(data_imputed, columns = test_data.columns, index=test_data.index)

    #Normalize the data using StandardScaler
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_imputed_df)

    #Convert the normalized data back to a DataFrame
    data_normalized_df = pd.DataFrame(data_normalized, columns = test_data.columns, index=test_data.index)

    model_features = model.feature_names_in_
    data_aligned = (data_normalized_df.T).reindex(columns=model_features, fill_value=0)

    if data_aligned.isnull().values.any():
        print("Missing values detected in the test data.")
    
    try: 
        predictions = model.predict(data_aligned)
        return predictions
    except ValueError as e:
        print(f"An error occurred: {e}")
        return "Not possible for prediction"

# Test the model using the output file of fetch_geo_sample_to_csv function
predictions = test_schizophrenia_model("schizophrenia_model.pkl", "GSM528851_data.csv", "scaler.pkl", "imputer.pkl")
print("PREDICTION:")
print(predictions)