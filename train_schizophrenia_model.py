import GEOparse #to download and parse GEO datasets
import pandas as pd #to manipulate data - mainly DataFrames
from sklearn.preprocessing import StandardScaler #to normalize and scale the data
from sklearn.impute import SimpleImputer #to handle missing values in the data
from sklearn.model_selection import train_test_split #to split the data into training and testing sets
from sklearn.linear_model import LogisticRegression #to train a logistic regression model
from sklearn.metrics import accuracy_score, roc_auc_score #to evaluate  the model
import joblib #to save the trained model

def train_schizophrenia_model():
    #Step 1: Download the data using GEOparse
    gse = GEOparse.get_GEO(geo="GSE21138")

    #Step 2: Inspect the metadata
    #Print the metadata to understand the structure of the dataset
    # print("Metadata:")
    # print(gse.metadata)

    #Step 3: Extract expression data
    #The expression data is stored in the 'VALUE' column of the samples
    data = gse.pivot_samples('VALUE')

    #Print the column names of the expression data
    print("\nColumn Names:")
    print(data.columns)
    #Print the first few rows of the raw data to understand its structure
    print("Raw Data:")
    print(data.head())

    #Step4: Handle missing values
    #We use SimpleImputer to fill missing values with the mean of each column
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)
    print("Imputed Data:")
    print(data_imputed)

    #Convert the imputed data back to a DataFrame
    data_imputed_df = pd.DataFrame(data_imputed, columns = data.columns, index=data.index)
    print("\nData after Imputation:")
    print(data_imputed_df)

    #Step5: Normalize the data
    #We use StandardScaler to normalize the data so that each feature has a mean of 0 and a standard deviation of 1
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_imputed)
    print("Data after Normalization:")
    print(data_normalized)

    #Convert the normalized data back to a DataFrame
    data_normalized_df = pd.DataFrame(data_normalized, columns = data.columns, index=data.index)
    print("\nData after Normalization:")
    print(data_normalized_df)

    #Step6: Create labels based on sample titles
    #Extract sample titles to determine the labels
    sample_info = gse.phenotype_data[["title"]].copy() #Create a copy to avoid the warning
    print("Sample info")
    print(sample_info)
    sample_info['label'] = sample_info['title'].apply(lambda x: 1 if 'Scz' in x else 0)
    print("\nSample Titles:")
    print(sample_info)

    #Ensure the indices match between sample_info and data_normalized_df
    sample_info.index = data_normalized_df.columns
    print("Sample info index aligned with columns")
    print(sample_info)
    #Transpose the normalized data to align samples as rows
    data_normalized_df = data_normalized_df.T
    print("Sample info transposed")
    print(data_normalized_df)

    #Merge the labels with the normalized data
    data_normalized_df['Schizophrenia'] = sample_info['label']
    print("\nData after Labeling:")
    print(data_normalized_df.head())

    #Step7: Prepare the data for modeling
    #Separate features and labels 
    # X will contain all the features (gene expression levels)
    # Y will contain the labels (0 for Control, 1 for Schizophrenia)
    X = data_normalized_df.drop(columns=['Schizophrenia'])
    y = data_normalized_df['Schizophrenia']
    print("\nFeatures:")
    print(X.head())
    print("\nLabels:")
    print(y.head())

    #Split the data into training and testing sets
    # test_size = 0.2 means 20% of the data will be used for testing, and 80% for training
    #random_state = 42 ensures reproducibility of the results
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    #Step8: train the logistic regression model with increased iteration and different solver
    model = LogisticRegression(solver='saga', max_iter=5000)
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    print(model)

    # Step9: evaluate the model
    # X_test contains the testing features
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy: .2f}')

    #with probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1] #Get the probability of the positive class
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f'ROC AUC: {roc_auc: .2f}')
    joblib.dump(model, 'schizophrenia_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(imputer, 'imputer.pkl')

train_schizophrenia_model()


