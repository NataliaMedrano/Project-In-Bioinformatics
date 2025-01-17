from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd 
from sklearn.preprocessing import StandardScaler    
from sklearn.impute import SimpleImputer
import os

#Initialize the Flask application
app = Flask(__name__)

#Load the trained model, scaler, and imputer from disk
model = joblib.load("schizophrenia_model.pkl")  
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")

@app.route('/')
def home():
    """
    Render the home page with the input form.
    This function is called when the user access the root URL.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle the form submission, process the input data, and return the prediction result.
    This function is called when the user submits the form on the home page.
    """
    if request.method == 'POST':
        #Check if the post request has the file part
        if 'input_file' not in request.files:
            return redirect(request.url)
        
        file = request.files['input_file']

        #If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            #Read the uploaded file into a DataFrame
            input_df = pd.read_csv(file, header = 0, index_col = 0, usecols=['ID_REF' , 'VALUE'])

            #Check for specific IDs and their values for schizophrenia - User case 01
            specific_ids = ['208817_at', '208818_s_at', '213981_at', '216204_at']
            threshold = 7.15
            high_risk = False

            for specific_id in specific_ids:
                if specific_id in input_df.index and input_df.loc[specific_id, 'VALUE']>threshold:
                    high_risk = True
                    break
            
            if high_risk:
                prediction = "Between 50 and 100"

            else:
            
            #Handle missing values using SimpleImputer
                data_imputed = imputer.fit_transform(input_df)

                #Convert the imputed data back to a DataFame
                data_imputed_df = pd.DataFrame(data_imputed, columns = input_df.columns, index=input_df.index)

                #Normalize the data using StandardScaler
                data_normalized = scaler.fit_transform(data_imputed_df)

                #Convert the normalized data back to a DataFrame
                data_normalized_df = pd.DataFrame(data_normalized, columns = input_df.columns, index=input_df.index)

                model_features = model.feature_names_in_
                data_aligned = (data_normalized_df.T).reindex(columns=model_features, fill_value=0)

                if data_aligned.isnull().values.any():
                    print("Missing values detected in the input data.")
                    return "Missing values detected in the test data"
                
                #Make predictions using the trained model
                try:
                    predictions = model.predict(data_aligned)
                    prediction = predictions[0] * 100 #Convert the probability to a percentage
                except ValueError as e:
                    print(f"An error occurred: {e}")
                    return "Not possible for prediction"
            
            #Render the result page with the prediction
            return render_template('result.html', prediction=prediction)
        
if __name__ == '__main__':
    #Run the Flask application in debug mode
    app.run(debug=True)