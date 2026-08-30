from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model and scaler from the EDA folder
model = joblib.load(r'EDA/random_forest_churn_model.pkl')
scaler = joblib.load(r'EDA/scaler.pkl')

# Home route for index.html
@app.route('/')
def home():
    return render_template('index.html')

# Form page route for churn.html
@app.route('/customer_churn_prediction')
def form_page():
    return render_template('churn.html')

# Predict churn risk after collecting data from the web form
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect customer name from the form input
        customer_name = request.form.get('full_name', 'Unknown Customer')
        
        # Collect all required feature values
        tenure = float(request.form['tenure'])
        contract = float(request.form['Contract'])
        monthly_charges = float(request.form['MonthlyCharges'])
        total_charges = float(request.form['TotalCharges'])
        
        # Create an array of features and apply the trained scaler
        features = np.array([[tenure, 
                            contract,
                            monthly_charges,
                            total_charges]])
        
        scaled_features = scaler.transform(features)
        
        # Run prediction using the random forest model
        prediction = model.predict(scaled_features)
        
        # Map numerical prediction to readable text ('Yes' or 'No')
        if prediction[0] == 1:
            churn_risk = "Yes"
        else:
            churn_risk = "No"
            
        # Format the output for the web page template
        output = f"Customer Name: {customer_name} | Churn Risk: {churn_risk}"

        # Print the result in your requested format to the VS Code terminal
        print(f"Data : Name : {customer_name} , Preduction --> \"{churn_risk}\"")

        return render_template('churn.html', prediction_text=output)

    except Exception as e:
        # Handle exceptions and print errors if any occur
        print(f"Error: {str(e)}")
        return render_template('churn.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)