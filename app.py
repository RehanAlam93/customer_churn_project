from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained machine learning model and scaler from the EDA folder
model = joblib.load(r'EDA\customer_churn_rf_model.pkl')
scaler = joblib.load(r'EDA\scaler.pkl')

# Home route to render the landing page (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Form page route to render the customer churn prediction form (churn.html)
@app.route('/customer_churn_prediction')
def form_page():
    return render_template('churn.html')

# Predict churn risk after collecting data from the web form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect customer name from the form input (default to 'Unknown Customer' if missing)
        customer_name = request.form.get('full_name', 'Unknown Customer')
        
        # Collect and parse all 10 required feature values from the form in exact order
        gender = float(request.form['gender'])
        tenure = float(request.form['tenure'])
        multiple_lines = float(request.form['MultipleLines'])
        internet_service = float(request.form['InternetService'])
        online_security = float(request.form['OnlineSecurity'])
        online_backup = float(request.form['OnlineBackup'])
        contract = float(request.form['Contract'])
        payment_method = float(request.form['PaymentMethod'])
        monthly_charges = float(request.form['MonthlyCharges'])
        total_charges = float(request.form['TotalCharges'])
        
        # Create a NumPy array with features in the exact required sequence (10 features)
        features = np.array([[gender,
                            tenure,
                            multiple_lines,
                            internet_service,
                            online_security,
                            online_backup,
                            contract,
                            payment_method,
                            monthly_charges,
                            total_charges]])
        
        # Apply the pre-fitted scaler to the input features
        scaled_features = scaler.transform(features)
        
        # Run prediction using the loaded random forest model
        prediction = model.predict(scaled_features)
        
        # Map the numerical prediction output to a readable string ('Yes' or 'No')
        if prediction[0] == 1:
            churn_risk = "Yes"
        else:
            churn_risk = "No"
            
        # Format the final output string to display on the web page template
        output = f"Customer Name: {customer_name} | Churn Risk: {churn_risk}"

        # Print the prediction log to the VS Code terminal
        print(f"Data : Name : {customer_name} , Prediction --> \"{churn_risk}\"")

        # Render the form template again along with the prediction result
        return render_template('churn.html', prediction_text=output)

    except Exception as e:
        # Handle unexpected exceptions, log the error, and display it on the UI
        print(f"Error: {str(e)}")
        return render_template('churn.html', prediction_text=f"Error: {str(e)}")

# Run the Flask development server in debug mode
if __name__ == '__main__':
    app.run(debug=True)