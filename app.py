from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model/gradient_boosting_churn_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# home route index.html
@app.route('/')
def home():
    return render_template('index.html')

# form page churn.html
@app.route('/customer_churn_prediction')
def form_page():
    return render_template('churn.html')

# predict churn risk after taking data from churn.html
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # collect customer name
        customer_name = request.form.get('full_name', 'Unknown Customer')
        
        # take all feature values
        gender = float(request.form['gender'])
        senior_citizen = float(request.form['SeniorCitizen'])
        partner = float(request.form['Partner'])
        dependents = float(request.form['Dependents'])
        tenure = float(request.form['tenure'])
        phone_service = float(request.form['PhoneService'])
        multiple_lines = float(request.form['MultipleLines'])
        internet_service = float(request.form['InternetService'])
        online_security = float(request.form['OnlineSecurity'])
        online_backup = float(request.form['OnlineBackup'])
        device_protection = float(request.form['DeviceProtection'])
        tech_support = float(request.form['TechSupport'])
        streaming_tv = float(request.form['StreamingTV'])
        streaming_movies = float(request.form['StreamingMovies'])
        contract = float(request.form['Contract'])
        paperless_billing = float(request.form['PaperlessBilling'])
        payment_method = float(request.form['PaymentMethod'])
        monthly_charges = float(request.form['MonthlyCharges'])
        total_charges = float(request.form['TotalCharges'])
        
        # make array and scale features
        features = np.array([[gender, senior_citizen, partner, dependents, tenure, 
                              phone_service, multiple_lines, internet_service, 
                              online_security, online_backup, device_protection, 
                              tech_support, streaming_tv, streaming_movies, 
                              contract, paperless_billing, payment_method, 
                              monthly_charges, total_charges]])
        
        scaled_features = scaler.transform(features)
        
        # find prediction
        prediction = model.predict(scaled_features)
        
        # set result format
        if prediction[0] == 1:
            churn_risk = "Yes"
        else:
            churn_risk = "No"
            
        output = f"Customer Name: {customer_name} | Churn Risk: {churn_risk}"

        # Terminal print format (CSV saving has been removed)
        print(f"Data : {customer_name} --> {output}")

        return render_template('churn.html', prediction_text=output)

    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template('churn.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)