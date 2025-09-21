import os
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Create the Flask application
app = Flask(__name__)

# Load the machine learning model
# You should train a model and save it as a .pkl file
# This is a placeholder for your actual model file
model = pickle.load(open('kidney_stone_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Handles the main page and prediction request.
    """
    prediction_text = None
    if request.method == 'POST':
        try:
            # Get data from the form
            gravity = float(request.form['gravity'])
            ph = float(request.form['ph'])
            osmo = float(request.form['osmo'])
            cond = float(request.form['cond'])
            urea = float(request.form['urea'])
            calc = float(request.form['calc'])

            # Create a numpy array for the model prediction
            # The order must match the features used for training
            features = np.array([[gravity, ph, osmo, cond, urea, calc]])

            # Make the prediction
            prediction = model.predict(features)
            
            # Get the result and format the message
            result = "Risk of Stone" if prediction[0] == 1 else "No Risk of Stone"
            prediction_text = f"The patient has a {result}."

        except ValueError:
            prediction_text = "Invalid input. Please enter numerical values."
        except Exception as e:
            prediction_text = f"An error occurred: {e}"

    return render_template('index.html', prediction_text=prediction_text)

# You would also need to have a trained model file
# (e.g., kidney_stone_model.pkl) in the same directory.
# This model is a key component of the backend logic.

if __name__ == '__main__':
    # You can change the port for local development
    # Use gunicorn or other WSGI server for production
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))