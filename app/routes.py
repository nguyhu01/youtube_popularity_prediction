from flask import Blueprint, render_template, request
import pickle
from model.pre_processing import Preprocessor  

main = Blueprint('main', __name__)

# Load the trained model
model = pickle.load(open('trained_model.pkl', 'rb'))

# Initialize the Preprocessor with the path to the dataset
preprocessor = Preprocessor('dataset/AI_ML_YT_Videos.csv')

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    title = request.form.get('title')
    channel = request.form.get('channel')
    published_date = request.form.get('published_date')  

    # Use the preprocessor to transform the input
    input_features = preprocessor.transform_input(title, channel, published_date)

    # Make a prediction
    prediction = model.predict([input_features])

    # Render the result template with prediction
    return render_template('result.html', prediction=prediction[0])

