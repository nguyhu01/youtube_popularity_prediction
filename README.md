# YouTube Video Popularity Predictor

## Overview
This project aims to predict the popularity of YouTube videos in the AI and ML niche using machine learning and NLP techniques.

## Dataset
The dataset consists of video titles, channels, views, likes, and other metrics from YouTube.

## Setup and Installation
- Clone this repository.
- Install required packages: `pip install -r requirements.txt`
- Start the Flask application by running: python run.py
- Open your web browser and enter the URL http://127.0.0.1:5000 to access the application.

## Structure

Youtube_Prediction/
│
├── .flaskenv                   # Environment variables for Flask
├── app/                        # Flask application directory
│   ├── __init__.py             # Initializes the Flask app and includes create_app function
│   ├── routes.py               # Contains route definitions for the Flask app
│   ├── templates/              # HTML templates for the application
│   │   ├── index.html          # The main page of the web app
│   │   └── result.html         # The page to display predictions or results
│   └── static/                 # Static files like CSS, JavaScript, and images
│       └── style.css           # CSS styles for the web app
│
├── model/                      # Directory for the machine learning model
│   ├── pre_processing.py       # Contains the Preprocessor class for data preprocessing
│   └── model.py                # Script to train and save the ML model
│
├── dataset/                    # Dataset directory 
│   └── AI_ML_YT_Videos.csv     # Dataset used by the ML model
│
├── run.py                      # Script to run the Flask application
├── trained_model.pkl           # The trained machine learning model (saved after training)
└── requirements.txt            # File specifying the dependencies for the project


## Contributing
Contributions to the project are welcome! 

## License
This project is licensed under the MIT License.
