# mpg-predictor

This Streamlit app predicts the **Miles Per Gallon (MPG)** of a car based on its specifications using a trained machine learning model.

## Features
- Takes 7 car features as input: Cylinders, Displacement, Horsepower, Weight, Acceleration, Model Year, and Origin.
- Uses a pre-trained regression model to predict MPG.
- Displays the predicted MPG and a simple bar chart visualization.

## Model
- Built using `scikit-learn`
- Trained on the Auto MPG dataset
- StandardScaler used to scale input features

## Requirements
See `requirements.txt` for all dependencies.

## Deployment
The app is deployed using [Streamlit Cloud](https://streamlit.io/cloud).

### 🌐 Live Demo
[Click here to try the app 🚀](https://mpg-predictor-tmzdkbqew9tlwim6jyqjg5.streamlit.app/)

## Project Structure
mpg-predictor/
│
├── autompgFile.py # Main Streamlit app file
├── model.pkl # Trained ML model
├── requirements.txt # Python dependencies
