import os
from src.logger import logging
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "src", "templates"))

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        logging.info("Before Prediction")
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        logging.info("Mid Prediction")
        result = predict_pipeline.predict(pred_df)
        print("After Prediction")
        logging.info("After Prediction")
        return render_template('home.html', results = result[0])

if __name__ == "__main__":
    app.run(host = "0.0.0.0")