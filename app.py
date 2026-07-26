import os

import pandas as pd

from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline, BranchNotOfferedException


application = Flask(__name__)

app = application


DATA_PATH = os.path.join("artifacts", "data.csv")


def get_dropdown_options():

    df = pd.read_csv(DATA_PATH)

    return {
        "college_names": sorted(df["college_name"].unique().tolist()),
        "score_types": sorted(df["score_type"].unique().tolist()),
        "seat_types": sorted(df["seat_type"].unique().tolist()),
        "branches": sorted(df["branch"].unique().tolist()),
    }


@app.route('/')
def index():

    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    options = get_dropdown_options()

    if request.method == 'GET':

        return render_template('home.html', results=None, error=None, **options)

    else:

        data = CustomData(
            college_name=request.form.get('college_name'),
            score_type=request.form.get('score_type'),
            seat_type=request.form.get('seat_type'),
            branch=request.form.get('branch')
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()

        try:

            results = predict_pipeline.predict(pred_df)

            return render_template(
                'home.html',
                results=round(float(results[0]), 2),
                error=None,
                **options
            )

        except BranchNotOfferedException as e:

            return render_template(
                'home.html',
                results=None,
                error=str(e),
                **options
            )


if __name__ == "__main__":

    app.run(host="0.0.0.0", debug=True)
