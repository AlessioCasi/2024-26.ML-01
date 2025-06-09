from flask import Flask, request, jsonify
import joblib
import pandas as pd
app = Flask(__name__)

# Load model once on startup
mymodel = joblib.load("artefatto.joblib")

@app.route('/infer', methods=['POST'])


@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()
    name = data.get('name', 'Stranger')
    param1 = data.get('param1')

    # Convert dict to DataFrame (1 row)
    df = pd.DataFrame([param1])

    # Preprocess df as needed (e.g. encode categorical columns, etc.)
    # For example, if your model requires encoded features:
    # df_processed = your_preprocessing_function(df)

    infer_result = mymodel.predict(df)  # pass DataFrame

    return jsonify({
        "message": f"Hello {name}!",
        "inference": infer_result.tolist()
    })

@app.route('/infer', methods=['GET'])
def infer_get():
    return "<h1> Ciao </h1>"

if __name__ == '__main__':
    app.run(debug=True)
