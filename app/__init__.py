from flask import request
from flask_api import FlaskAPI
from flask_cors import CORS, cross_origin

from app.models.predict import predict

app = FlaskAPI(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predict', methods=['POST'])
@cross_origin()
def example():
    proteins = request.json.get('proteins', [])
    drugs = request.json.get('drugs', [])
    return predict(proteins, drugs)

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    return response

if __name__ == "__main__":
    app.run(debug=True)
