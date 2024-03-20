import numpy as np
from flask import Flask , request , jsonify
import pickle
import numpy

model = pickle.load(open("classified.pkl","rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"
@app.route("/predict" , methods=["POST"])
def predict():
    age = request.form.get("age")
    gender = request.form.get("gender")

    stream = request.form.get("stream")
    internship = request.form.get("internship")
    cgpa = request.form.get("cgpa")
    backlog = request.form.get("backlog")

    input_query = np.array([[age,gender,stream,internship,cgpa,backlog]])
    result = model.predict(input_query)[0]


    return jsonify({"placement":str(result)})

if __name__ == "__main__":
    app.run(debug=True)
