import model

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
queries = []


@app.route("/", methods=['GET'])
def main():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def findSimilarities():
    queries.append(request.form['query'])
    return jsonify(model.k_NN(request.form['query'], int(request.form['model'])))


@app.route("/wehavegot", methods=['GET'])
def wehavegot():
    return jsonify(queries)


if __name__ == "__main__":
    app.run()

# http://127.0.0.1:5000/wehavegot
# http://127.0.0.1:5000/
