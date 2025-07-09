from flask import Flask, request, jsonify
from flask_cors import CORS
from gensim.models import KeyedVectors

app = Flask(__name__)
CORS(app)

# Carica il modello Word2Vec
model = KeyedVectors.load("word2vec.wordvectors", mmap='r')

@app.route('/sinonimi', methods=['GET'])
def sinonimi():
    word = request.args.get('word')
    try:
        result = model.most_similar(word, topn=10)
        return jsonify(result)
    except KeyError:
        return jsonify([])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
