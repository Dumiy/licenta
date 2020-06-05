import sys
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
app = Flask(__name__)

@app.route('/hello')
def hello():
    return 'Hello, World'


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 5500
    app.run(host='0.0.0.0',port=port,debug=False)