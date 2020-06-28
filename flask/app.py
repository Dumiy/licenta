import sys
from flask import Flask, render_template, request
from werkzeug import secure_filename
import pandas as pd
import os
import numpy as np

app.config['UPLOAD_FOLDER']='.\upload'

@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'

app = Flask(__name__)


@app.route('/hello',methods=['POST'])
def hello():
    return 'Hello, World'


if __name__ == "__main__":
    os.mkdir("\upload")
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 5500
    app.run(host='0.0.0.0',port=port,debug=False)