import sys
from flask import render_template,Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import os
import numpy as np

import os


UPLOAD_FOLDER = './/upload'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = "static\\files"


def allowed_image_filesize(filesize):
    print(filesize)
    if int(filesize) <= app.config["MAX_CONTENT_LENGTH"]:
        return True
    else:
        return False
@app.before_request
def handle_chunking():
    """
    Sets the "wsgi.input_terminated" environment flag, thus enabling
    Werkzeug to pass chunked requests as streams.  The gunicorn server
    should set this, but it's not yet been implemented.
    """

    transfer_encoding = request.headers.get("Transfer-Encoding", None)
    if transfer_encoding == u"chunked":
        request.environ["wsgi.input_terminated"] = True

@app.route('/upload-data',methods=["GET",'POST'])
def upload():
    if request.method == "POST":
        if request.files:
            #if not allowed_image_filesize(request.cookies.get("filesize")):
                #print("file exceeded file size")
                #return redirect(request.url)

            file = request.files["video"]
            if file.filename == "":
                print("Error at given name")
                return redirect(request.url)

            file.save(os.path.join(app.config["UPLOAD_FOLDER"],secure_filename(file.filename)))
            print("File saved")
    return render_template("upload.html")
with app.test_request_context():
    print(url_for("upload"))
if __name__ == "__main__":
    if not os.path.exists("upload"):
        os.mkdir("upload")
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 5500
    app.run(port=port,debug=True)
