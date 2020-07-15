import sys
from flask import render_template,Flask, flash, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import extract
from pytorch_i3d import InceptionI3d,InceptionI3d_CBAM
from torchvision import transforms
from PIL import Image
import torch
from torch.autograd import Variable
from collections import Counter
from cv2 import VideoWriter, VideoWriter_fourcc

UPLOAD_FOLDER = './/upload'

#model_to_use = 

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = "static/files"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
ALLOWED_EXTENSIONS = {'mp4', 'npy'}
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response
@app.route('/')
def hello():
    return render_template("home.html")


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
def predict(rgb,flow,cuda=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    i3d_rgb = InceptionI3d(400,in_channels=3)
    i3d_flow = InceptionI3d(400,in_channels=2)
    i3d_cbam = InceptionI3d(14,in_channels=3)
    i3d_class = InceptionI3d_CBAM(i3d_cbam)
    i3d_class.load_state_dict(torch.load("static/models/class_final.pt"))
    i3d_rgb.load_state_dict(torch.load("static/models/rgb_imagenet.pt"))
    i3d_flow.load_state_dict(torch.load("static/models/flow_imagenet.pt"))
    #i3d_rgb.replace_logits(157)
    #i3d_flow.replace_logits(157)
    i3d_flow.to(device)
    i3d_rgb.to(device)
    i3d_class.to(device)
    data_rgb = rgb.permute(0, 2, 1, 3, 4)
    data_flow = flow.permute(0, 2, 1, 3, 4)
    data_rgb = Variable(data_rgb.to(device), requires_grad = False)
    data_flow = Variable(data_flow.to(device), requires_grad = False)
    rgb_feature = []
    flow_feature = []
    labels = []
    result = []
    for (x,y) in zip(data_rgb,data_flow):
        with torch.no_grad():
            x,y = x.unsqueeze(0),y.unsqueeze(0)
            rgb_feature = i3d_rgb.extract_features(x)
            rgb_feature = rgb_feature.squeeze(3).squeeze(3).squeeze(2)
            flow_feature = i3d_flow.extract_features(y)
            flow_feature = flow_feature.squeeze(3).squeeze(3).squeeze(2)
            label = torch.argmax(i3d_class(x),dim=1)
            labels.append(label)
            result.append(torch.cat((rgb_feature,flow_feature),dim=1))
    result = torch.stack(result)
    return result.squeeze(1),labels

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum
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
            if file and allowed_file(file.filename):
                file.save(os.path.join(app.config["UPLOAD_FOLDER"],secure_filename(file.filename)))
                print("File saved")
                if ".npy" in file.filename:
                    output = np.array([np.load("static/files/"+file.filename)])
                    temp = output
                else:
                    output = extract.extract_frames("static/files/"+file.filename)
                dictionar = {
                0 : 'Abuse',
                1 : 'Arrest',
                2 : 'Arson',
                3 : 'Assault',
                4 : 'Burglary',
                5 : 'Explosion',
                6 : 'Fighting',
                7 : 'Normal',
                8 : 'RoadAccidents',
                9 : 'Robbery',
                10 : 'Shooting',
                11 : 'Shoplifting',
                12 : 'Stealing',
                13 : 'Vandalism'
                }
                flow = []
                test_transforms = transforms.Compose([transforms.CenterCrop(224),])
                print(output.shape)
                outputs = []
                for i in range(0,output.shape[0]):
                    new = []
                    for j in range(0,output[i].shape[0]):
                        new.append(np.array(test_transforms(Image.fromarray(output[i][j]))))
                    outputs.append(np.array(new))
                del output
                for i in outputs:
                    flow.append(extract.make_flow_frames(i))
                flow = np.array(flow)
                flow_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], # , 0.406
                                                    std=[0.229, 0.224, 0.225]), # , 0.225]
                                            ])
                output = np.array(outputs)
                outputs = []
                for i in range(0,output.shape[0]):
                    new = []
                    for j in range(0,output[i].shape[0]):
                        new.append(np.array(flow_transform(Image.fromarray(output[i][j]))))
                    outputs.append(np.array(new))
                del output
                outputs = np.array(outputs)
                flowe = []
                for i in range(0,flow.shape[0]):
                    new = []
                    for j in range(0,flow[i].shape[0]):
                        new.append(np.array(flow_transform(Image.fromarray(flow[i][j]))))
                    flowe.append(np.array(new))
                del flow
                flowe = np.array(flowe)
                print(outputs.shape)
                print(flowe.shape)
                print(file.filename)
                print(flowe[:,:,:2].shape)
                final,labels = predict(torch.Tensor(outputs),torch.Tensor(flowe[:,:,:2]))
                c = find_majority(labels)
                valoare= c[0]
                print(final.shape)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                clasifier = extract.Classifier()
                clasifier.load_state_dict(torch.load("static/models/final_clasifier.pt"))
                clasifier.to(device)
                clasifier.eval()
                anomaly = []
                normal = []
                for i in final:
                    s = clasifier(i.unsqueeze(0))
                    s = s.detach().cpu().numpy()
                    print(s.shape)
                    anomaly.append(np.exp(s[0][1]))
                    normal.append(np.exp(s[0][0]))
                    
                print(np.mean(anomaly))
                print(anomaly)
                print(np.mean(normal))
                print(normal)
                
                interval = np.linspace(0,3*final.shape[0],num=final.shape[0])
                print(interval.shape)
                plt.plot(interval,anomaly,label = "Anomaly")
                plt.legend()
                plt.savefig('static/anomaly.png')
                plt.clf()
                plt.plot(interval,normal,label = "Normal")
                plt.legend()
                plt.savefig('static/normal.png')
                plt.clf()
                video_file = file.filename
                if ".npy" in video_file:
                    width = 640
                    height = 480
                    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
                    fps = 30
                    video = VideoWriter('./static/files/numpy.avi', fourcc,float(6), (temp[0].shape[1], temp[0].shape[2]))
                    i = 0
                    for i in range(temp[0].shape[0]):
                        frame = temp[0][i]
                        i+=1
                        video.write(frame)
                    video.release()
                    video_file = "numpy.avi"
                print(valoare.tolist()[0])
                print(dictionar[valoare.tolist()[0]])
                return render_template("view.html",video_feed="static/files/"+video_file,predict=np.mean(anomaly),name=dictionar[valoare.tolist()[0]],graph1 ='static/anomaly.png',graph2='static/normal.png')
    return render_template("upload.html")
with app.test_request_context():
    print(url_for("upload"))

@app.route('/view')
def view():
     return render_template("view.html",video_feed="static/files/Abuse001_x264.mp4")

if __name__ == "__main__":
    if not os.path.exists("upload"):
        os.mkdir("upload")
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 5500
    app.run(host="0.0.0.0",port=port,debug=True)
