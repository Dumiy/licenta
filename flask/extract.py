import numpy as np
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
def extract_frames(file):
    output = []
    frames = []
    start = -1
    end = -1
    #print(folder+"/"+x[1]["name"])
    capture = cv2.VideoCapture(file)  # open the video using OpenCV
    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print(end)
        capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved
    i = 0
    while frame < end:  # lets loop through the frames until the end
        every, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if every:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            saved_count += 1  # increment our counter by one
            #print(saved_count)
            #print(end)
            if end >= 200:
                if saved_count  % 5 == 0:
                    frames.append(image)  # save the extracted image
            else:
                frames.append(image)
        frame += 1  # increment our frame count
    counter = 0
    #print(len(frames))
    capture.release()
    cv2.destroyAllWindows()  # after the while has finished close the capture
    if len(frames)<= 20 and len(frames) != 0:
        if len(frames) == 20:
            output.append(frames)
        frames.append(frames[:20-len(frames)])
        frames = np.array(frames)
        output.append(frames)
    else:
        image = np.array(frames)
        frames = []
        print(image.shape)
        if image.shape[0] > 20:
            for j in range(0,image.shape[0],20):
                new = []
                if j+20 > image.shape[0]:
                    for i in image[j:image.shape[0]-20]:
                        i = cv2.resize(i,(256,256))
                        new.append(i)
                        #cv2.imshow('frame',i)
                        #if cv2.waitKey(0) & 0xFF == ord('q'):
                            #break
                else:
                    for i in image[j:j+20]:
                        #print(j,j+20)
                        i = cv2.resize(i,(256,256))
                        new.append(i)
                        #cv2.imshow('frame',i)
                        #if cv2.waitKey(0) & 0xFF == ord('q'):
                            #break
                    new = np.array(new)
                    output.append(new)
                    #print(new.shape)
    return np.array(output)
def make_flow_frames(frames):
        new = []
        s = frames[0]
        prev = s #s
        prvs = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(s)
        hsv[..., 1] = 255
        for x in frames[1:]:
            next = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
            optical_flow = cv2.optflow.DualTVL1OpticalFlow_create(tau=0.6,theta=0.1,nscales=1,warps=1,epsilon=0.15)
            flow = optical_flow.calc(prvs, next, None)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[...,0] = ang * 180 / np.pi / 2
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            #flow = scale(flow,-1,1)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            prvs = next
            new.append(bgr)
        del frames
        frames = np.array(new)
        #print(frames.shape)
        return frames

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048,512)
        self.fc2 = nn.Linear(512,100)
        self.fc3 = nn.Linear(100,2)
        self.activation = nn.ReLU()
    def forward(self, y : torch.Tensor):
        output = []
        for x in y:
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = F.log_softmax(self.fc3(x),dim=0)
            output.append(x)
        x = torch.stack(output)
        return x