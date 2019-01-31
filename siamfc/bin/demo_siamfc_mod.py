import glob
import os
import pandas as pd
import argparse
import numpy as np
import cv2
import time
import sys
from random import randint
sys.path.append('/home/shaq/VisionLab/Projects/pytorch-yolo-v3')
sys.path.append(os.getcwd())
from detect import yolo
from fire import Fire
from tqdm import tqdm

from siamfc import SiamFCTracker

def main(video_dir, gpu_id,  model_path):
    # load videos
    filenames = sorted(glob.glob(os.path.join(video_dir, "img/*.jpg")),
           key=lambda x: int(os.path.basename(x).split('.')[0]))
    frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) for filename in filenames]
    first_frame = frames[0]

    #extract person detections and sort it from largest to smallest
    person_detections = list(set(filter(lambda x:x[1]=='person', yolo(first_frame))))
    person_detections.sort(key=lambda x: (x[0][2] * x[0][3]), reverse=True)
    
    title = video_dir.split('/')[-1]
    # starting tracking
    
    trackers = []
    bboxes = []
    bboxes_colours = []
    
    for i in range(len(person_detections)):
        trackers.append(SiamFCTracker(model_path, gpu_id))
        bboxes.append(person_detections[i][0])
        bboxes_colours.append((randint(0, 255) % 255, randint(0, 255), randint(0, 255)))

    for idx, frame in enumerate(frames):
        if idx == 0:
            for t in range(len(trackers)):
                trackers[t].init(frame, person_detections[t][0])
                bboxes[t] = (bboxes[t][0]-1, bboxes[t][1]-1,
                        bboxes[t][0]+ bboxes[t][2]-1, bboxes[t][1]+ bboxes[t][3]-1)
        
        else: 
            for t in range(len(trackers)):
                bboxes[t] = trackers[t].update(frame)
        # bbox xmin ymin xmax ymax
        for t in range(len(trackers)):
            frame = cv2.rectangle(frame,
                                (int(bboxes[t][0]), int(bboxes[t][1])),
                                (int(bboxes[t][2]), int(bboxes[t][3])),
                                bboxes_colours[t],
                                2)

        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.putText(frame, str(idx), (5, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        cv2.imshow(title, frame)
        cv2.waitKey(30)

if __name__ == "__main__":
    Fire(main)
