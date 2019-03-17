from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize, VideoCapture, imwrite
import os
from PIL import Image
import glob
import cv2
import numpy as np
import os

from os.path import isfile, join
from string import digits


def get_images_from_dir(path, extension='jpg'):
    image_list = [filename for filename in glob.glob(path + '/*.' + extension)]
    image_list.sort(key=lambda x: int(
        ''.join(c for c in x[-8:-4] if c in digits)))
    return image_list


def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = get_images_from_dir(pathIn)
    size = (0,0)
    for i, filename in enumerate(files):
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i, frame in enumerate(frame_array):
        out.write(frame)
    out.release()


def get_frames(video):
    vidcap = VideoCapture(video)
    success, image = vidcap.read()
    count = 0
    print(success)
    while success:
        # save frame as JPEG file
        imwrite("/home/husampa/Projects/siamfc-yolov3-tracker/VideoDeneme/chineserest/frames/frame%d.jpg" % count, image)
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


if __name__ == "__main__":
    imgs = get_images_from_dir("VideoDeneme/terrace0/frames")
    pathIn = '/home/husampa/Projects/siamfc-yolov3-tracker/dets/chineserest'
    pathOut = '/home/husampa/Projects/siamfc-yolov3-tracker/dets/chineserest.avi'
    fps = 25.0
    #convert_frames_to_video(pathIn, pathOut, fps)

    #get_frames("/home/husampa/Projects/siamfc-yolov3-tracker/VideoDeneme/chineserest/video/chineserestaruant.avi")
