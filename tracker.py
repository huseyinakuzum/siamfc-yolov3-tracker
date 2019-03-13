import os
import glob
from random import randint
import cv2
from yolov3.detect import yolo, prepare_model
from siamfc.siamfc import SiamFCTracker


class Tracker():
    def __init__(self, model_path='siamfc/models/siamfc_pretrained.pth', gpu_id=0):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.filenames = ''

    def get_filenames_frames(self, video_dir):
        filenames = sorted(glob.glob(os.path.join(video_dir, "img/*.jpg")),
                           key=lambda x: int(os.path.basename(x).split('.')[0]))
        if len(filenames) == 0:
            filenames = sorted(glob.glob(os.path.join(video_dir, "*.jpg")),
                               key=lambda x: int(os.path.basename(x).split('.')[0]))
        frames = [cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                  for filename in filenames]
        return filenames, frames

    def draw_frame(self, trackers, bboxes, bboxes_colours, frame_id):
        for t in range(len(trackers)):
            frame = cv2.rectangle(frame,
                                  (int(bboxes[t][0]), int(bboxes[t][1])),
                                  (int(bboxes[t][2]), int(bboxes[t][3])),
                                  bboxes_colours[t],
                                  2)
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.putText(frame, str(frame_id), (5, 20),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        return frame

    def detect(self, idx, model, inp_dim):
        person_detections = list(set(
            filter(lambda x: x[1] == 'person', yolo(self.filenames[idx], model, inp_dim))))
        return person_detections.sort(key=lambda x: (
            x[0][2] * x[0][3]), reverse=True)

    def track(self, video_dir):
        # load videos
        self.filenames, frames = self.get_filenames_frames(video_dir)

        # extract person detections and sort it from largest to smallest

        title = video_dir.split('/')[-1]
        # starting tracking

        trackers = []
        bboxes = []
        bboxes_colours = []
        person_detections = []

        model, inp_dim = prepare_model()

        for idx, frame in enumerate(frames):
            if idx % 20 == 0:
                trackers.clear()
                bboxes.clear()
                bboxes_colours.clear()
                person_detections.clear()
                person_detections = self.detect(idx, model, inp_dim)

                for i, ix in enumerate(person_detections):
                    trackers.append(SiamFCTracker(
                        self.model_path, self.gpu_id))
                    bboxes.append(ix[0])
                    bboxes_colours.append(
                        (randint(0, 255) % 255, randint(0, 255), randint(0, 255)))
                for t, tx in enumerate(trackers):
                    tx.init(frame, person_detections[t][0])
                    bboxes[t] = (bboxes[t][0]-1, bboxes[t][1]-1,
                                 bboxes[t][0] + bboxes[t][2]-1, bboxes[t][1] + bboxes[t][3]-1)

            else:
                for t, tx in enumerate(trackers):
                    bboxes[t] = tx.update(frame, idx)
            # bbox xmin ymin xmax ymax

            frame = self.draw_frame(trackers, bboxes, bboxes_colours, idx)
            cv2.imshow(title, frame)
            cv2.waitKey(30)

    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou
