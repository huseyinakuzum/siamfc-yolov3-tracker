from tracker import Tracker
from fire import Fire

def main():
    tracker = Tracker()
    tracker.track('/home/husampa/Projects/siamfc-yolov3-tracker/videos/bolt2')


if __name__ == "__main__":
    Fire(main)
