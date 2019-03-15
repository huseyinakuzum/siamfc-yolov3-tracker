from tracker import Tracker
from fire import Fire
import argparse


def arg_parse():
    """
    Parse arguements to the track module

    """
    parser = argparse.ArgumentParser(
        description='YOLOv3 - SiamFC Object Tracking Module')
    parser.add_argument("--video", dest='video', help="Video to run tracking upon",
                        default="/home/husampa/Projects/siamfc-yolov3-tracker/videos/bolt2", type=str)

    return parser.parse_args()


def main(args):
    tracker = Tracker()
    tracker.track('/home/husampa/Projects/siamfc-yolov3-tracker/videos/bolt2')


if __name__ == "__main__":
    args = arg_parse()
    Fire(main(args))
