import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from skimage.transform import resize
from skimage.io import imread
from skimage.color import gray2rgb
import matplotlib.pyplot as plt

from mtcnn.mtcnn import MTCNN

from unet import unet_small

INPUT_SHAPE = (128, 128)
TOPK = 70
THRESHOLD = 0.75


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, type=str, help="Path in input image\n")
    parser.add_argument("-o", "--output", required=False, type=str, help="Path in output image with detections\n")
    return parser.parse_args()


def get_top_k_predictions_mean(heatmap, top_k):
    height, width = heatmap.shape[:2]
    partition = np.partition(heatmap.ravel(), height * width - top_k)
    k_th_element = partition[-top_k]
    y_good, x_good = np.where(heatmap >= k_th_element)
    weights = heatmap[y_good, x_good]
    y_mean = (y_good * weights).sum() / weights.sum()
    x_mean = (x_good * weights).sum() / weights.sum()
    return {"x": x_mean, "y": y_mean, "max_score": np.max(partition)}


def main(args):
    try:
        image = imread(args.image)
    except Exception() as e:
        print("can't load image, error: ", e)
        return

    if len(image.shape) == 2:
        image = gray2rgb(image)

    if image.shape[-1] == 4:
        image = image[:,:, :3]

    detector = MTCNN()
    faces = detector.detect_faces(image)
    if len(faces) == 0:
        print("Can't detect faces\n")
        return

    face_box = faces[0]["box"]
    left_face = max(0, face_box[0])
    top_face = max(0, face_box[1])
    face_crop = image[top_face:top_face + face_box[3] + 1, left_face:left_face + face_box[2] + 1]

    model = unet_small(n_filters=12, input_shape=(None, None, 3), out_channels=2)
    model.load_weights("./checkpoint.cpt")

    prediction = model.predict(np.expand_dims(resize(face_crop, INPUT_SHAPE), 0))[0]
    prediction = resize(prediction, face_crop.shape[:2])

    left = get_top_k_predictions_mean(prediction[:,:,0], TOPK)
    right = get_top_k_predictions_mean(prediction[:,:,1], TOPK)

    if left["max_score"] > THRESHOLD:
        print("Left eye x: {0:.2f}, y: {1:.2f}".format(left["x"] + left_face, left["y"] + top_face))
    if right["max_score"] > THRESHOLD:
        print("Right eye x: {0:.2f}, y: {1:.2f}".format(right["x"] + left_face, right["y"] + top_face))

    if args.output:
        fig = plt.figure(figsize=(4,4))
        plt.imshow(face_crop)
        plt.axis("off")
        if left["max_score"] > THRESHOLD:
            plt.scatter(left["x"], left["y"], color="blue")
        if right["max_score"] > THRESHOLD:
            plt.scatter(right["x"], right["y"], color="blue")
        fig.savefig(args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
