"""
Evaluates a folder of video files or a single file with a xception binary
classification network.
Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>
Author: Andreas Rössler
"""
import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm

from network.models import model_selection
from network.models import TransferModel
from dataset.transform import xception_default_data_transforms


def get_boundingbox(face, width, height, scale=1.3, minsize=None): ##scale to 1.3
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region ##
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.
    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transforms['test'] ##call transform
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def test_full_image_network(video_path, model_path, output_path,
                            start_frame=0, end_frame=None, cuda=True):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    print('Starting: {}'.format(video_path))

    # Read
    reader = cv2.VideoCapture(video_path)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) # 视频的总帧数

    # Face detector
    face_detector = dlib.get_frontal_face_detector() ##功能：人脸检测画框 参数：无 返回值：默认的人脸检测器

    # Load model
    model, *_ = model_selection(modelname='xception', num_out_classes=2)  ##model
    if model_path is not None:
        model = torch.load(model_path)
        print('Model found in {}'.format(model_path))
    else:
        print('No model found, initializing random model.')
    if cuda:
        model = model.cuda()

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    while reader.isOpened():
        _, image = reader.read() ##image：每一帧的图像，是个三维矩阵；_：bool，读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        # Image size
        height, width = image.shape[:2]

        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # For now only take biggest face
            face = faces[0]

            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]

            # Actual prediction using our model
            ###prediction, output = predict_with_model(cropped_face, model,cuda=cuda)
            # ------------------------------------------------------------------

            ####
            #print(video_path.split('/')[-1].split('.')[0])
            save_dir = join(output_path, video_path.split('/')[4], video_path.split('/')[5], video_path.split('/')[6], video_path.split('/')[7],
                            video_path.split('/')[-1].split('.')[0])
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            #print(save_dir)
            save_path = "{}/{:>03d}.jpg".format(save_dir, frame_num)
            cv2.imwrite(save_path, cropped_face)
            cv2.waitKey(1)


        if frame_num >= end_frame:
            break
        
    pbar.close()
    
    reader.release()
    print('Finished! Output saved under {}'.format(output_path))
    print("Totally save {:d} pics".format(frame_num)) # 打印出所提取帧的总数

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str)
    p.add_argument('--model_path', '-mi', type=str, default=None)
    p.add_argument('--output_path', '-o', type=str,
                   default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()

    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        test_full_image_network(**vars(args))
    else:
        videos = os.listdir(video_path)
        for video in videos:
            args.video_path = join(video_path, video)
            test_full_image_network(**vars(args))



 
#python extract_from_video.py -i /data/ywang/DFDC/test/0/aaqkmjtoby.mp4 -o /data/ywang/DFDCExtract --cuda