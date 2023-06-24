# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path
import progressbar
import shutil
import numpy as np
import ast

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms

from shuffler.interface.pytorch.datasets import ImageDataset

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_img_size, non_max_suppression,
                           print_args, scale_coords, xyxy2xywh)
from utils.torch_utils import select_device


@torch.no_grad()
def run(
    weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
    in_db_file=None,
    rootdir=None,
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    out_db_file=None,
    dnn=False,  # use OpenCV DNN for ONNX inference
    batch_size=1,
    coco_category_id_to_name_map={},
):
    progressbar.streams.wrap_stderr()
    progressbar.streams.wrap_stdout()
    FORMAT = '[%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s'

    labels_to_names = ast.literal_eval(coco_category_id_to_name_map)

    transform_image = torchvision.transforms.Compose([
        lambda img: img.transpose((2, 0, 1)),
        lambda img: np.ascontiguousarray(img),
        lambda img: torch.Tensor(img),
        torchvision.transforms.Resize(imgsz),
    ])

    # Dataloader
    shutil.copyfile(in_db_file, out_db_file)
    dataset = ImageDataset(
        db_file=out_db_file,
        rootdir=rootdir,
        mode='w',
        used_keys=['image', 'imagefile', 'image_width', 'image_height'],
        transform_group={'image': transform_image})
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    imgsz = check_img_size(imgsz, s=model.stride)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup
    for batch in progressbar.progressbar(dataloader):
        images = batch['image']
        imagefiles = batch['imagefile']

        images = images.to(device)
        images = images.float()
        images /= 255  # 0 - 255 to 0.0 - 1.0

        # Inference
        pred = model(images, augment=augment, visualize=False)

        # NMS
        pred = non_max_suppression(pred,
                                   conf_thres,
                                   iou_thres,
                                   False,
                                   agnostic_nms,
                                   max_det=max_det)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            print('Found %d predictions in %s.' % (len(det), imagefiles[i]))
            old_height = batch['image_height'][i]
            old_width = batch['image_width'][i]
            new_height = imgsz[0]
            new_width = imgsz[1]

            for *xyxy, score, label in reversed(det):
                xyxy[0] *= (old_width / new_width)
                xyxy[1] *= (old_height / new_height)
                xyxy[2] *= (old_width / new_width)
                xyxy[3] *= (old_height / new_height)
                x1 = xyxy[0]
                y1 = xyxy[1]
                width = xyxy[2] - xyxy[0]
                height = xyxy[3] - xyxy[1]

                object_entry = (
                    imagefiles[i],
                    x1.item(),
                    y1.item(),
                    width.item(),
                    height.item(),
                    labels_to_names[label.item()],
                    score.item())
                s = 'objects(imagefile,x1,y1,width,height,name,score)'
                dataset.execute('INSERT INTO %s VALUES (?,?,?,?,?,?,?)' % s,
                                object_entry)

    dataset.conn.commit()
    dataset.close()


def parse_opt():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--in_db_file',
        help='Path to Shuffler database with images to run inference on. '
        'Will write to the same db.',
        required=True)
    parser.add_argument(
        '-o',
        '--out_db_file',
        help=
        'The path to a new Shuffler database, where detections will be stored.',
        default='examples/detected/epoch10-test.db')
    parser.add_argument('--rootdir',
                        help='Where image files in the db are relative to.',
                        required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--weights',
                        nargs='+',
                        type=str,
                        default=ROOT / 'yolov5s.pt',
                        help='model path(s)')
    parser.add_argument('--imgsz',
                        '--img',
                        '--img-size',
                        nargs='+',
                        type=int,
                        default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.25,
                        help='confidence threshold')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--max-det',
                        type=int,
                        default=1000,
                        help='maximum detections per image')
    parser.add_argument('--device',
                        default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms',
                        action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment',
                        action='store_true',
                        help='augmented inference')
    parser.add_argument('--dnn',
                        action='store_true',
                        help='use OpenCV DNN for ONNX inference')
    parser.add_argument(
        '--coco_category_id_to_name_map',
        default='{0: "stamp"}',
        help='A map (as a json string) from category id to its name.')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
