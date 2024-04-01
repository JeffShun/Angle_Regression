import argparse
import glob
import os
import sys
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.font_manager as fm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predictor import PredictModel, Predictor
from scipy.ndimage import zoom
import matplotlib.pylab as plt
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Test Angle Regression')

    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--input_path', default='./data/input', type=str)
    parser.add_argument('--output_path', default='./data/output', type=str)

    parser.add_argument(
        '--model_file',
        type=str,
        # default='../train/checkpoints/trt_model/model.engine'
        default='../train/checkpoints/Resnet50/150.pth'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='./test_config.yaml'
    )
    args = parser.parse_args()
    return args


def inference(predictor: Predictor, img: np.ndarray):
    pred_array = predictor.predict(img)
    return pred_array


def parse_label(label_path):
    with open(label_path) as f:
        label_data = json.load(f)
        angle = None
        for shape in label_data["shapes"]: 
            if shape["label"] == "angle":
                points = shape["points"]
                # 提取矩形框的左上角和右下角坐标
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                angle = np.arctan((y2 - y1) / (x2 - x1))
                break
    return angle

def draw_angle(img_show, angle):
    # 计算图像中间位置
    image_center = (img_show.shape[0] // 2, img_show.shape[1] // 2)
    # 计算线的起点
    line_start = image_center
    # 计算线的终点
    line_end_x = image_center[0] + img_show.shape[0] // 4
    line_end_y = int(line_start[1] + img_show.shape[1] // 4 * np.sin(angle))
    line_end = (line_end_x, line_end_y)

    # 绘制实线
    cv2.arrowedLine(img_show, line_start, line_end, (0, 0, 255), 1, cv2.LINE_AA)

    # 显示角度数值
    angle_text = f'{angle/np.pi*180:.2f}'
    text_position = (line_start[0], line_start[1] - 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_color = (0, 0, 255)
    cv2.putText(img_show, angle_text, text_position, font, font_scale, font_color, 1, cv2.LINE_AA)
    return img_show

def save_result(img, pred, save_path):
    img = (img-img.min())/(img.max()-img.min())
    img = (img * 255).astype("uint8")
    draw_img = np.zeros([img.shape[0], img.shape[1], 3])
    draw_img[:, :, 0] = img
    draw_img[:, :, 1] = img
    draw_img[:, :, 2] = img
    draw_img = draw_angle(draw_img, pred)
    cv2.imwrite(save_path, draw_img) 


def save_result_with_gt(img, pred, gt, save_path):
    img = (img-img.min())/(img.max()-img.min())
    img = (img * 255).astype("uint8")
    draw_img1 = np.zeros([img.shape[0], img.shape[1], 3])
    draw_img1[:, :, 0] = img
    draw_img1[:, :, 1] = img
    draw_img1[:, :, 2] = img
    draw_img2 = draw_img1.copy()
    draw1 = draw_angle(draw_img1, gt)
    draw2 = draw_angle(draw_img2, pred)
    draw_img = np.concatenate((draw1,draw2), 1)
    cv2.imwrite(save_path, draw_img)


def main(input_path, output_path, device, args):
    # TODO: 适配参数输入
    model_regression = PredictModel(
        model_f=args.model_file,
        config_f=args.config_file,
    )
    predictor_regression = Predictor(
        device=device,
        model=model_regression,
    )
    os.makedirs(output_path, exist_ok=True)
    img_dir = os.path.join(input_path, "imgs")
    label_dir = os.path.join(input_path, "labels")

    for sample in tqdm(os.listdir(img_dir)):  
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(img_dir, sample)))[0]
        try:
            gt = parse_label(os.path.join(label_dir, sample.replace(".dcm",".json")))
        except:
            gt = None
        pred = predictor_regression.predict(img)
        if gt is not None:        
            save_result_with_gt(img, pred, gt, os.path.join(output_path, f'{sample.replace(".dcm","")}.png'))
            # save_result(img, preds, label_map, os.path.join(output_path, f'{sample.replace(".dcm","")}.png'))
            raw_result_dir = os.path.join(output_path, "raw_result")
            os.makedirs(raw_result_dir, exist_ok=True)
            np.savez_compressed(os.path.join(raw_result_dir, sample.replace(".dcm",".npz")), pred=pred, label=gt)
        else:
            save_result(img, pred, os.path.join(output_path, f'{sample.replace(".dcm","")}.png'))



if __name__ == '__main__':
    args = parse_args()
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        device=args.device,
        args=args,
    )