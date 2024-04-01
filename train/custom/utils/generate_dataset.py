"""生成模型输入数据."""

import argparse
import glob
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image
import json
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./train_data/origin_data')
    parser.add_argument('--save_path', type=str, default='./train_data/processed_data')
    args = parser.parse_args()
    return args


def gen_lst(save_path, task, all_pids):
    save_file = os.path.join(save_path, task+'.txt')
    data_list = glob.glob(os.path.join(save_path, '*.npz'))
    num = 0
    with open(save_file, 'w') as f:
        for pid in all_pids:
            data = os.path.join(save_path, pid+".npz")
            if data in data_list:
                num+=1
                f.writelines(data.replace("\\","/") + '\n')
    print('num of data: ', num)

def get_max_box_count(data_path):
    max_box_count = 0
    for file_name in os.listdir(data_path):
        if file_name.endswith(".npz"):
            file_path = os.path.join(data_path, file_name)
            data = np.load(file_path, allow_pickle=True)
            label = data['label']
            max_box_count = max(label.shape[0],max_box_count) 
    print("max box num is {}".format(max_box_count))


def process_single(input):
    img_path, label_path, save_path, sample_id = input
    img_arr = np.array(Image.open(img_path))
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

        # For Debug
        img_show = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
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

        # 显示图像
        cv2.imshow('Image Angle', img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if angle:
        np.savez_compressed(os.path.join(save_path, f'{sample_id}.npz'), img=img_arr, angle=angle)


if __name__ == '__main__':
    args = parse_args()
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    for task in ["train","valid"]:
        print("\nBegin gen %s data!"%(task))
        img_dir = os.path.join(args.data_path, task, "imgs")
        label_dir = os.path.join(args.data_path, task, "labels")
        inputs = []
        all_ids = []
        for sample in tqdm(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, sample)
            sample_id = sample.replace('.png', '')
            label_path = os.path.join(label_dir, sample_id + ".json")
            inputs.append([img_path, label_path, save_path, sample_id])
            all_ids.append(sample_id)
        pool = Pool(1)
        pool.map(process_single, inputs)
        pool.close()
        pool.join()
        # 生成Dataset所需的数据列表
        gen_lst(save_path, task, all_ids)

    get_max_box_count(save_path)

    