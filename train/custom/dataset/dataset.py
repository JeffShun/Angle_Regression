"""data loader."""

import numpy as np
from torch.utils import data

class MyDataset(data.Dataset):
    def __init__(
            self,
            dst_list_file,
            transforms
        ):
        self.data_lst = self._load_files(dst_list_file)
        self._transforms = transforms

    def _load_files(self, file):
        data_list = []
        with open(file, 'r') as f:
            for line in f:
                data_list.append(line.strip())
        return data_list

    def __getitem__(self, idx):
        source_data = self._load_source_data(self.data_lst[idx])
        return source_data

    def __len__(self):
        return len(self.data_lst)

    def _load_source_data(self, file_name):
        data = np.load(file_name, allow_pickle=True)
        img = data['img']
        angle = data['angle']
        
        # transform前，数据必须转化为[C,H,W]的形状
        img = img[np.newaxis,:,:].astype(np.float32)
        angle = angle[np.newaxis].astype(np.float32)
        if self._transforms:
            img, angle = self._transforms(img, angle)

        # # For Debug
        # import cv2
        # img_show = img[0].numpy()
        # img_show = (img_show-img_show.min())/(img_show.max()-img_show.min())
        # img_show = (img_show*255).astype("uint8")
        # img_show = cv2.cvtColor(img_show, cv2.COLOR_GRAY2BGR)
        # # 计算图像中间位置
        # image_center = (img_show.shape[0] // 2, img_show.shape[1] // 2)
        # # 计算线的起点
        # line_start = image_center
        # # 计算线的终点
        # line_end_x = image_center[0] + img_show.shape[0] // 4
        # line_end_y = int(line_start[1] + img_show.shape[1] // 4 * np.sin(angle))
        # line_end = (line_end_x, line_end_y)

        # # 绘制实线
        # cv2.arrowedLine(img_show, line_start, line_end, (0, 0, 255), 1, cv2.LINE_AA)

        # # 显示角度数值
        # angle_text = f'{angle/np.pi*180:.2f}'
        # text_position = (line_start[0], line_start[1] - 10)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.4
        # font_color = (0, 0, 255)
        # cv2.putText(img_show, angle_text, text_position, font, font_scale, font_color, 1, cv2.LINE_AA)

        # # 显示图像
        # cv2.imshow('Image Angle', img_show)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
             
        return img, angle


