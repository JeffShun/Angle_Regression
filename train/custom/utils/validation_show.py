import cv2
import numpy as np


def save_result_as_image(images, regressions, labels, save_dir):
    for i in range(images.shape[0]):
        img = images[i,0].cpu().numpy()
        img = (img * 255).astype("uint8")
        draw_img1 = np.zeros([img.shape[0], img.shape[1], 3])
        draw_img1[:, :, 0] = img
        draw_img1[:, :, 1] = img
        draw_img1[:, :, 2] = img

        draw_img2 = draw_img1.copy()
        label_i = labels.cpu().numpy()[i]
        regressions_i = regressions.cpu().numpy()[i]
        draw1 = draw_angle(draw_img1, label_i)
        draw2 = draw_angle(draw_img2, regressions_i)
        draw_img = np.concatenate((draw1,draw2), 1)
        cv2.imwrite(save_dir+"/{}.png".format(i+1), draw_img)


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
    angle_text = f'{angle[0]/np.pi*180:.2f}'
    text_position = (line_start[0], line_start[1] - 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_color = (0, 0, 255)
    cv2.putText(img_show, angle_text, text_position, font, font_scale, font_color, 1, cv2.LINE_AA)
    return img_show
