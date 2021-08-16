#coding:utf-8
import numpy as np
import sys
import cv2

def draw(draw_panel, s_list):
    s_list = sorted(s_list)    

    for idx, line in enumerate(s_list):
        faceid = line.strip()
        imgpath = "image/%s.jpg"%faceid
        img = cv2.imread(imgpath)
        img2 = cv2.resize(img, (100, 100))
        row_num = idx // 10
        col_num = idx % 10
        draw_panel[row_num*100:(row_num+1)*100, col_num*100:(col_num+1)*100, :] = img2
    return draw_panel


if __name__ == "__main__":
    s1file, s2file, outfile = sys.argv[1], sys.argv[2], sys.argv[3]

    s1_list = open(s1file).readlines()
    s2_list = open(s2file).readlines()
    
    total_row = max(len(s1_list), len(s2_list)) // 10 + 1

    draw_panel1 = np.zeros((total_row*100, 1050, 3)).astype(np.uint8)
    draw_panel2 = np.zeros((total_row*100, 1000, 3)).astype(np.uint8)

    daw_panel1 = draw(draw_panel1, s1_list)
    daw_panel2 = draw(draw_panel2, s2_list)

    merge = np.concatenate((daw_panel1, daw_panel2), axis=1)
    cv2.imwrite(outfile, merge)
