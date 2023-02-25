# coding:utf-8
# 在圆圈上绘制文字并旋转
import time
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk
from threading import Thread
import random
import string
import json
import copy
import sys
from tqdm import tqdm
import os
import shutil
import numpy as np
os.chdir(sys.path[0])

def add_background2pic(bg,img,name,args):
    # background_picture = cv2.imread(background_picture)
    h,w = bg.shape[0],bg.shape[1]
    if w<args.windows_size or h<args.windows_size:
        return -1
    else:
        x = random.randint(0, w-args.windows_size)
        y = random.randint(0, h-args.windows_size)
        crop_bg = bg[y:y+args.windows_size,x:x+args.windows_size]
        res_img = merge_img(crop_bg, img, 0, crop_bg.shape[0], 0, crop_bg.shape[1])
        cv2.imencode('.png', res_img )[1].tofile("../ring_text_imgs+bg/"+name+'.png')

def add_alpha_channel(img):
    # 为jpg图像添加alpha通道
    b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道
    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
    return img_new
 
def merge_img(jpg_img, png_img, y1, y2, x1, x2):
    # 判断jpg图像是否已经为4通道
    if jpg_img.shape[2] == 3:
        jpg_img = add_alpha_channel(jpg_img)
    # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
    alpha_png = png_img[y1:y2, x1:x2, 3] / 255.0
    alpha_jpg = 1 - alpha_png
    # 开始叠加
    for c in range(0, 3):
        jpg_img[y1:y2, x1:x2, c] = ((alpha_jpg * jpg_img[y1:y2, x1:x2, c]) + (alpha_png * png_img[y1:y2, x1:x2, c]))
    return jpg_img

def gen_points(angle, points, R, up=1,centre_x=0, centre_y=0,):
    """
    Parameters
        ----------
        angle : Angle of text in the circle.
        points : Number of points selected on the circle (including starting point and ending point).
        R : Radius of circle.
        centre_x, centre_y : Center point coordinates

        Returns Take points evenly on the circle

    """
    start_ang = (180 - angle) / 2
    end_ang = 180 - start_ang
    t0 = np.linspace((start_ang / 180) * np.pi, (end_ang / 180) * np.pi, points, endpoint=True)

    x1 = R * np.cos(t0) + centre_x
    if up==1:
        y1 = -R * np.sin(t0) + centre_y
    else:
        y1 = centre_y+R * np.sin(t0)
    return list(zip(x1, y1))

def add_tag(img,up_bottom,tag,font,single_word_width,single_word_angle,center_loc,R,size_window,max_word_length,word_color):
    if up_bottom == "up":
        # 绘制文字
        for num,c in enumerate(tag):
            # 文字,需要Image类型
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            draw.text((center_loc[0]-single_word_width/2, center_loc[1] -
                    R-max_word_length), c, word_color, font=font)
            
            # 旋转，需要array类型
            img = np.array(img)
            #旋转前画的框
            # cv2.rectangle(img, [int(point)for point in points[0]],[int(point)for point in points[3]], (0, 0, 255) , 1, 4) 
            M = cv2.getRotationMatrix2D(center_loc, single_word_angle, 1.0)
            if num!=len(tag)-1:
                img = cv2.warpAffine(img, M, size_window, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)            



        #回正文字角度
        img = np.array(img)
        M = cv2.getRotationMatrix2D(center_loc, -(single_word_angle/2)*(len(tag)-1), 1.0)  
        img = cv2.warpAffine(img, M, size_window,
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if up_bottom == "bottom":
            #转到下环数字起点 
 
        M = cv2.getRotationMatrix2D(center_loc,single_word_angle*(len(tag)-1)/2, 1.0)  
        img = cv2.warpAffine(img, M, size_window,
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


        for num,c in enumerate(tag):
            # print(c)
            # 文字,需要Image类型
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            draw.text((center_loc[0]-single_word_width/2, center_loc[1] +
                    R+max_word_length-single_word_width), c, word_color, font=font)

            
            # 旋转，需要array类型
            img = np.array(img)
            M = cv2.getRotationMatrix2D(center_loc, -single_word_angle, 1.0)
            if num!=len(tag)-1:
                img = cv2.warpAffine(
                    img, M, size_window, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


        #回正角度
        img = np.array(img)
        M = cv2.getRotationMatrix2D(center_loc, single_word_angle*(len(tag)-1)/2, 1.0)  
        img = cv2.warpAffine(img, M, size_window,
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img

def draw_points(up_points,down_points,img):
    tk = 1
    for i in up_points:
        cv2.circle(img, [int(j) for j in i],tk, (255, 255,255 ), thickness=-1)
        tk+=1
    tk = 1
    for i in down_points:
        cv2.circle(img, [int(j) for j in i],tk, (255,225 ,0 ), thickness=-1)
        tk+=1
    cv2.imshow("draw_select_points",img)
    cv2.waitKey(0)

def Generate_images_and_labels(args): 
    # 窗口大小
    size_window = (args.windows_size,args.windows_size)
    # 旋转的圆心
    center_loc = (int(size_window[0]/2), int(size_window[1]/2))  
    # 创建画布
    img = Image.new("RGBA", size_window)  #  黑色背景 img = Image.new("RGB", size_window, "black")  # 黑色背景  
    
    if args.up_word_num!=0:
        #up_tag每个字分到的角度
        up_single_word_angle = int(args.Upper_ring_text_Angle/args.up_word_num)  
        #up_tag字体大小
        up_word_width = int((10*args.radius/args.up_word_num)*(args.Upper_ring_text_Angle/360))
        if up_word_width>100:
            up_word_width = 100
    else:
        up_single_word_angle = 0
        up_word_width = 0
    #up_tag字体型号
    up_font = ImageFont.truetype("../"+args.up_word_ttf, int(up_word_width))  # 第二个参数为size,即正方形的边长

    if args.bottom_word_num!=0:
        #bottom_tag每个字分到的角度
        bottom_single_word_angle = int(args.Bottom_ring_text_Angle/args.bottom_word_num)  
        #bottom_tag字体大小
        bottom_word_width = int((10*args.radius/args.bottom_word_num)*(args.Bottom_ring_text_Angle/360))
        if bottom_word_width>100:
            bottom_word_width = 100
    else:
        bottom_single_word_angle = 0
        bottom_word_width = 0
    #bottom_tag字体型号
    bottom_font = ImageFont.truetype("../"+args.bottom_word_ttf, int(bottom_word_width))  # 第二个参数为size,即正方形的边长


    img = np.array(img)
    
    # 画图，需要array类型 画圆圈
    cv2.circle(img, center_loc, args.radius+max(up_word_width,bottom_word_width), (0, 0, 255),3)
    
    if (args.windows_size/2) < args.radius + max(up_word_width,bottom_word_width):
        print("画布太小了！！！，画不下")
        sys.exit(0)

    # print(up_word_width,bottom_word_width)

    #_____绘制上环文字_____
    img = add_tag(img,"up",args.Upper_tag,up_font,up_word_width,up_single_word_angle,center_loc,args.radius,size_window,max(up_word_width,bottom_word_width),args.up_word_color)


    #_____绘制下环数字____
    img =  add_tag(img,"bottom",args.Bottom_tag,bottom_font,bottom_word_width,bottom_single_word_angle,center_loc,args.radius,size_window,max(up_word_width,bottom_word_width),args.bottom_word_color)

    #_____保存坐标点_____
    
    #保存的随机名字
    value = ''.join(random.sample(string.ascii_letters + string.digits, 10))

    with open("../labels/"+value+".txt", "a+",encoding='utf-8') as f:
        up_word_points = []
        bottom_word_points = []
        if args.up_word_num!=0:
            line_1 = gen_points(args.Upper_ring_text_Angle, args.up_point_number, args.radius+max(up_word_width,bottom_word_width), centre_x=center_loc[0], centre_y=center_loc[1])
            line_2 = gen_points(args.Upper_ring_text_Angle, args.up_point_number, args.radius+max(up_word_width,bottom_word_width)-up_word_width, centre_x=center_loc[0], centre_y=center_loc[1])
            up_word_points = line_1[::-1]+line_2
        if args.bottom_word_num!=0:
            line_3 = gen_points(args.Bottom_ring_text_Angle, args.bottom_point_number, args.radius+max(up_word_width,bottom_word_width), 0,centre_x=center_loc[0], centre_y=center_loc[1])
            line_4 = gen_points(args.Bottom_ring_text_Angle, args.bottom_point_number, args.radius+max(up_word_width,bottom_word_width)-bottom_word_width,0, centre_x=center_loc[0], centre_y=center_loc[1])
            bottom_word_points = line_3[::-1]+line_4

        line = [{"up_tag":args.Upper_tag,"up_points":up_word_points,
                "bottom_tag":args.Bottom_tag,"bottom_points":bottom_word_points,
                "center":[center_loc[0],center_loc[1]],"radius_in":args.radius,"radius_out":args.radius+max(up_word_width,bottom_word_width),
                "windows_size":args.windows_size}]
        line = json.dumps(line,ensure_ascii=False)
        f.write(line)
    
    draw_points(up_word_points,bottom_word_points,img)

    return img,value


def main(args,line):
    for dir in ["../imgs","../labels","../ring_text_imgs+bg"]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
            os.makedirs(dir)

    
    for num,i in enumerate(tqdm(range(len(line)))):
        if  i + args.up_word_num<= len(line):
            Upper_tag = line[i:i+args.up_word_num]
            Bottom_tag = line[i:i+args.bottom_word_num]
            args.Upper_tag = Upper_tag
            args.Bottom_tag = Bottom_tag
            img,value = Generate_images_and_labels(args)
            if args.save_pure_RingText_img==True:
                cv2.imencode('.png', img)[1].tofile("../imgs/"+value+'.png')
            if args.save_pure_RingText_img_with_background==True:
                bg_img_list = os.listdir("../back_groud")
                random.shuffle(bg_img_list)
                for bg_num,j in enumerate(bg_img_list):
                    bg = cv2.imread("../back_groud/"+j)
                    add_background2pic(bg,img,value,args)
                    break
        break

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    #————————上环文字————————
    parser.add_argument('--Upper_ring_text_Angle', type=int, default=90,help="0-360")
    parser.add_argument('--Upper_tag', type=str, default="3500",help="3500 or all,汉字种类数量")
    parser.add_argument('--up_word_num', type=int, default=10,help="up_tag的文字数量")
    parser.add_argument('--up_point_number', type=int, default=3 ,help="x>3,单侧坐标的数量") 
    parser.add_argument('--up_word_color', type=tuple, default=(0,225,0),help="Color,BGR") #绿
    parser.add_argument('--up_word_ttf', type=str, default="王汉宗中隶书简.ttf",help="方正隶变繁体.ttf、仿宋_GB2312.ttf、汉仪魏碑简.ttf、\
                                                                            经典隶书简.TTF、楷体_GB2312.ttf、宋体.ttf、王汉宗中隶书简.ttf、行楷.ttf")  
    #————————下环文字————————
    parser.add_argument('--Bottom_ring_text_Angle', type=int, default=90,help="0-360")
    parser.add_argument('--Bottom_tag', type=str, default="3500",help="3500 or all or number,汉字种类数量或者数字")
    parser.add_argument('--bottom_word_num', type=int, default=9,help="bottom_tag的文字数量")
    parser.add_argument('--bottom_point_number', type=int, default=8 ,help="x>3,单侧坐标的数量")   
    parser.add_argument('--bottom_word_color', type=tuple, default=(0,255,255),help="Color,BGR") #黄
    parser.add_argument('--bottom_word_ttf', type=str, default="王汉宗中隶书简.ttf")    
    
    
    
    #————————每张环形文字的背景图数量————————
    parser.add_argument('--bg_num', type=int, default=5 ,help="环形文字+?背景图")

    #————————环的内半径————————
    parser.add_argument('--radius', type=int, default=200,help="环的内半径")

    #————————背景图片的大小————————
    parser.add_argument('--windows_size', type=int, default=640,help="画布大小")
    
    #save1 无背景印章是否保存
    parser.add_argument('--save_pure_RingText_img', type=bool, default= True,help="是否保存无背景印章")  
    #save2 是否为无印章图片提供背景
    parser.add_argument('--save_pure_RingText_img_with_background', type=bool, default=True,help="是否执行背景替换并保存") 

    args = parser.parse_args()
    if args.Upper_tag == '3500':
        with open("../3500常用汉字.txt","r",encoding="utf-8") as f:
            line = f.readline()
        line = "好"*(args.up_word_num-1)+line+"好"*(args.up_word_num-1)
    if args.Upper_tag == 'all':
        line = []
        for i in range(0x4e00, 0x9fa5+1):
            line.append(chr(i))
        line = "好"*(args.up_word_num-1)+line+"好"*(args.up_word_num-1)    
    main(args,line)    