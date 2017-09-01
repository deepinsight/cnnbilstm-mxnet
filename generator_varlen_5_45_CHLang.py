#coding=utf-8

import random
import os
import re
import itertools as itt
from itertools import product
from PIL import Image, ImageDraw, ImageFont
import StringIO
import pygame
from han_gen_langmodel import *
pygame.init()

"""
基本：
1 图片size
2 字符个数
3 字符区域（重叠、等分）
4 字符位置（固定、随机）
5 字符size（所占区域大小的百分比）
6 字符fonts
7 字符 type （数字、字母、汉字、数学符号）
8 字符颜色
9 背景颜色

高级：
10 字符旋转
11 字符扭曲
12 噪音（点、线段、圈）
"""

NONHAN_SHRINKAGE = 0.6
TOTAL_WIDTH = 900
TOTAL_HEIGHT = 30
TOTAL_STD_CHAR = 1.0 * TOTAL_WIDTH/TOTAL_HEIGHT

def randRGB():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def cha_draw(cha, text_color, font, rotate,size_cha):
    im = Image.new(mode='RGBA', size=(size_cha*2, size_cha*2))
    #print size_cha*2,
    drawer = ImageDraw.Draw(im)
    drawer.text(xy=(0, 0), text=cha, fill=text_color, font=font) #text 内容，fill 颜色， font 字体（包括大小）
    if rotate:
        max_angle = 40 # to be tuned
        angle = random.randint(-max_angle, max_angle)
        im = im.rotate(angle, Image.BILINEAR, expand=1)
    im = im.crop(im.getbbox())
    return im


def choice_cha(chas):
    x = random.randint(0,len(chas))
    return chas[x-1]


def istruealpha(s):
    i = ord(s)
    return True if (i>=97 and i<=122) or (i>=65 and i<=90) else False

def captcha_draw_fixedstring(size_im, fixstring, 
        fonts=None, 
        overlap=0.0,
        rd_bg_color=False, rd_text_color=False, rd_text_pos=False, rd_text_size=False,
        rotate=False, noise=None, dir_path='', img_num = 0 , img_now = 0):
    
    drifterCH = 0.95
    global TOTAL_STD_CHAR
    """
        overlap: 字符之间区域可重叠百分比, 重叠效果和图片宽度字符宽度有关
        字体大小 目前长宽认为一致！！！
        所有字大小一致
        扭曲暂未实现
        noise 可选：point, line , circle
        fonts 中分中文和英文字体
        label全保存在label.txt 中，-文件第i行对应"i.jpg"的图片标签，i从1开始
    """
    label_list = get_label(fixstring)
    num_han = len([label for label in label_list if isLabelHan(label)])
    num_nonhan = len([label for label in label_list if not isLabelHan(label)])
    if len(label_list)!=(num_han+num_nonhan):
        print "Error: Length Unequal"
    
    num_add_bk = max(int((TOTAL_STD_CHAR - (num_han+num_nonhan*NONHAN_SHRINKAGE))/NONHAN_SHRINKAGE),0)
    fixstring = fixstring+' '*num_add_bk
    label_list = get_label(fixstring)
    num_han = len([label for label in label_list if isLabelHan(label)])
    num_nonhan = len([label for label in label_list if not isLabelHan(label)])

    
    
    rate_cha = 0.87 # rate to be tuned
    width_im, height_im = size_im
    width_cha = int(width_im / max ( num_han+num_nonhan*NONHAN_SHRINKAGE - overlap, 3)) # 字符区域宽度
    height_cha = height_im*1.2# 字符区域高度
    #print width_cha,height_cha
    bg_color = 'white'
    text_color = 'black'
    derx = 0
    dery = 0

    if rd_text_size:
        rate_cha = random.uniform(rate_cha-0.02, rate_cha) # to be tuned

    if rd_bg_color:
        bg_color = randRGB()
    im = Image.new(mode='RGB', size=size_im, color=bg_color) # color 背景颜色，size 图片大小

    drawer = ImageDraw.Draw(im)
    contents = []
    
    totalshrinkedcount = 0
    
    for i, cha in enumerate(fixstring):
        chaIsHan = isLabelHan(get_label(cha)[0])
        
        if chaIsHan:
            enflate = 1.0
            shrinkage = 1.0
        else:
            enflate = 1.0/NONHAN_SHRINKAGE
            shrinkage = NONHAN_SHRINKAGE
            
        size_cha = int(rate_cha*min(width_cha/0.85, height_cha/0.85) * 1.0)# 字符大小
        #print repr(cha),width_cha,enflate,height_cha
        if rd_text_color:
            text_color = randRGB()
        if rd_text_pos:
            derx = random.randint(0, max(width_cha-size_cha-5, 0))
            if not chaIsHan:
                derx = int((derx+random.randint(0,1))/2)
            dery = random.randint(0, max(height_cha-size_cha-5, 0))

        if not chaIsHan:
            
            font = ImageFont.truetype(fonts['eng'], size_cha)
            im_cha = cha_draw(cha, text_color, font, rotate, size_cha)
            im.paste(im_cha, (int(max(i, 0)*width_cha 
                                  - totalshrinkedcount*width_cha*(1-NONHAN_SHRINKAGE)) + derx + 2, dery + 3), im_cha) # 字符左上角位置
            contents.append(cha)
            
            totalshrinkedcount+=1
            
        else:
            font = pygame.font.Font(fonts['sc'], int(size_cha*drifterCH))
            rtext = font.render(cha, True, (0, 0, 0), (255, 255, 255))
 
            #pygame.image.save(rtext, "t.gif")
            sio = StringIO.StringIO()
            pygame.image.save(rtext, sio)
            sio.seek(0)
 
            im_cha = Image.open(sio)
            im.paste(im_cha, (int(max(i, 0)*width_cha - totalshrinkedcount*width_cha*(1-NONHAN_SHRINKAGE))+derx -2, dery -3 )) # 字符左上角位置
            contents.append(cha)
            #print size_cha,           

    if 'point' in noise:
        nb_point = 20
        color_point = randRGB()
        for i in range(nb_point):
            x = random.randint(0, width_im)
            y = random.randint(0, height_im)
            drawer.point(xy=(x, y), fill=color_point)
    if 'line' in noise:
        nb_line = 3
        for i in range(nb_line):
            color_line = randRGB()
            sx = random.randint(0, width_im)
            sy = random.randint(0, height_im)
            ex = random.randint(0, width_im)
            ey = random.randint(0, height_im)
            drawer.line(xy=(sx, sy, ex, ey), fill=color_line)
    if 'circle' in noise:
        nb_circle = 20
        color_circle = randRGB()
        for i in range(nb_circle):
            sx = random.randint(0, width_im-10)
            sy = random.randint(0, height_im-10)
            temp = random.randint(1,5)
            ex = sx+temp
            ey = sy+temp
            drawer.arc((sx, sy, ex, ey), 0, 360, fill=color_circle)

    if os.path.exists(dir_path ) == False: # 如果文件夹不存在，则创建对应的文件夹
        os.mkdir(dir_path )

    #print ''.join(contents),re.sub(u' ',u'-',''.join(contents))
    img_name = str(img_now) + '_' + \
                re.sub(' ','-',''.join(contents)) \
                + '.jpg'
    img_path = dir_path + img_name
    #print img_path,
    if img_now%20==0:
        print img_now, '/', img_num
    im.save(img_path)

def test_en_ch():
    
    size_im = (TOTAL_WIDTH, TOTAL_HEIGHT)
    overlaps = [0.0, 0.05, 1.0]
    rd_text_poss = [True, False]
    rd_text_sizes = [False]
    rd_text_colors = [False] # false 代表字体颜色全一致，但都是黑色，可以是[False, True]
    rd_bg_color = False
    
    noises = [ ['point'] ] # ['line'], ['line', 'point']
    rotates =[False]
    nb_chas = range(5,20)
    nb_image = 10000
    font_dir_en = 'fonts/english_lite'
    font_dir_sc = 'fonts/simpch_lite'
    
    font_paths = []
    num_pic = 0
    dir_folder = 0
    for dirpath_en, dirnames_en, filenames_en in os.walk(font_dir_en):
        for dirpath_ch, dirnames_ch, filenames_ch in os.walk(font_dir_sc):
            L = list(itt.product(filenames_en,filenames_ch))
            for filename_en,filename_ch in L:
                filepath_en = dirpath_en + os.sep + filename_en
                filepath_ch = dirpath_ch + os.sep + filename_ch
                font_paths.append({'eng':filepath_en, 'sc':filepath_ch})

    for i in range(0,nb_image):
        num_pic += 1
        overlap = random.choice(overlaps)
        rd_text_pos = random.choice(rd_text_poss)
        rd_text_size = random.choice(rd_text_sizes)
        rd_text_color = random.choice(rd_text_colors)
        
        fixstring , _ = gen_fixlen(fixlen=25, hanlenmax=2)
        
        #fixstring = u'{0: <40}'.format(fixstring)

        noise = random.choice(noises)
        rotate = random.choice(rotates)
        font_path = random.choice(font_paths)
        if num_pic % 1001 == 0:
            dir_folder += 1
        dir_name = 'train_data_varlen_5_45_CHLang'
        dir_path = '/data1/img_data/'+dir_name+'/'
        #print fixstring
        captcha_draw_fixedstring(size_im=size_im, fixstring=fixstring,
                overlap=overlap, rd_text_pos=rd_text_pos, rd_text_size=rd_text_size,
                rd_text_color=rd_text_color, rd_bg_color=rd_bg_color, noise=noise,
                rotate=rotate, dir_path=dir_path, fonts=font_path,img_num = nb_image,img_now = i)


if __name__ == "__main__":
    test_en_ch()
    #captcha_generator()
