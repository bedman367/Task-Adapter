import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import glob



frame_path = '/DATA/xxxxxx/Datasets/hmdb51/frames'
data_path = '/DATA/xxxxxx/hmdb51'
output = '/DATA/xxxxxx/Datasets/hmdb51/annotations'
dataset_list = ['base','val','novel']


if not os.path.exists(output):
    os.makedirs(output)

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])


for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        # train: 001-064  val:065-076  test:077-100
        #1-31 32-41 42-51
        if 'base' in dataset:
            # if (i%2 == 0):

            if i <= 30: #if i+1 <= 31: i <= 63
                f = open(os.path.join(output,'train.txt'),mode='a')
                for video_path in classfile_list:
                    path = video_path
                    name = os.path.basename(path)
                    name = os.path.splitext(name)[0]
                    name = os.path.join(frame_path,name)
                    frames = len(glob.glob(pathname=os.path.join(name,'*.*')))
                    label = i
                    f.write('{} {} {}\n'.format(name,frames,label))
                # file_list = file_list + classfile_list
                # label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset:
            # if (i%4 == 1):
            if i > 30 and i <= 40: #if i+1 > 31 and i+1 <=41: i > 63 and i <=75
                f = open(os.path.join(output,'val.txt'),mode='a')
                for video_path in classfile_list:
                    path = video_path
                    name = os.path.basename(path)
                    name = os.path.splitext(name)[0]
                    name = os.path.join(frame_path,name)
                    frames = len(glob.glob(pathname=os.path.join(name,'*.*')))
                    label = i
                    f.write('{} {} {}\n'.format(name,frames,label))
        if 'novel' in dataset:
            # if (i%4 == 3):
            if i > 40:#if i+1 >41: i > 75
                f = open(os.path.join(output,'test.txt'),mode='a')
                for video_path in classfile_list:
                    path = video_path
                    name = os.path.basename(path)
                    name = os.path.splitext(name)[0]
                    name = os.path.join(frame_path,name)
                    frames = len(glob.glob(pathname=os.path.join(name,'*.*')))
                    label = i
                    f.write('{} {} {}\n'.format(name,frames,label))

    print("%s -OK" %dataset)