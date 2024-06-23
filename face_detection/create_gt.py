import numpy as np
from scipy.io import loadmat

with open('dataset/wider_face_split/wider_face_val_bbx_gt.txt', 'r') as f:
    for line in f:
        if len(line.split("--"))>1:
            filename = line.split('/')[1].split('.jpg')[0]+".txt"
            fw = open("gt_hard/"+filename, 'w')
        elif len(line.split('\n')[0].split(" "))==11:
            info = np.array(line.split('\n')[0].split(" "))

            # If easy
            # if(np.all(info[4:10]==['0','0','0','0','0','0'])):
            #     file_line = "face "+info[0]+" "+info[1]+" "+str(int(info[0])+(int(info[2])))+" "+str(int(info[1])+(int(info[3])))+"\n"
            # else:
            #     file_line = "face "+info[0]+" "+info[1]+" "+str(int(info[0])+(int(info[2])))+" "+str(int(info[1])+(int(info[3])))+" difficult\n"

            # If hard
            file_line = "face "+info[0]+" "+info[1]+" "+str(int(info[0])+(int(info[2])))+" "+str(int(info[1])+(int(info[3])))+"\n"

            fw.write(file_line)