import glob
import os
from scipy.io import loadmat
import numpy as np
from PIL import Image
import cv2

colors = [(50, 0, 0), (100, 0, 0), (150, 0, 0), (200, 0, 0), (250, 0, 0), (0, 50, 0), (0, 100, 0), (0, 150, 0),
          (0, 200, 0), (0, 250, 0), (0, 0, 50), (0, 0, 100), (0, 0, 150), (0, 0, 200), (0, 0, 250), (50, 50, 50)]
#LSP
def lsp_dataset(mat_path, image_path, save_path,save_path1):
    """
    lsp数据集共2000张图片
    """
    joints = loadmat(mat_path)
    joints = joints["joints"].transpose(2, 0, 1)
    joints = joints[:, :, :]

    #num = 0
    for img_path in glob.glob("%s/*.jpg" % image_path):
        img_name = img_path.split("/")[-1].split(".")[0]
        img = Image.open(img_path)
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgh, imgw = img.shape[:2]
        num = int(img_name[2:])
        cen_points = joints[num-1, ...]

        points_num = cen_points.shape[-1]
        point_dict = {}
        ps = []
        for points_ in range(points_num):
            point_x = cen_points[0, points_]
            point_y = cen_points[1, points_]
            vi = cen_points[2, points_]
            if vi==0:
                vi = 2.0
            elif vi==2:
                print(name)
            point_dict[str(points_)] = [point_x/imgw, point_y/imgh,vi]
            # cv2.circle(img, (int(point_x), int(point_y)), 5, colors[points_],
            #                   thickness=-1)

            ps.append([int(point_x), int(point_y)])
        # x, y, w, h = cv2.boundingRect(np.array([ps]))
        # x = (x+w/2)/imgw
        # y = (y+h/2)/imgh
        # w = (w+6)/imgw
        # h = (h+6)/imgh
        x =0.5
        y = 0.5
        w =1
        h=1

        with open(os.path.join(save_path, img_name + ".txt"), "w") as f:
            f.write(str(0)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h))
            cv2.rectangle(img,(int(x*imgw-w*imgw/2),int(y*imgh-h*imgh/2)),
                          (int(x*imgw+w*imgw/2),int(y*imgh+h*imgh/2)),
                          (0,0,255),5)
            for i in point_dict:
                p = point_dict[i]
                f.write(" "+str(p[0]) + " " + str(p[1]) + " " + str(p[2]))
                cv2.circle(img, (int(p[0]*imgw), int(p[1]*imgh)), 5, colors[points_],
                           thickness=-1)
            f.write("\n")



            #img_txt.write(str(point_dict))
        f.close()
        #num += 1
        # 若不想看图片中关键点的位置是否准确，请注释掉后面两行
        # cv2.imshow("img", img)
        # cv2.waitKey()
        cv2.imwrite(save_path1+"/"+img_name+".jpg",img)

#FLIC
def flic_dataset(mat_path, image_path, save_path,save_path1):
    examples = loadmat(mat_path)
    examples = examples["examples"][0]
    joint_ids = ['lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri', 'lhip',
                 'lkne', 'lank', 'rhip', 'rkne', 'rank', 'leye', 'reye',
                 'lear', 'rear', 'nose', 'msho', 'mhip', 'mear', 'mtorso',
                 'mluarm', 'mruarm', 'mllarm', 'mrlarm', 'mluleg', 'mruleg',
                 'mllleg', 'mrlleg']
    available = ['lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri', 'lhip','rhip', 'head']#最后保留的9个节点

    for i, example in enumerate(examples):
        joint = example[2].T
        img_name = example[3][0]
        joints = dict(zip(joint_ids, joint))

        img =cv2.imread(image_path+"/"+img_name)
        img_name = img_name.split(".")[0]
        imgh, imgw = img.shape[:2]
        point_dict = {}
        ps = []
        #左眼，右眼，鼻子三个平均为head
        head = np.asarray(joints['reye']) + \
               np.asarray(joints['leye']) + \
               np.asarray(joints['nose'])
        head /= 3
        joints['head'] = head.tolist()
        for name in available:
            #joint_pos.append(joints[name])
            point = joints[name]
            point_dict[name] = [point[0]/imgw, point[1]/imgh,2.0]
            ps.append([int(point[0]), int(point[1])])
        x, y, w, h = cv2.boundingRect(np.array([ps]))
        x = (x+w/2)/imgw
        y = (y+h/2)/imgh
        w = (w+20)/imgw
        h = (h+20)/imgh
        with open(os.path.join(save_path, img_name + ".txt"), "w") as f:
            f.write(str(0) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h))
            cv2.rectangle(img, (int(x * imgw - w * imgw / 2), int(y * imgh - h * imgh / 2)),
                          (int(x * imgw + w * imgw / 2), int(y * imgh + h * imgh / 2)),
                          (0, 0, 255), 5)
            c =0
            for i in point_dict:
                p = point_dict[i]
                f.write(" " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]))
                cv2.circle(img, (int(p[0] * imgw), int(p[1] * imgh)), 5, colors[c],
                           thickness=-1)
            f.write("\n")
            c = c+1
            # img_txt.write(str(point_dict))
        f.close()

        # num += 1
        # 若不想看图片中关键点的位置是否准确，请注释掉后面两行
        # cv2.imshow("img", img)
        # cv2.waitKey()
        cv2.imwrite(save_path1 + "/" + img_name + ".jpg", img)


#数据集划分
def split_train_val(imgpath,txtpath,savepath):
    if not os.path.exists(savepath+"/images"):
        os.makedirs(savepath+"/images")
    if not os.path.exists(savepath+"/images/train"):
        os.makedirs(savepath+"/images/train")
    if not os.path.exists(savepath+"/images/val"):
        os.makedirs(savepath+"/images/val")
    if not os.path.exists(savepath + "/labels"):
        os.makedirs(savepath + "/labels")
    if not os.path.exists(savepath+"/labels/train"):
        os.makedirs(savepath+"/labels/train")
    if not os.path.exists(savepath+"/labels/val"):
        os.makedirs(savepath+"/labels/val")

    ps = os.listdir(image_path)
    trainls = int(len(ps)*0.9)
    import random
    random.shuffle(ps)
    valps = ps[trainls:]
    import shutil
    for name in ps:
        n = name.split(".")[0]
        if name in valps:
            shutil.copyfile(imgpath + "/" + name, savepath + "/images/val/" + name)
            shutil.copyfile(txtpath + "/" + n+".txt", savepath + "/labels/val/" + n+".txt")
        else:
            shutil.copyfile(imgpath+"/"+name,savepath+"/images/train/"+name)
            shutil.copyfile(txtpath + "/" + n + ".txt", savepath + "/labels/train/" + n + ".txt")

mat_path = ".../examples.mat" #数据集里面mat文件所在地址，包含mat文件名
image_path = ".../images" #数据集图像的地址，不包含图像名
save_path = ".../txt" #保存的txt的地址
save_path1 = ".../v1" #保存的可视化结果地址
save_path2 = ".../dataset" #保存的最后可用于yolov8-pose训练的数据集地址
#lsp数据集
lsp_dataset(mat_path, image_path, save_path,save_path1)
split_train_val(image_path,save_path,save_path2)
#flic数据集
#flic_dataset(mat_path, image_path, save_path,save_path1)
#split_train_val(image_path,save_path,save_path2)
