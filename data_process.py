import numpy as np
import open3d as o3d
import torch
import os
import laspy
from torch.utils import data
from scipy.spatial.transform import Rotation
from utils import get_point_cloud, farthest_point_sample


class OpenGF(data.Dataset):
    def __init__(self, root, dir="train"):
        for _, _, filelist in os.walk(root+"/"+dir):
            self.filelist = [root+"/"+dir+"/"+filename for filename in filelist]
        self.dir = dir

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        laz = laspy.read(self.filelist[index])
        name = self.filelist[index][self.filelist[index].rfind("/")+1:self.filelist[index].rfind(".")]
        inp, clz = laz.xyz, np.asarray(laz.classification)
        valid_idx = (clz != 0)
        inp, clz = inp[valid_idx], clz[valid_idx]
        # print(inp.shape)
        # pc = get_point_cloud(inp, color=[0, 0.651, 0.929], estimate_normal=True)
        # o3d.draw_geometries([pc], width=1000, height=800, window_name="open gf")
        coor_max, coor_min = np.max(inp, axis=0), np.min(inp, axis=0)
        # print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f  %d" % (coor_min[0], coor_max[0], coor_max[0]-coor_min[0], coor_min[1], coor_max[1], coor_max[1]-coor_min[1], coor_min[2], coor_max[2], coor_max[2]-coor_min[2], index))
        # print(inp.shape, clz.shape)
        # print(clz)
        # x, y的跨度是500，切分成跨度为50的块，也就是，每个场景能切出100块
        cross, sep = 500, 50
        w = cross // 50
        point_id = ((inp[:, 0]-coor_min[0]) // sep) * w + ((inp[:, 1]-coor_min[1]) // sep)
        for i in range(w*w):
            idx = (point_id == i)
            sub_pts, sub_cls = inp[idx], clz[idx]
            fps_idx = farthest_point_sample(torch.Tensor(sub_pts).unsqueeze(0), 8192)[0].numpy()
            sub_pts, sub_cls = sub_pts[fps_idx], sub_cls[fps_idx]
            # 画出切块的子场景
            # print(sub_pts.shape, sub_cls.shape)
            # sub_pc = get_point_cloud(sub_pts, color=[0, 0.651, 0.929], estimate_normal=True)
            # o3d.draw_geometries([sub_pc], width=1000, height=800, window_name="%d" % i)
            np.save("./OpenGF_8192/%s/%s.npy" % (self.dir, name), sub_pts)
            np.save("./OpenGF_8192/%s/%s_cls.npy" % (self.dir, name), sub_cls)


if __name__ == '__main__':
    opengf = OpenGF("E:/OpenGF_Exp", dir="train")
    print(len(opengf))
    for i in range(0, len(opengf)):
        _ = opengf[i]
