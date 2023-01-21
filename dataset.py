import numpy as np
import open3d as o3d
import torch
import os
from torch.utils import data
from utils import get_point_cloud, farthest_point_sample


class OpenGF(data.Dataset):
    def __init__(self, root, dir="train"):
        for _, _, filelist in os.walk(root+"/"+dir):
            self.filelist = [root+"/"+dir+"/"+filename for filename in filelist if not filename.endswith("cls.npy")]
        self.root, self.dir = root, dir

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        name = self.filelist[index][self.filelist[index].rfind("/")+1:self.filelist[index].rfind(".")]
        xyz = np.load(self.filelist[index])
        # 0: 非地面， 1: 地面
        clz = np.load(self.filelist[index][:self.filelist[index].rfind("/")+1]+name+"_cls.npy")-1

        xyz = (xyz - np.mean(xyz, axis=0, keepdims=True)) / 25
        # 查看坐标范围
        # coor_max, coor_min = np.max(xyz, axis=0), np.min(xyz, axis=0)
        # print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f  %d" % (coor_min[0], coor_max[0], coor_max[0] - coor_min[0], coor_min[1], coor_max[1], coor_max[1] - coor_min[1], coor_min[2], coor_max[2], coor_max[2] - coor_min[2], index))
        # 画图
        # pc = get_point_cloud(xyz, color=[0, 0.651, 0.929], estimate_normal=True)
        # np.asarray(pc.colors)[clz == 1] = np.array([1, 0.706, 0])
        # o3d.draw_geometries([pc], width=1000, height=800, window_name="open gf")

        # 特征是全1向量
        feats = np.ones(shape=(xyz.shape[0], 1))
        if self.dir == "val":
            # 验证集直接加载提前采样好的index，去除随机性
            # 因为我发现在这个数据集上，不同的随机采样会造成较大的结果差异，所以测试时消除随机性，保证每次测出来结果一样
            sample_idx = np.load(self.root+"/val_sample_idx/"+name+".npy")
            return torch.from_numpy(xyz).float(), torch.from_numpy(feats).float(), torch.from_numpy(clz).float(), torch.from_numpy(sample_idx).long()
        return torch.from_numpy(xyz).float(), torch.from_numpy(feats).float(), torch.from_numpy(clz).float()


if __name__ == '__main__':
    pass
