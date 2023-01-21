import torch
from torch import nn
from utils import knn, farthest_point_sample, index_points, square_distance, random_sample, feature_group


def relative_pos_transforming(xyz, neigh_idx, neighbor_xyz):
    batch_size, npts = xyz.shape[0], xyz.shape[1]
    xyz_tile = xyz.view(batch_size, npts, 1, 3).repeat([1, 1, neigh_idx.shape[-1], 1])
    # batch x npts x k x 3
    relative_xyz = xyz_tile - neighbor_xyz
    # batch x npts x k x 1
    relative_alpha = torch.unsqueeze(torch.atan2(relative_xyz[:, :, :, 1], relative_xyz[:, :, :, 0]), dim=3)
    # batch x npts x k
    relative_xydis = torch.sqrt(torch.sum(torch.square(relative_xyz[:, :, :, :2]), dim=-1))
    # batch x npts x k x 1
    relative_beta = torch.unsqueeze(torch.atan2(relative_xyz[:, :, :, 2], relative_xydis), dim=-1)
    # batch x npts x k x 1
    relative_dis = torch.sqrt(torch.sum(torch.square(relative_xyz), dim=3, keepdim=True))

    relative_info = torch.cat([relative_dis, xyz_tile, neighbor_xyz], dim=3)

    # negative exp of geometric distance
    exp_dis = torch.exp(-relative_dis)

    # volume of local region
    # batch x npts
    local_volume = torch.pow(torch.max(torch.max(relative_dis, dim=3)[0], dim=2)[0], 3)

    return relative_info, relative_alpha, relative_beta, exp_dis, local_volume


def gather_neighbour(features, neigh_idx):
    # b x n x c, b x n x k
    batch_size, num_points, num_dims, k = features.shape[0], features.shape[1], features.shape[2], neigh_idx.shape[2]
    idx = (neigh_idx + torch.arange(0, batch_size, device=features.device).contiguous().view(-1, 1, 1) * num_points).view(-1)
    features = features.contiguous().view(batch_size*num_points, num_dims)[idx, :]
    features = features.contiguous().view(batch_size, num_points, k, num_dims)
    return features


class LPR(nn.Module):
    def __init__(self):
        super(LPR, self).__init__()

    def forward(self, xyz, neigh_idx):
        # x: batch x npts x 3
        # b x npts x k x 3
        batch_size, npts = xyz.shape[0], xyz.shape[1]
        neighbor_xyz = gather_neighbour(xyz, neigh_idx)

        # Relative position transforming
        # b x n x k x 3  b x n x k x 1  b x n x k x 1  b x n x k x 1    b x n
        relative_info, relative_alpha, relative_beta, geometric_dis, local_volume = relative_pos_transforming(xyz, neigh_idx, neighbor_xyz)

        # Local direction calculation (angle)
        # b x n x 3
        neighbor_mean = torch.mean(neighbor_xyz, dim=2)
        # 自己和邻域中心的连接作为方向
        direction = xyz - neighbor_mean
        # b x n x k x 3
        direction_tile = direction.view(batch_size, npts, 1, 3).repeat([1, 1, neigh_idx.shape[-1], 1])
        # b x n x k x 1
        direction_alpha = torch.atan2(direction_tile[:, :, :, 1], direction_tile[:, :, :, 0]).unsqueeze(dim=3)
        # b x n x k
        direction_xydis = torch.sqrt(torch.sum(torch.square(direction_tile[:, :, :, :2]), dim=3))
        # b x n x k x 1
        direction_beta = torch.atan2(direction_tile[:, :, :, 2], direction_xydis).unsqueeze(dim=3)

        # Polar angle updating
        angle_alpha = relative_alpha - direction_alpha
        angle_beta = relative_beta - direction_beta
        # b x n x k x 2
        angle_updated = torch.cat([angle_alpha, angle_beta], dim=3)
        # Generate local spatial representation
        # b x n x k x 9
        local_rep = torch.cat([angle_updated, relative_info], dim=3)

        # Calculate volume ratio for GCF
        # b x n
        global_dis = torch.sqrt(torch.sum(torch.square(xyz), dim=2))
        # b x 1
        global_volume = torch.pow(torch.max(global_dis, dim=1)[0], 3).unsqueeze(dim=1)
        # b x n x 1
        lg_volume_ratio = (local_volume / global_volume).unsqueeze(dim=2)
        # b x n x k x 9   b x n x k x 1    b x n x 1
        return local_rep, geometric_dis, lg_volume_ratio


class DDAP(nn.Module):
    def __init__(self, d_in, d_out):
        super(DDAP, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(9, d_in, kernel_size=1, stride=1),
            nn.InstanceNorm2d(d_in),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(d_in, d_out//2, kernel_size=1, stride=1),
            nn.InstanceNorm2d(d_out//2),
            nn.LeakyReLU(0.2)
        )
        self.dap_fc1 = nn.Linear(d_in*2+2, d_in*2, bias=False)
        self.dap_fc2 = nn.Linear(d_out+2, d_out, bias=False)
        self.dap_conv1d1 = nn.Sequential(
            nn.Conv1d(d_in*2, d_out//2, kernel_size=1, stride=1),
            nn.InstanceNorm1d(d_out//2),
            nn.LeakyReLU(0.2)
        )
        self.dap_conv1d2 = nn.Sequential(
            nn.Conv1d(d_out, d_out, kernel_size=1, stride=1),
            nn.InstanceNorm1d(d_out),
            nn.LeakyReLU(0.2)
        )
        self.softmax = nn.Softmax(dim=1)
        self.local_polar_representation = LPR()

    def dualdis_att_pool(self, feature_set, f_dis, g_dis, opid):
        # b x n x k x [id1: 2c, d_out]  b x n x k x 1  b x n x k x 1  [d_out//2, d_out]
        batch_size = feature_set.shape[0]
        num_points = feature_set.shape[1]
        num_neigh = feature_set.shape[2]
        d = feature_set.shape[3]
        d_dis = g_dis.shape[3]

        # bn x k x d
        f_reshaped = feature_set.view(-1, num_neigh, d)
        # bn x k x 1
        f_dis_reshaped = f_dis.view(-1, num_neigh, d_dis) * 0.1
        # bn x k x 1
        g_dis_reshaped = g_dis.view(-1, num_neigh, d_dis)
        concat = torch.cat([g_dis_reshaped, f_dis_reshaped, f_reshaped], dim=2)

        # weight learning
        # bn x k x d
        if opid == 1:
            att_activation = self.dap_fc1(concat)
        else:
            att_activation = self.dap_fc2(concat)
        att_scores = self.softmax(att_activation)
        # dot product
        f_lc = f_reshaped * att_scores
        # sum
        # bn x d
        f_lc = torch.sum(f_lc, dim=1)
        # b x n x d
        f_lc = f_lc.view(batch_size, num_points, d)
        # shared MLP
        # b x n x d_out
        f_lc = f_lc.permute([0, 2, 1])
        if opid == 1:
            f_lc = self.dap_conv1d1(f_lc)
        else:
            f_lc = self.dap_conv1d2(f_lc)
        f_lc = f_lc.permute([0, 2, 1])
        # b x n x d_out
        return f_lc

    def forward(self, xyz, feature, neigh_idx):
        # xyz: b x n x 3
        # feature: b x n x c
        # LPR
        # b x n x k x 5, b x n x k x 1, b x n x 1
        local_rep, g_dis, lg_volume_ratio = self.local_polar_representation(xyz, neigh_idx)

        # b x n x k x c
        local_rep = self.conv1(local_rep.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])
        # b x n x k x c
        f_neighbours = gather_neighbour(feature, neigh_idx)
        # b x n x k x (c+c)
        f_concat = torch.cat([f_neighbours, local_rep], dim=3)
        # b x n x k x 1
        f_dis = self.cal_feature_dis(feature, f_neighbours)
        # b x n x d_out//2
        f_lc = self.dualdis_att_pool(f_concat, f_dis, g_dis, opid=1)

        # 2
        # b x n x k x d_out//2
        local_rep = self.conv2(local_rep.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])
        # b x n x k x d_out//2
        f_neighbours = gather_neighbour(f_lc, neigh_idx)
        # b x n x k x d_out
        f_concat = torch.cat([f_neighbours, local_rep], dim=3)
        # b x n x k x 1
        f_dis = self.cal_feature_dis(f_lc, f_neighbours)
        # b x n x d_out
        f_lc = self.dualdis_att_pool(f_concat, f_dis, g_dis, opid=2)

        return f_lc, lg_volume_ratio

    def cal_feature_dis(self, feature, f_neighbours):
        """
        Calculate the feature distance
        """
        # b x n x c, b x n x k x c
        batch_size, n, c = feature.shape[0], feature.shape[1], feature.shape[2]
        feature_tile = feature.view(batch_size, n, 1, c).repeat([1, 1, f_neighbours.shape[2], 1])
        # b x n x k x c
        feature_dist = feature_tile - f_neighbours
        feature_dist = torch.mean(torch.abs(feature_dist), dim=3).unsqueeze(dim=3)
        feature_dist = torch.exp(-feature_dist)

        return feature_dist


class SCF(nn.Module):
    def __init__(self, d_in, d_out):
        super(SCF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(d_in, d_out//2, kernel_size=1, stride=1),
            nn.InstanceNorm1d(d_out//2),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_out, d_out*2, kernel_size=1, stride=1),
            nn.InstanceNorm1d(d_out*2),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(d_in, d_out*2, kernel_size=1, stride=1),
            nn.InstanceNorm1d(d_out*2),
            nn.LeakyReLU(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(4, d_out*2, kernel_size=1, stride=1),
            nn.InstanceNorm1d(d_out*2),
            nn.LeakyReLU(0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(d_out*4, d_out, kernel_size=1, stride=1),
            nn.InstanceNorm1d(d_out),
            nn.LeakyReLU(0.2)
        )
        self.local_context_learning = DDAP(d_out//2, d_out)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, xyz, feature, neigh_idx):
        """
        SCF
        """
        # Local Contextual Features
        # MLP 1
        f_pc = self.conv1(feature.permute([0, 2, 1])).permute([0, 2, 1])
        # Local Context Learning (LPR + DDAP)
        # b x n x dout, b x n x 1
        f_lc, lg_volume_ratio = self.local_context_learning(xyz, f_pc, neigh_idx)
        # MLP 2
        # f_lc = helper_tf_util.conv2d(f_lc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training, activation_fn=None)
        # b x n x 2d_out
        f_lc = self.conv2(f_lc.permute([0, 2, 1])).permute([0, 2, 1])
        # MLP Shotcut
        # shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID', activation_fn=None, bn=True, is_training=is_training)
        shortcut = self.conv3(feature.permute([0, 2, 1])).permute([0, 2, 1])
        # Global Contextual Features
        # b x n x 4
        f_gc = torch.cat([xyz, lg_volume_ratio], dim=2)
        # f_gc = helper_tf_util.conv2d(f_gc, d_out * 2, [1, 1], name + 'lg', [1, 1], 'VALID', activation_fn=None, bn=True, is_training=is_training)
        # b x n x 2d_out
        f_gc = self.conv4(f_gc.permute([0, 2, 1])).permute([0, 2, 1])
        # b x n x d_out 我很确定这一行是他写错了
        # return self.leaky_relu(torch.cat([f_lc + shortcut, f_gc], dim=2))
        return self.conv5(torch.cat([f_lc + shortcut, f_gc], dim=2).permute([0, 2, 1])).permute([0, 2, 1])


class UpSample(nn.Module):
    def __init__(self, in_channel, mlp):
        super(UpSample, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.InstanceNorm1d(out_channel))
            last_channel = out_channel
        self.leaky_relu = nn.LeakyReLU(0.2)

    #                 多    少    多       少
    def forward(self, xyz1, xyz2, features1, features2):
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)
        #
        # points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = features2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(features2, idx) * weight.view(B, N, 3, 1), dim=2)

        if features1 is not None:
            # points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([features1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = self.leaky_relu(bn(conv(new_points)))
        new_points = new_points.permute(0, 2, 1)
        return new_points


class SCFNet(nn.Module):
    def __init__(self):
        super(SCFNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=1, stride=1),
            nn.InstanceNorm1d(8),
            nn.LeakyReLU()
        )
        self.encoder = nn.ModuleList()
        self.encoder.append(SCF(8, 16))
        self.encoder.append(SCF(16, 64))
        self.encoder.append(SCF(64, 128))
        self.encoder.append(SCF(128, 256))
        self.encoder.append(SCF(256, 512))
        self.last_encoder_layer = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, stride=1),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(0.2)
        )
        self.npts = [2048, 1024, 512, 256, 128]

        self.up1 = UpSample(768, [256])
        self.up2 = UpSample(384, [128])
        self.up3 = UpSample(192, [64])
        self.up4 = UpSample(80, [16])
        self.up5 = UpSample(32, [16])

        self.clasify_layer = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, stride=1),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 32, kernel_size=1, stride=1),
            nn.InstanceNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv1d(32, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, xyz, features, sampled_idx=None):
        # xyz: b x 8192 x 3, feature: b x 8192 x c
        # b x 8192 x 8
        features = self.fc(features.permute([0, 2, 1])).permute([0, 2, 1])

        # b x 8192 x 16, b x 4096 x 16, b x 2048 x 64, b x 1024 x 128, b x 512 x 256, b x 256 x 512
        xyz_list, feature_list = [], []
        npts_prefix = 0
        for i, en_layer in enumerate(self.encoder):
            # b x n x k
            neigh_idx = knn(16, xyz, xyz)
            features = en_layer(xyz, features, neigh_idx)
            if i == 0:
                feature_list.append(features)
                xyz_list.append(xyz)
            # b x n_sample
            # sample_idx = farthest_point_sample(xyz, self.npts[i])
            if sampled_idx is None:
                sample_idx = random_sample(xyz, self.npts[i])
            else:
                sample_idx = sampled_idx[:, npts_prefix:npts_prefix+self.npts[i]]
                npts_prefix += self.npts[i]
            xyz = index_points(xyz, sample_idx)
            # b x n_sample x k
            idx = index_points(neigh_idx, sample_idx)
            features = torch.max(index_points(features, idx), dim=2)[0]

            feature_list.append(features)
            xyz_list.append(xyz)
        features = self.last_encoder_layer(features.permute([0, 2, 1])).permute([0, 2, 1])
        features = self.up1(xyz_list[-2], xyz_list[-1], feature_list[-2], features)
        features = self.up2(xyz_list[-3], xyz_list[-2], feature_list[-3], features)
        features = self.up3(xyz_list[-4], xyz_list[-3], feature_list[-4], features)
        features = self.up4(xyz_list[-5], xyz_list[-4], feature_list[-5], features)
        features = self.up5(xyz_list[-6], xyz_list[-5], feature_list[-6], features)
        # print(xyz.shape, features.shape)
        result = self.clasify_layer(features.permute([0, 2, 1])).permute([0, 2, 1])
        return result


if __name__ == '__main__':
    net = SCFNet()
    device = torch.device("cuda:0")
    net.to(device)

    xyz = torch.randn(3, 8192, 3).to(device)
    features = torch.randn(3, 8192, 3).to(device)
    y = net(xyz, features)
    print(y.shape)