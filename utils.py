import numpy as np
import open3d as o3d
import torch
from torch import nn
from sklearn.metrics import precision_recall_fscore_support


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def random_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long)
    for i in range(B):
        centroids[i, :] = torch.from_numpy(np.random.permutation(N)[:npoint]).long()
    centroids = centroids.to(device)
    return centroids


def processbar(current, totle):
    process_str = ""
    for i in range(int(20*current/totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|  %d / %d" % (process_str, current, totle)


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def feature_group(features, sample_idx):
    # batch_size x n x c, batch_size x sample_n x k
    # return: batch_size x sample_n x k x c
    batch_size, n, c, device = features.shape[0], features.shape[1], features.shape[2], features.device
    # b x (sample_n x k)
    idx = torch.arange(batch_size, dtype=torch.long).to(device).view(batch_size, 1)*n + sample_idx.view(batch_size, -1)
    return features.contiguous().view(batch_size*n, c)[idx.contiguous().view(-1)].contiguous().view(batch_size, sample_idx.shape[1], sample_idx.shape[2], c)


def pc_normalize(pc):
    pmin = np.min(pc, axis=0)
    pmax = np.max(pc, axis=0)
    pc -= (pmin + pmax) / 2
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc *= 1.0 / scale
    return pc


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


# def knn(x, k):
#     pairwise_distance = square_distance(x, x)
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
#     return idx


def knn(k, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=False)
    return group_idx


def get_point_cloud(pts, color=None, estimate_normal=False):
    pc = o3d.PointCloud()
    pc.points = o3d.Vector3dVector(pts)
    if color is not None:
        pc.colors = o3d.Vector3dVector(np.array([color]*pts.shape[0]))
    if estimate_normal:
        o3d.estimate_normals(pc)
    return pc


################### loss ######################

def weighted_bce_loss(prediction, gt):
    loss = nn.BCELoss(reduction='none')
    gt = gt.float()
    class_loss = loss(prediction.view(-1), gt)

    weights = torch.ones_like(gt)
    w_negative = gt.sum() / gt.size(0)
    w_positive = 1 - w_negative

    weights[gt >= 0.5] = w_positive
    weights[gt < 0.5] = w_negative
    w_class_loss = torch.mean(weights * class_loss)

    #######################################
    # get classification precision and recall
    predicted_labels = prediction.detach().cpu().round().numpy()
    cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels,
                                                                      average='binary')
    return w_class_loss, cls_precision, cls_recall


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, predict, gt):
        # batch_size x n, batchsize x n
        batch_size = predict.shape[0]
        result = torch.zeros(batch_size, 3)
        loss = 0
        for i in range(batch_size):
            w_class_loss, cls_precision, cls_recall = weighted_bce_loss(predict[i], gt[i])
            result[i, :] = torch.Tensor([w_class_loss.item(), cls_precision, cls_recall])
            loss += w_class_loss
        loss /= batch_size
        return loss, result