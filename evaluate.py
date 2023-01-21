import numpy as np
import torch
from utils import processbar, WeightedBCELoss
from model import SCFNet
from dataset import OpenGF
from torch.utils.data import DataLoader

device = torch.device("cuda")
seg_classes = {'All Scenes': [0, 1]}
loss_fn = WeightedBCELoss()


def evaluate(net, test_loader):
    net.eval()
    loss_val, process, correct, process_pts = 0, 0, 0, 0
    precision_val, recall_val, acc_val = 0, 0, 0
    # 指标统计量
    num_part = 2
    total_seen_class = [0 for _ in range(num_part)]
    total_correct_class = [0 for _ in range(num_part)]
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    shape_ious_cls = {cat: [] for cat in seg_classes.keys()}
    seg_label_to_cat = {}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat
    with torch.no_grad():
        for pts, features, label, sample_idx in test_loader:
            pts, features, label, sample_idx = pts.float().to(device), features.to(device), label.long().to(device), sample_idx.to(device)
            point_num = pts.shape[1]
            out = net(pts, features, sample_idx)
            out = out.view(pts.shape[0], -1, 1)
            cur_pred_val = out.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((pts.shape[0], point_num)).astype(np.int32)
            target = label.cpu().data.numpy()
            for i in range(pts.shape[0]):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                logits = np.concatenate([1-logits, logits], axis=1)
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            correct += (out.view(-1).round() == label.view(-1)).sum(dim=0).item()
            # loss = loss_fn(out.view(-1, num_part), label.view(-1))
            loss, result = loss_fn(out, label)
            loss_val += loss.item()
            process += pts.shape[0]
            process_pts += point_num

            loss_precision_recall_sum = torch.sum(result, dim=0)
            precision_val, recall_val = precision_val + loss_precision_recall_sum[1].item(), recall_val + loss_precision_recall_sum[2].item()
            for i in range(pts.shape[0]):
                acc_val += (out[i].view(-1).round() == label[i]).sum(dim=0).item() / pts.shape[1]
            cur_mean_acc, cur_mean_precision, cur_mean_recall = acc_val / process, precision_val / process, recall_val / process

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(pts.shape[0]):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))
                shape_ious_cls[cat].append(part_ious)
            print("\r测试进度：%s  本批loss:%.5f   precision: %.5f  recall: %.5f  acc: %.5f" % (
                processbar(process, len(test_loader.dataset)), loss.item(), cur_mean_precision, cur_mean_recall, cur_mean_acc), end="")

    all_shape_ious = []
    all_shape_ious_cls = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
        for iou in shape_ious_cls[cat]:
            all_shape_ious_cls.append(iou)
        shape_ious_cls[cat] = np.mean(np.array(shape_ious_cls[cat]), axis=0)

    mean_shape_ious = np.mean(list(shape_ious.values()))
    # 所有指标汇总统计
    accuracy = correct / (process * point_num)
    class_avg_accuracy = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)).item()
    class_avg_iou = mean_shape_ious.item()
    instance_avg_iou = np.mean(all_shape_ious).item()
    # print("\ncls iou: ")
    # print(shape_ious)
    mean_precision, mean_recall = precision_val / len(test_loader.dataset), recall_val / len(test_loader.dataset)
    instance_avg_iou1_2 = np.mean(np.array(all_shape_ious_cls), axis=0)
    print("\ntest finished!  accuracy (pixAcc): %.5f   instance avg acc (OA): %.5f  class avg iou: %.5f  instance avg iou (mIoU): %.5f  IoU1: %.5f  IoU2: %.5f" % (
        accuracy, class_avg_accuracy, class_avg_iou, instance_avg_iou, instance_avg_iou1_2[0].item(), instance_avg_iou1_2[1].item()))

    return accuracy, class_avg_accuracy, class_avg_iou, instance_avg_iou, mean_precision, mean_recall


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    params_save_path = "./params/SCFNet-OpenGF8192.pth"
    net = SCFNet()
    net.to(device)
    net.load_state_dict(torch.load(params_save_path))
    val_dataset = OpenGF("./OpenGF_8192", dir="val")
    val_loader = DataLoader(dataset=val_dataset, batch_size=5, shuffle=False)
    OA, mAcc, _, mIoU, precision, recall = evaluate(net, val_loader)