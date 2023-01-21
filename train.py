import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import OpenGF
from utils import WeightedBCELoss, processbar
from model import SCFNet
from evaluate import evaluate

device = torch.device("cuda:0")

batch_size = 3
epoch = 50
learning_rate = 0.01
min_learning_rate = 0.0001
learning_rate_decay_gamma = 0.95
loss_fn = WeightedBCELoss()
params_save_path = "./params/SCFNet-OpenGF8192.pth"

net = SCFNet()
net.to(device)
optimizer = torch.optim.Adam(lr=learning_rate, params=net.parameters())

train_dataset = OpenGF("./OpenGF_8192", dir="train")
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = OpenGF("./OpenGF_8192", dir="val")
val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False)


def update_lr(optimizer, gamma=0.5):
    global learning_rate
    learning_rate = max(learning_rate*gamma, min_learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    print("lr update finished  cur lr: %.5f" % learning_rate)


if __name__ == '__main__':
    max_iou = 0
    for epoch_count in range(1, 1+epoch):
        net.train()
        loss_val, precision_val, recall_val, acc_val, processed = 0, 0, 0, 0, 0
        for xyz, feats, clz in train_loader:
            xyz, feats, clz = xyz.to(device), feats.to(device), clz.to(device)
            predict = net(xyz, feats)
            loss, result = loss_fn(predict, clz)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val += loss.item()
            processed += result.shape[0]
            loss_precision_recall_sum = torch.sum(result, dim=0)
            precision_val, recall_val = precision_val+loss_precision_recall_sum[1].item(), recall_val+loss_precision_recall_sum[2].item()
            for i in range(xyz.shape[0]):
                acc_val += (predict[i].view(-1).round() == clz[i]).sum(dim=0).item() / xyz.shape[1]
            cur_mean_acc, cur_mean_precision, cur_mean_recall = acc_val/processed, precision_val/processed, recall_val/processed
            print("\r进度：%s  本批loss:%.5f   precision: %.5f  recall: %.5f  acc: %.5f" % (processbar(processed, len(train_dataset)), loss.item(), cur_mean_precision, cur_mean_recall, cur_mean_acc), end="")
        mean_acc, mean_precision, mean_recall = acc_val/len(train_dataset), precision_val/len(train_dataset), recall_val/len(train_dataset)
        print("\repoch: %d  本轮loss:%.5f   precision: %.5f  recall: %.5f  acc: %.5f" % (epoch_count, loss_val, mean_precision, mean_recall, mean_acc), end="")
        print("开始测试...")
        OA, mAcc, _, mIoU, precision, recall = evaluate(net, val_loader)
        f1_score = 2*precision*recall/(precision+recall)
        if max_iou < mIoU:
            max_iou = mIoU
            print("save...")
            torch.save(net.state_dict(), params_save_path)
            print("save finished !!!")
        # 每轮后 lr = lr * 0.95
        update_lr(optimizer, learning_rate_decay_gamma)