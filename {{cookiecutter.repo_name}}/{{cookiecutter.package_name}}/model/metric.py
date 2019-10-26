import torch


def top_1_acc(output, target):
    return top_k_acc(output, target, k=1)


def top_3_acc(output, target):
    return top_k_acc(output, target, k=3)


def top_k_acc(output, target, k):
    pred = torch.topk(output, k, dim=1)[1]
    assert pred.shape[0] == len(target)
    correct = 0
    for i in range(k):
        correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
