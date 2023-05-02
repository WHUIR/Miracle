import torch


def cal_recall(label, predict, ks):
    """
    label，predict，ks都是数组。
    label是标签构成的数组，[int, int,..., int]
    predict是预测结果构成的数据, [tensor, tensor,..., tensor]
    ks是k
    :param label:
    :param predict:
    :param ks:
    :return:
    """
    label = torch.cat(label, dim=0).unsqueeze(-1)
    predict = torch.cat(predict, dim=0).cpu()
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    recall = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return recall


def cal_ndcg(label, predict, ks):
    """
    label，predict，ks都是数组。
    label是标签构成的数组，[int, int,..., int]
    predict是预测结果构成的数据, [tensor, tensor,..., tensor]
    ks是k
    :param label:
    :param predict:
    :param ks:
    :return:
    """
    label = torch.cat(label, dim=0).unsqueeze(-1)
    predict = torch.cat(predict, dim=0).cpu()
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def cal_recall_analysis(label, predict, ks=10):
    """
    label，predict，ks都是数组。
    label是标签构成的数组，[int, int,..., int]
    predict是预测结果构成的数据, [tensor, tensor,..., tensor]
    ks是k
    :param label:
    :param predict:
    :param ks:
    :return:
    """
    label = torch.cat(label, dim=0).unsqueeze(-1)
    predict = torch.cat(predict, dim=0).cpu()
    max_ks = ks
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    recall = hit.sum(dim=-1)
    return recall


def cal_ndcg_analysis(label, predict, ks=10):
    """
    label，predict，ks都是数组。
    label是标签构成的数组，[int, int,..., int]
    predict是预测结果构成的数据, [tensor, tensor,..., tensor]
    ks是k
    :param label:
    :param predict:
    :param ks:
    :return:
    """
    label = torch.cat(label, dim=0).unsqueeze(-1)
    predict = torch.cat(predict, dim=0).cpu()
    max_ks = ks
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    max_dcg = dcg(torch.tensor([1] + [0] * (ks-1)))
    predict_dcg = dcg(hit[:, :ks])
    return predict_dcg/max_dcg






if __name__ == '__main__':
    label = torch.randint(low=0, high=99, size=[100000]).tolist()
    predict = [torch.randint(low=1, high=100, size=[100]) for i in range(len(label))]
    ks = [10, 20, 100]
    # print(cal_recall(label, predict, ks))
    print(cal_ndcg(label, predict, ks))