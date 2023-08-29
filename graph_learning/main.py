
import torch

import data_processor
import models
from config import opt
from sklearn import metrics
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F


def train(**kwargs):
    opt._parse(kwargs)
    adj, features, labels, idx_train, idx_val, idx_test, adj_lists = data_processor.load_data(opt)
    train = data_processor.Dataload(labels, idx_train)
    val = data_processor.Dataload(labels, idx_val)
    test = data_processor.Dataload(labels, idx_test)
    if opt.model is 'GCN':
        model = getattr(models, opt.model)(nfeat=features.shape[1], nhid=opt.nhid, nclass=labels.max().item() + 1,
                                           dropout=opt.dropout).train()
    elif opt.model is 'GAT' or opt.model is 'SpGAT':
        model = getattr(models, opt.model)(nfeat=features.shape[1], nhid=opt.nhid, nclass=labels.max().item() + 1,
                                           dropout=opt.dropout, alpha=opt.alpha, nheads=opt.nheads).train()
    elif opt.model is 'GraphSage':
        model = getattr(models, opt.model)(num_layers=opt.num_layers, input_size=features.shape[1], out_size=opt.nhid,
                                           num_classes=labels.max().item() + 1, adj_lists=adj_lists, device=opt.device,
                                           gcn=False, agg_func='MEAN')

    else:
        print("Please input the correct model name: GCN, GAT or Graphsage")
        return
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model = model.to(opt.device)
    adj = adj.to(opt.device)
    features = features.to(opt.device)   # 将模型以及在模型中需要使用到的矩阵加载到设备中
    train_dataloader = DataLoader(train, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_dataloader = DataLoader(test, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    criterion = F.nll_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    lr = opt.lr

    best_val_acc = 0.
    for epoch in range(opt.max_epoch):
        for trains, labels in tqdm(train_dataloader):
            labels = labels.to(opt.device)
            trains = trains.to(opt.device)
            optimizer.zero_grad()
            if opt.model is 'GCN' or opt.model is 'GAT' or opt.model is 'SpGAT':
                outputs = model(features, adj)
                loss = criterion(outputs[trains], labels)
                loss.backward()
                optimizer.step()
            elif opt.model is 'GraphSage':
                outputs = model(features, trains)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        lr = lr * opt.lr_decay
        for param_groups in optimizer.param_groups:
            param_groups['lr'] = lr
        val_acc = evaluate(opt, model, val_dataloader, epoch, features, adj)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save()
        model.train()
    model.load('./checkpoints/' + opt.model)
    evaluate(opt, model, test_dataloader, 'Test', features, adj)


def evaluate(opt, model, val_dataloader, epoch, features, adj):

    model.eval()
    loss_total = 0
    predict_all = list()
    labels_all = list()
    criterion = F.nll_loss
    with torch.no_grad():
        for evals, labels in tqdm(val_dataloader):
            labels = labels.to(opt.device)
            evals = evals.to(opt.device)
            if opt.model is 'GCN' or opt.model is 'GAT'or opt.model is 'SpGAT':
                outputs = model(features, adj)
                loss = criterion(outputs[evals], labels)
                loss_total += loss
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs[evals].data, 1)[1].cpu().numpy()
                labels = list(labels)
                predic = list(predic)
                labels_all.extend(labels)
                predict_all.extend(predic)
            elif opt.model is 'GraphSage':
                outputs = model(features, evals)
                loss = criterion(outputs, labels)
                loss_total += loss
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels = list(labels)
                predic = list(predic)
                labels_all.extend(labels)
                predict_all.extend(predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    print("The acc for Epoch %s is %f" % (str(epoch), acc))
    return acc


if __name__ == '__main__':
    train()