import rdkit
import os
from DeepCoSI_GraphGenerate import *
from MyUtils_V4 import *
from DeepCoSI_Model import *
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import time
import warnings
import torch,csv

torch.set_default_tensor_type('torch.FloatTensor')
import pandas as pd

warnings.filterwarnings('ignore')
import argparse
import datetime

limit = None
num_process = 12
path_marker = '/'
np.set_printoptions(threshold=1e6)
class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def run_a_train_epoch(model, loss_fn, train_dataloader, optimizer, device):
    # training model for one epoch
    model.train()
    for i_batch, batch in enumerate(train_dataloader):
        model.zero_grad()
        bg1, bg2, _, Ys, _ = batch
        bg1, bg2, Ys = bg1.to(device), bg2.to(device), Ys.to(device)
        outputs = model(bg1,bg2)
        loss = loss_fn(outputs, Ys)
        loss.backward()
        optimizer.step()


def run_a_eval_epoch(model, validation_dataloader, device):
    true = []
    pred = []
    key = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            # DeepCoSI_Model.zero_grad()
            bg1, bg2, _, Ys, keys = batch
            bg1, bg2, Ys = bg1.to(device), bg2.to(device),Ys.to(device)
            outputs = model(bg1,bg2)
            true.append(Ys.data.cpu().numpy())
            pred.append(outputs.data.cpu().numpy())
            key.append(keys)
    return true, pred, key

cmd = 'mkdir ./example/trainset ; mkdir ./example/validset ; mkdir ./example/testset ; mkdir ./example/stats ; mkdir ./example/models'
os.system(cmd)
# path for RDKit molecular file
data_path = './example/RDKit_mols'
keys = os.listdir(data_path)

postive_keys, negetive_keys = [], []
for key in keys:
    if key.split('_')[1] == '1':
        postive_keys.append(key)
    else:
        negetive_keys.append(key)
# .csv to store the information of cysteine position and data set splitting.
split_file = list(csv.reader(open('./example/split.csv','r')))
split_dic = {}
for line in split_file[1:]:
    sample_id = line[0]
    # split information
    split_set = line[1]
    # chain ID in PDB
    chain = line[2]
    # residue ID in PDB
    resi_id = int(line[3])
    split_dic[sample_id] = [split_set,chain,resi_id]

train_keys, train_cys, valid_keys, valid_cys, test_keys, test_cys= [], [], [], [], [], []
for key in keys:
    sample_id = key.split('_')[0]
    try:
        if split_dic[sample_id][0] == 'train':
            train_keys.append(key)
            train_cys.append((split_dic[sample_id][1],split_dic[sample_id][2]))
        elif split_dic[sample_id][0] == 'val':
            valid_keys.append(key)
            valid_cys.append((split_dic[sample_id][1], split_dic[sample_id][2]))
        elif split_dic[sample_id][0] == 'test':
            test_keys.append(key)
            test_cys.append((split_dic[sample_id][1], split_dic[sample_id][2]))
    except:
        continue

train_dirs = [data_path + path_marker + key for key in train_keys]
train_labels = [eval(key.split('_')[1]) for key in train_keys]
valid_dirs = [data_path + path_marker + key for key in valid_keys]
valid_labels = [eval(key.split('_')[1]) for key in valid_keys]
test_dirs = [data_path + path_marker + key for key in test_keys]
test_labels = [eval(key.split('_')[1]) for key in test_keys]

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    # paras for training
    argparser.add_argument('--gpuid', type=str, default='0', help="gpu id for training model")
    argparser.add_argument('--lr', type=float, default=10 ** -3.5, help="Learning rate")
    argparser.add_argument('--epochs', type=int, default=5000, help="Number of epochs in total")
    argparser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    argparser.add_argument('--tolerance', type=float, default=0.0, help="early stopping tolerance")
    argparser.add_argument('--patience', type=int, default=70, help="early stopping patience")
    argparser.add_argument('--l2', type=float, default=0.00, help="L2 regularization")
    argparser.add_argument('--repetitions', type=int, default=5, help="the number of independent runs")

    # paras for model
    argparser.add_argument('--graph_feat_size', type=int, default=256)
    argparser.add_argument('--num_layers', type=int, default=3, help='the number of intra-molecular layers')
    argparser.add_argument('--d_FC_layer', type=int, default=200, help='the hidden layer size of task networks')
    argparser.add_argument('--n_FC_layer', type=int, default=2, help='the number of hidden layers of task networks')
    argparser.add_argument('--dropout', type=float, default=0.25, help='dropout ratio')
    argparser.add_argument('--n_tasks', type=int, default=1)

    argparser.add_argument('--num_workers', type=int, default=0,
                           help='number of workers for loading data in Dataloader')
    argparser.add_argument('--dic_path_suffix', type=str, default='0')

    # paras for TorchANI setting
    argparser.add_argument('--EtaR', type=float, default=4.00, help='EtaR')
    argparser.add_argument('--ShfR', type=float, default=3.17, help='ShfR')
    argparser.add_argument('--Zeta', type=float, default=8.00, help='Zeta')
    argparser.add_argument('--ShtZ', type=float, default=3.14, help='ShtZ')

    args = argparser.parse_args()
    print(args)

    lr, epochs, batch_size, num_workers, = args.lr, args.epochs, args.batch_size, args.num_workers
    tolerance, patience, l2, repetitions = args.tolerance, args.patience, args.l2, args.repetitions

    # paras for model
    graph_feat_size, num_layers = args.graph_feat_size, args.num_layers
    d_FC_layer, n_FC_layer, dropout, n_tasks = args.d_FC_layer, args.n_FC_layer, args.dropout, args.n_tasks
    dic_path_suffix = args.dic_path_suffix

    # paras for TorchANI setting
    EtaR, ShfR, Zeta, ShtZ = args.EtaR, args.ShfR, args.Zeta, args.ShtZ

    # generating the graph using multi process

    train_dataset = GraphDatasetGenerate(keys=train_keys[:limit], labels=train_labels[:limit],cyss=train_cys[:limit] ,
                                           data_dirs=train_dirs[:limit], EtaR=EtaR, ShfR=ShfR, Zeta=Zeta, ShtZ=ShtZ,split = 0,
                                           graph_ls_path='./example/trainset',
                                           graph_dic_path='./example/graphs',
                                           num_process=num_process, path_marker=path_marker)
    valid_dataset = GraphDatasetGenerate(keys=valid_keys[:limit], labels=valid_labels[:limit],cyss=valid_cys[:limit],
                                           data_dirs=valid_dirs[:limit], EtaR=EtaR, ShfR=ShfR, Zeta=Zeta, ShtZ=ShtZ,split = 0,
                                           graph_ls_path='./example/validset',
                                           graph_dic_path='./example/graphs',
                                           num_process=num_process, path_marker=path_marker)
    test_dataset = GraphDatasetGenerate(keys=test_keys[:limit], labels=test_labels[:limit],cyss=test_cys[:limit],
                                          data_dirs=test_dirs[:limit], EtaR=EtaR, ShfR=ShfR, Zeta=Zeta, ShtZ=ShtZ,split = 0,
                                          graph_ls_path='./example/testset',
                                          graph_dic_path='./example/graphs',
                                          num_process=num_process, path_marker=path_marker)
    stat_res = []

    print('the number of train data:', len(train_dataset))
    print('the number of valid data:', len(valid_dataset))
    print('the number of test data:', len(test_dataset))

    # For average calculation.
    train_auc_total = 0
    valid_auc_total = 0
    test_auc_total = 0

    train_pr_auc_total = 0
    valid_pr_auc_total = 0
    test_pr_auc_total = 0

    # # Train model
    for repetition_th in range(repetitions):
        dt = datetime.datetime.now()
        filename = './example/models/DeepCoSI_{}_{:02d}_{:02d}_{:02d}_{:d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond)
        print('Independent run %s' % repetition_th)
        print('model file %s' % filename)
        set_random_seed(repetition_th)
        train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                                       collate_fn=collate_fn,drop_last=True)
        valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn,drop_last=True)

        # model
        DeepCoSI_Model = DeepCoSIPredictor(node_feat_size=94, edge_feat_size=20, num_layers=num_layers,
                             graph_feat_size=graph_feat_size,
                             d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer, dropout=dropout)
        print('number of parameters : ', sum(p.numel() for p in DeepCoSI_Model.parameters() if p.requires_grad))
        if repetition_th == 0:
            print(DeepCoSI_Model)
        device = torch.device("cuda:%s" % args.gpuid if torch.cuda.is_available() else "cpu")
        DeepCoSI_Model.to(device)
        optimizer = torch.optim.Adam(DeepCoSI_Model.parameters(), lr=lr, weight_decay=l2)

        stopper = EarlyStopping(mode='lower', patience=patience, tolerance=tolerance, filename=filename)
        loss_fn = FocalLoss(gamma=2, alpha=len(negetive_keys) / (len(negetive_keys) + len(postive_keys)))

        for epoch in range(epochs):
            st = time.time()
            # train
            run_a_train_epoch(DeepCoSI_Model, loss_fn, train_dataloader, optimizer, device)

            # validation
            train_true, train_pred, _ = run_a_eval_epoch(DeepCoSI_Model, train_dataloader, device)
            valid_true, valid_pred, _ = run_a_eval_epoch(DeepCoSI_Model, valid_dataloader, device)

            train_true = np.concatenate(np.array(train_true), 0)
            train_pred = np.concatenate(np.array(train_pred), 0)

            valid_true = np.concatenate(np.array(valid_true), 0)
            valid_pred = np.concatenate(np.array(valid_pred), 0)

            train_loss = loss_fn(torch.tensor(train_pred, dtype=torch.float),
                                 torch.tensor(train_true, dtype=torch.float))
            valid_loss = loss_fn(torch.tensor(valid_pred, dtype=torch.float),
                                 torch.tensor(valid_true, dtype=torch.float))

            train_auc = roc_auc_score(train_true, train_pred)
            valid_auc = roc_auc_score(valid_true, valid_pred)

            early_stop = stopper.step(1-valid_auc, DeepCoSI_Model)
            end = time.time()
            if early_stop:
                break
            print(
                "epoch:%s \t train_auc:%.4f \t valid_auc:%.4f \t time:%.3f s" % (
                    epoch, train_auc, valid_auc, end - st))

        # load the best model
        stopper.load_checkpoint(DeepCoSI_Model)
        train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn,drop_last=True)
        valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn,drop_last=True)
        test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                      collate_fn=collate_fn,drop_last=True)
        train_true, train_pred, train_keys = run_a_eval_epoch(DeepCoSI_Model, train_dataloader, device)
        valid_true, valid_pred, valid_keys = run_a_eval_epoch(DeepCoSI_Model, valid_dataloader, device)
        test_true, test_pred, test_keys = run_a_eval_epoch(DeepCoSI_Model, test_dataloader, device)

        # metrics
        train_true = np.concatenate(np.array(train_true), 0).flatten()
        train_pred = np.concatenate(np.array(train_pred), 0).flatten()

        train_keys = np.concatenate(np.array(train_keys), 0).flatten()
        valid_keys = np.concatenate(np.array(valid_keys), 0).flatten()
        test_keys = np.concatenate(np.array(test_keys), 0).flatten()

        valid_true = np.concatenate(np.array(valid_true), 0).flatten()
        valid_pred = np.concatenate(np.array(valid_pred), 0).flatten()
        test_true = np.concatenate(np.array(test_true), 0).flatten()
        test_pred = np.concatenate(np.array(test_pred), 0).flatten()

        pd_tr = pd.DataFrame(
            {'key': train_keys, 'train_true': train_true, 'train_pred': train_pred})
        pd_va = pd.DataFrame(
            {'key': valid_keys, 'valid_true': valid_true, 'valid_pred': valid_pred})
        pd_te = pd.DataFrame(
            {'key': test_keys, 'test_true': test_true, 'test_pred': test_pred})

        pd_tr.to_csv(
            './example/stats/train_{}_{:02d}_{:02d}_{:02d}_{:d}_tr.csv'.format(
                dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)
        pd_va.to_csv(
            './example/stats/valid_{}_{:02d}_{:02d}_{:02d}_{:d}_va.csv'.format(
                dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)
        pd_te.to_csv(
            './example/stats/test_{}_{:02d}_{:02d}_{:02d}_{:d}_te.csv'.format(
                dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)

        train_loss = loss_fn(torch.tensor(train_pred, dtype=torch.float), torch.tensor(train_true, dtype=torch.float))
        valid_loss = loss_fn(torch.tensor(valid_pred, dtype=torch.float), torch.tensor(valid_true, dtype=torch.float))
        test_loss = loss_fn(torch.tensor(test_pred, dtype=torch.float), torch.tensor(test_true, dtype=torch.float))
        train_auc = roc_auc_score(train_true, train_pred)
        valid_auc = roc_auc_score(valid_true, valid_pred)
        test_auc = roc_auc_score(test_true, test_pred)

        precision, recall, thresholds = precision_recall_curve(train_true, train_pred)
        train_pr_auc = auc(recall, precision)
        precision, recall, thresholds = precision_recall_curve(valid_true, valid_pred)
        valid_pr_auc = auc(recall, precision)
        precision, recall, thresholds = precision_recall_curve(test_true, test_pred)
        test_pr_auc = auc(recall, precision)

        print("train_loss:%.4f \t valid_loss:%.4f test_loss:%.4f" % (train_loss, valid_loss, test_loss))
        print("train_auc:%.4f \t valid_auc:%.4f test_auc:%.4f" % (train_auc, valid_auc, test_auc))
        print("train_pr_auc:%.4f \t valid_pr_auc:%.4f test_pr_auc:%.4f" % (train_pr_auc, valid_pr_auc, test_pr_auc))

        train_auc_total += train_auc
        valid_auc_total += valid_auc
        test_auc_total += test_auc

        train_pr_auc_total += train_pr_auc
        valid_pr_auc_total += valid_pr_auc
        test_pr_auc_total += test_pr_auc

    print('***average result ***')
    print("train_auc:%.4f \t valid_auc:%.4f test_auc:%.4f" % (train_auc_total/(repetitions), valid_auc_total/(repetitions), test_auc_total/(repetitions)))
    print("train_pr_auc:%.4f \t valid_pr_auc:%.4f test_pr_auc:%.4f" % (
    train_pr_auc_total / (repetitions), valid_pr_auc_total / (repetitions), test_pr_auc_total / (repetitions)))