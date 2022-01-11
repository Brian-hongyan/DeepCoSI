import pickle, traceback
from rdkit import Chem as ch
import os
from DeepCoSI_GraphGenerate import *
from MyUtils_V4 import *
from DeepCoSI_Model import *
import warnings
import torch, csv
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
torch.set_default_tensor_type('torch.FloatTensor')
import pandas as pd

warnings.filterwarnings('ignore')
import argparse
import datetime


def pdb_process(pdb):
    pdb_file = open(f'./{pdb}.pdb', 'r')
    pdb_file_new = open(f'./build/{pdb}_process.pdb', 'w')
    lines = pdb_file.readlines()
    for line in lines:
        if line.startswith('HEADER') or line.startswith('TITLE') or line.startswith('SSBOND') or line.startswith(
                'ATOM'):
            pdb_file_new.write(line)


def detect_cys(pdb):
    pdb_lines = open(f'./build/{pdb}_process.pdb', 'r').readlines()
    links = []
    ssbonds = []
    ls = []
    ls_id = []
    i = 0
    
    for line in pdb_lines:
        if line.startswith('LINK'):
            # (chain,resi_id)
            links.append((line[21:22], line[22:26].strip()))
            links.append((line[51:52], line[52:56].strip()))
        elif line.startswith('ATOM'):
            
            break
    for line in pdb_lines:
        if line.startswith('SSBOND'):
            # (chain,resi_id)
            ssbonds.append((line[15], line[17:21].strip()))
            ssbonds.append((line[29], line[31:35].strip()))
        elif line.startswith('ATOM'):
            break
    for pdb_line in pdb_lines:
        if pdb_line.startswith('ATOM'):
            if pdb_line[17:20] == 'CYS':
                if (pdb_line[21], pdb_line[22:26].strip()) not in links and (
                        pdb_line[21], pdb_line[22:26].strip()) not in ssbonds:
                    if pdb_line[22:26].strip() not in ls_id:
                        i += 1
                        ls.append([i, pdb, pdb_line[21], pdb_line[22:26].strip()])
                        ls_id.append(pdb_line[22:26].strip())
    return ls


def get_pocket(samlpe_id, pdb, chain, position, distance, jobname):
    pdb_file = './build/' + pdb + '_process.pdb'
    selected_file = f'./build/{jobname}_pocket/' + f'{pdb}_{chain}_{position}' + '_pocket.pdb'
    filecontent = "from chimera import runCommand \n"
    filecontent += "runCommand('open 0 %s') \n" % pdb_file
    filecontent += "runCommand('select :%s.%s z<%d') \n" % (position, chain, distance)
    filecontent += "runCommand('write format pdb selected 0 %s') \n" % selected_file
    filecontent += "runCommand('close 0')"
    filename = f'./build/{job_name}_chimera_py/' + f'{pdb}_{chain}_{position}' + '.py'
    with open(filename, 'w') as f:
        f.write(filecontent)
    try:
        cmdline = 'module load chimera && chimera --nogui --silent --script %s' % filename
        os.system(cmdline)
    except:
        print('complex %s generation failed...' % samlpe_id)


def pdb_to_mol(samlpe_id, pdb, chain, position, jobname):
    try:
        mol = ch.MolFromPDBFile(f'./build/{jobname}_pocket/' + f'{pdb}_{chain}_{position}' + '_pocket.pdb')
        if mol.GetNumAtoms() > 10:
            file = open(f'./build/{jobname}_pickle/{pdb}_{chain}_{position}', 'wb')
            pickle.dump(mol, file)
            file.close()
    except:
        traceback.print_exc()

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def run_a_eval_epoch(model, validation_dataloader, device):
    true = []
    pred = []
    key = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            # DeepCoSI_Model.zero_grad()
            bg1, bg2, _, Ys, keys = batch
            bg1, bg2, Ys = bg1.to(device), bg2.to(device), Ys.to(device)
            outputs = model(bg1, bg2)
            true.append(Ys.data.cpu().numpy())
            pred.append(outputs.data.cpu().numpy())
            key.append(keys)
    return true, pred, key


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('pdb', type=str,help="file name for .pdb")
    argparser.add_argument('job_name', type=str, help="job name")
    argparser.add_argument('--n', type=int, default=1,help="number of processors")
    args = argparser.parse_args()
    print(args)
    correction = 0.2
    pdb = args.pdb.replace('.pdb','')
    job_name = args.job_name
    comline = f'mkdir build/{job_name}_pocket && mkdir build/{job_name}_pickle && mkdir build/{job_name}_dataset && mkdir build/{job_name}_graphs && mkdir build/{job_name}_chimera_py'
    os.system(comline)
    dt = datetime.datetime.now()
    filename = 'DeepCoSI_{}_{:02d}_{:02d}_{:02d}_{:d}.pth'.format(
        dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond)
    # process pdb
    print('Processing PDB...')
    pdb_process(pdb)
    print('Done process.')
    # detect flexible cysteine
    print('Detecting cysteines...')
    cys_list = detect_cys(pdb)
    cys_csv = open(f'./build/{job_name}_cysteines.csv', 'w', newline='')
    mycsv = csv.writer(cys_csv)
    for cys in cys_list:
        mycsv.writerow(cys)
    cys_csv.close()
    num_cys = len(cys_list)
    if num_cys == 0:
        print('No flexible cysteine for %s.' % pdb)
    else:
        # generate pocket
        print('%d cysteine(s) detected.' % num_cys)
        print('Generating pockets...')
        for cys in cys_list:
            get_pocket(cys[0], cys[1], cys[2], cys[3], 15, job_name)
            pdb_to_mol(cys[0], cys[1], cys[2], cys[3], job_name)
        print('Generating pockets done.')
        # print('Generating graph.')
        limit = None
        num_process = args.n
        path_marker = '/'
        np.set_printoptions(threshold=1e6)
        # path for RDKit molecular file
        data_path = f'./build/{job_name}_pickle'
        keys = os.listdir(data_path)
        test_keys, test_cys = [], []
        for key in keys:
            sample_id = key.split('_')[0]
            try:
                test_keys.append(key)
                test_cys.append((key.split('_')[1], int(key.split('_')[2])))
            except:
                continue
        test_dirs = [data_path + path_marker + key for key in test_keys]
        test_labels = [3 for key in test_keys]
        test_dataset = GraphDatasetGenerate(keys=test_keys[:limit], labels=test_labels[:limit], cyss=test_cys[:limit],
                                            data_dirs=test_dirs[:limit], EtaR=4.00, ShfR=3.17, Zeta=8.00, ShtZ=3.14,
                                            split=0,
                                            graph_ls_path='./build/%s_dataset' % job_name,
                                            graph_dic_path='./build/%s_graphs' % job_name,
                                            num_process=num_process, path_marker=path_marker)
        Local = './codes/DeepCoSI_model.pth'
        print('Load model %s' % Local)
        dt = datetime.datetime.now()
        # model
        DeepCoSI_Model = DeepCoSIPredictor(node_feat_size=94, edge_feat_size=20, num_layers=3,
                                           graph_feat_size=256,
                                           d_FC_layer=200, n_FC_layer=2, dropout=0.25)
        print('number of parameters : ', sum(p.numel() for p in DeepCoSI_Model.parameters() if p.requires_grad))
        device = torch.device("cuda:%s" % args.gpuid if False else "cpu")
        DeepCoSI_Model.to(device)
        DeepCoSI_Model.load_state_dict(torch.load(Local,map_location=torch.device('cpu'))['model_state_dict'])
        test_dataloader = DataLoaderX(test_dataset, 1, shuffle=False, num_workers=0,
                                      collate_fn=collate_fn, drop_last=True)
        test_true, test_pred, te_keys = run_a_eval_epoch(DeepCoSI_Model, test_dataloader, device)
        test_true = np.concatenate(np.array(test_true), 0).flatten()
        test_pred = np.concatenate(np.array(test_pred), 0).flatten()+correction
        te_keys = np.concatenate(np.array(te_keys), 0).flatten()
        pd_te = pd.DataFrame(
            {'key': te_keys,  'probability': test_pred})
        pd_te.to_csv(f'./build/{job_name}_result.csv')
        print('Job success!')
