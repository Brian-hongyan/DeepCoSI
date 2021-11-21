from rdkit import Chem
import dgl
import torch
from dgllife.utils import BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, \
    atom_is_aromatic, ConcatFeaturizer, bond_type_one_hot, atom_hybridization_one_hot, \
    one_hot_encoding, atom_formal_charge, atom_num_radical_electrons, bond_is_conjugated, \
    bond_is_in_ring, bond_stereo_one_hot
import pickle
import os
from dgllife.utils import BaseBondFeaturizer
from functools import partial
import warnings
import multiprocessing
from itertools import repeat
import numpy as np
import traceback
from torchani import SpeciesConverter, AEVComputer

converter = SpeciesConverter(['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'])
warnings.filterwarnings('ignore')

# the chirality information defined in the AttentiveFP
def chirality(atom):
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


class MyAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_filed='h'):
        super(MyAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_filed: ConcatFeaturizer([partial(atom_type_one_hot,
                                                                         allowable_set=['C', 'N', 'O', 'S', 'F', 'P',
                                                                                        'Cl', 'Br', 'I', 'B', 'Si',
                                                                                        'Fe', 'Zn', 'Cu', 'Mn', 'Mo'],
                                                                         encode_unknown=True),
                                                                 partial(atom_degree_one_hot,
                                                                         allowable_set=list(range(6))),
                                                                 atom_formal_charge, atom_num_radical_electrons,
                                                                 partial(atom_hybridization_one_hot,
                                                                         encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 # A placeholder for aromatic information,
                                                                 atom_total_num_H_one_hot, chirality])})

class MyBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_filed='e'):
        super(MyBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_filed: ConcatFeaturizer([bond_type_one_hot, bond_is_conjugated, bond_is_in_ring,
                                                                 partial(bond_stereo_one_hot, allowable_set=[
                                                                     Chem.rdchem.BondStereo.STEREONONE,
                                                                     Chem.rdchem.BondStereo.STEREOANY,
                                                                     Chem.rdchem.BondStereo.STEREOZ,
                                                                     Chem.rdchem.BondStereo.STEREOE],
                                                                         encode_unknown=True)])})


def D3_info(a, b, c):
    '''
    To calculate edge 3D information.
    :param a:
    :param b:
    :param c:
    :return:
    '''
    # Angle
    ab = b - a
    ac = c - a
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)

    # Area
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_


# claculate the 3D info for each directed edge
def D3_info_cal(nodes_ls, g):
    if len(nodes_ls) > 2:
        Angles = []
        Areas = []
        Distances = []
        for node_id in nodes_ls[2:]:
            angle, area, distance = D3_info(g.ndata['pos'][nodes_ls[0]].numpy(), g.ndata['pos'][nodes_ls[1]].numpy(),
                                            g.ndata['pos'][node_id].numpy())
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [np.max(Angles) * 0.01, np.sum(Angles) * 0.01, np.mean(Angles) * 0.01, np.max(Areas), np.sum(Areas),
                np.mean(Areas),
                np.max(Distances) * 0.1, np.sum(Distances) * 0.1, np.mean(Distances) * 0.1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]

AtomFeaturizer = MyAtomFeaturizer()
BondFeaturizer = MyBondFeaturizer()

def get_SG_index(mol, chain, resi):
    '''
    To locate the SG atom of Cys in mol.
    :param mol:
    :param chain: chain ID
    :param resi: resi num
    :return:int: The node index of SG atom
    '''
    num_atoms = mol.GetNumAtoms()
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        resi_inform = atom.GetPDBResidueInfo()
        if resi_inform.GetChainId() == chain and resi_inform.GetResidueNumber() == resi and resi_inform.GetName().strip() == 'SG':
            if resi_inform.GetName().strip() == 'SG':
                index = i
                break
            else:
                print(resi_inform.GetName().strip())
    return index

def graphs_from_mol(dir, key, label, cys, graph_dic_path, path_marker='\\', EtaR=4.00, ShfR=0.5, Zeta=8.00, ShtZ=0):
    """
    node features: AtomFeaturizer, TorchANI
    edge feature: BondFeaturizer, D3_info_cal,
    :param dir: the path for the RDKit molecular
    :param key: the key for the sample
    :param label: the label for the sample
    :param cys: the position for cysteine
    :param dis_threshold: the distance threshold to determine the atom-pair interactions
    :param graph_dic_path: the path for storing the generated graph
    :param path_marker: '\\' for window and '/' for linux

    """

    add_self_loop = False
    if not os.path.exists(graph_dic_path + path_marker + key + '.bin'):
        try:
            with open(dir, 'rb') as f:
                mol = pickle.load(f)
            # pocket graph
            g1 = dgl.DGLGraph()
            # cysteine interaction graph
            g2 = dgl.DGLGraph()
            # add nodes
            num_atoms_m1 = mol.GetNumAtoms()
            g1.add_nodes(num_atoms_m1)
            g2.add_nodes(num_atoms_m1)
            if add_self_loop:
                nodes = g1.nodes()
                g1.add_edges(nodes, nodes)

            # add edges for g1
            num_bonds1 = mol.GetNumBonds()
            src1 = []
            dst1 = []
            for i in range(num_bonds1):
                bond1 = mol.GetBondWithIdx(i)
                u = bond1.GetBeginAtomIdx()
                v = bond1.GetEndAtomIdx()
                src1.append(u)
                dst1.append(v)
            src_ls1 = np.concatenate([src1, dst1])
            dst_ls1 = np.concatenate([dst1, src1])
            g1.add_edges(src_ls1, dst_ls1)

            # assign atom features
            # RDKit-based physicochemical feature
            g1.ndata['h'] = torch.zeros(num_atoms_m1, AtomFeaturizer.feat_size('h'), dtype=torch.float)  # init 'h'
            g1.ndata['h'] = AtomFeaturizer(mol)['h']

            # TorchANI-based 3D feature
            AtomicNums = []
            for i in range(num_atoms_m1):
                AtomicNums.append(mol.GetAtomWithIdx(i).GetAtomicNum())
            Corrds = mol.GetConformer().GetPositions()
            AtomicNums = torch.tensor(AtomicNums, dtype=torch.long)
            Corrds = torch.tensor(Corrds, dtype=torch.float64)
            AtomicNums = torch.unsqueeze(AtomicNums, dim=0)
            Corrds = torch.unsqueeze(Corrds, dim=0)
            res = converter((AtomicNums, Corrds))
            pbsf_computer = AEVComputer(Rcr=12.0, Rca=12.0, EtaR=torch.tensor([EtaR]), ShfR=torch.tensor([ShfR]),
                                        EtaA=torch.tensor([3.5]), Zeta=torch.tensor([Zeta]),
                                        ShfA=torch.tensor([0]), ShfZ=torch.tensor([ShtZ]), num_species=9)
            outputs = pbsf_computer((res.species, res.coordinates))
            if torch.any(torch.isnan(outputs.aevs[0].float())):
                print(key)
            g1.ndata['h'] = torch.cat([g1.ndata['h'], outputs.aevs[0].float()], dim=-1)

            # assign edge features
            # RDKit-based physicochemical feature
            g1.edata['e'] = torch.zeros(g1.number_of_edges(), BondFeaturizer.feat_size('e'), dtype=torch.float)  # init 'e'
            efeats1 = BondFeaturizer(mol)['e']  # 重复的边存在！
            g1.edata['e'] = torch.cat([efeats1[::2], efeats1[::2]])

            # 3D feature
            # init 'pos'
            g1.ndata['pos'] = torch.zeros([g1.number_of_nodes(), 3], dtype=torch.float)
            g1.ndata['pos'] = torch.tensor(mol.GetConformers()[0].GetPositions(), dtype=torch.float)
            # calculate the 3D info for g
            src_nodes, dst_nodes = g1.find_edges(range(g1.number_of_edges()))
            src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
            neighbors_ls = []
            for i, src_node in enumerate(src_nodes):
                tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
                neighbors = g1.predecessors(src_node).tolist()
                neighbors.remove(dst_nodes[i])
                tmp.extend(neighbors)
                neighbors_ls.append(tmp)
            D3_info_ls = list(map(partial(D3_info_cal, g=g1), neighbors_ls))
            D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
            g1.edata['e'] = torch.cat([g1.edata['e'], D3_info_th], dim=-1)
            g1.ndata.pop('pos')
            # detect the nan values in the D3_info_th
            if torch.any(torch.isnan(D3_info_th)):
                status = False
            else:
                status = True

            src2 = []
            dst2 = []

            edge_feature = []
            distance_matrix = Chem.Get3DDistanceMatrix(mol)
            index = get_SG_index(mol,cys[0],cys[1])

            for i in range(len(distance_matrix)):
                if distance_matrix[index,i] <= 7 and i != index and mol.GetBondBetweenAtoms(index, i) is None:
                    src2.append(index)
                    dst2.append(i)
                    edge_feature.append(distance_matrix[index,i])
            edge_feature = torch.tensor(edge_feature)
            g2.add_edges(src2, dst2)
            g2.edata.update({'h': edge_feature})


        except:
            g1 = None
            g2 = None
            index = None
            status = False
            traceback.print_exc()
            print(F"Error {key}")
        if status:
            with open(graph_dic_path + path_marker + key + '.bin', 'wb') as f:
                pickle.dump({'g1': g1, 'g2': g2, 'SG':[index],'key': key, 'label': label}, f)


def collate_fn(data_batch):

    graphs1, graphs2 , SGs, Ys, keys = map(list, zip(*data_batch))
    bg1 = dgl.batch(graphs1)
    bg2 = dgl.batch(graphs2)
    SGs = torch.tensor(SGs)
    Ys = torch.unsqueeze(torch.stack(Ys, dim=0), dim=-1)
    return bg1, bg2, SGs, Ys, keys

class GraphDatasetGenerate(object):

    """
    This class is used for generating graph objects.
    """

    def __init__(self, keys=None, labels=None, cyss = None, data_dirs=None, graph_ls_path=None, graph_dic_path=None, num_process=6,
                 path_marker='/', EtaR=4.00, ShfR=3.17, Zeta=8.00, ShtZ=3.14, split=None):
        """
        :param keys: the keys for the samples, list
        :param labels: the corresponding labels for the samples, list
        :param cyss: cys position, [(chain, resi_id),...]
        :param data_dirs: the corresponding data_dirs for the samples, list
        :param graph_ls_path: the cache path for the total graphs objects
        :param graph_dic_path: the cache path for the separate graphs objects (dic) for each sample
        :param num_process: the numer of process used to generate the graph objects
        :param add_3D: add the 3D geometric features to the edges of graphs
        :param path_marker: '\\' for windows and '/' for linux
        """
        self.origin_keys = keys
        self.origin_labels = labels
        self.origin_data_dirs = data_dirs
        self.cyss = cyss
        self.graph_ls_path = graph_ls_path
        self.graph_dic_path = graph_dic_path
        self.num_process = num_process
        self.path_marker = path_marker
        self.EtaR = EtaR
        self.ShfR = ShfR
        self.Zeta = Zeta
        self.ShtZ = ShtZ
        self.spl = split
        self._pre_process()


    def _pre_process(self):
        if os.path.exists(self.graph_ls_path+self.path_marker+'data.bin' ):
            print('Loading previously saved dgl graphs and corresponding data...')
            with open(self.graph_ls_path+self.path_marker+'data.bin', 'rb') as f:
                data = pickle.load(f)
            self.graphs1 = data['g1']
            self.graphs2 = data['g2']
            self.SGs = data['SGs']
            self.keys = data['keys']
            self.labels = data['labels']
        else:
            graph_dic_paths = repeat(self.graph_dic_path, len(self.origin_data_dirs))
            path_markers = repeat(self.path_marker, len(self.origin_data_dirs))
            EtaRs = repeat(self.EtaR, len(self.origin_data_dirs))
            ShfRs = repeat(self.ShfR, len(self.origin_data_dirs))
            Zetas = repeat(self.Zeta, len(self.origin_data_dirs))
            ShtZs = repeat(self.ShtZ, len(self.origin_data_dirs))
            print('Generating graphs...')

            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(graphs_from_mol,
                         zip(self.origin_data_dirs, self.origin_keys,  self.origin_labels, self.cyss,graph_dic_paths,
                             path_markers, EtaRs, ShfRs, Zetas, ShtZs))
            pool.close()
            pool.join()
            # collect the generated graph for each sample
            self.graphs1 = []
            self.graphs2 = []
            self.SGs = []
            self.labels = []
            self.keys = []

            for key in self.origin_keys:
                try:
                    with open(self.graph_dic_path + self.path_marker + key + '.bin', 'rb') as f:
                        graph_dic = pickle.load(f)
                        self.graphs1.append(graph_dic['g1'])
                        self.graphs2.append(graph_dic['g2'])
                        self.SGs.append(graph_dic['SG'])
                        self.labels.append(graph_dic['label'])
                        self.keys.append(graph_dic['key'])
                except:
                    continue
            with open(self.graph_ls_path + self.path_marker + 'data.bin', 'wb') as f:
                pickle.dump({'g1': self.graphs1,'g2':self.graphs2, 'SGs':self.SGs, 'keys': self.keys, 'labels': self.labels}, f)

    def __getitem__(self, indx):
        return self.graphs1[indx], self.graphs2[indx], self.SGs[indx], torch.tensor(self.labels[indx], dtype=torch.float), self.keys[indx]

    def __len__(self):
        return len(self.labels)