import dgl.function as fn
import torch
torch.set_default_tensor_type('torch.FloatTensor')
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.glob import WeightAndSum
import dgl




class AttentiveGRU1(nn.Module):
    """Update node features with attention and GRU.

    This will be used for incorporating the information of edge features
    into node features for message passing.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, edge_feats, node_feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Previous edge features.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class AttentiveGRU2(nn.Module):
    """Update node features with attention and GRU.

    This will be used in GCN layers for updating node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_hidden_size : int
        Size for the intermediate edge (bond) representations.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        edge_logits : float32 tensor of shape (E, 1)
            The edge logits based on which softmax will be performed for weighting
            edges within 1-hop neighborhoods. E represents the number of edges.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Previous node features. V represents the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, node_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))

class GetContext(nn.Module):
    """Generate context for each node by message passing at the beginning.

    This layer incorporates the information of edge features into node
    representations so that message passing needs to be only performed over
    node representations.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge (bond) features.
    graph_feat_size : int
        Size of the learned graph representation (molecular fingerprint).
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def apply_edges1(self, edges):
        """Edge feature update.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges

        Returns
        -------
        dict
            Mapping ``'he1'`` to updated edge features.
        """
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        """Edge feature update.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges

        Returns
        -------
        dict
            Mapping ``'he2'`` to updated edge features.
        """
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        """Incorporate edge features and update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])

class GCNLayer(nn.Module):
    """GCNLayer for updating node features.

    This layer performs message passing over node representations and update them.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    graph_feat_size : int
        Size for the graph representations to be computed.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GCNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.bn_layer = nn.BatchNorm1d(graph_feat_size)

    def apply_edges(self, edges):
        """Edge feature generation.

        Generate edge features by concatenating the features of the destination
        and source nodes.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges.

        Returns
        -------
        dict
            Mapping ``'he'`` to the generated edge features.
        """
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        """Perform message passing and update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.

        Returns
        -------
        float32 tensor of shape (V, graph_feat_size)
            Updated node features.
        """
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.bn_layer(self.attentive_gru(g, logits, node_feats))

class PocketGNNLayer(nn.Module):

    """
    The framework of this layer was inspired by AttentiveFP:

    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    This layer performs message passing on pocket graph and returns the final state of atoms.

    Parameters
    ----------
    node_feat_size : int
        Size for the input node features.
    edge_feat_size : int
        Size for the input edge features.
    num_layers : int
        Number of GCN layers. Default to 2.
    graph_feat_size : int
        Size for the graph representations to be computed. Default to 200.
    dropout : float
        The probability for performing dropout. Default to 0.
    """

    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(PocketGNNLayer, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gcn_layers = nn.ModuleList()
        self.sum_node_feats = 0
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNLayer(graph_feat_size, graph_feat_size, dropout))

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        node_feats : float32 tensor of shape (V, graph_feat_size)
            Updated node representations.
        """
        node_feats = self.init_context(g, node_feats, edge_feats)
        self.sum_node_feats = node_feats
        for gcn in self.gcn_layers:
            node_feats = gcn(g, node_feats)
            # Summarize the outputs from each GCN layer to prevent the over-smooth of the graph.
            self.sum_node_feats = self.sum_node_feats + node_feats
        return self.sum_node_feats

class CysInteractLayer(nn.Module):

    """
    This layer encodes non-covalent interactions between cysteine and other pocket atoms.

    Parameters
    ----------
    in_dim : int
        Size for the node features comes from PocketGNNLayer + Size for the edge features in Cys-interaction graph.
    out_dim : int
        Size for the edge feature which encoded with the interaction information.
    dropout : float
         The probability for performing dropout. Default to 0.
    """
    def __init__(self, in_dim, out_dim, dropout=0.):  # in_dim = graph module1 output dim + 1
        super(CysInteractLayer, self).__init__()

        # the MPL for update the edge state
        self.mpl = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU())
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def edge_update(self, edges):

        """
        Represent the interaction information with edge feature 'h', which comes from node states and initial edge features.

        Parameters
        ----------
        edges : EdgeBatch
            Container for a batch of edges

        Returns
        -------
        dict
            Mapping 'h' to updated edge features.

        """

        return {'h': self.mpl(torch.cat([edges.data['h'], edges.data['m']], dim=1))}

    def forward(self, bg, node_feats, edge_feats):
        """Performs non-covalent interactions encoding.

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Initial edge features. E for the number of edges.

        Returns
        -------
        edge_feats : float32 tensor of shape (V, edge_feat_size)
            Updated edge feature which encoded with the interaction information.
        """
        bg.ndata['h'] = node_feats
        bg.edata['h'] = edge_feats
        with bg.local_scope():
            bg.apply_edges(dgl.function.u_add_v('h', 'h', 'm'))
            bg.apply_edges(self.edge_update)
            new_feature = bg.edata.pop('h')
            return self.bn_layer(self.dropout(new_feature))

class CysReadout(nn.Module):
    """
        This class aggregate the interaction information stored in edge features.

        Parameters
        ----------
        in_dim : int
            Size for the edge features which comes from CysInteractLayer.
    """
    def __init__(self, in_dim):
        super(CysReadout, self).__init__()
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Tanh()
        )

    def forward(self, g, edge_feats):
        '''
        Performs non-covalent interactions aggregation.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Updated edge features which encoded with the interaction information. E for the number of edges.

        Returns
        -------
        agg_interaction : float32 tensor of shape (edge_feat_size)
            Aggregated edge feature which represent the 'reactivity' of cysteine.
        '''
        with g.local_scope():
            g.edata['h'] = edge_feats
            g.edata['w'] = self.atom_weighting(g.edata['h'])
            agg_interaction = dgl.sum_edges(g, 'h', 'w')
        return agg_interaction

class Readout(nn.Module):

    """
        This class combines the two representations from PocketGNNLayer and CysInteractLayer respectively.

        Parameters
        ----------
        in_feats : int
            Size for the two representations.
    """
    def __init__(self, in_feats):
        super(Readout, self).__init__()
        self.in_feats = in_feats
        self.read_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Tanh()
        )

    def forward(self, cys_readout, pocket_read_out):
        '''

        Combine the representations.

        Parameters
        ----------

        cys_readout : float32 tensor of shape (cys_readout_size)
            Cysteine representation from CysInteractLayer.

        pocket_read_out : float32 tensor of shape (pocket_readout_size)
            Pocket representation from PocketGNNLayer.

        Returns
        -------
        readous_combined : float32 tensor of shape (readout_size)
            Combined representation for cysteine ligandability.
        '''

        weight1 = self.read_weighting(cys_readout)
        weight2 = self.read_weighting(pocket_read_out)
        readous_combined = (weight1 * cys_readout) + (weight2 * pocket_read_out)
        return readous_combined


class FC(nn.Module):
    """

    MLP for classification.

    Parameters
    ----------
    d_graph_layer : int
        Size for the readout of two graphs.
    d_FC_layer : int
        Size of MLP hidden layer.
    n_FC_layer : int
        Number of hidden layer.
    dropout : float
        The probability for performing dropout.

    """
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, 1))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        return torch.sigmoid(h)


class DeepCoSIPredictor(nn.Module):
    """
        The predictor of DeepCoSI.

        Parameters
        ----------
        node_feat_size : int
            Size for the input node features.
        edge_feat_size : int
            Size for the input edge features.
        num_layers : int
            Number of GCN layers.
        graph_feat_size : int
            Size for the graph representations to be computed.
        d_FC_layer : int
            Size of MLP hidden layer.
        n_FC_layer : int
            Number of hidden layer.
        dropout : float
            The probability for performing dropout.

    """
    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, d_FC_layer, n_FC_layer, dropout):
        super(DeepCoSIPredictor, self).__init__()
        # pocket
        self.pocket_layer = PocketGNNLayer(node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout)
        # pocket readout
        self.pocket_readout = WeightAndSum(graph_feat_size)

        # cys
        self.cys_layer = CysInteractLayer(graph_feat_size + 1, graph_feat_size, dropout)
        # cys readout
        self.cys_readout = CysReadout(graph_feat_size)

        # combine readout
        self.readout = Readout(graph_feat_size)

        # MLP predictor
        self.FC = FC(graph_feat_size, d_FC_layer, n_FC_layer, dropout)

    def forward(self, bg1, bg2):
        '''

        Identify ligandable cysteine with DeepCoSI.

        Parameters
        ----------

        bg1 : DGLGraph
            DGLGraph for a batch of pocket graphs.
        bg2 : DGLGraph
            DGLGraph for a batch of Cys-interaction graphs.

        Returns
        -------
        float32 tensor of shape (G, 1)
            Prediction for the graphs in the batch. G for the number of graphs.
        '''
        atom_feats = bg1.ndata.pop('h')
        bond_feats = bg1.edata.pop('e')
        atom_feats_updated = self.pocket_layer(bg1, atom_feats, bond_feats)
        pocket_readouts = self.pocket_readout(bg1, atom_feats_updated)
        interactions = self.cys_layer(bg2,atom_feats_updated,bg2.edata.pop('h').view(-1,1).float())
        cys_readout = self.cys_readout(bg2,interactions)
        readout = self.readout(cys_readout,pocket_readouts)
        return self.FC(readout)