import torch
# import torch_geometric
from torch_geometric import data as torch_geometric_data
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch_cluster
import numpy as np
import math, random, json
from tqdm import tqdm

def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

class TripleDataGNN(torch_geometric_data.Data):
    def __init__(self, x_anchor, edge_index_anchor, edge_attr_anchor, seq_anchor, name_anchor, x_pos,
                 edge_index_pos, edge_attr_pos, seq_pos, name_pos, x_neg, edge_index_neg, edge_attr_neg,
                 seq_neg, name_neg, label, pos_sim):
        super().__init__()
        self.x_anchor, self.x_pos, self.x_neg = x_anchor, x_pos, x_neg
        self.edge_index_anchor, self.edge_index_pos, self.edge_index_neg = edge_index_anchor, edge_index_pos, \
                                                                           edge_index_neg
        self.edge_attr_anchor, self.edge_attr_pos, self.edge_attr_neg = edge_attr_anchor, edge_attr_pos, edge_attr_neg
        self.name_anchor, self.name_pos, self.name_neg = name_anchor, name_pos, name_neg
        self.seq_anchor, self.seq_pos, self.seq_neg = seq_anchor, seq_pos, seq_neg
        self.label = label
        self.pos_sim = pos_sim

    def __inc__(self, key, value, *args, **kwargs):
        features = ['edge_index', 'edge_attr', 'name', 'seq']
        anchor_features = [f + '_anchor' for f in features]
        pos_features = [f + '_pos' for f in features]
        neg_features = [f + '_neg' for f in features]
        if key in anchor_features:
            return self.x_anchor.size(0)
        elif key in pos_features:
            return self.x_pos.size(0)
        elif key in neg_features:
            return self.x_neg.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

class TripletMultilabelDataset(Dataset):
    def __init__(self, config, subset_pids=None, logger=None):
        super().__init__()
        self.config = config
        with open(config.data_file) as f:
            self.data_dict = json.load(f)
        if subset_pids is not None:
            self.data_dict = {k: v for k, v in self.data_dict.items() if k in subset_pids}
        self.logger = logger
        self.logger.info(f'Loaded {len(self.data_dict)} proteins from {config.data_file}')
        self.topk = config.topk
        self.num_rbf = config.num_rbf
        self.num_positional_embeddings = config.num_positional_embeddings
        self.graph_type = config.graph_type
        self.edge_type = config.edge_type
        self.edge_threshold = config.edge_threshold
        self.seq_emb = config.seq_emb
        self.device = 'cpu'
        self.scalar_only = config.scalar_only
        self.ground_truth_file = config.ground_truth_file
        self.n_classes = config.n_classes
                
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                              'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                              'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                              'N': 2, 'Y': 18, 'M': 12, 'X': 20, 'U': 21, 'B': 22,
                              'Z': 23, 'J': 24, 'O': 25}  # Modified: added 'X': 0
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}
        for i in list(self.data_dict.keys()):
            coords = self.data_dict[i]["coordinates"]
            self.data_dict[i]['coordinates'] = list(zip(coords['N'], coords['CA'], coords['C'], coords['O']))
            
        self.seq_id = list(self.data_dict.keys())
        self.id2label, self.label2id = self.parse_hierarchical_label(self.seq_id)
        self.data_len = len(self.seq_id)
        
        self.id2graphs = self.get_graphs(list(self.data_dict.keys()))
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        anchor_id = self.seq_id[index]
        anchor = self.id2graphs[anchor_id]
        anchor_label = random.choice(self.id2label[anchor_id])
        pos, neg, pos_label, neg_label, pos_sim = self.get_pair(anchor_id, anchor_label)
        t_anc = torch.tensor(anchor_label).view(1, -1)
        t_pos = torch.tensor(pos_label).view(1, -1)
        t_neg = torch.tensor(neg_label).view(1, -1)
        label = torch.cat([t_anc, t_pos, t_neg], dim=0).view(1, 3, -1)
        triplet = None
        triplet = TripleDataGNN(anchor.x, anchor.edge_index, anchor.edge_attr, anchor.seq,
                                anchor.name, pos.x, pos.edge_index, pos.edge_attr, pos.seq,
                                pos.name, neg.x, neg.edge_index, neg.edge_attr, neg.seq, neg.name,
                                label, pos_sim)
        triplet.num_nodes = len(anchor.x) + len(pos.x) + len(neg.x)
        
        return triplet

    def parse_hierarchical_label(self, pids):
        id2label, label2id = {}, {}
        with open(self.ground_truth_file) as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            pid, labels = line.strip().split()
            labels = labels.split(',')
            id2label[pid] = []
            for label in labels:
                h1, h2, h3, h4 = [int(i) if i != 'n1' else 0 for i in label.split('.')]
                if h1 not in label2id:
                    label2id[h1] = {}
                if h2 not in label2id[h1]:
                    label2id[h1][h2] = {}
                if h3 not in label2id[h1][h2]:
                    label2id[h1][h2][h3] = {}
                if h4 not in label2id[h1][h2][h3]:
                    label2id[h1][h2][h3][h4] = []
                id2label[pid].append([h1, h2, h3, h4])
                label2id[h1][h2][h3][h4].append(pid)
        
        return id2label, label2id
        
    def get_id2graphs(self):
        return self.id2graphs
    
    def get_rnd_label(self, labels, is_pos, existing_label=None):
        n_labels = len(labels)
        # print('1', n_labels)
        # if alternative labels are available, ensure difference between existing and new label
        if n_labels > 1 and existing_label is not None:
            labels = [label for label in labels if label != existing_label]
            n_labels -= 1
        # print('2', n_labels)
        rnd_idx = np.random.randint(0, n_labels)

        i = iter(labels)
        for _ in range(rnd_idx):
            next(i)
        rnd_label = next(i)
        # do not accidentaly draw the same label; instead draw again if necessary
        if existing_label is not None and rnd_label == existing_label:
            if is_pos:  # return the label itself for positives
                # Allow positives to have the same class as the anchor (relevant for rare classes)
                return existing_label
            else:
                # if there exists no negative sample for a certain combination of anchor and similarity-level
                return None
        return rnd_label
    
    def get_rnd_candidates(self, anchor_label, similarity_level, is_pos):

        # Get CATH classification of anchor sample
        class_n, arch, topo, homo = anchor_label

        if similarity_level == 0:  # No similarity - different class
            rnd_class = self.get_rnd_label(
                self.label2id.keys(), is_pos, class_n)
            rnd_arch = self.get_rnd_label(
                self.label2id[rnd_class].keys(), is_pos)
            rnd_topo = self.get_rnd_label(
                self.label2id[rnd_class][rnd_arch].keys(), is_pos)
            rnd_homo = self.get_rnd_label(
                self.label2id[rnd_class][rnd_arch][rnd_topo].keys(), is_pos)
            candidates = self.label2id[rnd_class][rnd_arch][rnd_topo][rnd_homo]

        elif similarity_level == 1:  # Same class but different architecture
            rnd_arch = self.get_rnd_label(
                self.label2id[class_n].keys(), is_pos, arch)
            rnd_topo = self.get_rnd_label(
                self.label2id[class_n][rnd_arch].keys(), is_pos)
            rnd_homo = self.get_rnd_label(
                self.label2id[class_n][rnd_arch][rnd_topo].keys(), is_pos)
            candidates = self.label2id[class_n][rnd_arch][rnd_topo][rnd_homo]

        elif similarity_level == 2:  # Same Class & Architecture but different topo
            rnd_topo = self.get_rnd_label(
                self.label2id[class_n][arch].keys(), is_pos, topo)
            rnd_homo = self.get_rnd_label(
                self.label2id[class_n][arch][rnd_topo].keys(), is_pos)
            candidates = self.label2id[class_n][arch][rnd_topo][rnd_homo]

        elif similarity_level == 3:  # Same Class & Architecture & topo but different homo
            rnd_homo = self.get_rnd_label(
                self.label2id[class_n][arch][topo].keys(), is_pos, homo)
            candidates = self.label2id[class_n][arch][topo][rnd_homo]

        # Highest similarity - different homology class (only relevent for positives)
        elif similarity_level == 4:
            candidates = self.label2id[class_n][arch][topo][homo]

        else:
            raise NotImplementedError

        return candidates

    def check_triplet(self, anchor_label, pos_label, neg_label, neg_hardness, pos_hardness):
        assert neg_hardness < pos_hardness, print("Neg sample more similar than pos sample")

        for i in range(0, pos_hardness):
            assert anchor_label[i] == pos_label[i], print(f'Pos label not overlapping:\nDiff: {pos_hardness}\nanchor: {anchor_label}\npos: {pos_label}\nneg: {neg_label}')
            
        for j in range(0, neg_hardness):
            assert anchor_label[j] == neg_label[j], print(f'Neg label not overlapping:\nDiff: {neg_hardness}\nanchor: {anchor_label}\npos: {pos_label}\nneg: {neg_label}')
            
        assert anchor_label[neg_hardness] != neg_label[neg_hardness], print("Neg label not different from anchor")
        
    def check_triplet_pid(self, anchor_id, pos_id, neg_id, neg_hardness, pos_hardness):
        def similarity(label1, label2):
            sim = 0
            n = len(label1)
            for i in range(n):
                if label1[i] == label2[i]:
                    sim += 1
                else:
                    break
            return sim
        valid = True
        anchor_labels = self.id2label[anchor_id]
        pos_labels = self.id2label[pos_id]
        neg_labels = self.id2label[neg_id]
        for anchor_label in anchor_labels:
            for pos_label in pos_labels:
                for neg_label in neg_labels:
                    pos_sim = similarity(anchor_label, pos_label)
                    neg_sim = similarity(anchor_label, neg_label)
                    if pos_sim <= neg_sim:
                        valid = False
                        break
        # assert valid, print(f'Invalid triplet:\nanchor: {anchor_id}\npos: {pos_id}\nneg: {neg_id}')
        
        return valid
    
    def get_pair(self, anchor_id, anchor_label, hardness_level=None, verbose=False):
        pos, neg = None, None
        pos_label, neg_label = None, None

        while pos is None or neg is None:
            neg_similarity = np.random.randint(self.n_classes)
            pos_similarity = neg_similarity + 1
            try:
                neg_candidates = self.get_rnd_candidates(
                    anchor_label, neg_similarity, is_pos=False)  # get set of negative candidates
                neg_id = random.choice(neg_candidates)  # randomly pick one of the neg. candidates
                # check whether the neg sample has another label that is positive with the anchor
                for possible_neg_id in self.id2label[neg_id]:
                    for possible_anchor_id in self.id2label[anchor_id]:
                        if possible_neg_id[:pos_similarity] == possible_anchor_id[:pos_similarity]:
                            continue
                neg_label = random.choice(self.id2label[neg_id])  # get label of randomly picked neg.
                neg = self.id2graphs[neg_id]  # get embedding of randomly picked neg.

                # repeat the same for the positive sample
                pos_candidates = self.get_rnd_candidates(
                    anchor_label, pos_similarity, is_pos=True)
                pos_id = random.choice(pos_candidates)

                # ensure that we do not randomly pick the same protein as anchor and positive
                if pos_id == anchor_id and len(pos_candidates) > 1:
                    while pos_id == anchor_id:  # re-draw from the pos. candidates if possible
                        pos_id = random.choice(pos_candidates)
                # if there is only one protein in a superfamily (anchor==positive without other candidates),
                # re-start picking process
                elif pos_id == anchor_id and len(pos_candidates) == 1:
                    continue

                pos = self.id2graphs[pos_id]
                pos_label = random.choice(self.id2label[pos_id])
                # if we successfully picked anchor, positive and negative candidates, do same sanity checks
                if pos_label is not None and neg_label is not None:
                    # try:
                    #     # self.check_triplet(anchor_label, pos_label,
                    #     #                 neg_label, neg_similarity, pos_similarity)
                    #     self.check_triplet_pid(anchor_id, pos_id, neg_id, neg_similarity, pos_similarity)
                    # except:
                    #     print(anchor_id, pos_id, neg_id, anchor_label, pos_label, neg_label)
                    if not self.check_triplet_pid(anchor_id, pos_id, neg_id, neg_similarity, pos_similarity):
                        continue
                else:  # if no triplet could be formed for a given combination of similarities/classes
                    continue

            except NotImplementedError:  # if you try to create triplets for a class level that is not yet
                # implemented in get_rnd_candidates
                print(anchor_id, anchor_label)
                raise NotImplementedError

            except KeyError:
                # if get_rnd_label returned None because no negative could be found
                # for a certain combination of anchor protein and similarity-lvl
                # re-start picking process
                continue

        if verbose:
            print('#### Example ####')
            print('Anc ({}) label: {}'.format(anchor_id, anchor_label))
            print('Pos ({}) label: {}'.format(pos_id, random.choice(self.id2label[pos_id])))
            print('Neg ({}) label: {}'.format(neg_id, random.choice(self.id2label[neg_id])))
            print('#### Example ####')

        return pos, neg, pos_label, neg_label, pos_similarity

    def get_graphs(self, pids):
        id2graph = dict()
        for i in tqdm(pids, desc='generating graphs'):
        # for i in cath_ids:
            id2graph[i] = self.featurize_as_graph(self.data_dict[i], i, self.graph_type)

        return id2graph
    
    def featurize_as_graph(self, protein, pid, graph_type='gat'):
        name = pid
        # print(name)
        with torch.no_grad():
            coords = torch.as_tensor(protein['coordinates'], dtype=torch.float32)
            # print(coords.size())
            # seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']], dtype=torch.long)
            seq = protein['seq']

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            if self.edge_type == "topk":
                edge_index = torch_cluster.knn_graph(X_ca, k=self.topk)
            elif self.edge_type == 'radius':
                edge_index = torch_cluster.radius_graph(X_ca, r=self.edge_threshold)
            else:
                raise NotImplementedError

            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                                                 (node_s, node_v, edge_s, edge_v))
        data = None
        if self.scalar_only:
            x = node_s
            edge_attr = edge_s
        else:
            x = torch.cat([X_ca, node_s, node_v.reshape((node_v.shape[0], -1))], dim=1)
            edge_attr = torch.cat([edge_s, edge_v.reshape((edge_v.shape[0], -1))], dim=1)
        # x = x / torch.linalg.norm(x)
        # edge_attr = edge_attr / torch.linalg.norm(edge_attr)
        data = torch_geometric_data.Data(x=x, edge_index=edge_index, seq=seq, name=name, edge_attr=edge_attr)
        
        return data

    def _check(self, X_ca, edge_index):
        num_nodes = len(X_ca)
        max_dist = torch.zeros(num_nodes)
        num_edges = len(edge_index[0])
        for i in range(num_edges):
            max_dist[edge_index[0][i]] = max(torch.dist(X_ca[edge_index[0][i]], X_ca[edge_index[1][i]]), max_dist[edge_index[0][i]])
            max_dist[edge_index[1][i]] = max(torch.dist(X_ca[edge_index[0][i]], X_ca[edge_index[1][i]]), max_dist[edge_index[1][i]])
        # print(max_dist.min().item())
        return max_dist.min().item() >= 8

    def _get_edges_threshold(self, X_ca, threshold=8.0):
        edge_index = [[], []]
        knn_edges = torch_cluster.knn_graph(X_ca, k=self.topk)
        num_knn_edges = len(knn_edges[0])
        for i in num_knn_edges:
            if torch.dist(X_ca[knn_edges[0][i]], X_ca[knn_edges[1][i]]) <= threshold:
                edge_index[0].append(knn_edges[0][i])
                edge_index[1].append(knn_edges[1][i])
        
        return torch.tensor(edge_index)

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design

        X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(self, edge_index,
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec


class ProteinGraphDataset(Dataset):
    def __init__(self, data_file, config, logger) -> None:
        super().__init__()
        self.config = config
        with open(data_file) as f:
            self.data_dict = json.load(f)
        self.logger = logger
        self.logger.info(f'Loaded {len(self.data_dict)} proteins from {data_file}')
        self.topk = config.topk
        self.num_rbf = config.num_rbf
        self.num_positional_embeddings = config.num_positional_embeddings
        self.graph_type = config.graph_type
        self.edge_type = config.edge_type
        self.edge_threshold = config.edge_threshold
        self.seq_emb = config.seq_emb
        self.device = 'cpu'
        self.scalar_only = config.scalar_only
        self.ground_truth_file = config.ground_truth_file
        self.n_classes = config.n_classes
                
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                              'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                              'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                              'N': 2, 'Y': 18, 'M': 12, 'X': 20, 'U': 21, 'B': 22,
                              'Z': 23, 'J': 24, 'O': 25}  # Modified: added 'X': 0
        self.num_to_letter = {v: k for k, v in self.letter_to_num.items()}
        for i in list(self.data_dict.keys()):
            coords = self.data_dict[i]["coordinates"]
            self.data_dict[i]['coordinates'] = list(zip(coords['N'], coords['CA'], coords['C'], coords['O']))
            
        self.seq_id = list(self.data_dict.keys())
        self.data_len = len(self.seq_id)
        
        self.id2graphs = self.get_graphs(list(self.data_dict.keys()))
        
    def __getitem__(self, index):
        return self.id2graphs[self.seq_id[index]]
    
    def __len__(self):
        return self.data_len

    def get_id2label(self):
        id2label = {}
        for pid, graph in self.id2graphs.items():
            id2label[pid] = graph.label
        return id2label
    
    def get_graphs(self, pids):
        id2graph = dict()
        for i in tqdm(pids, desc='generating graphs'):
        # for i in cath_ids:
            id2graph[i] = self.featurize_as_graph(self.data_dict[i], i, self.graph_type)

        return id2graph
    
    def featurize_as_graph(self, protein, pid, graph_type='gat'):
        name = pid
        # print(name)
        with torch.no_grad():
            coords = torch.as_tensor(protein['coordinates'], dtype=torch.float32)
            # print(coords.size())
            # seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']], dtype=torch.long)
            seq = protein['seq']

            mask = torch.isfinite(coords.sum(dim=(1, 2)))
            coords[~mask] = np.inf

            X_ca = coords[:, 1]
            if self.edge_type == "topk":
                edge_index = torch_cluster.knn_graph(X_ca, k=self.topk)
            elif self.edge_type == 'radius':
                edge_index = torch_cluster.radius_graph(X_ca, r=self.edge_threshold)
            else:
                raise NotImplementedError

            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                                                 (node_s, node_v, edge_s, edge_v))
        data = None
        if self.scalar_only:
            x = node_s
            edge_attr = edge_s
        else:
            x = torch.cat([X_ca, node_s, node_v.reshape((node_v.shape[0], -1))], dim=1)
            edge_attr = torch.cat([edge_s, edge_v.reshape((edge_v.shape[0], -1))], dim=1)
        # x = x / torch.linalg.norm(x)
        # edge_attr = edge_attr / torch.linalg.norm(edge_attr)
        data = torch_geometric_data.Data(x=x, edge_index=edge_index, seq=seq, name=name, edge_attr=edge_attr, label=protein['ec'])
        
        return data

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design

        X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(self, edge_index,
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec
    
if __name__ == '__main__':
    import easydict
    import logging
    config = easydict.EasyDict({'data_file': '../data/cath_splits/train_0.json', 'topk': 30, 'num_rbf': 16, 'num_positional_embeddings': 16, 'graph_type': 'gat', 'edge_type': 'topk', 'edge_threshold': 8.0, 'seq_emb': 'esm', 'device': 'cpu', 'scalar_only': False, 'ground_truth_file': '../data/ec-data/pdb2ec_multilabel.txt', 'n_classes': 4})
    # load logger
    logger = logging.getLogger(__name__)
    
    dataset = TripletMultilabelDataset(config, logger=logger)
    # dataset = ProteinGraphDataset(config.data_file, config, logger)
    
    for i in range(len(dataset)):
        print(dataset[i])
        print(dataset[i].label)
        input()