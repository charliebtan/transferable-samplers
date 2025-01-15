import networkx as nx
import networkx.algorithms.isomorphism as iso
import scipy
from networkx import isomorphism

from src.models.components.utils import create_adjacency_list


def align_topology(sample, reference, atom_types):
    sample = sample.reshape(-1, 3)
    all_dists = scipy.spatial.distance.cdist(sample, sample)
    # sns.clustermap(all_dists)
    adj_list_computed = create_adjacency_list(all_dists, atom_types)
    G_reference = nx.Graph(reference)
    G_sample = nx.Graph(adj_list_computed)
    # not same number of nodes
    if len(G_sample.nodes) != len(G_reference.nodes):
        return sample, False
    for i, atom_type in enumerate(atom_types):
        G_reference.nodes[i]["type"] = atom_type
        G_sample.nodes[i]["type"] = atom_type

    nm = iso.categorical_node_match("type", -1)
    GM = isomorphism.GraphMatcher(G_reference, G_sample, node_match=nm)
    is_isomorphic = GM.is_isomorphic()
    initial_idx = list(GM.mapping.keys())
    final_idx = list(GM.mapping.values())
    sample[initial_idx] = sample[final_idx]
    # print(is_isomorphic)
    return sample, is_isomorphic
