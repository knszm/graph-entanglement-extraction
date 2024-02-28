import networkx as nx
# from qutip import tensor, basis, qeye, Qobj, sigmax, sigmay, sigmaz, ket2dm
# from qutip.qip.operations import cz_gate
# from math import sqrt
import logging
from itertools import combinations,product
from utils import prepare_graph_state  # is_projector, prepare_projector
from quantumstates import PureSpinState, SpinStateEnsemble
# from qutip import tensor, basis, qeye, Qobj, sigmax, sigmay, sigmaz, ket2dm
# from qutip.qip.operations import cz_gate
# from math import sqrt
import logging
from itertools import combinations
import numpy as np

import networkx as nx

from quantumstates import PureSpinState, SpinStateEnsemble
from utils import prepare_graph_state, select_random_elements, pseudonormal_regular_unit_var_pts  # is_projector, prepare_projector, 
from copy import deepcopy

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class WeightedGraphState(PureSpinState):
    warning_issued_pure = False
    warning_issued_reduced = False

    def __init__(self, N, edge_list, weight_list, probability=1):
        """
        Initialize a WeightedGraphState with N qubits, an edge list, weights, and an optional probability.
        """

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(N))
        for edge, weight in zip(edge_list,weight_list):
            v1,v2=edge
            self.graph.add_edge(v1,v2,weight=weight)
        graphstate = prepare_graph_state(self.graph, N)

        super().__init__(N, graphstate, probability)


class WeightedGraphStateEnsemble(SpinStateEnsemble):

    def __init__(self, N, ensemble_data, probability_function=None):
        """
        Initialize a GraphStateEnsemble with N qubits and ensemble_data, a list of dictionaries.
        Each dictionary must contain the key 'edge_list'
        """

        self.graph_states = []

        for ensemble_element in ensemble_data:

            graph_state_list_element = deepcopy(ensemble_element)

            edge_list = ensemble_element['edge_list']
            weight_list = ensemble_element['weight_list']
            local_unitaries = ensemble_element.get('local_unitaries', None)

            graph_state = WeightedGraphState(N, edge_list, weight_list)

            if local_unitaries:
                graph_state.apply_local_unitary(local_unitaries)

            graph_state_list_element['state'] = graph_state

            self.graph_states.append(graph_state_list_element)
        super().__init__(N, self.graph_states, probability_function)

def ensemble_with_sqrtcz(N, edge_list, edge_failure_probability):
    
    ensemble_data = []
    

    probability = 1-edge_failure_probability*len(edge_list)

    ensemble_data.append({'edge_list': edge_list, 'probability': probability, 'weight_list': [np.pi for _ in edge_list]})


    for removed_edge in edge_list:
        remaining_edges=[edge for edge in edge_list if edge!=removed_edge]
        probability=edge_failure_probability
        ensemble_data.append({'edge_list': remaining_edges, 'probability': probability, 'weight_list': [np.pi for _ in remaining_edges]})

    for idx1,changed_edge1 in enumerate(edge_list):
        for idx2,changed_edge2 in enumerate(edge_list):
            if idx2>idx1:
                for s1,s2 in product([-1,1],[-1,1]):
                    edge_weights=[np.pi for _ in edge_list]
                    edge_weights[idx1]=s1*np.pi/2
                    edge_weights[idx2]=s2*np.pi/2
                    
                    probability=edge_failure_probability*s1*s2
                    ensemble_data.append({'edge_list': edge_list, 'probability': probability, 'weight_list': edge_weights})

    return WeightedGraphStateEnsemble(N, ensemble_data)


def n_weighted_graph_states_ensemble(N,edge_list,edge_failure_probability,no_generated_states):
    
    probability=1/no_generated_states
    ensemble_data = []
    epsilon = 2 * np.arcsin(np.sqrt(edge_failure_probability))
    phases=pseudonormal_regular_unit_var_pts(no_generated_states)*epsilon
    ordering=np.argsort(-np.abs(phases))
    #print(ordering,phases)
    phases=phases[ordering]
    #print(phases)
    for phase in phases:
        ensemble_data.append({'edge_list': edge_list, 'probability': probability, 'weight_list': [np.pi+phase for _ in edge_list]})
    return WeightedGraphStateEnsemble(N, ensemble_data)

def two_weighted_graph_states_ensemble(N, edge_list, edge_failure_probability, ignore_check=False):
    if not ignore_check:
        assert 10*edge_failure_probability<1/len(edge_list)
    epsilon = 2 * np.arcsin(np.sqrt(edge_failure_probability))
    weights= [[np.pi+ sign*epsilon for _ in edge_list] for sign in [-1,1]]
    ensemble_data = []
    

    probability = 1/2

    ensemble_data.append({'edge_list': edge_list, 'probability': probability, 'weight_list': [np.pi+epsilon for _ in edge_list]})
    ensemble_data.append({'edge_list': edge_list, 'probability': probability, 'weight_list': [np.pi-epsilon for _ in edge_list]})

    return WeightedGraphStateEnsemble(N, ensemble_data)