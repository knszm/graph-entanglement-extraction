import networkx as nx

import logging
from itertools import combinations
from utils import prepare_graph_state  
from quantumstates import PureSpinState, SpinStateEnsemble

import logging
from itertools import combinations

import networkx as nx

from quantumstates import PureSpinState, SpinStateEnsemble
from utils import prepare_graph_state, select_random_elements  
from copy import deepcopy

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class GraphState(PureSpinState):
    warning_issued_pure = False
    warning_issued_reduced = False

    def __init__(self, N, edge_list, probability=1):
        """
        Initialize a GraphState with N qubits, an edge list, and an optional probability.
        """

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(N))
        self.graph.add_edges_from(edge_list)
        graphstate = prepare_graph_state(self.graph, N)

        super().__init__(N, graphstate, probability)


class GraphStateEnsemble(SpinStateEnsemble):

    def __init__(self, N, ensemble_data, probability_function=None):
        """
        Initialize a GraphStateEnsemble with N qubits and ensemble_data, a list of dictionaries.
        Each dictionary must contain the key 'edge_list'
        """

        self.graph_states = []

        for ensemble_element in ensemble_data:

            graph_state_list_element = deepcopy(ensemble_element)

            edge_list = ensemble_element['edge_list']
            local_unitaries = ensemble_element.get('local_unitaries', None)

            graph_state = GraphState(N, edge_list)

            if local_unitaries:
                # Assuming apply_local_unitary is implemented in the GraphState class
                graph_state.apply_local_unitary(local_unitaries)

            graph_state_list_element['state'] = graph_state

            self.graph_states.append(graph_state_list_element)
        super().__init__(N, self.graph_states, probability_function)







def subgraph_and_z_monte_carlo_ensemble(N, edge_list, edge_failure_probability, z_flip_probability, no_generated_graphs, add_full=False):
    """
    Generates graph states with some edges missing and qubits Z-flipped.
    Basically: start with every edge, remove some of them, flip some qubits.
    """
    ensemble_data = []
    total_probability = 0
    for _ in range(no_generated_graphs):
        remaining_edges  = select_random_elements(edge_list,1-edge_failure_probability)
        flipped_vertices = select_random_elements(list(range(N)),z_flip_probability)
        unitary_dict = {i: 'z' for i in flipped_vertices}

        probability = 1
        total_probability += probability

        ensemble_data.append({'edge_list': remaining_edges, 'probability': probability, 'local_unitaries': unitary_dict})
        # Log the generated edge subset and its probability
        logger.info(f"Generated edge subset: {remaining_edges}, Probability: {probability}")

    if add_full:
        ensemble_data.append({'edge_list': edge_list, 'probability': 1e-9})
    for substate in ensemble_data: #renormalization to unit prob.
        substate['probability'] /= total_probability

    return GraphStateEnsemble(N, ensemble_data)

def subgraph_and_z_one_error_ensemble(N, edge_list, edge_failure_probability, z_flip_probability):
    """
    Generates an ensemble of graph states: the full one, and one pure state for each
     - missing edge,
     - qubit Z-flipped.
    Approximation valid for low edge failure/z flip probabilities. The one-edge-missing and one-flip states are then the dominant ones.
    """
    ensemble_data = []
    total_probability = 0


    if edge_failure_probability>0:
        for rem_edge in edge_list:
            missing_edges=[rem_edge]
            remaining_edges  = [edge for edge in edge_list if edge not in missing_edges]#select_random_elements(edge_list,1-edge_failure_probability)

            probability = edge_failure_probability
            total_probability += probability

            ensemble_data.append({'edge_list': remaining_edges, 'probability': probability})
            # Log the generated edge subset and its probability
            logger.info(f"Generated edge subset: {remaining_edges}, Probability: {probability}")
    if z_flip_probability>0:
        for flipped_vertex in range(N):

            flipped_vertices = [flipped_vertex] #select_random_elements(list(range(N)),z_flip_probability)
            unitary_dict = {i: 'z' for i in flipped_vertices}

            probability = z_flip_probability
            total_probability += probability

            ensemble_data.append({'edge_list': edge_list, 'probability': probability, 'local_unitaries': unitary_dict})
            # Log the generated edge subset and its probability
            logger.info(f"Generated z flip: {flipped_vertex}, Probability: {probability}")

    # the entire unaffected graph:
    ensemble_data.append({'edge_list': edge_list, 'probability': 1 - total_probability})

    return GraphStateEnsemble(N, ensemble_data)