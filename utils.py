
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, isequal, basis, cz_gate, cphase
import random
import networkx as nx
from scipy.stats import sem
import numpy as np
from itertools import product
import multiprocessing
import tqdm
from scipy.special import erfinv


def same_state(state1,state2):
    return isequal(state1,state2)
def is_projector(operator, tolerance=None):
    """
    Check if the given operator is a projector.
    """
    return isequal(operator*operator,operator,tol=tolerance)

def string_representation_of_projector(N, qubit_dict):
    str = ''
    for i in range(N):
        if i in qubit_dict:
            pauli_operator, measurement_result = qubit_dict[i]

            if   pauli_operator in ('x', 'X'):
                pauli_op = 'x'
            elif pauli_operator in ('y', 'Y'):
                pauli_op = 'y'
            elif pauli_operator in ('z', 'Z'):
                pauli_op = 'z'
            else:
                raise ValueError("Invalid Pauli operator {pauli_operator} inside {qubit_dict} of {N} qubits. Choose 'x', 'y', or 'z'. ")

            if measurement_result not in ('+', '-', '0', '1', 0, 1, -1):
                raise ValueError("Invalid measurement result. Choose '+', '-', '0', or '1'.")

            if measurement_result in ('+', '1', 1):
                sign = '+'
            elif measurement_result in ('-', '0', 0, -1):
                sign = '-'

            str += pauli_op
            str += sign
        else:
            str += 'id'

    return str

def string_representation_of_pauli(N, qubit_dict):
    str = ''
    for i in range(N):
        if i in qubit_dict:
            pauli_operator, measurement_result = qubit_dict[i]

            if   pauli_operator in ('x', 'X'):
                pauli_op = 'x'
            elif pauli_operator in ('y', 'Y'):
                pauli_op = 'y'
            elif pauli_operator in ('z', 'Z'):
                pauli_op = 'z'
            else:
                raise ValueError("Invalid Pauli operator {pauli_operator} inside {qubit_dict} of {N} qubits. Choose 'x', 'y', or 'z'. ")

            str += pauli_op
        else:
            str += '1'

    return str

def prepare_projector(N, qubit_dict):
    projectors = []
    if len(qubit_dict)==1:
        [i]=qubit_dict.keys()
        pauli_operator, measurement_result = qubit_dict[i]
        if i!=0:
            ns=[2 for _ in range(i)]
            projector=qeye(ns)
            projectors.append(projector)
        if   pauli_operator in ('x', 'X'):
            pauli_op = sigmax()
        elif pauli_operator in ('y', 'Y'):
            pauli_op = sigmay()
        elif pauli_operator in ('z', 'Z'):
            pauli_op = sigmaz()
        else:
            raise ValueError("Invalid Pauli operator {pauli_operator} inside {qubit_dict} of {N} qubits. Choose 'x', 'y', or 'z'. ")
        if measurement_result not in ('+', '-', '0', '1', 0, 1, -1):
            raise ValueError("Invalid measurement result. Choose '+', '-', '0', or '1'.")

        if measurement_result in ('+', '1', 1):
            sign = 1
        elif measurement_result in ('-', '0', 0, -1):
            sign = -1

        projector = (qeye(2) + sign * pauli_op) / 2
        projectors.append(projector)
        
        if i!=N-1:
            ns=[2 for _ in range(i+1,N)]
            projector=qeye(ns)
            projectors.append(projector)
    else:
        for i in range(N):
            if i in qubit_dict:
                pauli_operator, measurement_result = qubit_dict[i]

                if   pauli_operator in ('x', 'X'):
                    pauli_op = sigmax()
                elif pauli_operator in ('y', 'Y'):
                    pauli_op = sigmay()
                elif pauli_operator in ('z', 'Z'):
                    pauli_op = sigmaz()
                else:
                    raise ValueError("Invalid Pauli operator {pauli_operator} inside {qubit_dict} of {N} qubits. Choose 'x', 'y', or 'z'. ")

                if measurement_result not in ('+', '-', '0', '1', 0, 1, -1):
                    raise ValueError("Invalid measurement result. Choose '+', '-', '0', or '1'.")

                if measurement_result in ('+', '1', 1):
                    sign = 1
                elif measurement_result in ('-', '0', 0, -1):
                    sign = -1

                projector = (qeye(2) + sign * pauli_op) / 2
            else:
                projector = qeye(2)

            projectors.append(projector)

    return tensor(projectors)

def prepare_pauli_operator(N, qubit_dict):
    projectors = []
    
    for i in range(N):
        if i in qubit_dict:
            pauli_operator = qubit_dict[i]

            if   pauli_operator in ('x', 'X'):
                pauli_op = sigmax()
            elif pauli_operator in ('y', 'Y'):
                pauli_op = sigmay()
            elif pauli_operator in ('z', 'Z'):
                pauli_op = sigmaz()
            else:
                raise ValueError("Invalid Pauli operator {pauli_operator} inside {qubit_dict} of {N} qubits. Choose 'x', 'y', or 'z'. ")



            projector = (pauli_op) 
        else:
            projector = qeye(2)

        projectors.append(projector)

    return tensor(projectors)


def prepare_graph_state(graph,N):
    """
    Prepare a graph state given a graph.
    """
    initial_state = prepare_plus_state(N)
    graph_state = apply_controlled_z_gates(initial_state, graph, N)
    return graph_state

def prepare_plus_state(N):
    """
    Prepare the initial state (a tensor product of N qubit |+> states).
    """
    plus_state = (basis(2, 0) + basis(2, 1)).unit()
    initial_state = tensor([plus_state] * N)

    return initial_state

def apply_controlled_z_gates(state, graph, N):
    """
    Apply controlled-Z gates to the given state based on the graph edges.
    """
    for edge in graph.edges:
        q1, q2 = edge
        if graph.edges[edge]:
            weight=graph.edges[edge]['weight']
            cz_gate_obj = cphase(weight, N, q1, q2)
        else:
            cz_gate_obj = cz_gate(N, q1, q2)
        state = cz_gate_obj * state

    return state

def prepare_intertwined_grid_with_endpoints(length):
    """
    a `crazy graph` prepared atop 2xlength grid. endpoints are 0 and 1. each layer also connected!
    measure with Y to get Bell.
    @param length: no vertices between endpoints
    @return:
    """
    G = nx.Graph()

    G.add_node('in')
    G.add_node('out')

    for i in range(length):
        G.add_node((i, 0))
        G.add_node((i, 1))
        G.add_edge((i, 0), (i, 1))
    for i in range(length - 1):
        for a, b in product([0, 1], [0, 1]):
            G.add_edge((i, a), (i + 1, b))
    for a in [0, 1]:
        G.add_edge('in', (0, a))
        G.add_edge('out', (length - 1, a))
    assert list(G.nodes)[:2]==['in','out']
    H = nx.from_numpy_array(nx.adjacency_matrix(G).todense())
    return H

def prepare_intertwined_twist_with_endpoints(length):
    """
    a `crazy graph` prepared atop 2xlength grid. endpoints are 0 and 1. each layer disconnected!
    measure with Xs to get Bell.
    @param length: no vertices between endpoints
    @return:
    """
    G = nx.Graph()

    G.add_node('in')
    G.add_node('out')

    for i in range(length):
        G.add_node((i, 0))
        G.add_node((i, 1))
    for i in range(length - 1):
        for a, b in product([0, 1], [0, 1]):
            G.add_edge((i, a), (i + 1, b))
    for a in [0, 1]:
        G.add_edge('in', (0, a))
        G.add_edge('out', (length - 1, a))
    assert list(G.nodes)[:2]==['in','out']
    H = nx.from_numpy_array(nx.adjacency_matrix(G).todense())
    return H

def prepare_path_cross(length,nghz):
    """
    a `crazy graph` prepared atop 2xlength grid. endpoints are 0 and 1. each layer disconnected!
    measure with Xs to get Bell.
    @param length: no vertices between endpoints
    @return:
    """
    G = nx.Graph()
    for k in range(nghz):
        G.add_node('term'+str(k))

    for k in range(nghz):
        for i in range(length-1):
            G.add_node((k,i, 0))
            #G.add_node((k,i, 1))
    G.add_node((length-1,0))
    #G.add_node((length-1,1))

    for k in range(nghz):
        for i in range(length - 2):
            #for a, b in product([0, 1], [0,]):
            G.add_edge((k,i, 0), (k,i + 1, 0))
        #for a, b in product([0, 1], [0, 1]):
        G.add_edge((k,length-2, 0), (length-1, 0))
    for k in range(nghz):
        G.add_edge('term'+str(k), (k,0, 0))
    #assert list(G.nodes)[:2]==['in','out']
    H = nx.from_numpy_array(nx.adjacency_matrix(G).todense())
    return H

def prepare_crazy_cross(length,nghz):
    """
    a `crazy graph` prepared atop 2xlength grid. endpoints are 0 and 1. each layer disconnected!
    measure with Xs to get Bell.
    @param length: no vertices between endpoints
    @return:
    """
    G = nx.Graph()
    for k in range(nghz):
        G.add_node('term'+str(k))

    for k in range(nghz):
        for i in range(length-1):
            G.add_node((k,i, 0))
            G.add_node((k,i, 1))
    G.add_node((length-1,0))
    G.add_node((length-1,1))

    for k in range(nghz):
        for i in range(length - 2):
            for a, b in product([0, 1], [0, 1]):
                G.add_edge((k,i, a), (k,i + 1, b))
        for a, b in product([0, 1], [0, 1]):
                G.add_edge((k,length-2, a), (length-1, b))
    for k in range(nghz):
        for a in [0, 1]:
            G.add_edge('term'+str(k), (k,0, a))
    #assert list(G.nodes)[:2]==['in','out']
    H = nx.from_numpy_array(nx.adjacency_matrix(G).todense())
    return H

def select_random_elements(input_list, probability):
    new_list = []
    for item in input_list:
        if random.random() < probability:
            new_list.append(item)
    return new_list


def mean_and_error(dataset):
    return np.mean(dataset),sem(dataset)

def parallel_for(func, iterable, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm.tqdm(pool.imap(func, iterable), total=len(iterable)))
    
    return results

def pseudonormal_regular_unit_var_pts(npts):
    es=[]
    for p in np.linspace(0,1,npts+1,endpoint=False)[1:]:
        es.append(erfinv(2*p-1))
    es=np.array(es/np.std(es))
    return es