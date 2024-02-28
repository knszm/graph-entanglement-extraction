import pdb

from qutip import tensor, basis, qeye, Qobj, sigmax, sigmay, sigmaz, ket2dm, expect, Qobj, concurrence
from qutip.qip.operations import cz_gate
from math import sqrt
import logging
from utils import is_projector, prepare_projector, prepare_pauli_operator
from abc import ABC, abstractmethod
import copy
import random
import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SpinState(ABC):
    """
    Base class for spin states.
    """

    def __init__(self):
        self.N = None

    @abstractmethod
    def probability(self):
        pass

    @abstractmethod
    def apply_projection_operator(self, projector):
        pass

    @abstractmethod
    def apply_local_unitary(self, *args):
        pass

    @abstractmethod
    def get_reduced_state(self, qubits, normalized=None):
        pass

    @abstractmethod
    def expval(self, operator, normalized=None):
        pass

    def measure_pauli_operator(self, qubit, pauli_operator, measurement_result):
        """
        Project the input state onto the eigenvector of the Pauli operator with the given measurement result.
        """

        projector = prepare_projector(self.N, {qubit: (pauli_operator, measurement_result)})

        return self.apply_projection_operator(projector)

    def measure_pauli_operator_sequence(self, pauli_operator_dict):
        """
        Project the input state onto the eigenvector of the Pauli operator sequence with given measurement results.
        """

        projector = prepare_projector(self.N, pauli_operator_dict)

        return self.apply_projection_operator(projector)

    def expval_pauli_operator_sequence(self, pauli_operator_dict, normalized=None):
        """
        Calculate the expectation value of a Pauli operator or a tensor product of Pauli operators.
        
        Args:
        pauli_operator_dict: A dictionary where keys are qubit indices and values are the corresponding
        Pauli operators ('I', 'X', 'Y', 'Z').
        normalized: Boolean or None. If True, return the expectation value for the normalized state.
                    If False/None, return the expectation value for the non-normalized state.


        Returns:
        The expectation value of the Pauli operator or tensor product of Pauli operators.
        """

        # Prepare the Pauli operator in qutip format
        pauli_operator = prepare_pauli_operator(self.N, pauli_operator_dict)

        return self.expval(pauli_operator, normalized=normalized)

    def expval_pauli_operator(self, qubit, pauli_operator, normalized=None):
        return self.expval_pauli_operator_sequence({qubit: pauli_operator}, normalized)

    def probs_of_pauli_measurement(self, qubit, pauli_operator, normalized=False):
        """
        Calculate the probabilities of measurement results for a given Pauli operator.
        
        Args:
        qubit, pauli_operator: qubit index and pauli operator (['X','Y','Z''])
        normalized: Boolean. If True, calculate probabilities using the normalized state.
                    If False, calculate probabilities using the non-normalized state. The resulting probabilities

        Returns:
        A dictionary with keys '+', '-' representing the probabilities of the corresponding measurement results.
        """


        plus_projection_operator = prepare_projector(self.N, {qubit: (pauli_operator, '+')})

        plus_probability = self.expval(plus_projection_operator, normalized=True)

        if (np.isclose(plus_probability, 1)):
            plus_probability = 1
        if (np.isclose(plus_probability, 0)):
            plus_probability = 0
        minus_probability = 1 - plus_probability

        if not( plus_probability >= 0 and minus_probability >= 0):
            pdb.set_trace()

        if normalized is True:
            return {'+': plus_probability, '-': minus_probability}
        else:
            return {'+': plus_probability * self.probability(), '-': minus_probability * self.probability()}


    def perform_pauli_measurement(self, qubit, pauli_operator):
        """
        Generate a measurement result '+' or '-' based on the probabilities
        returned from the probs_of_pauli_measurement function.
        Then applies the measurement and returns the result (+ or -).

        Args:
        qubit: The qubit index.
        pauli_operator: The Pauli operator ('X', 'Y', or 'Z').

        Returns:
        A string '+' or '-' representing the measurement result.
        """
        probs = self.probs_of_pauli_measurement(qubit, pauli_operator, normalized=True)
        sign = random.choices(['+', '-'], weights=[probs['+'], probs['-']])[0]
        self.measure_pauli_operator(qubit, pauli_operator, sign)
        return sign
    def concurrence_of_reduced_state(self, qubits):
        state =  self.get_reduced_state(qubits, normalized=True)
        return concurrence(state)

""" def __deepcopy__(self, memo):
    # Create a new instance of the class without calling the constructor
    new_instance = self.__class__.__new__(self.__class__)

    # Add the new instance to the memo dictionary to avoid infinite recursion
    memo[id(self)] = new_instance

    # Deep copy all attributes from the original object to the new instance
    for attr_name, attr_value in self.__dict__.items():
        setattr(new_instance, attr_name, copy.deepcopy(attr_value, memo))

    return new_instance
"""


class PureSpinState(SpinState):
    warning_issued_pure = False
    warning_issued_reduced = False

    def __init__(self, N, state, probability=1):
        """
        Initialize a PureSpinState with N qubits, an edge list, and an optional probability.
        """

        super().__init__()
        self.N = N
        self.internal_probability = probability
        self.state = copy.deepcopy(state)
        self._normalize()

    def _normalize(self):
        """
        Normalize the internal state and move the norm to internal probability.
        All the expectation values etc. stay the same, just shifts the numbers.
        """
        norm = self.state.norm()

        if norm == 0:
            logger.info("Norm zero state encountered.")
        else:
            self.state = self.state / norm

        probability_multiplier = norm ** 2

        self.internal_probability *= probability_multiplier

    def probability(self):
        return self.internal_probability

    def apply_projection_operator(self, projector):
        """
        Apply a projection operator to the state.
        """
        #if not is_projector(projector):
        #    raise ValueError("The given operator is not a projector.")

        projected_state = projector * self.state

        self.state = projected_state
        self._normalize()


    def apply_local_unitary(self, unitary_dict):
        pauli_operator = prepare_pauli_operator(self.N, unitary_dict)

        transformed_state = pauli_operator * self.state

        self.state = transformed_state
        self._normalize()


    def get_qutip_state(self, normalized=None):

        """
        Return the qutip state, optionally normalized.
        """

        if normalized is None:
            normalized = False
            if not PureSpinState.warning_issued_pure:
                logger.warning("Returning a non-normalized pure state.")
                PureSpinState.warning_issued_pure = True
        elif normalized == False:
            logger.warning("Returning a non-normalized pure state on purpose (weird).")

        if normalized:
            return copy.deepcopy(self.state)
        else:
            return copy.deepcopy(self.state) * sqrt(self.probability())

    def get_reduced_state(self, qubits, normalized=None):
        """
        Return the reduced state, tracing out qubits not in the list, and optionally normalized.
        """

        reduced_density_matrix = self.state.ptrace(qubits)

        # Check if the result is a ket -- see https://github.com/qutip/qutip/issues/2129
        if reduced_density_matrix.isket:
            # Turn the ket into a density operator
            reduced_density_matrix = ket2dm(reduced_density_matrix)

        if normalized is None:
            normalized = False
            if not PureSpinState.warning_issued_reduced:
                logger.warning("Returning a non-normalized reduced state.")
                PureSpinState.warning_issued_reduced = True
        elif normalized is False:
            logger.info("Returning a non-normalized reduced state on purpose.")

        if normalized:
            assert np.isclose(reduced_density_matrix.tr(),1)
            #if not np.isclose(reduced_density_matrix.tr(),1,atol=1e-5):
            #    raise ValueError(reduced_density_matrix.tr())
            return reduced_density_matrix
        else:
            return reduced_density_matrix * self.probability()

    def expval(self, operator, normalized=False):
        """
        Calculate the expectation value of an operator.
        
        Args:
        operator: a qutip operator.
        normalized: Boolean or None. If True, return the expectation value for the normalized state.
                    If False/None, return the expectation value for the non-normalized state.


        Returns:
        The expectation value of the operator over a state. If normalized!=True, it's multiplied by the internal probability of measuring the state.
        """

        qutip_state = self.get_qutip_state(normalized=True)

        expectation_value = expect(operator, qutip_state)

        if normalized is False:
            expectation_value *= self.probability()

        return expectation_value
    def __deepcopy__(self, memodict={}):
        return PureSpinState(self.N,self.state,self.internal_probability)


class MixedSpinState(SpinState):
    def __init__(self, N, ensemble_data):
        """
        Initialize a MixedSpinState with N qubits and ensemble_data, a list of dictionaries.
        """
        super().__init__()
        self.N = N
        self.state = Qobj()

        for ensemble_element in ensemble_data:
            state_list_element = copy.deepcopy(ensemble_element)
            self.state += ket2dm(state_list_element['state'].state) * state_list_element['probability']

    def probability(self):
        return self.state.tr()

    def apply_projection_operator(self, projector):
        assert is_projector(projector)
        self.state = projector * self.state * projector.dag()

    def apply_local_unitary(self, *args):
        raise NotImplementedError

    def get_reduced_state(self, qubits, normalized=False):
        """
        Return the reduced state, tracing out qubits not in the list, and optionally normalized.
        """

        reduced_density_matrix = self.state.ptrace(qubits)

        if normalized:
            return reduced_density_matrix
        else:
            return reduced_density_matrix * self.probability()

    def expval(self, operator, normalized=False):
        qutip_state = self.state

        expectation_value = expect(operator, qutip_state)

        if normalized is False:
            expectation_value *= self.probability()

        return expectation_value


class SpinStateEnsemble(SpinState):
    def __init__(self, N, ensemble_data, probability_function=None):
        """
        Initialize a SpinStateEnsemble with N qubits and ensemble_data, a list of dictionaries.
        each element of ensemble_data should have the key "state": a PureSpinState instance.
        An additional key "probability" is a probability multiplier for the (default) case that the optional
        probability_function is not provided in the constructor.
        """
        super().__init__()
        self.N = N

        if probability_function is None:
            probability_function = (lambda state_list_element: state_list_element['probability'])

        self.probability_function = probability_function

        self.states = []

        for ensemble_element in ensemble_data:
            state_list_element = copy.deepcopy(ensemble_element)

            # graph_state_list_element['reduced_states_cache']={}

            self.states.append(state_list_element)

    def probability(self):
        total_probability = 0
        for state_data in self.states:
            state = state_data['state']
            external_probability = self.probability_function(state_data)

            total_probability += state.probability() * external_probability

        return total_probability

    def apply_projection_operator(self, projector):
        for state in self.states:
            state['state'].apply_projection_operator(projector)

    def apply_local_unitary(self, *args):
        raise NotImplementedError

    def get_reduced_state(self, qubits, normalized=False):
        """
        Return the reduced state of the ensemble, tracing out qubits not in the list.
        """
        total_reduced_state = Qobj()

        for state_data in self.states:
            state = state_data['state']

            reduced_state = state.get_reduced_state(qubits, normalized=False)

            external_probability = self.probability_function(state_data)

            total_reduced_state += reduced_state * external_probability

        if normalized:
            total_reduced_state /= self.probability()

        return total_reduced_state

    def expval(self, operator, normalized=False):
        total_expval = 0.

        for state_data in self.states:
            state = state_data['state']

            state_expval = state.expval(operator)

            external_probability = self.probability_function(state_data)

            total_expval += state_expval * external_probability

        # total_reduced_state = sum(reduced_states, Qobj())
        if normalized:
            return total_expval / self.probability()
        else:
            return total_expval
