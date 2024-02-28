
#from itertools import combinations

class RandomSubgraphStateEnsemble(GraphStateEnsemble):
    
    def __init__(self, N, edge_list, max_edge_failure_probability, cutoff_probability=None):
        """
        Constructor for RandomSubgraphStateEnsemble.
        Generates graph states with some edges missing - starting with every edge in place, then some are removed.
        cutoff_probability controls how many subgraphs are produced, along with edge_failure_probability.
        Probabilities of graphs generated so far are estimated (with edge_failure_probability) and the sum is compared with cutoff_probability. If sum>cutoff, graph generation stops.
        The final ensemble is then renormalized to unit probability.
        """
        ensemble_data = []
        total_probability = 0

        for r in range(len(edge_list) + 1):
            for missing_edges in combinations(edge_list, r):
                remaining_edges = [edge for edge in edge_list if edge not in missing_edges]

                probability = (max_edge_failure_probability ** len(missing_edges)) * ((1 - max_edge_failure_probability) ** len(remaining_edges))
                total_probability += probability

                ensemble_data.append({'edge_list': remaining_edges})
                # Log the generated edge subset and its probability
                logger.info(f"Generated edge subset: {remaining_edges}, Probability: {probability}")
            logger.info(f"Total probability up to {r} missing: {total_probability} vs cutoff: {cutoff_probability}")
            if cutoff_probability is not None and total_probability > cutoff_probability:
                break


        super().__init__(N, ensemble_data)
        self.edge_list=edge_list
        self.max_edge_failure_probability=max_edge_failure_probability
        self.total_probability=total_probability
        self.probability_function=self._probability_function

    def _probability_function(self,graph_state_data):
        graph_state = graph_state_data['graph_state']
        edge_list = graph_state_data['edge_list']
        #local_unitaries = graph_state_data['local_unitaries']
        #reduced_states_cache = graph_state_data['reduced_states_cache']

        remaining_edges = edge_list
        missing_edges = [edge for edge in self.edge_list if edge not in remaining_edges]
        probability = (self.edge_failure_probability ** len(missing_edges)) * ((1 - self.edge_failure_probability) ** len(remaining_edges))
        probability /= self.total_probability  # renormalization, usually we don't generate *all* subgraph states
        return probability


    def update_failure_probability(self, new_edge_failure_probability):
        """
        Update the edge failure probability, check if it's higher than the maximal
        edge failure probability, and renormalize the probabilities.
        """
        if new_edge_failure_probability > self.max_edge_failure_probability:
            raise ValueError("New edge failure probability is higher than the maximal allowed.")

        self.edge_failure_probability = new_edge_failure_probability

        total_probability = 0
        for graph_state_data in self.graph_states:

            remaining_edges = graph_state_data['edge_list']
            missing_edges = [edge for edge in self.edge_list if edge not in remaining_edges]
            
            probability = (self.edge_failure_probability ** len(missing_edges)) * ((1 - self.edge_failure_probability) ** len(remaining_edges))
            total_probability += probability

        # Renormalize the probabilities
        self.total_probability = total_probability
        self.edge_failure_probability=new_edge_failure_probability

    def __call__(self, new_edge_failure_probability):
        self.update_failure_probability(new_edge_failure_probability)
        return self
