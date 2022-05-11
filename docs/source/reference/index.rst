Reference
=========

Search Evolution Operator
-------------------------
To create the search evolution operator given an adjacency matrix,
use the ``CoinedModel.EvolutionOperator_SearchCoinedModel(AdjMatrix)`` function.

.. py:function:: CoinedModel.EvolutionOperator_SearchCoinedModel(AdjMatrix)
    
    Return The search evolution operator for the coined model given the
    adjacency matrix of a graph.

    :param AdjMatrix: Adjacency Matrix.
    :type AdjMatrix: scipy.csr_matrix format.
    :return: Search evolution operator.
    :rtype: scipy.csr_matrix???
