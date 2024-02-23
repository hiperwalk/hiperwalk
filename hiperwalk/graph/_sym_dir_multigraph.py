from .graph import *

class SDMultigraph(Graph):
    r"""
    Class for managing symmetric directed multigraph.

    Functions for managing a symmetric directed multigraph are provided.
    Only a reference to the underlying multigraph is stored.
    """

    def __init__(self, graph):
        # underlying multigraph
        self.graph = graph
        # TODO: different behavior if graph is simple or multigraph

    def arc(self, number):
        r"""
        Convert a numerical label to arc notation.
    
        Given an integer that represents the numerical label of an arc,
        this method returns the corresponding arc in ``(tail, head)`` 
        representation.
    
        Parameters
        ----------
        number : int
            The numerical label of the arc.
    
        Returns
        -------
        (int, int)
            The arc represented in ``(tail, head)`` notation.
        """

        adj_matrix = self._adj_matrix
        head = adj_matrix.indices[number]
        #TODO: binary search
        for tail in range(len(adj_matrix.indptr)):
            if adj_matrix.indptr[tail + 1] > number:
                break
        return (tail, head)

    def arc_number(self, arc):
        r"""
        Return the numerical label of the arc.

        Parameters
        ----------
        arc:
            int:
                The arc's numerical label itself is passed
                as argument.
            (tail, head):
                Arc in arc notation.

        Returns
        -------
        label: int
            Numerical label of the arc.

        Examples
        --------
        If arc ``(0, 1)`` exists, the following commands return
        the same result.

        .. testsetup::

            import networkx as nx
            from sys import path
            path.append('../..')
            import hiperwalk as hpw
            nxg = nx.cycle_graph(10)
            adj_matrix = nx.adjacency_matrix(nxg)
            graph = hpw.Graph(adj_matrix)

        >>> graph.arc_number(0) #arc number 0
        0
        >>> graph.arc_number((0, 1)) #arc as tuple
        0
        """
        if not hasattr(arc, '__iter__'):
            num_arcs = self.number_of_arcs()
            if arc < 0 and arc >= num_arcs:
                raise ValueError("Arc value out of range. "
                                 + "Expected arc value from 0 to "
                                 + str(num_arcs - 1))
            return int(arc)

        tail = self._graph.vertex_number(arc[0])
        head = self._graph.vertex_number(arc[1])
        # TODO: the behavior may change after updating neighbors()
        # TODO: the behavior will change for multigraphs
        arc_number = self._adj_matrix.indptr[tail]

        offset = np.where(self.neighbors(head) == tail)
        if len(offset) != 1:
            raise ValueError("Inexistent arc " + str(arc) + ".")
        offset = offset[0]

        arc_number += offset
        return arc_number

    def arcs_with_tail(self, tail):
        r"""
        Return all arcs that have the given tail.
        """
        arcs_lim = self._adj_matrix.indptr
        return np.arange(arcs_lim[tail], arcs_lim[tail + 1])

    def arcs_with_tail(self, tail):
        r"""
        Return all arcs that have the given tail.
        """
        arcs_lim = self._adj_matrix.indptr
        return np.arange(arcs_lim[tail], arcs_lim[tail + 1])

    def number_of_arcs(self):
        r"""
        Determine the cardinality of the arc set.

        In simple graphs, the cardinality of the arc set is 
        equal to twice the number of edges. 
        However, for graphs containing loops, the 
        cardinality is incremented by one for each loop.
        """

        return self._adj_matrix.sum()
