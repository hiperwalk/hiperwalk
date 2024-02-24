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

        if self.graph.is_simple():
            adj_matrix = self._adj_matrix
            head = adj_matrix.indices[number]

            from warnings import warn
            warn("Use interval_binary_search")

            for tail in range(len(adj_matrix.indptr)):
                if adj_matrix.indptr[tail + 1] > number:
                    break
            return (tail, head)

        raise NotImplementedError("arc() for multigraphs")
        # Use interval_binary_search

    def arc_number(self, arc):
        r"""
        Return the numerical label of the arc.

        Parameters
        ----------
        arc: (int, int)
            Arc in arc notation, i.e. (tail, head).

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
        if self.graph.is_simple():
            # if not hasattr(arc, '__iter__'):
            #     num_arcs = self.number_of_arcs()
            #     if arc < 0 and arc >= num_arcs:
            #         raise ValueError("Arc value out of range. "
            #                          + "Expected arc value from 0 to "
            #                          + str(num_arcs - 1))
            #     return int(arc)

            tail = self.graph.vertex_number(arc[0])
            head = self.graph.vertex_number(arc[1])

            first_arc = self._adj_matrix.indptr[tail]
            offset = self.graph._neighbor_index(tail, head)

            return first_arc + offset

        raise NotImplementedError("arc_number() for multigraphs.")

    def arcs_with_tail(self, tail):
        r"""
        Return all arcs that have the given tail.
        """
        tail = self._graph.vertex_number(tail)
        indptr = self.graph._adj_matrix.indptr

        if self.graph.is_simple():
            indptr = self._adj_matrix.indptr
            return np.arange(indptr[tail], indptr[tail + 1])

        data = self.graph.data
        first_arc = data[indptr[tail] - 1] if tail > 0 else 0
        last_arc = data[indptr[tail + 1] - 1]
        return np.arange(first_arc, last_arc)

    def number_of_arcs(self):
        r"""
        Determine the cardinality of the arc set.

        In simple graphs, the cardinality of the arc set is 
        equal to twice the number of edges. 
        However, for graphs containing loops, the 
        cardinality is incremented by one for each loop.
        """
        return self.graph._adj_matrix.data[-1]

    # TODO: override self.graph methods
