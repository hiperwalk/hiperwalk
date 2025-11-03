import numpy as np
from .graph import Graph

class SDMultigraph(Graph):
    r"""
    Class for managing symmetric directed multigraph.

    Functions for managing a symmetric directed multigraph are provided.
    Only a reference to the underlying multigraph is stored.
    """

    def __init__(self, graph):
        # underlying multigraph
        self.graph = graph

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
        if self.is_underlying_simple():
            return self.graph._find_entry(number)

        # multiarc
        raise NotImplementedError()

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
        try:
            tail = self.vertex_number(arc[0])
            head = self.vertex_number(arc[1])
            out_degree = 1
            multiedge = 0

            if not self.is_underlying_simple():
                # multigraph
                out_degree = self.graph.number_of_edges(tail, head)
                multiedge = arc[2]

            entry = self.graph._entry(tail, head)
            entry += multiedge - out_degree
            return entry

        except (TypeError, IndexError):
            # TypeError if python int
            # IndexError if numpy int
            return arc

        raise NotImplementedError("arc_number() for multigraphs.")

    def arcs_with_tail(self, tail):
        r"""
        Return all arcs that have the given tail.
        """
        tail = self.vertex_number(tail)
        try:
            indptr = self.graph._adj_matrix.indptr

            if self.is_underlying_simple():
                indptr = self.graph._adj_matrix.indptr
                return np.arange(indptr[tail], indptr[tail + 1])

            data = self.graph._adj_matrix.data
            first_arc = data[indptr[tail] - 1] if tail > 0 else 0
            last_arc = data[indptr[tail + 1] - 1]
            return np.arange(first_arc, last_arc)
        except:
            neigh = self.graph.neighbors(tail)
            arcs = np.array([self.arc_number((tail, n)) for n in neigh])
            return arcs


    def number_of_arcs(self):
        r"""
        Determine the cardinality of the arc set.

        In simple graphs, the cardinality of the arc set is 
        equal to twice the number of edges. 
        However, for graphs containing loops, the 
        cardinality is incremented by one for each loop.
        """
        if self.is_underlying_simple():
            try:
                return self.graph._adj_matrix.indptr[-1]
            except AttributeError:
                num_edges = self.number_of_edges() << 1
                return  num_edges - self.number_of_loops()

        return self.graph._adj_matrix.data[-1]

    def is_underlying_simple(self):
        r"""
        Return if the underlying (multi)graph is simple.
        """
        return self.graph.is_simple()

    #####################################
    ### Overriding self.graph methods ###
    #####################################


    def adjacent(self, u, v):
        return self.graph.adjacent(u, v)

    def _neighbor_index(self, vertex, neigh):
        return self.graph._neighbor_index(vertex, neigh)

    def neighbors(self, vertex):
        return self.graph.neighbors(vertex)

    def number_of_vertices(self):
        return self.graph.number_of_vertices()

    def number_of_edges(self):
        r"""
        Return number of edges of the underlying (multi)graph.
        """
        return self.graph.number_of_edges()

    def number_of_loops(self):
        return self.graph.number_of_loops()

    def degree(self, vertex):
        return self.graph.degree(vertex)

    def vertex_number(self, vertex):
        return self.graph.vertex_number(vertex)

    def laplacian_matrix(self):
        return self.graph.laplacian_matrix()

    def is_simple(self):
        r"""
        Directed multigraph is not simple
        """
        return False

    def previous_arc(self, arc):
        g = self.graph

        if (not g.is_simple() or not hasattr(g, '_basis')):
            return None

        arc = self.arc(arc)

        tail = g.vertex_coordinates(arc[0])
        head = g.vertex_coordinates(arc[1])
        direction = head - tail

        prev_head = tail
        prev_tail = tail - direction
        if not g._valid_vertex(prev_tail, exception=False):
            prev_tail = head

        return self.arc_number((prev_tail, prev_head))
