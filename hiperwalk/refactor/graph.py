import numpy as np
import networkx as nx
from scipy.sparse import issparse, csc_array, diags

def _binary_search(v, elem, start = 0, end = None) -> int:
    r"""
    This function expects a sorted array and performs a binary search on the subarray
    v[start:end], looking for the element 'elem'.
    If the element is found, the function returns the index of the element.
    If the element is not found, the function returns -1.
    This is an implementation of binary search following Cormen's algorithm.
    It is used to improve the time complexity of the search process.
    """
    for i in range(len(v) - 1):
        if v[i] > v[i + 1]:
            raise ValueError("The array is not sorted.")
    if elem < v[start]:
        raise ValueError("The element is less than the minimum value in the array.")
    elif elem >= v[len(v) - 1]:
        return len(v) - 1
    elif elem == v[start]:
        return start
    control_index_v = {}
    for index, value in enumerate(v):
        if value not in control_index_v:
            control_index_v[value] = index
    using_control_index = list(control_index_v.keys())
    if end is None:
        end = len(using_control_index)-1
    while start < end:
        mid = (start + end) // 2
        if elem <= using_control_index[mid]:
            end = mid
        else:
            start = mid + 1
        if((elem >= using_control_index[start] and elem < using_control_index[start+1])):
            return control_index_v[using_control_index[start]]
    print(using_control_index)
    print("Start: ", start, "|End: ", end, "|Elem: ", elem, "|Mid: ", mid, "|using_control_index[mid]: ", using_control_index[mid], "|using_control_index[start]: ", using_control_index[start], "|using_control_index[end]: ", using_control_index[end])
    return -1

# class Graph:
#     """
#     Represents a simple graph with nodes and edges.

#     This class defines the graph structure used for implementing a quantum walk. It encapsulates all necessary properties and functionalities of the graph required for the quantum walk dynamics.
#     """
#     def __init__(self, adj_matrix):
#         if not hasattr(adj_matrix, '__iter__'):
#             raise ValueError('adj_matrix should be an iterable')
#         for row in adj_matrix:
#             if len(row) != len(adj_matrix):
#                 raise ValueError('adj_matrix should be a square matrix')
#             for val in row:
#                 if (val != 0) and (val != 1):
#                     raise ValueError('adj_matrix should be a binary matrix')
#         for i in len(adj_matrix):
#             if adj_matrix[i][i] != 0:
#                 raise ValueError('adj_matrix should not have self loops')
#         if all(hasattr(adj_matrix, attr) for attr in
#                ['__len__', 'edges', 'nbunch_iter', 'subgraph',
#                 'is_directed']):
#             adj_matrix = nx.convert_matrix.to_scipy_sparse_array(
#                     adj_matrix).astype(np.int8)
        
#         if not issparse(adj_matrix):
#             adj_matrix = csc_array(adj_matrix, dtype=np.int8)
        
#         self._adj_matrix = adj_matrix
#         self._coloring = None
    
#     def arc_number(self, *args):
#         r"""
#         Return the numerical label of the arc.

#         Parameters
#         ----------
#         *args:
#             int:
#                 The arc's numerical label itself is passed
#                 as argument.
#             (tail, head):
#                 Arc in arc notation,
#             tail, head:
#                 Arc in arc notation,
#                 but ``tail``and ``head``are passed as different arguments, not as a tuple.
        
#         Returns
#         -------
#         label: int
#             Numerical label of the arc.
        
#         Examples
#         --------
#         If arc ``(0, 1)``exists, the following commands return
#         the same result.

#         .. testsetup::
            
#             import networkx as nx
#             from sys import path
#             path.append('../..')
#             import hiperwalk as hpw
#             nxg = nx.cycle_graph(10)
#             adj_matrix = nx.adjacency_matrix(nxg)
#             graph = hpw.Graph(adj_matrix)
        
#         >>> graph.arc_number(0) #arc number 0
#         0
#         >>> graph.arc_number((0, 1)) #arc as tuple
#         0
#         >>> graph.arc_number(0, 1) #tail and head in separate arguments
#         0
#         """
#         arc = (args[0], args[1]) if len(args) == 2 else args[0]

#         if not hasattr(arc, '__iter__'):
#             num_arcs = self.number_of_arcs()
#             if arc < 0 or arc >= num_arcs:
#                 raise ValueError("Arc value out of range. "
#                                  + "Expected arc value from 0 to "
#                                  + str(num_arcs - 1))
#             return int(arc)
        
#         tail, head = arc
#         arc_number = _binary_search(self._adj_matrix.indices, head,
#                                     start = self._adj_matrix.indptr[tail],
#                                     end = self._adj_matrix.indptr[tail + 1])
#         if arc_number == -1:
#             raise ValueError("Inexistent arc " + str(arc) + ".")
#         return arc_number
    
#     def arc(self, number):
#         r"""
#         Convert a numerical label to arc notation.

#         Given an integert that represents the numerical label of an arc,
#         this method returns the corresponding arc in ``(tail, head)``
#         representation.

#         Parameters
#         ----------
#         number: int
#             Numerical label of the arc.
        
#         Returns
#         -------
#         (int, int)
#             The arc represented in ``(tail, head)`` notation.
#         """

#         adj_matrix = self._adj_matrix
#         head = adj_matrix.indices[number]
#         #TODO: binary search
#         for tail in range(len(adj_matrix.indptr)):
#             if adj_matrix.indptr[tail + 1] > number:
#                 break
#         return (tail, head)