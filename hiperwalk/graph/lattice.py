from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse
from .graph import Graph
from warnings import warn

class Lattice(ABC, Graph):
    r"""
    Abstract class for Lattice graphs.

    Lattice graphs are embedabble in a Euclidian Space,
    forming a regular tiling.

    Notes
    -----
    Given an embedding,
    directions can be assigned to the arcs.
    """

    @abstractmethod
    def next_arc(self, arc):
        r"""
        Next arc with respect to the default embedding.

        Parameters
        ----------
        arc
            The arc in any of the following notations.

            * arc notation: tuple of vertices
                In ``(tail, head)`` format where
                ``tail`` and ``head`` must be valid vertices.
            * arc number: int.
                The numerical arc label.

        Returns
        -------
        Next arc in the same notation as the ``arc`` argument.

        See Also
        --------
        arc
        arc_number
        """
        raise NotImplementedError

    @abstractmethod
    def previous_arc(self, arc):
        r"""
        Previous arc with respect to the default embedding.

        Parameters
        ----------
        arc
            The arc in any of the following notations.

            * arc notation: tuple of vertices
                In ``(tail, head)`` format where
                ``tail`` and ``head`` must be valid vertices.
            * arc number: int.
                The numerical arc label.

        Returns
        -------
        Previous arc in the same notation as the ``arc`` argument.

        See Also
        --------
        arc
        arc_number
        """
        raise NotImplementedError
