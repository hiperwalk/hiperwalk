from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse
from .graph import Graph
from warnings import warn

class Lattice(ABC, Graph):

    @abstractmethod
    def next_arc(self, arc):
        r"""
        Next arc in an embeddable graph.

        Parameters
        ----------
        arc
            The arc in any of the following notations.

            * arc notation: tuple of vertices
                In ``(tail, head)`` format where
                ``tail`` and ``head`` must be valid vertices.
            * arc label: int.
                The arc label (number).

        Returns
        -------
        Next arc in the same notation as the ``arc`` argument.

        See Also
        --------
        arc
        arc_label
        """
        raise NotImplementedError

    @abstractmethod
    def previous_arc(self, arc):
        r"""
        Previous arc in an embeddable graph.

        Parameters
        ----------
        arc
            The arc in any of the following notations.

            * arc notation: tuple of vertices
                In ``(tail, head)`` format where
                ``tail`` and ``head`` must be valid vertices.
            * arc label: int.
                The arc label (number).

        Returns
        -------
        Previous arc in the same notation as the ``arc`` argument.

        See Also
        --------
        arc
        arc_label
        """
        raise NotImplementedError
