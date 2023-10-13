class Simulator():
    r"""
    TODO: docs
    """

    def __init__(self, matrix):
        self._matrix = matrix

    def set_matrix(self, matrix):
        r"""
        """
        # TODO: check dimensions

    def get_matrix(self, copy=True):
        r"""
        TODO
        """
        return np.copy(self._matrix) if copy else self._matrix

    def simulate(self, power=None, vector=None, hpc=True):
        r"""
        TODO DOCS
        """
        # TODO: check vector dimensions
        return None
