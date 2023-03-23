import numpy as np
from .segment import Segment

class Line(Segment):
    r"""
    Class for managing quantum walk on the line.

    For simulating the quantum walk on the "infinite" line,
    it is necessary to know the initial condition and the
    number of steps beforehand
    in order to allocate the necessary resources.

    Parameters
    ----------
    num_steps : int
        Number of steps to be simulated.
    state_entries : tuple
        Check :meth:`state` for details of valid entries.
    entry_type : {'vertex_dir', 'arc_notation', 'arc_order'}
        Check :meth:`state` for details.

    Notes
    -----
    It is built on top of :class:`Segment`,
    so a walk on the line is not simulated.
    But rather a walk on a sufficiently large segment.

    """

    def __init__(self, num_steps, state_entries, entry_type='vertex_dir'):
        valid_entry_types = ['vertex_dir', 'arc_notation']
        if entry_type not in valid_entry_types:
            raise ValueError("Invalid argument for entry_type."
                             + "Expected either of"
                             + str(valid_entry_types))

        self._shift, right_vert = self.__get_extreme_vertices(
            state_entries, entry_type)
        self._shift -= num_steps

        shifted_entries = self.__shift_state_entries(
                state_entries, entry_type
        )

        num_vert = right_vert - self._shift + num_steps + 1

        super().__init__(num_vert)

        self._num_steps = num_steps
        self._initial_condition = self.state(shifted_entries, entry_type)


    def __get_extreme_vertices(self, state_entries, entry_type):
        # returns leftmost and rightmost vertices
        left_vert = state_entries[0][1]
        right_vert = left_vert
        for i in range(len(state_entries)):
            left_vert = (state_entries[i][1]
                        if state_entries[i][1] < left_vert
                        else left_vert)
            right_vert = (state_entries[i][1]
                        if state_entries[i][1] > right_vert
                        else right_vert)

        return left_vert, right_vert

    def __shift_state_entries(self, state_entries, entry_type):
        def __shift_entry(entry, shift, type):
            shifted_entry = entry[:]
            shifted_entry[1] -= shift
            if entry_type == 'arc_notation':
                shifted_entry[2] -= shift

            return shifted_entry

        shifted_state = [__shift_entry(entry, self._shift, type)
                         for entry in state_entries]

        return shifted_state

    def simulate_walk(self, evolution_operator, save_interval=0,
                      hpc=False):
        r"""
        Starts quantum walk simulation.

        Since the initial condition and number of steps must
        passed as arguments to the constructor,
        the respective parameters are ommited.
        """
        return super().simulate_walk(
            evolution_operator, self._initial_condition,
            self._num_steps, save_interval, hpc
        )
