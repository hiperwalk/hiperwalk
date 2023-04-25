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
    max_steps : int
        Maximum number of steps that can be simulated.
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

    def __init__(self, max_steps, state_entries, entry_type='vertex_dir'):
        valid_entry_types = ['vertex_dir', 'arc_notation']
        if entry_type not in valid_entry_types:
            raise ValueError("Invalid argument for entry_type."
                             + "Expected either of"
                             + str(valid_entry_types))

        self._max_steps = int(max_steps)

        self._shift, right_vert = self.__get_extreme_vertices(
            state_entries, entry_type)
        self._shift -= self._max_steps

        shifted_entries = self.__shift_state_entries(
                state_entries, entry_type
        )

        num_vert = right_vert - self._shift + self._max_steps + 1

        super().__init__(num_vert)

        self._initial_condition = self.state(shifted_entries,
                                             type=entry_type)


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

    def simulate(self, time_range=None, evolution_operator=None,
                 hpc=True):
        r"""
        Simulate coined quantum walk on the line.

        Analogous to :meth:`qwalk.BaseWalk.simulate` but uses the
        ``initial_condition`` sent to the constructor
        (See :class:`Line`)
        and the maximum number of evolution operator application is
        limited by the ``max_steps`` value sent to the constructor.

        If ``time_range=None``, uses ``time_range=max_steps``.


        Raises
        ------
        ValueError
            If ``time_range``'s ``end``  time exceeds ``max_steps``.
        """
        if time_range is None:
            time_range = self._max_steps

        time_range = self._time_range_to_tuple(time_range)
        if time_range[1] > self._max_steps:
            raise ValueError(
                "`time_range` requested more steps than allowed. "
                + "Check if the value of ``end`` in ``time_range`` is "
                + "less than or equal to the value of ``max_steps`` "
                + "sent to the constructor."
            )

        return super().simulate(time_range, self._initial_condition,
                                evolution_operator, hpc)
