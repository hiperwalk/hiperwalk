import numpy as np
from .segment import Segment

class Line(Segment):
    def __init__(self, num_steps, state_entries, entry_type='vertex_dir'):
        r"""
        .. todo::
            Check which is the rightmost vertex pointing righttowards
            and the leftmost vertex pointing leftwards
        """
        
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

        print(state_entries)
        print(shifted_entries)
        num_vert = right_vert - self._shift + num_steps + 1
        print(num_vert)

        super().__init__(num_vert)

        self._num_steps = num_steps
        self._initial_condition = self.state(state_entries, entry_type)


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
        super().simulate_walk(evolution_operator, self._initial_condition,
                              self._num_steps, save_interval, hpc)
