from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
from hiperwalk.graph.graph import _interval_binary_search
import unittest

class TestBinarySearch(unittest.TestCase):
    def setUp(self):
        self.sorted_array = [1, 1, 2, 3, 3, 4, 4, 7, 8, 8, 9, 10]
        self.sorted_array2 = [1, 2, 2, 3, 3, 4, 4, 7, 8, 8, 9, 10]
        self.sorted_array3 = [0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6]
        self.start = 4
        self.end = 8
    
    def test_binary_search_element_less_than_the_minimum_value_in_the_array(self):
        self.assertEqual(_interval_binary_search(self.sorted_array, 0), -1)
        self.assertEqual(_interval_binary_search(
                            self.sorted_array3, -1, start=1, end=4),
                         0)
    
    def test_binary_search_first_element_in_the_array(self):
        self.assertEqual(_interval_binary_search(self.sorted_array, 1), 1)
        self.assertEqual(_interval_binary_search(self.sorted_array2, 1), 0)
    
    def test_binary_search_element_greater_than_the_maximum_value_in_the_array(self):
        self.assertEqual(_interval_binary_search(self.sorted_array, 11), 11)
    
    def test_binary_search_element_in_the_middle_of_the_array(self):
        self.assertEqual(_interval_binary_search(self.sorted_array, 4), 6)
        self.assertEqual(_interval_binary_search(self.sorted_array2, 5), 6)
        self.assertEqual(_interval_binary_search(self.sorted_array2, 6), 6)
        self.assertEqual(_interval_binary_search(self.sorted_array2, 7), 7)
        self.assertEqual(_interval_binary_search(self.sorted_array, 8), 9)
        self.assertEqual(_interval_binary_search(self.sorted_array3, 0, self.start, self.end), 4)
        self.assertEqual(_interval_binary_search(self.sorted_array3, 2, self.start, self.end), 5)
        self.assertEqual(_interval_binary_search(self.sorted_array3, 4, self.start, self.end), 6)
        self.assertEqual(_interval_binary_search(self.sorted_array3, 6, self.start, self.end), 7)
        self.assertEqual(_interval_binary_search(self.sorted_array3, 40, self.start, self.end), 7)
