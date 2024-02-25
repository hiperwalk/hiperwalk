from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
from hiperwalk.refactor.graph import _binary_search
import unittest

class TestBinarySearch(unittest.TestCase):
    def setUp(self):
        self.sorted_array = [1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10]
        self.unsorted_array = [1, 3, 10, 8, 8, 2, 5, 1, 9, 7, 6, 4, 6, 4, 3]
    
    def test_binary_search_element_in_the_array_unsorted(self):
        with self.assertRaises(ValueError) as e:
            _binary_search(self.unsorted_array, 4)
        self.assertEqual(str(e.exception), "The array is not sorted.")
    
    def test_binary_search_element_less_than_the_minimum_value_in_the_array(self):
        with self.assertRaises(ValueError) as e:
            _binary_search(self.sorted_array, 0)
        self.assertEqual(str(e.exception), "The element is less than the minimum value in the array.")
    
    def test_binary_search_first_element_in_the_array(self):
        # the result must be 0
        self.assertEqual(_binary_search(self.sorted_array, 1), 0)
    
    def test_binary_search_element_greater_than_the_maximum_value_in_the_array(self):
        #the result must be the last index of the array
        self.assertEqual(_binary_search(self.sorted_array, 11), 14)
    
    def test_binary_search_element_in_the_middle_of_the_array(self):
        # the binary search can receive a value between two indexes and the result must be the minimum of these two indexes
        self.assertEqual(_binary_search(self.sorted_array, 4), 5)
    
    def test_binary_search_element_greater_than_the_maximum_value_in_the_array(self):
        self.assertEqual(_binary_search(self.sorted_array, 20), 14)
