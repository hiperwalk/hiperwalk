from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
from hiperwalk.refactor.graph import _interval_binary_search
import unittest

class TestBinarySearch(unittest.TestCase):
    def setUp(self):
        self.sorted_array = [1, 1, 2, 3, 3, 4, 4, 7, 8, 8, 9, 10]
        self.sorted_array2 = [1, 2, 2, 3, 3, 4, 4, 7, 8, 8, 9, 10]
    
    def test_binary_search_element_less_than_the_minimum_value_in_the_array(self):
        with self.assertRaises(ValueError) as e:
            _interval_binary_search(self.sorted_array, 0)
        self.assertEqual(str(e.exception), "The element is less than the minimum value in the array.")
    
    def test_binary_search_first_element_in_the_array(self):
        self.assertEqual(_interval_binary_search(self.sorted_array, 1), 1)
    
    def test_binary_search_first_element_in_the_array2(self):
        self.assertEqual(_interval_binary_search(self.sorted_array2, 1), 0)
    
    def test_binary_search_element_greater_than_the_maximum_value_in_the_array(self):
        self.assertEqual(_interval_binary_search(self.sorted_array, 11), 11)
    
    def test_binary_search_element_in_the_middle_of_the_array(self):
        self.assertEqual(_interval_binary_search(self.sorted_array, 4), 6)
    
    def test_binary_search_element_in_the_middle_of_the_array2(self):
        self.assertEqual(_interval_binary_search(self.sorted_array2, 5), 6)
    
    def test_binary_search_element_in_the_middle_of_the_array3(self):
        self.assertEqual(_interval_binary_search(self.sorted_array2, 6), 6)
    
    def test_binary_search_element_in_the_middle_of_the_array4(self):
        self.assertEqual(_interval_binary_search(self.sorted_array2, 7), 7)
    
    def test_binary_search_element_in_the_middle_of_the_array5(self):
        self.assertEqual(_interval_binary_search(self.sorted_array, 8), 9)
