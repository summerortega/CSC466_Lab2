import unittest
from c45 import C45Tree


class MyTestCase(unittest.TestCase):
    def test_c45_fit(self):
        my_tree = C45Tree()
        assert my_tree is not None

    def test_induce_c45(self):
        my_tree = C45Tree()




if __name__ == '__main__':
    unittest.main()
