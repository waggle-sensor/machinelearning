import unittest

from Data import data


##############################################

class TestData(unittest.TestCase):
    @classmethod
    def setUp(cls) -> None:
        pass

    @classmethod
    def tearDown(cls) -> None:
        pass

    def test_load_data_py(self):
        x = data.load_data_py()
        self.assertEqual(x,"Loaded data.py")

    def test_load_data_py(self):
        data_class = data.DataManager("string")
        self.assertEqual(x,"Loaded data.py")


##############################################

if __name__ == "__main__":
    unittest.main()