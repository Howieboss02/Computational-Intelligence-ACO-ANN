import unittest


class MyTestCase(unittest.TestCase):
    def test_something_false(self):
        self.assertEqual(True, False, "Should be false")
    def test_something_true(self):
        self.assertEqual(2 + 2, 4, "Should be equal to 4")


if __name__ == '__main__':
    unittest.main()
