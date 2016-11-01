import azimuth
import azimuth.model_comparison
import numpy as np
import unittest
import pandas

class SavedModelTests(unittest.TestCase):
    """
    This unit test checks that the predictions for 1000 guides match the predictions we expected in Nov 2016.
    This unit test can fail due to randomness in the model (e.g. random seed, feature reordering).
    """

    def test_predictions(self):
        df = pandas.read_csv('1000guides.csv', index_col=0)
        predictions = azimuth.model_comparison.predict(np.array(df['guide'].values), None, None)
        self.assertTrue(np.allclose(predictions, df['Stable prediction'].values, atol=1e-3))

if __name__ == '__main__':
    unittest.main()
