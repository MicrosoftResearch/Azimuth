import azimuth
import azimuth.model_comparison
import numpy as np
import unittest
import os

class SavedModelTest(unittest.TestCase):
    def test_predictions(self):
        path_to_pickled_models = os.path.join(os.path.dirname(azimuth.__file__), 'saved_models')

        data = [['TGGAGGCTGCTTTACCCGCTGTGGGGGCGC', 254, 87, 0.5309, 0.5655],
                ['CGTCTCCGGGTTGGCCTTCCACTGGGGCAG', 216, 74, 0.6762, 0.6784],
                ['CCCTCAGCATCCTTCGGAAAGCTCTGGACA', 80, 27, 0.5258, 0.4890]]

        for d in data:
            prediction_full = azimuth.model_comparison.predict(np.array([d[0]]), np.array([d[1]]), np.array([d[2]]), model_file=os.path.join(path_to_pickled_models, 'V3_model_full.pickle'))[0]
            prediction_nopos = azimuth.model_comparison.predict(np.array([d[0]]), np.array([None]), np.array([None]), model_file=os.path.join(path_to_pickled_models, 'V3_model_nopos.pickle'))[0]

            message =  "\n\n\n"
            message += "WARNING!!! The predictions don't match"
            message +=  "Full model prediction: %.4f  \t  Correct prediction: %.4f" % (d[3], prediction_full)
            message +=  "No-pos model prediction: %.4f  \t  Correct prediction: %.4f" % (d[4], prediction_nopos)

            self.assertTrue(np.allclose([prediction_full, prediction_nopos], [d[3], d[4]], atol=1e-3), msg=message)
