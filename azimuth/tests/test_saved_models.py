import azimuth
import azimuth.model_comparison
import numpy as np
import unittest

class SavedModelTests(unittest.TestCase):
    def test_predictions(self):

        data = [['TGGAGGCTGCTTTACCCGCTGTGGGGGCGC', 254, 87, 0.5335, 0.5909],
                ['CGTCTCCGGGTTGGCCTTCCACTGGGGCAG', 216, 74, 0.6783, 0.6632],
                ['CCCTCAGCATCCTTCGGAAAGCTCTGGACA', 80, 27, 0.4898, 0.4461]]

        for d in data:
            prediction_full = azimuth.model_comparison.predict(np.array([d[0]]), np.array([d[1]]), np.array([d[2]]))[0]
            prediction_nopos = azimuth.model_comparison.predict(np.array([d[0]]), None, None)[0]

            message =  "\n\n\n"
            message += "WARNING!!! The predictions don't match\n"
            message +=  "Full model prediction: %.4f  \t  Correct prediction: %.4f\n" % (d[3], prediction_full)
            message +=  "No-pos model prediction: %.4f  \t  Correct prediction: %.4f\n" % (d[4], prediction_nopos)

            self.assertTrue(np.allclose([prediction_full, prediction_nopos], [d[3], d[4]], atol=1e-3), msg=message)
