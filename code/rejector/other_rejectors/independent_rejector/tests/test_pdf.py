from independent_rejector.pdf import PDF
from independent_rejector.prediction import Prediction
from independent_rejector.pdfs import PDFs
import unittest
import numpy as np
from scipy.integrate import simps


class TestPDF(unittest.TestCase):
    def setUp(self):
        predictions = []
        for p in np.linspace(0, 0.5, 100):
            prediction = Prediction(
                "positive", "positive", p, "positive", None)
            predictions.append(prediction)

        self.predictions = predictions

    def test_init(self):
        kde = PDFs.estimator(self.predictions)
        pdf = PDF(self.predictions, 1.0, kde)

        # The top of the PDF function should be around 2 since 0.5*2=1.
        # Explanation: all points are between 0 and 0.5, so the area under the PDF curve
        # should be equal to 1 between 0 and 0.5.
        self.assertAlmostEqual(max(pdf.pdf_y), 2, 1)
        total_area = simps(pdf.pdf_y, pdf.pdf_x)
        self.assertAlmostEqual(total_area, 1, 1)

    def test_integral(self):
        kde = PDFs.estimator(self.predictions)
        pdf = PDF(self.predictions, 1.0, kde)
        self.assertTrue(0.97 <= pdf.integral(1.0) <= 1.0)
        self.assertTrue(0.9 <= pdf.integral(0.5) <= 1.0)
        self.assertTrue(0.47 <= pdf.integral(0.25) <= 0.5)
        self.assertTrue(0 <= pdf.integral(0) <= 0.1)


if __name__ == '__main__':
    unittest.main()
