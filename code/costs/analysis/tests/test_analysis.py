import unittest
from analysis import Analysis
import pandas as pd


class TestAnalysis(unittest.TestCase):
    def test_magnitude_estimates(self):
        df = pd.DataFrame({'METP1h.': ['Hateful', 'Not hateful', 'Not hateful'],
                           'METP1s.': ['Agree', 'Disagree', 'Neutral'],
                           'METP1v.': [10.0, 50.0, None],
                           'METN1h.': ['Hateful', 'Not hateful', 'Not hateful'],
                           'METN1s.': ['Agree', 'Disagree', 'Neutral'],
                           'METN1v.': [10.0, 50.0, None],
                           'MEFP1h.': ['Hateful', 'Not hateful', 'Not hateful'],
                           'MEFP1s.': ['Agree', 'Disagree', 'Neutral'],
                           'MEFP1v.': [10.0, 50.0, None],
                           'MEFN1h.': ['Hateful', 'Not hateful', 'Not hateful'],
                           'MEFN1s.': ['Agree', 'Disagree', 'Neutral'],
                           'MEFN1v.': [10.0, 50.0, None],
                           'MEREJ1h.': ['Hateful', 'Not hateful', 'Not hateful'],
                           'MEREJ1s.': ['Agree', 'Disagree', 'Neutral'],
                           'MEREJ1v.': [10.0, 50.0, None]})

        expected = pd.DataFrame({'METP1': [10.0, -50.0, 0.0],
                                 'METN1': [10.0, -50.0, 0.0],
                                 'MEFP1': [10.0, -50.0, 0.0],
                                 'MEFN1': [10.0, -50.0, 0.0],
                                 'MEREJ1': [10.0, -50.0, 0.0]
                                 })
        mes = Analysis.magnitude_estimates(data=df, num_scenarios=1)
        self.assertTrue(mes.equals(expected))

    def test_s100_values(self):
        df = pd.DataFrame({'S100TP1h.': ['Hateful', 'Not hateful', 'Not hateful'],
                           'S100TP1s.': ['Agree', 'Disagree', 'Neutral'],
                           'S100TP1a[SQ001].': [10.0, None, None],
                           'S100TP1d[SQ001].': [None, 50.0, None],
                           'S100TN1h.': ['Hateful', 'Not hateful', 'Not hateful'],
                           'S100TN1s.': ['Agree', 'Disagree', 'Neutral'],
                           'S100TN1a[SQ001].': [10.0, None, None],
                           'S100TN1d[SQ001].': [None, 50.0, None],
                           'S100FP1h.': ['Hateful', 'Not hateful', 'Not hateful'],
                           'S100FP1s.': ['Agree', 'Disagree', 'Neutral'],
                           'S100FP1a[SQ001].': [10.0, None, None],
                           'S100FP1d[SQ001].': [None, 50.0, None],
                           'S100FN1h.': ['Hateful', 'Not hateful', 'Not hateful'],
                           'S100FN1s.': ['Agree', 'Disagree', 'Neutral'],
                           'S100FN1a[SQ001].': [10.0, None, None],
                           'S100FN1d[SQ001].': [None, 50.0, None],
                           'S100REJ1h.': ['Hateful', 'Not hateful', 'Not hateful'],
                           'S100REJ1s.': ['Agree', 'Disagree', 'Neutral'],
                           'S100REJ1a[SQ001].': [10.0, None, None],
                           'S100REJ1d[SQ001].': [None, 50.0, None]})

        expected = pd.DataFrame({'S100TP1': [10.0, -50.0, 0.0],
                                 'S100TN1': [10.0, -50.0, 0.0],
                                 'S100FP1': [10.0, -50.0, 0.0],
                                 'S100FN1': [10.0, -50.0, 0.0],
                                 'S100REJ1': [10.0, -50.0, 0.0]
                                 })
        mes = Analysis.s100_values(data=df, num_scenarios=1)
        self.assertTrue(mes.equals(expected))


if __name__ == '__main__':
    unittest.main()
