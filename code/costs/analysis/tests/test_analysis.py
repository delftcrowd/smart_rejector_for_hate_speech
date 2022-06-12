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

    def test_pivot_value(self):
        df = pd.DataFrame({'G20Q51[SQ001].': [-100, -10, -1],
                           'G20Q51[SQ002].': [-50, -5, -0.5],
                           'G20Q51[SQ003].': [-10, -1, -0.1],
                           'G20Q51[SQ005].': [10, 1, 0.1],
                           'G20Q51[SQ006].': [50, 5, 0.5],
                           'G20Q51[SQ007].': [100, 10, 1],
                           })

        s = pd.Series({'G20Q51[SQ001].': -100,
                       'G20Q51[SQ002].': -20,
                       'G20Q51[SQ003].': -2,
                       'G20Q51[SQ005].': 2,
                       'G20Q51[SQ006].': 4,
                       'G20Q51[SQ007].': 400,
                       })

        pivot_value = Analysis.pivot_value(s)
        self.assertEquals(pivot_value, 88)

    def test_normalize(self):
        data = pd.DataFrame({'G20Q51[SQ001].': [-100, -10, -1],
                             'G20Q51[SQ002].': [-50, -5, -0.5],
                             'G20Q51[SQ003].': [-10, -1, -0.1],
                             'G20Q51[SQ005].': [10, 1, 0.1],
                             'G20Q51[SQ006].': [30, 3, 0.3],
                             'G20Q51[SQ007].': [100, 10, 1]
                             })

        mes = pd.DataFrame({'METP1': [100.0, 10.0, 1.0],
                            'METN1': [60.0, 6.0, 0.6],
                            'MEFP1': [-10.0, -1.0, -0.1],
                            'MEFN1': [-100.0, -10.0, -1],
                            'MEREJ1': [-30.0, -3.0, -0.3]
                            })

        expected = pd.DataFrame({'METP1': [2.0, 2.0, 2.0],
                                 'METN1': [1.2, 1.2, 1.2],
                                 'MEFP1': [-0.2, -0.2, -0.2],
                                 'MEFN1': [-2.0, -2.0, -2.0],
                                 'MEREJ1': [-0.6, -0.6, -0.6]
                                 })

        normalized_mes = Analysis.normalize(data, mes)
        self.assertTrue(normalized_mes.equals(expected))


if __name__ == '__main__':
    unittest.main()
