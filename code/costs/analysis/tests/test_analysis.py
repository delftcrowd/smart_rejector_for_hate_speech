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
        s100 = Analysis.s100_values(data=df, num_scenarios=1)
        self.assertTrue(s100.equals(expected))

    def test_hatefulness(self):
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
                           'MEREJ1v.': [10.0, 50.0, None],
                           'S100TP1h.': ['Hateful', 'Not hateful', 'Not hateful'],
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
                           'S100REJ1d[SQ001].': [None, 50.0, None]
                           })

        expected = pd.DataFrame({'Hateful_METP1': [1.0, 0.0, 0.0],
                                 'Hateful_METN1': [1.0, 0.0, 0.0],
                                 'Hateful_MEFP1': [1.0, 0.0, 0.0],
                                 'Hateful_MEFN1': [1.0, 0.0, 0.0],
                                 'Hateful_MEREJ1': [1.0, 0.0, 0.0],
                                 'Hateful_S100TP1': [1.0, 0.0, 0.0],
                                 'Hateful_S100TN1': [1.0, 0.0, 0.0],
                                 'Hateful_S100FP1': [1.0, 0.0, 0.0],
                                 'Hateful_S100FN1': [1.0, 0.0, 0.0],
                                 'Hateful_S100REJ1': [1.0, 0.0, 0.0]})

        hatefulness = Analysis.hatefulness(data=df, num_scenarios=1)
        self.assertTrue(hatefulness.equals(expected))

    def test_pivot_value(self):
        s = pd.Series({'G20Q51[SQ001].': -100,
                       'G20Q51[SQ002].': -20,
                       'G20Q51[SQ003].': -2,
                       'G20Q51[SQ005].': 2,
                       'G20Q51[SQ006].': 4,
                       'G20Q51[SQ007].': 400,
                       })

        pivot_value = Analysis.pivot_value(s)
        self.assertEqual(pivot_value, 88)

    def test_normalize(self):
        data = pd.DataFrame({'G20Q51[SQ001].': [-100, -10, -1],
                             'G20Q51[SQ002].': [-50, -5, -0.5],
                             'G20Q51[SQ003].': [-10, -1, -0.1],
                             'G20Q51[SQ005].': [10, 1, 0.1],
                             'G20Q51[SQ006].': [30, 3, 0.3],
                             'G20Q51[SQ007].': [100, 10, 1],
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

    def test_reliabilty(self):
        mes = pd.DataFrame({'METP1': [2.0, 2.0, 2.0],
                            'METN1': [0.6, 1.2, 1.2],
                            'MEFP1': [-0.2, -0.2, -0.2],
                            'MEFN1': [-2.0, -2.0, -2.0],
                            'MEREJ1': [-0.6, -0.6, -0.6],
                            'S100TP1': [20.0, 20.0, 20.0],
                            'S100TN1': [8.0, 12.0, 12.0],
                            'S100FP1': [-2.0, -2.0, -2.0],
                            'S100FN1': [-20.0, -20.0, -20.0],
                            'S100REJ1': [-6.0, -6.0, -6.0],
                            'Hateful_METP1': [True, False, False],
                            'Hateful_S100FN1': [False, False, False]
                            })

        alpha = Analysis.reliability(mes)
        self.assertAlmostEqual(alpha, 0.9945, 4)

        alpha = Analysis.reliability(mes, type="TP")
        self.assertEqual(alpha, 1.0)

        alpha = Analysis.reliability(mes, type="TN")
        self.assertAlmostEqual(alpha, 0.9098, 4)

        alpha = Analysis.reliability(mes, type="FP")
        self.assertEqual(alpha, 1.0)

        alpha = Analysis.reliability(mes, type="FN")
        self.assertEqual(alpha, 1.0)

        alpha = Analysis.reliability(mes, scale="ME")
        self.assertAlmostEqual(alpha, 0.9882, 4)

        alpha = Analysis.reliability(mes, scale="S100")
        self.assertAlmostEqual(alpha, 0.9948, 4)

        mes = pd.DataFrame({'METN1': [1.2, 1.2, 1.2],
                            'METN2': [0.8, 0.8, 0.8],
                            'S100TN1': [12.0, 12.0, 12.0],
                            'S100TN2': [5.5, 6.0, 6.0]
                            })

        alpha = Analysis.reliability(mes, scale="ME", type="TN")
        self.assertEqual(alpha, 1.0)

        alpha = Analysis.reliability(mes, scale="S100", type="TN")
        self.assertAlmostEqual(alpha, 0.9964, 4)

    def test_calculate_mean(self):
        data = pd.DataFrame({'METP1': [2.0, 2.0, 2.0],
                            'METN1': [0.6, 1.2, 1.2],
                             'MEFP1': [-0.2, -0.2, -0.2],
                             'MEFN1': [-2.0, -2.0, -2.0],
                             'MEREJ1': [-0.6, -0.6, -0.6],
                             'S100TP1': [20.0, 20.0, 20.0],
                             'S100TN1': [8.0, 12.0, 12.0],
                             'S100FP1': [-2.0, -2.0, -2.0],
                             'S100FN1': [-20.0, -20.0, -20.0],
                             'S100REJ1': [-6.0, -6.0, -6.0],
                             'Hateful_METP1': [True, False, False],
                             'Hateful_S100TP1': [False, False, False]
                             })

        mean = Analysis.calculate_mean(data, scale='ME', type='TP')
        self.assertEqual(mean, 2.0)

        mean = Analysis.calculate_mean(data, scale='ME', type='TN')
        self.assertEqual(mean, 1.0)

        mean = Analysis.calculate_mean(data, scale='ME', type='FP')
        self.assertEqual(mean, -0.2)

        mean = Analysis.calculate_mean(data, scale='ME', type='FN')
        self.assertEqual(mean, -2.0)

        mean = Analysis.calculate_mean(data, scale='ME', type='REJ')
        self.assertEqual(mean, -0.6)

        mean = Analysis.calculate_mean(data, scale='S100', type='TP')
        self.assertEqual(mean, 20.0)

        mean = Analysis.calculate_mean(data, scale='S100', type='TN')
        self.assertAlmostEqual(mean, 10.667, 3)

        mean = Analysis.calculate_mean(data, scale='S100', type='FP')
        self.assertEqual(mean, -2.0)

        mean = Analysis.calculate_mean(data, scale='S100', type='FN')
        self.assertEqual(mean, -20.0)

        mean = Analysis.calculate_mean(data, scale='S100', type='REJ')
        self.assertEqual(mean, -6.0)

    def test_filter_slow_subjects(self):
        data = pd.DataFrame({'startdate.': ['2022-06-12 12:00:0', '2022-06-12 12:00:0', '2022-06-12 12:00:0'],
                             'submitdate.': ['2022-06-12 13:00:0', '2022-06-12 13:00:0', '2022-06-12 13:00:0']})

        durations = Analysis.append_durations(
            data).filter(regex="duration")

        expected = pd.DataFrame({'duration': [3600.0, 3600.0, 3600.0]})

        self.assertTrue(durations.equals(expected))

        data = pd.DataFrame({'startdate.': ['2022-06-12 12:00:0'] * 100,
                             'submitdate.': ['2022-06-12 13:00:0'] * 99 + ['2022-06-12 12:01:0']})

        durations = Analysis.append_durations(
            data).filter(regex="duration")

        expected = pd.DataFrame({'duration': [3600.0] * 99 + [None]})

        # The last subject contains a None value because it's more than 3 times the stdv below the mean duration.
        self.assertTrue(durations.equals(expected))

    def test_convert_to_boxplot_data(self):
        data = pd.DataFrame({'METP1': [1.5, 2.0, 2.5],
                             'S100TP1': [20.0, 25.0, 35.0],
                            'Hateful_METP1': [True, False, False],
                             'Hateful_S100TP1': [False, False, False]
                             })

        plot_data = Analysis.convert_to_boxplot_data(data)

        expected = pd.DataFrame({'(Dis)agreement': [1.5, 2.0, 2.5, 20.0, 25.0, 35.0],
                                 'Scale': ["ME", "ME", "ME", "100-level", "100-level", "100-level"],
                                 'Scenario': ["TP1", "TP1", "TP1", "TP1", "TP1", "TP1"]
                                 })

        self.assertTrue(plot_data.equals(expected))


if __name__ == '__main__':
    unittest.main()
