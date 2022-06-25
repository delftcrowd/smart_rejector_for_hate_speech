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

        expected = pd.DataFrame({'TP1': [10.0, -50.0, 0.0],
                                 'TN1': [10.0, -50.0, 0.0],
                                 'FP1': [10.0, -50.0, 0.0],
                                 'FN1': [10.0, -50.0, 0.0],
                                 'REJ1': [10.0, -50.0, 0.0]
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

        expected = pd.DataFrame({'TP1': [10.0, -50.0, 0.0],
                                 'TN1': [10.0, -50.0, 0.0],
                                 'FP1': [10.0, -50.0, 0.0],
                                 'FN1': [10.0, -50.0, 0.0],
                                 'REJ1': [10.0, -50.0, 0.0]
                                 })
        s100 = Analysis.s100_values(data=df, num_scenarios=1)
        self.assertTrue(s100.equals(expected))

    def test_hatefulness(self):
        df_mes = pd.DataFrame({'METP1h.': ['Hateful', 'Not hateful', 'Not hateful'],
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

        df_s100 = pd.DataFrame({
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

        expected = pd.DataFrame({'Hateful_TP1': [1.0, 0.0, 0.0],
                                 'Hateful_TN1': [1.0, 0.0, 0.0],
                                 'Hateful_FP1': [1.0, 0.0, 0.0],
                                 'Hateful_FN1': [1.0, 0.0, 0.0],
                                 'Hateful_REJ1': [1.0, 0.0, 0.0]})

        hatefulness = Analysis.hatefulness(data=df_mes, scale="ME", num_scenarios=1)
        self.assertTrue(hatefulness.equals(expected))

        hatefulness = Analysis.hatefulness(data=df_s100, scale="S100", num_scenarios=1)
        self.assertTrue(hatefulness.equals(expected))

    def test_attention_checks(self):
        df = pd.DataFrame({'attention1.': ['Blue', 'Red', 'Blue'],
                           'attention2.': ['Orange', 'Orange', 'Blue']
                           })

        expected = pd.DataFrame({'attention_checks_passed': [1.0, 0.0, 0.0]})

        attention_checks = Analysis.attention_checks(data=df)
        self.assertTrue(attention_checks.equals(expected))
        self.assertTrue(Analysis.any_failed_attention_checks(expected))

        none_failed = pd.DataFrame(
            {'attention_checks_passed': [1.0, 1.0, 1.0]})
        self.assertFalse(Analysis.any_failed_attention_checks(none_failed))

    def test_normalize(self):
        data = pd.DataFrame({'G20Q51[SQ001].': [-100, -10, -1],
                             'G20Q51[SQ002].': [-50, -5, -0.5],
                             'G20Q51[SQ003].': [-10, -1, -0.1],
                             'G20Q51[SQ005].': [10, 1, 0.1],
                             'G20Q51[SQ006].': [30, 3, 0.3],
                             'G20Q51[SQ007].': [100, 10, 1],
                             })

        mes = pd.DataFrame({'TP1': [100.0, 10.0, 1.0],
                            'TN1': [60.0, 6.0, 0.6],
                            'FP1': [-10.0, -1.0, -0.1],
                            'FN1': [-100.0, -10.0, -1],
                            'REJ1': [-30.0, -3.0, -0.3]
                            })

        expected = pd.DataFrame({'TP1': [1.0, 1.0, 1.0],
                                 'TN1': [0.6, 0.6, 0.6],
                                 'FP1': [-0.1, -0.1, -0.1],
                                 'FN1': [-1.0, -1.0, -1.0],
                                 'REJ1': [-0.3, -0.3, -0.3]
                                 })

        normalized_mes = Analysis.normalize(data, mes)

        self.assertTrue(normalized_mes.equals(expected))

    def test_reliabilty(self):
        mes = pd.DataFrame({'TP1': [2.0, 2.0, 2.0],
                            'TP2': [2.0, 2.0, 2.0],
                            'TN1': [3.0, 3.0, 3.0],
                            'TN2': [4.0, 4.0, 4.0],
                            'FP1': [-0.2, -0.2, -0.2],
                            'FP2': [-0.3, -0.3, -0.3],
                            'FN1': [-2.0, -2.0, -2.0],
                            'FN2': [-2.0, -2.0, -2.0],
                            'REJ1': [-0.5, -0.6, -0.6],
                            'REJ2': [-0.6, -0.6, -0.6],
                            'Hateful_TP1': [True, False, False]})

        s100 = pd.DataFrame({
                            'TP1': [20.0, 20.0, 20.0],
                            'TP2': [20.0, 20.0, 20.0],
                            'TN1': [8.0, 12.0, 12.0],
                            'TN2': [8.0, 12.0, 12.0],
                            'FP1': [-2.0, -2.0, -2.0],
                            'FP2': [-3.0, -3.0, -3.0],
                            'FN1': [-19.0, -16.0, -17.0],
                            'FN2': [-51.0, -55.0, -45.0],
                            'REJ1': [-6.0, -6.0, -6.0],
                            'REJ2': [-6.0, -6.0, -6.0],
                            'Hateful_FN1': [False, False, False]
                            })

        alpha = Analysis.reliability(mes, scale="ME", type="TN")
        self.assertEqual(alpha, 1.0)

        alpha = Analysis.reliability(mes, scale="S100", type="FP")
        self.assertEqual(alpha, 1.0)

        alpha = Analysis.reliability(s100, scale="S100", type="FN")
        self.assertAlmostEqual(alpha, 0.9590, 4)

        alpha = Analysis.reliability(mes, scale="ME")
        self.assertAlmostEqual(alpha, 0.9997, 4)

        alpha = Analysis.reliability(s100, scale="S100")
        self.assertAlmostEqual(alpha, 0.9905, 4)

        mes = pd.DataFrame({'TN1': [1.2, 1.2, 1.2],
                            'TN2': [0.8, 0.8, 0.8]})

        s100 = pd.DataFrame({
                            'TN1': [12.0, 12.0, 12.0],
                            'TN2': [5.5, 6.0, 6.0]
                            })

        alpha = Analysis.reliability(mes, scale="ME", type="TN")
        self.assertEqual(alpha, 1.0)

        alpha = Analysis.reliability(s100, scale="S100", type="TN")
        self.assertAlmostEqual(alpha, 0.9964, 4)

    def test_calculate_mean(self):
        mes = pd.DataFrame({'TP1': [2.0, 2.0, 2.0],
                            'TN1': [0.6, 1.2, 1.2],
                            'FP1': [-0.2, -0.2, -0.2],
                            'FN1': [-2.0, -2.0, -2.0],
                            'REJ1': [-0.6, -0.6, -0.6],
                            'Hateful_TP1': [True, False, False]})

        s100 = pd.DataFrame({
            'TP1': [20.0, 20.0, 20.0],
            'TN1': [8.0, 12.0, 12.0],
            'FP1': [-2.0, -2.0, -2.0],
            'FN1': [-20.0, -20.0, -20.0],
            'REJ1': [-6.0, -6.0, -6.0],
            'Hateful_TP1': [False, False, False]
        })

        mean = Analysis.calculate_mean(mes,  type='TP')
        self.assertEqual(mean, 2.0)

        mean = Analysis.calculate_mean(mes,  type='TN')
        self.assertEqual(mean, 1.0)

        mean = Analysis.calculate_mean(mes,  type='FP')
        self.assertEqual(mean, -0.2)

        mean = Analysis.calculate_mean(mes,  type='FN')
        self.assertEqual(mean, -2.0)

        mean = Analysis.calculate_mean(mes,  type='REJ')
        self.assertEqual(mean, -0.6)

        mean = Analysis.calculate_mean(s100,  type='TP')
        self.assertEqual(mean, 20.0)

        mean = Analysis.calculate_mean(s100,  type='TN')
        self.assertAlmostEqual(mean, 10.667, 3)

        mean = Analysis.calculate_mean(s100,  type='FP')
        self.assertEqual(mean, -2.0)

        mean = Analysis.calculate_mean(s100, type='FN')
        self.assertEqual(mean, -20.0)

        mean = Analysis.calculate_mean(s100, type='REJ')
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

    def test_convert_to_dual_boxplot_data_individual(self):
        mes = pd.DataFrame({
            'TP1': [1.5, 2.0, 2.5],
            'Hateful_TP1': [True, False, False]
        })

        s100 = pd.DataFrame({
            'TP1': [20.0, 25.0, 35.0],
            'Hateful_TP1': [False, False, False]
        })

        plot_data = Analysis.convert_to_dual_boxplot_data(data_mes=mes, data_s100=s100, show_individual=True)

        expected = pd.DataFrame({'(Dis)agreement': [150.0, 200.0, 250.0, 20.0, 25.0, 35.0],
                                'Scenario': ["TP1", "TP1", "TP1", "TP1", "TP1", "TP1"],
                                 'Scale': ["ME", "ME", "ME", "100-level", "100-level", "100-level"],

                                 })

        self.assertTrue(plot_data.equals(expected))

    def test_convert_to_dual_boxplot_data_grouped(self):
        mes = pd.DataFrame({
            'TP1': [1.5, 2.0, 2.5],
            'Hateful_TP1': [True, False, False]
        })

        s100 = pd.DataFrame({
            'TP1': [20.0, 25.0, 35.0],
            'Hateful_TP1': [False, False, False]
        })

        plot_data = Analysis.convert_to_dual_boxplot_data(data_mes=mes, data_s100=s100, show_individual=False)

        expected = pd.DataFrame({'(Dis)agreement': [150.0, 200.0, 250.0, 20.0, 25.0, 35.0],
                                 'Scenario': ["TP", "TP", "TP", "TP", "TP", "TP"],
                                 'Scale': ["ME", "ME", "ME", "100-level", "100-level", "100-level"],
                                 })

        self.assertTrue(plot_data.equals(expected))

    def test_convert_to_boxplot_data_grouped(self):
        mes = pd.DataFrame({
            'TP1': [1.5, 2.0, 2.5],
            'Hateful_TP1': [True, False, False]
        })

        plot_data = Analysis.convert_to_boxplot_data(data=mes, show_individual=False)

        expected = pd.DataFrame({'(Dis)agreement': [1.5, 2.0, 2.5],
                                'Scenario': ["TP", "TP", "TP"]
                                 })

        self.assertTrue(plot_data.equals(expected))

    def test_convert_to_boxplot_data_individual(self):
        mes = pd.DataFrame({
            'TP1': [1.5, 2.0, 2.5],
            'Hateful_TP1': [True, False, False]
        })

        plot_data = Analysis.convert_to_boxplot_data(data=mes, show_individual=True)

        expected = pd.DataFrame({'(Dis)agreement': [1.5, 2.0, 2.5],
                                'Scenario': ["TP1", "TP1", "TP1"]
                                 })

        self.assertTrue(plot_data.equals(expected))

    def test_convert_to_stackedbar_data(self):
        mes = pd.DataFrame({
            'TP1': [1.5, 2.0, 2.5],
            'TP2': [1.5, 2.0, 2.5],
            'Hateful_TP1': [True, False, False],
            'Hateful_TP2': [False, False, False]
        })

        s100 = pd.DataFrame({
            'TP1': [20.0, 25.0, 35.0],
            'TP2': [20.0, 25.0, 35.0],
            'Hateful_TP1': [False, True, True],
            'Hateful_TP2': [False, False, False],
        })

        data = mes.append(s100)
        plot_data = Analysis.convert_to_stackedbar_data(data)
        expected = pd.DataFrame({'Scenario': ['TP1', 'TP2'],
                                 'Hateful': [50.0, 0.0], "Not hateful": [50.0, 100.0]})

        self.assertTrue(plot_data.equals(expected))


if __name__ == '__main__':
    unittest.main()
