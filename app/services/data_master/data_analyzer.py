import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd


class DataAnalyzer:
    """Class for analyzing results"""

    @staticmethod
    def analyze(X, y_true, y_pred, keys):
        """
        Creates colored table and save it on path "table_save_path" (.xlsx)
        Creates bar chart and save it on path "diagram_save_path" (.png)
        """

        predicted = []
        actual = []

        for words, true_tags, pred_tags in zip(X, y_true, y_pred):

            predicted_example = dict.fromkeys(keys, '')
            actual_example = dict.fromkeys(keys, '')

            for word, true_tag, pred_tag in zip(words, true_tags, pred_tags):
                if true_tag in actual_example.keys():
                    actual_example[true_tag] += f'{word} '

                if pred_tag in predicted_example.keys():
                    predicted_example[pred_tag] += f'{word} '

            actual.append({key: value.rstrip() for key, value in actual_example.items()})
            predicted.append({key: value.rstrip() for key, value in predicted_example.items()})

        df_actual = pd.DataFrame({key: [wine.get(key) for wine in actual] for key in keys})
        df_predicted = pd.DataFrame({key: [wine.get(key) for wine in predicted] for key in keys})

        matches = dict.fromkeys(keys + ['All'], 0)
        matched_indices = []
        false_negative = dict.fromkeys(keys, 0)
        false_negative_indices = []
        false_positive = dict.fromkeys(keys, 0)
        false_positive_indices = []

        for index, row in df_predicted.iterrows():

            flag_all = True

            for col_index, column in enumerate(keys):

                if row[column] == df_actual.iloc[index][column]:

                    matches[column] += 1

                    matched_indices.append((index, col_index))
                else:

                    flag_all = False

                    if df_actual.iloc[index][column]:

                        false_negative[column] += 1

                        false_negative_indices.append((index, col_index))
                    else:
                        false_positive[column] += 1

                        false_positive_indices.append((index, col_index))

            if flag_all:
                matches['All'] += 1

        for key in matches:

            if key == 'All': continue

            false_positive[key] += matches[key]

            false_negative[key] += false_positive[key]

        style.use('seaborn-v0_8-darkgrid')

        fig = plt.figure(figsize=(18, 8))

        index = list(range(1, len(matches) + 1))

        index2 = list(range(1, len(false_positive) + 1))

        analyze_res = dict(sorted(matches.items(), key=lambda x: x[1], reverse=True))

        false_positive = {key: false_positive[key] for key in analyze_res if key != 'All'}

        false_negative = {key: false_negative[key] for key in analyze_res if key != 'All'}

        plt.barh(index2, false_negative.values(), 0.6, label='Кол-во ложноотрицательных ошибок', color='tab:blue')
        plt.barh(index2, false_positive.values(), 0.6, label='Кол-во ложноположительных ошибок', color='tab:orange')
        plt.barh(index, analyze_res.values(), 0.6, label='Кол-во совпадений', color='tab:green')

        plt.title('Результаты')

        plt.yticks(index, analyze_res.keys())

        fig.legend(loc='upper right')

        for index, (key, value) in enumerate(analyze_res.items()):

            if value >= 100: plt.text(0, index + .8, str(value))

            if key != 'All':

                false_negative[key] -= false_positive[key]

                false_positive[key] -= value

                if false_positive[key] >= 100:
                    plt.text(value, index + .8, str(false_positive[key]))

                if false_negative[key] >= 100:
                    plt.text(value + false_positive[key], index + .8, str(false_negative[key]))

        return df_predicted, df_actual, fig, matched_indices, false_positive_indices, false_negative_indices
