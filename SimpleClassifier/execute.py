from dataset.sportsmans_height import Sportsmanheight
from model.simple_classifier import Classifier
import numpy as np
import pandas as pd
from config.cfg import cfg
import plotly.graph_objects as go
import copy


def prepare_data():
    dataset = Sportsmanheight()()
    confidence = Classifier()(dataset['height'])
    right_class = dataset['class']

    indices = confidence.argsort()
    indices = indices[::-1]
    confidence = confidence[indices]
    right_class = right_class[indices]
    return confidence, right_class


def find_tp_tn_fp_fn(confidence: np.ndarray, right_class: np.ndarray):
    unique_confidence = np.unique(confidence)[::-1]
    tp = np.zeros(unique_confidence.size)
    tn = np.zeros(unique_confidence.size)
    fp = np.zeros(unique_confidence.size)
    fn = np.zeros(unique_confidence.size)

    for i in range(unique_confidence.size):
        threshold = unique_confidence[i]
        prediction = np.zeros(right_class.size)
        prediction[confidence >= threshold] = 1
        tp[i] = np.count_nonzero((prediction == 1) & (right_class == 1))
        tn[i] = np.count_nonzero((prediction == 0) & (right_class == 0))
        fp[i] = np.count_nonzero((prediction == 1) & (right_class == 0))
        fn[i] = np.count_nonzero((prediction == 0) & (right_class == 1))

    return tp, tn, fp, fn, unique_confidence


def calculate_metrics(tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    a = fp / (fp + tn)
    b = fn / (fn + tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    f1_score[0] = 0
    return accuracy, a, b, precision, recall, f1_score


def visualise_pr_curve(precision: np.ndarray, recall: np.ndarray, accuracy: np.ndarray,
                       f1_score: np.ndarray, confidence: np.ndarray, plot_title=''):
    text = []
    for i in range(accuracy.size):
        text.append('Accuracy: ' + str(accuracy[i]) + '<br>F1 scope: '
                    + str(f1_score[i]) + '<br>Confidence: ' + str(confidence[i]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall,
                             y=precision,
                             hovertext=text,
                             mode='lines',
                             name='PR curve'))
    fig.update_layout(title=plot_title,
                      xaxis_title='Recall',
                      yaxis_title='Precision')
    fig.update_traces(hoverinfo='all',
                      hovertemplate='Recall: %{x}<br>Precision: %{y}<br>Accuracy: %{hovertext}')
    fig.show()


def make_steps(precision: np.ndarray):
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    return precision


def calculate_area(x: np.ndarray, y: np.ndarray):
    return np.trapz(y, x)


def visualise_roc_curve(a: np.ndarray, b: np.ndarray, plot_title=''):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a,
                             y=1 - b,
                             hovertext=b,
                             mode='lines',
                             name='ROC curve'))
    fig.update_layout(title=plot_title,
                      xaxis_title='alpha',
                      yaxis_title='1 - beta')
    fig.update_traces(hoverinfo='all',
                      hovertemplate='Alpha: %{x}<br>Beta: %{hovertext}')
    fig.show()


if __name__ == '__main__':
    confidence_arr, right_class_arr = prepare_data()
    tp_arr, tn_arr, fp_arr, fn_arr, confidence_arr = find_tp_tn_fp_fn(confidence_arr, right_class_arr)
    accuracy_arr, a_arr, b_arr, precision_arr, recall_arr, f1_score_arr = calculate_metrics(tp_arr, tn_arr, fp_arr,
                                                                                            fn_arr)

    recall_arr = np.append(recall_arr, 1)
    precision_arr = np.append(precision_arr, 0)
    accuracy_arr = np.append(accuracy_arr, 0.5)
    f1_score_arr = np.append(f1_score_arr, 0)
    confidence_arr = np.append(confidence_arr, 0)

    precision_arr = make_steps(precision_arr)
    visualise_pr_curve(precision=precision_arr,
                       recall=recall_arr,
                       accuracy=accuracy_arr,
                       f1_score=f1_score_arr,
                       confidence=confidence_arr,
                       plot_title=f'Precision-Recall Curve. Area = {round(calculate_area(recall_arr, precision_arr), 4)}')

    visualise_roc_curve(a=a_arr,
                        b=b_arr,
                        plot_title=f'ROC Curve. Area = {round(calculate_area(a_arr, 1 - b_arr), 4)}')
