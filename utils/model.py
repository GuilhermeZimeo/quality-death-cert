"""Viz methods to evaluate predictive models behavior and performance."""

import pandas as pd
import plotly.graph_objs as go

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve
)


STD_COLOR = 'rgb(250, 250, 250)'
POS_CLASS = 'Positive class'
NEG_CLASS = 'Negative class'


def curve_auc(clf, X_val, y_val,
              plot_type='roc',
              backgr_color=STD_COLOR):
    """Plot ROC or Precision-Recall curve as well as the area under the curve.

    It computes the probabilities on X_val, compares with y_val and
    then plots the desired curve ('roc' or 'pr') with respective auc.

    Parameters
    ----------
    clf : classifier object
        classifier object : a fitted classifier that
        presents the predict_proba method.       

    X_val : array-like
        Training-val array (array, DataFrame, scipy cr matrix, etc).

    y_val: array-like
        Target-vat array (array, pandas Series, pandas DataFrame

    plot_type : string
        The desired plot - currently supported: 'roc' or 'pr'.

    backgr_color : string
        String to define the background color of the plot.
        For example, use 'rgb(250, 250, 250)' for gray color.

    Returns
    -------
    data : list of go.Scattergl objects
        All data to be plotted (e.g. plot of interest, diagnonal line).

    layout : go.Layout object
        The layout of the plot (labels, color).

    """
    # retrieve the predicted probabilities for X_val
    pred_array = clf.predict_proba(X_val)

    pred_df = pd.DataFrame(pred_array)
    pred_prob = pred_df.iloc[:, 1]

    data = list()

    # considering the desired case, get data and compute AUC
    if plot_type == 'roc':
        x_data, y_data, _ = roc_curve(y_val, pred_prob)
        y_label = 'True Positive Rate'
        x_label = 'False Positive Rate'
        title = 'ROC Curve'
        ref_value = 0.5
        x_line = [0, 1]
        y_line = [0, 1]
        auc = roc_auc_score(y_val, pred_prob)

    elif plot_type == 'pr':
        y_data, x_data, _ = precision_recall_curve(y_val, pred_prob)
        y_label = 'Precision'
        x_label = 'Recall'
        title = 'Precision-Recall Curve'
        # baseline -> proportion of positive class: p/(p+n)
        ref_value = y_val.value_counts(normalize=True)[1]
        x_line = [0, 1]
        y_line = [ref_value, ref_value]
        auc = average_precision_score(y_val, pred_prob)

    else:
        raise Exception(f'Unknown plot_type: {plot_type}')

    # plot curve of interest
    data.append(go.Scattergl(
        x=x_data,
        y=y_data,
        hoverinfo='x, y',
        mode='lines',
        name=f'AUC: {auc:.2f}',
        line=dict(color='black')
    ))

    # plot reference diagonal line
    data.append(go.Scattergl(
        x=x_line,
        y=y_line,
        mode='lines',
        name=f'Ref: {ref_value:.2f}',
        line=dict(color='black', dash='dash')
    ))

    # set the layout of plot
    layout = go.Layout(
        xaxis=dict(title=x_label, hoverformat='.0%'),
        yaxis=dict(title=y_label, hoverformat='.0%'),
        title=dict(text=title),
        plot_bgcolor=backgr_color,
        paper_bgcolor=backgr_color,
    )

    return data, layout


def probs_hist(clf, X_val, y_val,
               backgr_color=STD_COLOR):
    """Plot histograms depicting the predicted probabilities per class.

    Parameters
    ----------
    clf : classifier object
        A fitted classifier that presents the predict_proba method.

    X_val : array-like
        Validation-set data (array, DataFrame, scipy cr matrix, etc).

    y_val: array-like
        Validation-set target (array, pandas Series).

    backgr_color : string
        String to define the background color of the plot.
        For example, use 'rgb(250, 250, 250)' for gray color.

    Returns
    -------
    data : list of go.Histogram objects
        All data to be plotted (e.g. plot of interest, diagnonal line).

    layout : go.Layout object
        The layout of the plot (labels, color).

    """
    # retrieve the predicted probabilities for X_val
    pred_array = clf.predict_proba(X_val)

    pred_df = pd.DataFrame(pred_array)
    pred_prob = pred_df.iloc[:, 1]

    results = pd.DataFrame()
    results['prob'] = pred_prob
    results['label'] = y_val.values

    results['color'] = results['label'].apply(
        lambda x: 'red' if x else 'blue')

    results['label'] = results['label'].apply(
        lambda x: 'Label: ' + str(x))

    title = dict(text='Probabilities distribution per original class')
    xaxis_start = 0
    xaxis_end = 1
    xaxis_size = 0.05
    thr = 0.5

    data = [go.Histogram(
        x=results[results['label'] == label]['prob'],
        name=label,
        histnorm='percent',
        hoverinfo='y, name, x',
        autobinx=False,
        xbins=dict(
            start=xaxis_start,
            end=xaxis_end,
            size=xaxis_size
        ),
        opacity=0.5,
        marker=dict(color=results[results['label'] == label]['color'])
    ) for label in results['label'].dropna().unique()]

    layout = go.Layout(
        title=title,
        xaxis=dict(title='Predicted Prob.',
                   range=[xaxis_start, xaxis_end + xaxis_size]),
        yaxis=dict(title='Percent (per class)', hoverformat='.1f'),
        barmode='overlay',
        showlegend=True,
        shapes=[{
            'type': 'line',
            'x0': thr,
            'x1': thr,
            'y0': 0,
            'y1': 100,
            'line': {'color': 'gold', 'dash': 'dash'}
        }],
        plot_bgcolor=backgr_color,
        paper_bgcolor=backgr_color
    )

    return data, layout
