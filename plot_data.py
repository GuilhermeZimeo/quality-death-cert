"""Create relevant figures."""
import os
import shap
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from PIL import Image
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from utils.georef import read_ibge_shapes, check_links
from utils.model import curve_auc, probs_hist


def main(args=None):
    """Create and store all figures."""
    df = pd.read_csv(os.path.join('data', 'preproc', 'merged_data.csv'))

    # Figures 1 - 3
    plot_binary_quality_map(df)
    plot_quality_boxplot(df)
    plot_quality_scatter(df)

    X_train, y_train, _ = read_features(split='train')
    best_estimator = read_best_estimator()
    X_check, y_check, _ = read_features(split='test')

    # Figure 4
    for plot_type in ['roc', 'pr']:
        plot_curve_auc(best_estimator, X_check, y_check, plot_type=plot_type)

    # Figures 5- 6
    plot_probs_hist(best_estimator, X_check, y_check)
    plot_shap_values(best_estimator, X_train, y_train, X_check, y_check)

    # Figure 7
    df_plot = plot_model_residuals(df, best_estimator)


def _export_figure(fig, pathout, filename):
    """Workaround to export plotly figure as tiff."""
    # ensure path exists
    if not os.path.exists(pathout):
        os.makedirs(pathout)

    fig.update_layout(width=1450, height=900)

    # firstly save as png
    filename = os.path.join(pathout, f'{filename}.png')
    fig.write_image(filename)

    # convert to tiff
    img = Image.open(filename)
    os.remove(filename)
    img.save(filename.replace('.png', '.tiff'), dpi=(300, 300))


def plot_binary_quality_map(df, col='Data Quality (%)'):
    """Create map of binary quality of death certificates per municipality."""
    # consider median value as threshold
    thres = df[col].median()
    # create binary column of interest
    df['binary'] = df[col].apply(
        lambda x: 'Poor quality' if x >= thres else 'Good quality')

    shp_path = os.path.join('data', 'shapes')
    to_link = 'CODMUNOCOR'
    # get geodf of interest
    geodf = read_ibge_shapes(shp_path)
    # make eventual adjustment of df and geodf links
    geodf = check_links(df, geodf, to_link)

    _, ax = plt.subplots(figsize=(1600/300, 1600/300))
    ax.axis('off')

    geodf.plot(color='white', edgecolor='black', linewidth=0.3, ax=ax)
    geodf.plot(
        column='binary', categorical=True, cmap='bwr',
        ax=ax, legend=True, alpha=0.7)

    # save figure
    pathout = os.path.join('data', 'figures')
    if not os.path.exists(pathout):
        os.makedirs(pathout)

    filename = 'Figure1.tiff'
    plt.savefig(
        os.path.join(pathout, filename), dpi=300)
    plt.clf()


def plot_quality_boxplot(df, col='Data Quality (%'):
    """Plot boxplot of quality (%) per variable over all municipalities."""
    # retrieve columns of interest depending on input col
    if 'Data' in col:
        value_vars = [
            col for col in df.columns if col.endswith('(%)') and
            (not col.startswith('Type')) and ('GBD' not in col)
        ]
        label = 'Poor quality (%) based on missing or non-expected values'
    elif 'GBD' in col:
        value_vars = [
            col for col in df.columns if col.endswith('(%)') and
            (col.startswith('Type') or ('GBD' in col))
        ]
        label = 'Poor quality (%) based on garbage codes'
    # create melt dataframe based on the columns of interest
    df_melt = df.melt(
        id_vars=['Municipality'],
        value_vars=value_vars,
        value_name='Quality (%)')

    # remove repetitive ' Quality (%)' from variable names
    df_melt['variable'] = df_melt['variable'].apply(
        lambda x: x.split(' Quality (%)')[0])

    df_melt['variable'].replace({
        'DTOBITO': 'Date of occurrence',
        'DTATESTADO': 'Date of registration',
        'DTNASC': 'Date of birth',
        'LOCOCOR': 'Place of occurrence',
        'ATESTANTE': 'Certifier',
        'SEXO': 'Sex',
        'ESTCIV': 'Marital status',
        'CAUSABAS_O': 'Cause of death',
        'Data': 'All',
        'GBD': 'All',
    }, inplace=True)

    fig = px.box(
        df_melt,
        x='variable',
        y='Quality (%)',
        labels={
            'Quality (%)': label,
            'variable': ''}
    )

    pathout = os.path.join('data', 'figures')

    filename = 'Figure2'
    fig.update_layout(
        plot_bgcolor='white',
        font=dict(size=24),
    )
    fig.update_yaxes(showgrid=True, gridcolor='LightGrey')

    _export_figure(fig, pathout, filename)


def plot_quality_scatter(df):
    """Scatter data and gbd quality stratified by GeoSES quartiles."""
    df_copy = df.copy()
    df_copy.dropna(subset=['GeoSES'], inplace=True)
    df_copy['GeoSES'] = df_copy['GeoSES'].round(2)
    col_q = 'GeoSES_Q'
    df_copy[col_q] = pd.qcut(df_copy['GeoSES'], 4)
    df_copy.sort_values(by=col_q, inplace=True)

    rename_dict = {}

    for quant in df_copy[col_q].unique():
        idx = (df_copy[col_q] == quant)
        rho = spearmanr(
            df_copy[idx]['Data Quality (%)'],
            df_copy[idx]['GBD Quality (%)'],
            nan_policy='omit')
        rename_dict[quant] = f'{quant}, rho={rho.correlation:.2f}'

    df_copy[col_q] = df_copy[col_q].cat.rename_categories(rename_dict)

    fig = px.scatter(
        df_copy,
        'Data Quality (%)',
        'GBD Quality (%)',
        facet_col=col_q,
        marginal_x='histogram',
        trendline='ols',
        facet_col_wrap=2,
        width=1200,
        height=800,
        labels={
            'GBD Quality (%)': 'Poor quality (%) based on garbage codes',
            'Data Quality (%)': 'Poor quality (%) based on missing or non-expected values',  # noqa
        })

    for i in ['', 3, 4]:
        fig['layout'][f'xaxis{i}']['title']['text'] = ''

    fig.update_layout({
        'plot_bgcolor': 'white',
    })
    fig.update_yaxes(showgrid=True, gridcolor='LightGrey')
    fig.update_xaxes(showgrid=True, gridcolor='LightGrey')

    pathout = os.path.join('data', 'figures')
    filename = 'Figure3'
    _export_figure(fig, pathout, filename)


def read_features(split='train'):
    """Read features related to split of interest (train, val or test)."""
    # note: also possible to read whole dataset if suffix is provided as None
    if split is None:
        suffix = ''
    else:
        suffix = f'_{split}'

    Xc = pd.read_csv(os.path.join('data', 'features', f'X{suffix}.csv'))
    yc = pd.read_csv(os.path.join('data', 'features', f'y{suffix}.csv'))
    gc = pd.read_csv(os.path.join('data', 'features', f'groups{suffix}.csv'))

    return Xc, yc, gc


def read_best_estimator():
    """Load best estimator to memory."""
    return joblib.load(os.path.join('data', 'model', 'best_estimator.pkl'))


def _get_feature_names(X_train):
    """Get feature names to be plotted."""
    features_dict = {
        'P_DOCTORS': 'Physicians density',
        'P_HOSPBEDS': 'Hospital beds density',
        'DEATH_RATE': 'Death rate in 2010',
        'URGEMERG': 'Facilities with emergency (%)',
        'VINC_SUS': 'Facilities with public care (%)'
    }

    names = []
    for feature in X_train.columns.tolist():
        if feature in features_dict.keys():
            names.append(features_dict[feature])
        else:
            names.append(feature)

    return names


def plot_curve_auc(best_estimator, X_test, y_test, plot_type='roc'):
    """Plot ROC or precision-recall curve and compute the AUC."""
    data, layout = curve_auc(
        best_estimator, X_test, y_test, plot_type=plot_type)
    layout.autosize = False
    layout.height = 600
    layout.width = 600
    layout.xaxis.gridcolor = 'rgb(200, 200, 200)'
    layout.yaxis.gridcolor = 'rgb(200, 200, 200)'
    fig = go.Figure(data, layout)

    fig.update_layout({
        'plot_bgcolor': 'white',
        'paper_bgcolor':'white',
    })
    fig.update_yaxes(showgrid=True, gridcolor='LightGrey')
    fig.update_xaxes(showgrid=True, gridcolor='LightGrey')

    pathout = os.path.join('data', 'figures')
    if plot_type == 'roc':
        suffix = 'a'
    else:
        suffix = 'b'
    filename = f'Figure4{suffix}'
    _export_figure(fig, pathout, filename)


def plot_probs_hist(best_estimator, X_test, y_test):
    """Plot the histogram of the model probabilities and expected outcome."""
    data, layout = probs_hist(best_estimator, X_test, y_test)
    data[0].name = 'Good quality'
    data[1].name = 'Poor quality'
    layout.autosize = False
    layout.yaxis.range = [0, 20]
    layout.xaxis.title.text = 'Predicted probability'
    fig = go.Figure(data, layout)

    fig.update_yaxes(showgrid=True, gridcolor='LightGrey')
    fig.update_xaxes(showgrid=True, gridcolor='LightGrey')

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=24),
    )

    pathout = os.path.join('data', 'figures')
    filename = 'Figure5'
    _export_figure(fig, pathout, filename)


def plot_shap_values(best_estimator, X_train, y_train, X_test, y_test):
    """Assess and plot the impact of each feature on shap values."""
    shap.initjs()

    best_estimator['estimator'].fit(X_train, y_train)

    explainer = shap.TreeExplainer(best_estimator['estimator'])
    shap_values = explainer.shap_values(X_test, y_test)

    names = _get_feature_names(X_test)

    shap.summary_plot(
        shap_values[1],
        X_test,
        feature_names=names,
        plot_type='dot',
        max_display=X_test.shape[1],
        show=False
    )

    # save figure
    pathout = os.path.join('data', 'figures')
    filename = os.path.join(pathout, 'Figure6.tiff')
    if not os.path.exists(pathout):
        os.makedirs(pathout)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.clf()


def plot_model_residuals(df, best_estimator):
    """Create map highlighting the residuals based on the predictive model."""
    df_copy = df.dropna(subset=['GeoSES']).reset_index()
    df_copy['UF'] = df_copy['UF_x'].copy()

    X, y, groups = read_features(split=None)
    group_kfold = GroupKFold(n_splits=5)

    # specify columns to be displayed by hover
    hover_info = [
        'Municipality', 'UF', 'Region',
        'Data Quality (%)', 'Probability', 'Residual']

    cols_to_keep = hover_info + ['CODMUNOCOR']
    df_plot = pd.DataFrame(columns=cols_to_keep)

    # use GroupKFold splits to estimate residuals in all possible test sets
    for train_index, test_index in group_kfold.split(X, y, groups):

        # fit estimator and predict on test data
        best_estimator['estimator'].fit(
            X.iloc[train_index, :], y.iloc[train_index])
        df_cur = df_copy.iloc[test_index, :].copy()

        # get model probability for class 1 (bad quality)
        y_prob = best_estimator['estimator'].predict_proba(
            X.iloc[test_index, :])
        df_cur['Probability'] = y_prob[:, 1].reshape(-1, 1)
        df_cur['y_test'] = y.iloc[test_index].copy()

        # compute residual estimation (class - probability)
        df_cur['Residual'] = df_cur['y_test'] - df_cur['Probability']

        df_plot = df_plot.append(df_cur[cols_to_keep], ignore_index=True)

    df_plot['CODMUNOCOR'] = df_plot['CODMUNOCOR'].astype(str)

    shp_path = os.path.join('data', 'shapes')
    to_link = 'CODMUNOCOR'

    df_plot['Residual_cat'] = df_plot['Residual'].apply(
        lambda x: -1 if x < -0.5 else -0.5 if x < 0 else 0.5 if x <= 0.5 else 1
    )
    # get geodf of interest
    geodf = read_ibge_shapes(shp_path)
    # make eventual adjustment of df and geodf links
    geodf = check_links(df_plot, geodf, to_link)

    _, ax = plt.subplots(figsize=(1600/300, 1600/300))
    ax.axis('off')
    geodf.plot(color='white', edgecolor='black', linewidth=0.3, ax=ax)
    geodf.plot(
        column='Residual_cat', categorical=True, cmap='bwr',
        ax=ax, legend=True, alpha=0.7)

    # save figure
    pathout = os.path.join('data', 'figures')
    if not os.path.exists(pathout):
        os.makedirs(pathout)

    filename = 'Figure7.tiff'
    plt.savefig(os.path.join(pathout, filename), dpi=300)
    plt.clf()

    return df_plot


if __name__ == '__main__':
    main()
