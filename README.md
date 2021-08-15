# Factors associated with the quality of death certification in Brazilian municipalities: a data-driven non-linear model

The provided scripts aim to reproduce the model (fit_data.py) and the figures (plot_data.py) as described in the paper.

For that, we provide the preprocessed data under /data/preproc/ as well as the IBGE shapes file under /data/shapes/ .

Also, under /utils/ one may find further Python scripts that are necessary to run the code.

After running the scripts, the figures - as in the paper - shall be available under /data/figures/ .

Note: the scripts were tested on 2021-08-01 with following Python (3.8.10) main packages:

- geopandas 0.9.0
- lightgbm 3.2.0
- matplotlib 3.3.4
- pandas 1.2.2
- plotly 4.14.3
- plotly-express 0.4.1
- scikit-learn 0.24.1
- scipy 1.6.2
- shap 0.39.0
