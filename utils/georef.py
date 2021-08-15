"""Set of functions that allow for plotting of static and interactive maps."""
import os
import pandas as pd
import geopandas as gpd

# ignore pandas warning
pd.options.mode.chained_assignment = None


def read_ibge_shapes(shp_path=''):
    """Read desired IBGE shapes (municipio) as geopandas dataframe.

    Parameters
    ----------
    shp_path : string
        Local path from where the shapes file shall be loaded.
        Default: empty string (''), denoting the current directory.

    Returns
    -------
    geodf : geopandas DataFrame
        Dataframe with shapes names, codes and respective polygons.

    """
    shp_file = 'LM_MUNICIPIO_2007.shp'
    geodf = gpd.read_file(os.path.join(shp_path, shp_file))  
    geodf = geodf.rename({'nm_nng': 'nome'}, axis=1)

    return geodf


def check_links(df, geodf, to_link):
    """Perform consistent checks between provided df and geodf.

    This function checks the dtype of the linkage columns,
    make sure that their values present the expected lengths,
    make sure that all rows in geodf have a non-missing geocode,
    and fix the length of geodf['geocodico] for a special case.

    Finally, it also merges df with geodf and returns the latter.

    Parameters
    ----------
    df : pandas DataFrame
        Original dataframe (df) with to_link column to be checked.

    geodf : geopandas DataFrame
        Original geoDataFrame (geodf) with geocodico column to be checked.

    to_link : string
        Name of the df column that will be used as key to merge with geodf.
        Thus, the data in this column should present the respective codes.

    Returns
    -------
    geodf: geopandas DataFrame
        Fixed geoDataFrame merged with pandas DataFrame.

    """
    # to_link expected to be string (as in geodf)
    if str(df[to_link].dtypes) in ['int64', 'float64']:
        df[to_link] = df[to_link].astype('str')

    # force all geodf columns to be lower-case
    geodf.columns = geodf.columns.str.lower()

    # municipality codes can have 6 or 7 digits
    lenref = [6, 7]
    left_link = 'geocodigo'

    # make sure the median number of digits in to_link has the expected length
    med_len = df[to_link].dropna().apply(lambda x: len(x)).median()
    assert(med_len in lenref), 'Column to link has unexpected digits length.'

    # drop eventual missing geocode info
    geodf = geodf[~geodf[left_link].isna()]

    # for the particular case in which to_link has 6 digits,
    # we need to adapt the geocodico column to match it
    # by simply removing the last digit (as not required)
    if (med_len == 6):
        geodf[left_link] = geodf[left_link].apply(lambda x: str(x)[:-1])

    # make sure crs attribute is properly defined
    geodf.crs = {'proj': 'longlat', 'ellps': 'GRS80', 'no_defs': True}

    # merge geodf with provided dataframe
    geodf = geodf.merge(df, left_on=left_link, right_on=to_link, how='inner')

    return geodf
