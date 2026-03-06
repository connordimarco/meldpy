"""
l1_coordinates.py
-----------------
Coordinate transformation utilities.

Currently provides GSE -> GSM rotation via SpacePy.
The transformation is applied in-place on a DataFrame's named columns.
"""
import spacepy.coordinates as sc

from spacepy.time import Ticktock


def gse_to_gsm(df, cols):
    """Rotate vector columns from GSE to GSM in-place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose index is a DatetimeIndex.
    cols : list[str]
        Three column names representing the X, Y, Z components.

    Returns
    -------
    df : pd.DataFrame
        The same DataFrame with the named columns overwritten in GSM.
    """
    if df.empty:
        return df

    try:
        # SpacePy wants vectors as Nx3 and matching timestamps.
        vec_gse = df[cols].values
        times = df.index.to_pydatetime()
        t = Ticktock(times, 'UTC')
        # Convert coordinate frame in one shot.
        c_gse = sc.Coords(vec_gse, 'GSE', 'car', ticks=t)
        c_gsm = c_gse.convert('GSM', 'car')
        vec_gsm = c_gsm.data

        # Write converted vectors back into the same columns.
        df[cols[0]] = vec_gsm[:, 0]
        df[cols[1]] = vec_gsm[:, 1]
        df[cols[2]] = vec_gsm[:, 2]
    except Exception as e:
        print(f"Error in Coordinate Transformation: {e}")

    return df
