"""Utilitiy functions to deal cnr Classes."""

import numpy as np
import pandas as pd


def predict_response(rloc, rpert):
    """Predict response of a rperturbation based on local response matrix.

    Parameter:
    ---------
    rloc : (dict of) pandas.DataFrame or numpy.array

    rpert : (dict of) pandas.DataFrame or numpy.array

    Returns
    -------
    pd.DataFrame or np.array depending on input
    """
    # Check if dimensions are OK
    if isinstance(rloc, dict):
        prediction = dict()
        assert isinstance(rpert, dict)
        assert rloc.keys() == rpert.keys()
        for key in rloc.keys():
            assert (np.shape(rloc[key])[0] ==
                    np.shape(rloc[key])[1] == len(rpert[key]))
            prediction[key] = - np.dot(np.linalg.inv(rloc[key]), rpert[key])
            # Change array back to df if input was df
            if isinstance(rloc[key], pd.DataFrame):
                prediction[key] = pd.DataFrame(prediction[key],
                                               index=rloc[key].index,
                                               columns=rpert[key].columns)
    else:
        assert np.shape(rloc)[0] == np.shape(rloc)[1] == len(rpert)
        prediction = - np.dot(np.linalg.inv(rloc), rpert)
        if isinstance(rloc, pd.DataFrame):
            prediction = pd.DataFrame(prediction,
                                      index=rloc.index,
                                      columns=rpert.columns)

    return prediction


def error(rglob_predicted, rglob, sum=False):
    """Calculate error in prediction.

    Parameters:
    rglob_predicted : np.array like or dict thereof

    rglob : np.array like or dict thereof
        Measured responses

    Optional:
    sum (bool): Default = True

    Returns:
        If sum = True, (dictionary of) sum of squares of errors
        If sum = False, (dictionary of) np.array with error terms per
            response element

    """
    assert sum in [True, False]
    # If a dictionary is provided, get error of all entries
    if isinstance(rglob_predicted, dict):
        assert isinstance(rglob, dict)
        assert set(rglob_predicted.keys()) == set(rglob.keys())

        err = dict()
        for key in rglob.keys():
            err[key] = rglob_predicted[key] - rglob[key]
            if sum:
                err[key] = np.sum(np.square(np.array(err[key])))
    else:
        err = rglob_predicted - rglob
        if sum:
            err = np.sum(np.square(np.array(err)))
    return err
