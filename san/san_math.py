import numpy as np

def mat_inv_2x2(array):
    """compute inverse of 2x2 matrix, broadcasted

    Parameters
    ----------
    array : numpy.ndarray
        array containing 2x2 matrices in last dimension. Returns inverse array of shape array.shape

    Returns
    -------
    matinv
        matrix inverse array
    """

    a = array[..., 0, 0]
    b = array[..., 0, 1]
    c = array[..., 1, 0]
    d = array[..., 1, 1]

    det = a * d - b * c

    matinv = np.array([[d, -b], [-c, a]]) / det
    if matinv.ndim > 2:
        for i in range(matinv.ndim - 2):
            matinv = np.moveaxis(matinv, -1, 0)

    return matinv


def mat_inv_3x3(array):
    """compute inverse of 3x3 matrix, broadcasted

    Parameters
    ----------
    array : numpy.ndarray
        array containing 3x3 matrices in last dimension. Returns inverse array of shape array.shape

    Returns
    -------
    matinv
        matrix inverse array
    """

    a = array[..., 0, 0]  # row 1
    b = array[..., 0, 1]
    c = array[..., 0, 2]

    d = array[..., 1, 0]  # row 2
    e = array[..., 1, 1]
    f = array[..., 1, 2]

    g = array[..., 2, 0]  # row 3
    h = array[..., 2, 1]
    i = array[..., 2, 2]

    # determine cofactor elements
    ac = e * i - f * h
    bc = -(d * i - f * g)
    cc = d * h - e * g
    dc = -(b * i - c * h)
    ec = a * i - c * g
    fc = -(a * h - b * g)
    gc = b * f - c * e
    hc = -(a * f - c * d)
    ic = a * e - b * d

    # get determinant
    det = a * ac + b * bc + c * cc  # second term's negative is included in cofactor term
    det = det[..., np.newaxis, np.newaxis]

    # Assemble adjucate matrix (transpose of cofactor)
    arrayinv = np.asarray([[ac, bc, cc], [dc, ec, fc], [gc, hc, ic]]).T / det

    return arrayinv
