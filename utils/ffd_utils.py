import numpy as np
from scipy.special import comb
import warnings
warnings.filterwarnings("error")



def bernstein_poly(n, v, stu):
    coeff = comb(n, v)
    weights = coeff * ((1 - stu) ** (n - v)) * (stu ** v)
    return weights


def trivariate_bernstein(stu, lattice):
    if len(lattice.shape) != 4 or lattice.shape[3] != 3:
        raise ValueError('lattice must have shape (L, M, N, 3)')
    l, m, n = (d - 1 for d in lattice.shape[:3])
    lmn = np.array([l, m, n], dtype=np.int32)
    v = mesh3d(
        np.arange(l+1, dtype=np.int32),
        np.arange(m+1, dtype=np.int32),
        np.arange(n+1, dtype=np.int32),
        dtype=np.int32)
    stu = np.reshape(stu, (-1, 1, 1, 1, 3))
    weights = bernstein_poly(n=lmn, v=v, stu=stu)
    weights = np.prod(weights, axis=-1, keepdims=True)
    return np.sum(weights * lattice, axis=(1, 2, 3))


def mesh3d(x, y, z, dtype=np.float32):
    grid = np.empty(x.shape + y.shape + z.shape + (3,), dtype=dtype)
    grid[..., 0] = x[:, np.newaxis, np.newaxis]
    grid[..., 1] = y[np.newaxis, :, np.newaxis]
    grid[..., 2] = z[np.newaxis, np.newaxis, :]
    return grid


def extent(x, *args, **kwargs):
    """ retutn the upper boundary and low boundary """

    return np.min(x, *args, **kwargs), np.max(x, *args, **kwargs)




def xyz_to_stu(xyz, origin, stu_axes):
    if stu_axes.shape == (3,):
        stu_axes = np.diag(stu_axes)
        # raise ValueError(
        #     'stu_axes should have shape (3,), got %s' % str(stu_axes.shape))
    # s, t, u = np.diag(stu_axes)
    assert(stu_axes.shape == (3, 3))
    s, t, u = stu_axes

    tu = np.cross(t, u)
    su = np.cross(s, u)
    st = np.cross(s, t)

    diff = xyz - origin

    # TODO: vectorize? np.dot(diff, [tu, su, st]) / ...


    tu_divide = handling_inf(np.dot(diff, tu),np.dot(s, tu))
    su_divide = handling_inf(np.dot(diff, su),np.dot(t, su))
    st_divide = handling_inf(np.dot(diff, st),np.dot(u, st))

    stu = np.stack([
        tu_divide,
        su_divide,
        st_divide

    ], axis=-1)
    return stu





def handling_inf(A,B):
    with np.errstate(divide='ignore', invalid='ignore'):
        C = A / B
        # 可以参考的对错误处理的方式
        #     arrayC[arrayC == np.inf] = 0  # 对inf的错误进行修正，不会修正-inf
        C[~ np.isfinite(C)] = 0  # 对 -inf, inf, NaN进行修正，置为0

        return C



def stu_to_xyz(stu_points, stu_origin, stu_axes):
    if stu_axes.shape != (3,):
        raise NotImplementedError()
    return stu_origin + stu_points*stu_axes


def get_stu_control_points(dims):
    stu_lattice = mesh3d(
        *(np.linspace(0, 1, d+1) for d in dims), dtype=np.float32)
    stu_points = np.reshape(stu_lattice, (-1, 3))
    return stu_points


def get_control_points(dims, stu_origin, stu_axes):
    stu_points = get_stu_control_points(dims)
    xyz_points = stu_to_xyz(stu_points, stu_origin, stu_axes)
    return xyz_points


def get_stu_deformation_matrix(stu, dims):
    v = mesh3d(
        *(np.arange(0, d+1, dtype=np.int32) for d in dims),
        dtype=np.int32)
    v = np.reshape(v, (-1, 3))

    #
    weights = bernstein_poly(
        n=np.array(dims, dtype=np.int32),
        v=v,
        stu=np.expand_dims(stu, axis=-2))

    #
    b = np.prod(weights, axis=-1)
    return b


def get_deformation_matrix(xyz, dims, stu_origin, stu_axes):
    stu = xyz_to_stu(xyz, stu_origin, stu_axes)
    return get_stu_deformation_matrix(stu, dims)


def get_ffd(xyz, dims, stu_origin=None, stu_axes=None):
    if stu_origin is None or stu_axes is None:
        if not (stu_origin is None and stu_axes is None):
            raise ValueError(
                'Either both or neither of stu_origin/stu_axes must be None')
        stu_origin, stu_axes = get_stu_params(xyz)
    b = get_deformation_matrix(xyz, dims, stu_origin, stu_axes)
    p = get_control_points(dims, stu_origin, stu_axes)
    return b, p





def get_stu_params(xyz):
    minimum, maximum = extent(xyz, axis=0)
    stu_origin = minimum
    # stu_axes = np.diag(maximum - minimum)
    stu_axes = maximum - minimum
    return stu_origin, stu_axes



def calculate_ffd(points,n=3):
    dims = (n,) * 3
    return get_ffd(points, dims)


