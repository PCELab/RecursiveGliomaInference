import dolfin as dl
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import scipy.io
import math

class InterpolatedParameter(dl.UserExpression):
    def __init__(self, X, Y, image, **kwargs):
        super().__init__(**kwargs)
        self.X = X * 0.25  # voxel scaling in mm
        self.Y = Y * 0.25
        self.image = image

    def eval(self, values, x):
        interp_handle = RegularGridInterpolator((self.X, self.Y), self.image)
        values[0] = interp_handle(x)

    def value_shape(self):
        return ()  # scalar field

def interp(file_loc, mat_name, V):
    mat = scipy.io.loadmat(file_loc)[mat_name]
    mat = np.fliplr(mat.T)
    x, y = mat.shape
    mat_interp = InterpolatedParameter(np.linspace(0, x, x), np.linspace(0, y, y), mat, degree=1)
    return dl.interpolate(mat_interp, V)

def vector2Function(v, V, name="v"):
    f = dl.Function(V, name=name)
    f.vector().set_local(v.get_local())
    return f
