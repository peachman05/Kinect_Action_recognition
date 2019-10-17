import numpy as np

def cart2pol(point_o, point_i):
    x_o, y_o = point_o
    x_i, y_i = point_i
    x_dif = x_i - x_o
    y_dif = y_i - y_o
    rho = np.sqrt(x_dif**2 + y_dif**2)
    phi = np.degrees(np.arctan2(y_dif, x_dif))
    return [rho, phi]

def pol2cart(point_pol, point_ori):
    rho, phi = point_pol
    x_o, y_o = point_ori
    phi = np.radians(phi)
    x_i = rho * np.cos(phi) + x_o
    y_i = rho * np.sin(phi) + y_o
    return [x_i, y_i]

origin = [2, 3]
x_i = [1, 3]
polar_p = cart2pol(origin, x_i)
polar_p[1] = polar_p[1] - 45
print(pol2cart(polar_p, origin))