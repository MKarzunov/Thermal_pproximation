import numpy as np
from scipy.optimize import least_squares, minimize, basinhopping
import matplotlib.pyplot as plt


def solve_equation(oil_cp, oil_mass_flow, air_cp, air_mass_flow, f_rad, k_rad, t_oil_in, t_air_in):
    # x = [power, t_oil_out, t_air_out, t_oil_av, t_air_av]
    eq1 = [1.0, oil_cp * oil_mass_flow, 0.0, 0.0, 0.0]
    an1 = oil_cp * oil_mass_flow * t_oil_in
    eq2 = [1.0, 0.0, -air_cp * air_mass_flow, 0.0, 0.0]
    an2 = -air_cp * air_mass_flow * t_air_in
    eq3 = [1.0, 0.0, 0.0, -f_rad * k_rad, f_rad * k_rad]
    an3 = 0.0
    eq4 = [0.0, -0.5, 0.0, 1.0, 0.0]
    an4 = 0.5 * t_oil_in
    eq5 = [0.0, 0.0, -0.5, 0.0, 1.0]
    an5 = 0.5 * t_air_in
    coefficients = np.array([eq1, eq2, eq3, eq4, eq5])
    answers = np.array([an1, an2, an3, an4, an5])
    return np.linalg.solve(coefficients, answers)


print(solve_equation(4200, 0.1, 1000, 0.2, 2, 45, 40, 20))
print(solve_equation(4200, 0.1, 1000, 0.2, 2, 100, 40, 20))
print(solve_equation(4200, 0.1, 1000, 0.2, 2, 1000, 40, 20))
print(solve_equation(4200, 0.1, 1000, 0.2, 2, 10000, 40, 20))


def get_k(d_tube, eta, f_op, fin_length, oil_mass_flow, air_speed, oil_props):
    delta = 2e-3
    aluminum_conductivity = 230
    oil_density = oil_props[0]
    oil_conductivity = oil_props[1]
    oil_kin_viscosity = oil_props[2]
    oil_cp = oil_props[3]
    oil_vol_flow = oil_mass_flow / oil_density
    tube_area = np.pi * (d_tube ** 2 / 4)
    oil_speed = oil_vol_flow / tube_area
    re_oil = oil_speed * d_tube / oil_kin_viscosity
    pr_oil = oil_kin_viscosity * oil_density * oil_cp / oil_conductivity
    if re_oil <= 2000:
        nu_oil = 3.66
    else:
        nu_oil = 0.023 * re_oil ** 0.8 * pr_oil ** 0.4
    alpha_oil = nu_oil * oil_conductivity / d_tube

    air_kin_viscosity = 15e-6
    air_conductivity = 0.022
    re_air = air_speed * fin_length / air_kin_viscosity
    nu_air = 0.032 * re_air ** 0.8
    alpha_air = nu_air * air_conductivity / fin_length

    first = 1 / (alpha_air * eta)
    second = delta / aluminum_conductivity
    third = 1 / (alpha_oil * f_op)
    return 1 / (first + second + third)


print('-' * 20)
k = get_k(0.01, 0.49, 1.33, 0.01, 0.1, 20, [1000, 0.58, 0.66e-6, 4200])
print(k)
print(solve_equation(4200, 0.1, 1000, 0.2, 2, k, 40, 20))
