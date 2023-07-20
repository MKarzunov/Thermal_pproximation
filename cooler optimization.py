import numpy as np
from scipy.optimize import least_squares, minimize, basinhopping
import matplotlib.pyplot as plt

# ASA 0929 230/400V (50Hz) (1.5 kW)
# Oil type ISO VG 46

# density, conductivity, kinematic viscosity, heat capacity
iso_46_props = (860, 0.132, 46.6e-6, 1942.67)

# theta = [f_rad, d_tube, eta, f_op, fin_length, l_tube]

plot_vol_flow = np.array(
    [49.99011794871795, 88.20600512820513, 153.49001025641027, 201.1771948717949, 243.64994871794875,
     299.95204102564105, 364.2634358974359, 404.0979846153847, 443.93253333333337, 484.2228205128206,
     524.5131076923077, 562.461276923077, 600.4094461538461])
plot_p_spec = np.array(
    [1.2171298474358976, 1.8340764333333335, 2.4372561730769235, 2.7157759974358977, 2.882851252564103,
     3.1126270871794874, 3.275092664102565, 3.3658708833333337, 3.456649102564103, 3.5281774551282057,
     3.5997057602564104, 3.657639411538462, 3.7155730628205137])

pressure_vol_flow = np.array([0.0, 52.51037435897437, 103.76692820512821, 150.69810256410258, 202.85903076923077,
                              262.5098923076923, 300.0774307692308, 345.50162564102567, 398.76475897435904,
                              450.71934358974363,
                              500.73258461538467, 554.7155538461539, 600.1548615384615])
pressure_loss = np.array([0.0, 0.17028338461538464, 0.38161955897435906, 0.626422076923077, 0.9286652974358977,
                          1.4629227743589746, 1.7999686974358977, 2.1968336358974363, 2.6523654102564107,
                          3.108078938461539,
                          3.5441338871794876, 4.008169676923077, 4.4051709589743595])


# theta = [f_rad, d_tube, eta, f_op, fin_length, l_tube]


def solve_equation(theta, oil_vol_flow, oil_props=iso_46_props, air_cp=1005, air_mass_flow=5.74, t_oil_in=40,
                   t_air_in=20):
    oil_density = oil_props[0]
    oil_vol_flow = oil_vol_flow / 6e+4
    oil_mass_flow = oil_density * oil_vol_flow
    oil_cp = oil_props[3]
    f_rad = theta[0]
    air_area = 1031 * 1100 * 0.02 * 1e-6
    air_density = 1.205
    air_speed = air_mass_flow / (air_density * air_area)
    k_rad = get_k(theta[1], theta[2], theta[3], theta[4], oil_mass_flow, air_speed, oil_props)
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
    res = np.linalg.solve(coefficients, answers)
    # res = [power, t_oil_out, t_air_out, t_oil_av, t_air_av]
    p_spec = res[0] / (t_oil_in - t_air_in)
    return p_spec / 1000


def get_k(d_tube, eta, f_op, fin_length, oil_mass_flow, air_speed, oil_props=iso_46_props):
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
    # if re_oil <= 2000:
    #     nu_oil = 3.66
    # else:
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


# theta = [f_rad, d_tube, eta, f_op, fin_length, l_tube]


def get_fanning(re: float) -> float:
    denominator = np.sqrt((8 / re) ** 10 + (re / 36500) ** 20)
    second_in_sum = (2.21 * np.log(re / 7)) ** 10
    two_by_f = (1 / denominator + second_in_sum) ** (1 / 5)
    return 2 / two_by_f


def pressure_drop(theta, q, oil_props=iso_46_props):
    ro = oil_props[0]
    _lambda = oil_props[1]
    nu = oil_props[2]

    dh = theta[1]
    l_tube = theta[5]
    cs_area = np.pi * (dh ** 2) / 4
    a_wet = np.pi * dh * l_tube

    q = q / 6e+4
    um = q / cs_area
    reinolds = (um * dh) / nu
    print(reinolds)
    fanning = get_fanning(reinolds)
    return (fanning * (a_wet / cs_area) * (0.5 * ro * um ** 2)) / 1e+5


def optimized_fun(coefs):
    res = 0
    for flow_perf, perf, flow_pres, pres in zip(plot_vol_flow, plot_p_spec, pressure_vol_flow, pressure_loss):
        res += ((solve_equation(coefs, flow_perf) - perf)) ** 2
        res += ((pressure_drop(coefs, flow_pres) - pres)) ** 2
    return res


theta_first = (2, 0.025, 0.80, 1.6, 0.001, 2)
minimizer_kwargs = {'method': 'Nelder-Mead', 'bounds': ((0, 4), (2e-3, 5e-2), (0, 1), (1, 100), (1e-3, 1), (1, 20))}
result = basinhopping(optimized_fun, theta_first, minimizer_kwargs=minimizer_kwargs)

# result = minimize(optimized_fun, theta_first, method='Nelder-Mead', options={'maxfev': 10000, 'maxiter': 5000},
#                   bounds=((0, 3), (2e-3, 5e-2), (0, 1), (1, 100), (5e-3, 1), (1, 20)))


print('-' * 50)
# theta_res = (16, 0.025, 0.80, 1.6, 0.2, 2)
theta_res = result.x
# theta_res[5] = 0.0005
print(result)
# theta = [f_rad, d_tube, eta, f_op, fin_length, l_tube]
print(theta_res)

plt.figure(1)
plt.plot(plot_vol_flow, plot_p_spec, label='original')
optim_performance = [solve_equation(theta_res, flow) for flow in plot_vol_flow]
plt.plot(plot_vol_flow, optim_performance, label='optimized')
plt.title('Performance')
plt.legend()
plt.figure(2)
plt.plot(pressure_vol_flow, pressure_loss, label='original')
optim_pressure = pressure_drop(theta_res, pressure_vol_flow)
plt.plot(pressure_vol_flow, optim_pressure, label='optimized')
plt.title('Pressure drop')
plt.legend()
plt.show()
