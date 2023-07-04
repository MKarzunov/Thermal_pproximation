import numpy as np
from scipy.optimize import least_squares, minimize, basinhopping
import matplotlib.pyplot as plt

# MDT STD Resistance
x = np.array([1.496681142857143, 4.514585632653061, 7.44835069387755, 10.448388408163265,
              13.655478857142857, 17.711789224489795, 22.425386612244896])
y = np.array([0.018544964719101123, 0.012999032359550562, 0.010753430337078651,
              0.009464173707865169, 0.00848128426966292, 0.00775302382022472, 0.007031696179775281])

# MDT XP Resistance
# y = np.array([0.016114378426966294, 0.011306980898876405, 0.009352813033707866,
#               0.008224950786516853, 0.007376081573033708, 0.006749440449438203, 0.00613060786516854])


# MDT STD Pressure Drop
x_pressure = np.array(
    [1.5013507789740344, 7.526142520582647, 10.307069791006967, 13.497012995566815, 16.333569499683346,
     19.49506031665611, 22.518245775807475])
y_pressure = np.array(
    [0.02310142405063291, 0.08928056962025316, 0.16484156645569617, 0.26924651898734175, 0.45573083860759483,
     0.6451825316455696, 1.01871207278481])


# MDT XP Pressure Drop
# y_pressure = np.array(
#     [0.023662088607594934, 0.1018643987341772, 0.1955640981012658, 0.31717776898734173, 0.535710237341772,
#      0.759040316455696, 1.201657816455696])


def resistance(theta, q, liquid=(1050.440, 3499, 1.464e-6, 4.108e-1)):
    """
    :param liquid:
    :param theta: 0 - R_cond, 1 - Dh, 2 - cross-section area, 3 - a_wet
    :param q:
    :return:
    """
    eta = 0.49
    ro = liquid[0]
    cp = liquid[1]
    nu = liquid[2]
    _lambda = liquid[3]
    r_cond = theta[0]
    dh = theta[1]
    cs_area = theta[2]
    a_wet = theta[3]

    q = q / 6e+4
    reinolds = (q * dh) / (cs_area * nu)
    print(reinolds)
    nusselt = 0.21 * pow(reinolds, 0.61)
    r_conv = dh / (eta * nusselt * _lambda * a_wet)
    r_cal = 1 / (2 * ro * q * cp)
    # r_cal = 0
    return r_cond + r_conv + r_cal


def get_fanning(re: float) -> float:
    denominator = np.sqrt((8 / re) ** 10 + (re / 36500) ** 20)
    second_in_sum = (2.21 * np.log(re / 7)) ** 10
    two_by_f = (1 / denominator + second_in_sum) ** (1 / 5)
    return 2 / two_by_f


def pressure_drop(theta, q, liquid=(1050.440, 3499, 1.464e-6, 4.108e-1)):
    ro = liquid[0]
    nu = liquid[2]
    _lambda = liquid[3]
    dh = theta[1]
    cs_area = theta[2]
    a_wet = theta[3]

    q = q / 6e+4
    um = q / cs_area
    reinolds = (um * dh) / nu
    fanning = get_fanning(reinolds)
    return (fanning * (a_wet / cs_area) * (0.5 * ro * um ** 2)) / 1e+5


def optimized_fun(coefs):
    resistance_array = (((resistance(coefs, x) - y) / y) * 1.075) ** 4
    pressure_array = ((pressure_drop(coefs, x_pressure) - y_pressure) / y_pressure) ** 4
    result_array = np.hstack((resistance_array, pressure_array))
    return np.sum(result_array)


# minimizer_kwargs = {'method': 'Nelder-Mead', 'options': {'maxfev': 1600, 'maxiter': 1600},
#                     'bounds': [(0, 6e-3), (0, 0.015), (0, 2e-4), (0, 1e-1)]}
# result = basinhopping(optimized_fun, [4.000e-03, 1.473e-03, 9.188e-05, 1.000e-01], minimizer_kwargs=minimizer_kwargs)

# result = minimize(optimized_fun, [0.001, 1e-3, 1e-6, 1e-6], method='Nelder-Mead', options={'maxfev': 1600, 'maxiter': 1600})
# print(result)
#
# theta_res = result.x
theta_res = [4.417e-03, 1.493e-03, 9.165e-05, 1.000e-01]

liquid = (1050.440, 3499, 1.633e-5, 2.675e-2)

plt.figure(1)
plt.xlabel('Volumetric flow, l/m')
plt.ylabel('Thermal resistance, K/W')
plt.plot(x, y, label='original')
y_res = resistance(theta_res, x)
plt.plot(x, y_res, label='optimized')
pms_data = (980, 1632, 1e-5, 0.167)
y_pms = resistance(theta_res, x, pms_data)
plt.plot(x, y_pms, label='ПМС-10')
plt.legend()

plt.figure(2)
plt.xlabel('Volumetric flow, l/m')
plt.ylabel('Pressure drop, bar')
plt.plot(x_pressure, y_pressure, label='original')
y_res_pressure = pressure_drop(theta_res, x_pressure)
plt.plot(x_pressure, y_res_pressure, label='optimized')
y_pms_pressure = pressure_drop(theta_res, x_pressure, pms_data)
plt.plot(x_pressure, y_pms_pressure, label='ПМС-10')
plt.legend()

plt.show()
