import optimization as opt


class HeatExchanger:
    def __init__(self):
        self.power = 1e+3
        self.temperature = None

    def heat_liquid(self, liquid_in, recorder_self, recorder_liq):
        resistance = opt.resistance([4.417e-03, 1.493e-03, 9.165e-05, 1.000e-01], liquid_in.vol_flow,
                                    [liquid_in.density, liquid_in.heat_capacity, liquid_in.viscosity,
                                     liquid_in.conductivity])
        self.temperature = resistance * liquid_in.vol_flow + liquid_in.temperature
        recorder_self.append(self.temperature)
        recorder_liq.append(liquid_in.temperature)
        delta_temperature = self.power / (liquid_in.heat_capacity * liquid_in.vol_flow * liquid_in.density)
        liquid_in.temperature += delta_temperature


class Liquid:
    def __init__(self):
        self.density = 1050.440
        self.heat_capacity = 3499
        self.viscosity = 1.464e-6
        self.conductivity = 4.108e-1
        self.pressure = 1e+5
        self.temperature = 40
        self.vol_flow = 1e-4


he = HeatExchanger()
glycol = Liquid()
slf = list()
lq = list()
for _ in range(100):
    he.heat_liquid(glycol, slf, lq)
pass
