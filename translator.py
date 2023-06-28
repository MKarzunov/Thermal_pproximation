x_coef = 24 / 147
y_coef = 0.02 / 89

file = open('MDT XP Resistance.txt', 'r')
t = file.readlines()
file.close()
xs = t[1::4]
ys = t[2::4]


def translate(vector: list, coef: float):
    return [float(item.split()[-1]) * coef for item in vector]


print(translate(xs, x_coef))
print(translate(ys, y_coef))

# file = open('MDT XP Pressure Drop.csv', 'w')
# file.write('Q L/min, Pressure Drop bar\n')
# for q, r in zip(translate(xs, x_coef), translate(ys, y_coef)):
#     file.write(f'{q}, {r}\n')

