import csv
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import line

f = open('compare.csv', 'r')
reader = csv.reader(f, delimiter=',')

x, y = np.zeros(400), np.zeros(400)

line_count = 0
for row in reader:
    if line_count == 0:
        line_count += 1
    else:
        x[line_count - 1], y[line_count - 1] = row[0], row[1]
        line_count += 1

plt.scatter(x, y)
plt.show()