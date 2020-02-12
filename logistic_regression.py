import numpy as np
import matplotlib.pyplot as plt

def draw(x1, x2):
    line = plt.plot(x1, x2)
    plt.pause(0.0001)
    line[0].remove()

def sigmoid(score):
    return 1/(1+ np.exp(-score))

def calculate_error(line_parameters, points, y):
    m = points.shape[0]
    p = sigmoid(points * line_parameters)
    cross_entropy = -(1/m)*(np.log(p).transpose() * y + np.log(1-p).transpose()*(1-y))
    return cross_entropy

def gradient_descent(line_parameters, points, y, alpha):
    for i in range(2000):
        m = points.shape[0]
        p = sigmoid(points * line_parameters)
        gradient = (points.transpose() * (p - y))*(1/m)
        line_parameters = line_parameters - (gradient * alpha)
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -b / w2 + x1 * (- w1 /w2 )
        draw(x1, x2)

num_points = 100
np.random.seed(0)
bias = np.ones(num_points)
top_region = np.array([np.random.normal(10, 2, num_points), np.random.normal(12, 2, num_points), bias]).transpose()
bottom_region = np.array([np.random.normal(5, 2, num_points), np.random.normal(6, 2, num_points), bias]).transpose()
all_points = np.vstack((top_region, bottom_region))
line_parameters = np.matrix([0, 0, 0]).transpose()


y = np.array([np.zeros(num_points), np.ones(num_points)]).reshape(num_points * 2, 1)

_, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
gradient_descent(line_parameters, all_points, y, 0.06)
plt.show()
