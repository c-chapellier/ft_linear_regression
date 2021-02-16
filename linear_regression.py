import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

class linear_regression:

  def __init__(self, dataset_path, learning_rate = 1, iterations = 100):
    self.__learning_rate = learning_rate
    self.__iterations = iterations
    self.__df = pd.read_csv(dataset_path)
    self.__x = self.__normalize(self.__df['km'])
    self.__y = self.__normalize(self.__df['price'])
    self.__errors = np.empty((3, self.__iterations))
    self.__theta0 = 0
    self.__theta1 = 0
    self.__thetas_path = "./thetas"
    self.__is_trained = False

  @staticmethod
  def __normalize(numpyArray):
    return (numpyArray - numpyArray.min(axis=0)) / (numpyArray.max(axis=0) - numpyArray.min(axis=0))

  @staticmethod
  def __denormalize(numpyArray, base):
    return numpyArray * (base.max(axis=0) - base.min(axis=0)) + base.min(axis=0)

  def __write_thetas(self):
    y_0 = self.__denormalize(self.__estimate_price(self.__x[0]), self.__df['price'])
    y_1 = self.__denormalize(self.__estimate_price(self.__x[1]), self.__df['price'])
    x_0 = self.__denormalize(self.__x[0], self.__df['km'])
    x_1 = self.__denormalize(self.__x[1], self.__df['km'])
    theta1_denormalized = (y_1 - y_0) / (x_1 - x_0)
    theta0_denormalized = y_0 - (theta1_denormalized * x_0)
    with open(self.__thetas_path, "w") as file:
      file.write(str(theta0_denormalized) + " ")
      file.write(str(theta1_denormalized))

  def __read_thetas(self):
    with open(self.__thetas_path, "r") as file:
      tmp = file.readlines()[0].split(" ")
      self.__theta0 = float(tmp[0])
      self.__theta1 = float(tmp[1])

  def estimate_price_from_file(self, x):
    try:
      self.__read_thetas()
    except IOError:
      print("File not accessible")
      return np.nan
    return self.__estimate_price(x)

  def __estimate_price(self, x):
    return self.__theta0 + (self.__theta1 * x)

  def train(self, starting_t0=0, starting_t1=1):
    m = len(self.__x)
    self.__theta0 = starting_t0
    self.__theta1 = starting_t1
    for itr in range(self.__iterations):
      self.__errors[0][itr] = self.__theta0
      self.__errors[1][itr] = self.__theta1
      error_cost = 0
      cost_theta0 = 0
      cost_theta1 = 0
      for i in range(m):
        error = self.__estimate_price(self.__x[i]) - self.__y[i]
        error_cost += (error ** 2) / m
        cost_theta0 += error
        cost_theta1 += self.__x[i] * error
      self.__theta0 -= self.__learning_rate * (cost_theta0 / m)
      self.__theta1 -= self.__learning_rate * (cost_theta1 / m)
      self.__errors[2][itr] = error_cost
    self.__write_thetas()
    self.__is_trained = True

  def plot(self):
    if not self.__is_trained:
      print("Model not trained.")
      exit()
    y_pred = self.__theta0 + (self.__theta1 * self.__x)
    y_pred_denormalize = self.__denormalize(y_pred, self.__df['price'])
    x_denormalize = self.__denormalize(self.__x, self.__df['km'])
    y_denormalize = self.__denormalize(self.__y, self.__df['price'])
    plt.scatter(x_denormalize, y_denormalize)
    plt.plot(x_denormalize, y_pred_denormalize)
    plt.ylabel("price")
    plt.xlabel("mileage")
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(self.__errors[2]) + 1), self.__errors[2], color='red', linewidth=5)
    plt.ylabel("error")
    plt.xlabel("iterations")
    plt.show()

  def plot_3d_cost(self):
    if not self.__is_trained:
      print("Model not trained.")
      exit()
    grid_size = 40
    thetas0 = np.empty((1, grid_size))
    thetas1 = np.empty((1, grid_size))
    cost_grid = np.empty((grid_size, grid_size))
    m = len(self.__x)

    thetas0[0] = np.linspace(-5, 5, grid_size)
    thetas1[0] = np.linspace(-5, 5, grid_size)
    # for i in range(grid_size):
    #   thetas0[0][i] = self.__theta0 + ((i - (grid_size / 2)) * (self.__theta0 / 0.5))
    # for i in range(grid_size):
    #   thetas1[0][i] = self.__theta1 + ((i - (grid_size / 2)) * (self.__theta1 / 0.5))

    for i in range(grid_size):
      for j in range(grid_size):
        cost = 0
        for k in range(m):
          error = (thetas0[0][i] + (thetas1[0][j] * self.__x[k])) - self.__y[k]
          cost += (error ** 2) / m
        cost_grid[j][i] = cost

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    thetas0, thetas1 = np.meshgrid(thetas0, thetas1)

    # create circle base
    x0, y0, radius = self.__theta0, self.__theta1, 3.5
    r = np.sqrt((thetas0 - x0)**2 + (thetas1 - y0)**2)
    inside = r < radius

    surf = ax.plot_trisurf(thetas0[inside], thetas1[inside], cost_grid[inside], alpha=.5, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.scatter(self.__errors[0][:-1], self.__errors[1][:-1], self.__errors[2][:-1], c='blue', s=10)
    ax.scatter(self.__errors[0][-1], self.__errors[1][-1], self.__errors[2][-1], c='red', s=100)
    plt.title("cost of the model with thetas")
    ax.set_xlabel("theta0")
    ax.set_ylabel("theta1")
    ax.set_zlabel("cost")
    plt.show()


if __name__ == "__main__":
  argc = len(sys.argv)
  if argc < 2:
    print("arg[1] must be train or predict.")
    exit()
  model = linear_regression("./data.csv", learning_rate = 0.1, iterations = 1000)
  if sys.argv[1] == "train":
    if argc > 3:
      try:
        value, value2 = float(sys.argv[2]), float(sys.argv[3])
      except ValueError:
        print("arg[2] and arg[3] with train must be the starting thetas, numbers.")
        print("-2 and -1 give a beautiful graph.")
        exit()
      model.train(value, value2)
    else:
      model.train()
    model.plot()
    model.plot_3d_cost()
  elif sys.argv[1] == "predict":
    if argc < 3:
      print("arg[2] with predict must be the mileage.")
      exit()
    try:
      value = float(sys.argv[2])
    except ValueError:
      print("arg[2] with predict must be the mileage, a number.")
      exit()
    print(model.estimate_price_from_file(value))
