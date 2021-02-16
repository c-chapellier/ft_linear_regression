import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class linear_regression:

  def __init__(self, dataset_path, learning_rate = 1, iterations = 100):
    self.__learning_rate = learning_rate
    self.__iterations = iterations
    self.__df = pd.read_csv(dataset_path)
    self.__x = self.__normalize(self.__df['km'])
    self.__y = self.__normalize(self.__df['price'])
    self.__errors = []
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

  def train(self):
    self.__theta0 = 0
    self.__theta1 = 1
    for itr in range(self.__iterations):
      error_cost = 0
      cost_theta0 = 0
      cost_theta1 = 0
      for i in range(len(self.__x)):
        error = self.__estimate_price(self.__x[i]) - self.__y[i]
        error_cost += error ** 2
        cost_theta0 += error
        cost_theta1 += self.__x[i] * error
      self.__theta0 -= self.__learning_rate * (cost_theta0 / len(self.__x))
      self.__theta1 -= self.__learning_rate * (cost_theta1 / len(self.__x))
      self.__errors.append(error_cost)
    self.__write_thetas()
    self.__is_trained = True

  def plot(self):
    if self.__is_trained:
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
      plt.plot(np.arange(1, len(self.__errors) + 1), self.__errors, color='red', linewidth=5)
      plt.ylabel("error")
      plt.xlabel("iteration")
      plt.show()
    else:
      print("Model not trained.")

  def plot_3d_cost(self):
    grid_size = 40
    cost_grid = np.empty((grid_size, grid_size))

    X = np.empty((1, grid_size))
    Y = np.empty((1, grid_size))

    m = len(self.__x)
    print(self.__theta0, self.__theta1)

    for x_i in range(grid_size):
      X[0][x_i] = self.__theta0 + ((x_i - (grid_size / 2)) * (self.__theta0 / 10))
    for y_i in range(grid_size):
      Y[0][y_i] = self.__theta1 + ((y_i - (grid_size / 2)) * (self.__theta1 / 5))

    for x_i in range(grid_size):
      for y_i in range(grid_size):
        cost = 0
        for i in range(m):
          error = (X[0][x_i] + (Y[0][y_i] * self.__x[i])) - self.__y[i]
          cost += (error ** 2) / m
        cost_grid[x_i][y_i] = cost

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(X, Y)

    # Gx, Gy = np.gradient(cost_grid)
    # G = ((Gx ** 2) + (Gy ** 2)) ** 0.5
    # N = G / G.max()
    # surf = ax.plot_surface(X, Y, cost_grid, rstride=1, cstride=1, facecolors=cm.jet(N), linewidth=0, antialiased=False, shade=False)

    surf = ax.plot_surface(X, Y, cost_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    plt.title("cost of the model with thetas");
    plt.xlabel("theta0")
    plt.ylabel("theta1")
    plt.show()

if __name__ == "__main__":
  argc = len(sys.argv)
  if argc < 2:
    print("arg[1] must be train or predict")
    exit()
  model = linear_regression("./data.csv")
  if sys.argv[1] == "train":
    model.train()
    model.plot()
  elif sys.argv[1] == "predict":
    if argc < 3:
      print("arg[2] with predict must be the mileage")
      exit()
    try:
      value = float(sys.argv[2])
    except ValueError:
      print("arg[2] with predict must be the mileage, a number")
      exit()
    print(model.estimate_price_from_file(value))
