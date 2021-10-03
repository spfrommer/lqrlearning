import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import control

import click
import socket
import threading
import os
import numpy as np
import matplotlib.pyplot as plt
import colorama

from dataclasses import dataclass

from lqrlearning.utils import dirs, file_utils, pretty, torch_utils

import pdb

torch.set_default_tensor_type('torch.cuda.FloatTensor')

@dataclass
class System:
    A: np.ndarray
    B: np.ndarray

    Q: np.ndarray
    R: np.ndarray

    W: np.ndarray

    def state_n(self):
        return self.A.shape[0]

    def input_n(self):
        return self.B.shape[1]

@dataclass
class Trajectory:
    xs: np.ndarray
    us: np.ndarray
    ws: np.ndarray

    def timesteps(self):
        return self.us.shape[0]


def make_system(state_n, input_n):
    A = np.random.normal(size=(state_n, state_n))
    A /= np.linalg.norm(A)
    B = np.random.normal(size=(state_n, input_n))

    Q = np.eye(state_n)
    R = np.eye(input_n)

    W = (1/4) * np.eye(state_n)

    return System(A, B, Q, R, W)

def simulate(system: System, policy, x0: np.ndarray, timesteps: int):
    xs = np.zeros((timesteps + 1, system.state_n()))
    us = np.zeros((timesteps, system.input_n()))
    ws = np.zeros((timesteps, system.state_n()))

    xs[0] = x0

    for t in range(timesteps):
        x = xs[t]
        u = policy(x)
        w = np.random.multivariate_normal(np.zeros(system.state_n()), system.W)

        xs[t+1] = system.A @ x + system.B @ u + w
        us[t] = u
        ws[t] = w

    return Trajectory(xs, us, ws)

def cost(system: System, trajectory: Trajectory):
    total_cost = sum([x @ system.Q @ x + u @ system.R @ u
                     for (x, u) in zip(trajectory.xs[:-1], trajectory.us)])
    return (1 / trajectory.timesteps()) * total_cost

def dlqr_calculate(G, H, Q, R):
  '''
  Adapted from Lucas Bellinaso's github comment.
  '''
  from scipy.linalg import solve_discrete_are, inv, eig
  P = solve_discrete_are(G, H, Q, R)  #Solução Ricatti
  K = inv(H.T@P@H + R)@H.T@P@G    #K = (B^T P B + R)^-1 B^T P A

  from numpy.linalg import eigvals
  eigs = np.array([eigvals(G-H@K)]).T
  return K, P, eigs

def lqr_policy(system):
    K, P, e = dlqr_calculate(system.A, system.B, system.Q, system.R)
    return lambda x: - K @ x


@click.command()
@click.option('--epochs', default=1000)
@click.option('--state_n', default=4)
@click.option('--input_n', default=2)
@click.option('--timesteps', default=100)
def run(epochs, state_n, input_n, timesteps):
    colorama.init()
    # torch_utils.launch_tensorboard(dirs.out_path('tensorboard'), 6006)

    pretty.section_print('Making system')

    x0 = np.ones(state_n)
    system = make_system(state_n, input_n)
    policy = lqr_policy(system)
    trajectory = simulate(system, policy, x0, timesteps)

    print(trajectory.xs)
    print(cost(system, trajectory))



if __name__ == "__main__":
    run()
