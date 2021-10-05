import torch
import torch.nn as nn
import torch.distributions as dists
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import control

import click
import random
import numpy as np
import matplotlib.pyplot as plt
import colorama

from dataclasses import dataclass

from lqrlearning.utils import dirs, file_utils, pretty, torch_utils

import pdb

torch.manual_seed(0)

torch.set_default_tensor_type('torch.cuda.FloatTensor')

@dataclass
class System:
    A: Tensor
    B: Tensor

    Q: Tensor
    R: Tensor

    W: Tensor

    def state_n(self):
        return self.A.shape[0]

    def input_n(self):
        return self.B.shape[1]

@dataclass
class Trajectory:
    xs: Tensor
    us: Tensor
    ws: Tensor

    def timesteps(self):
        return self.us.shape[0]


def make_system(state_n, input_n):
    A = torch.normal(mean=0.0, std=1.0, size=(state_n, state_n))
    A /= torch.norm(A)
    B = torch.normal(mean=0.0, std=1.0, size=(state_n, input_n))

    Q = torch.eye(state_n)
    R = torch.eye(input_n)

    W = (1/4) * torch.eye(state_n)

    return System(A, B, Q, R, W)

def simulate(system: System, policy, x0: Tensor, timesteps: int):
    wdist = dists.MultivariateNormal(torch.zeros(system.state_n()), system.W)

    xs = [x0]
    us = []
    ws = []

    xs[0] = x0

    for t in range(timesteps):
        x = xs[t].detach()
        # x = xs[t]
        u = policy(x)
        w = wdist.sample()

        xs.append(system.A @ x + system.B @ u + w)
        us.append(u)
        ws.append(w)

    return Trajectory(torch.stack(xs, dim=0), torch.stack(us, dim=0), torch.stack(ws, dim=0))

def cost(system: System, trajectory: Trajectory):
    total_cost = sum([x @ system.Q @ x + u @ system.R @ u
                     for (x, u) in zip(trajectory.xs[:-1], trajectory.us)])
    return (1 / trajectory.timesteps()) * total_cost


def dlqr_calculate(G, H, Q, R):
  '''
  Adapted from Lucas Bellinaso's github comment.
  '''
  G = torch_utils.numpy(G)
  H = torch_utils.numpy(H)
  Q = torch_utils.numpy(Q)
  R = torch_utils.numpy(R)

  from scipy.linalg import solve_discrete_are, inv, eig
  P = solve_discrete_are(G, H, Q, R)  #Solução Ricatti
  K = inv(H.T@P@H + R)@H.T@P@G    #K = (B^T P B + R)^-1 B^T P A

  from numpy.linalg import eigvals
  eigs = np.array([eigvals(G-H@K)]).T
  return torch.tensor(K).float(), torch.tensor(P).float(), torch.tensor(eigs).float()

def lqr_policy(system):
    K, P, e = dlqr_calculate(system.A, system.B, system.Q, system.R)
    return lambda x: - K @ x


class LQRModel(nn.Module):
    def __init__(self, system: System):
        super().__init__()
        # self.theta = nn.Parameter(0.1 * torch.ones(system.state_n(), system.state_n()), requires_grad=True)
        init_theta = torch.eye(system.state_n())
        init_theta[0,2] = 0.5
        init_theta[3,2] = 2.5
        init_theta[1,0] = 8.0
        self.theta = nn.Parameter(init_theta, requires_grad=True)
        self.system = system

    def forward(self, x):
        A, B, R, theta = self.system.A, self.system.B, self.system.R, self.theta

        inner = R + B.t() @ theta.t() @ theta @ B
        K = -inner.inverse() @ B.t() @ theta.t() @ theta @ A

        return K @ x

def learned_policy(system: System):
    model = LQRModel(system)

    return lambda x: model(x)

def train_policy(system: System, policy, optimal_policy, timesteps: int, epochs):
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)
    K = 10

    file_utils.create_empty_directory(dirs.out_path('tensorboard'))
    writer = SummaryWriter(dirs.out_path('tensorboard'))

    for epoch in range(epochs):

        trajectories = []
        trajectories_optimal = []
        for _ in range(K):
            x0 = torch.normal(mean=0.0, std=1.0, size=(system.state_n(),))
            # x0 = torch.ones(system.state_n())
            trajectories.append(simulate(system, policy, x0, timesteps))
            trajectories_optimal.append(simulate(system, optimal_policy, x0, timesteps))


        cum_cost = (1 / K) * sum([cost(system, traj) for traj in trajectories])
        cum_cost.backward()
        optimizer.step()

        optimal_cost = (1 / K) * sum([cost(system, traj) for traj in trajectories_optimal])

        writer.add_scalars('Loss/train', {'learned': cum_cost, 'optimal': optimal_cost}, epoch)
        print(cum_cost)
        print(policy.theta)


@click.command()
@click.option('--epochs', default=1000)
@click.option('--state_n', default=4)
@click.option('--input_n', default=2)
@click.option('--timesteps', default=100)
def run(epochs, state_n, input_n, timesteps):
    colorama.init()
    torch_utils.launch_tensorboard(dirs.out_path('tensorboard'), 6006)

    pretty.section_print('Making system')

    system = make_system(state_n, input_n)

    policy = LQRModel(system)
    optimal_policy = lqr_policy(system)
    train_policy(system, policy, optimal_policy, timesteps, epochs)

    # x0 = torch.ones(state_n)
    # trajectory = simulate(system, policy, x0, timesteps)
    # print(trajectory.xs)
    # print(cost(system, trajectory))

if __name__ == "__main__":
    run()
