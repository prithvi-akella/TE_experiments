# from atnmy.mpac_cmd import *
from mpac_cmd import *
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import pdb

# Reduced Order controller class.  Double check satisfaction with end plots
# after cosntruction.
class Red_order_controller:
    def __init__(self, goal, obstacles, obs_radius = 0.3):
        DO = obs_radius*np.ones((1,3))

        # Controller Params
        scale = 0.05
        Kp = 1*scale
        alpha = 0.2
        Kv = 5*scale

        # Initial Conditions
        xinit = np.array([0,0]).T
        vinit = np.array([0,0]).T
        yinit = np.vstack((xinit, vinit))

        # Dimension of the problem
        dim = len(goal)

        # Store Parameters in Dict
        self.par = {'xgoal' : goal, 'xO': obstacles, 'DO': DO, 'Kp': Kp,
            'alpha' : alpha, 'Kv': Kv, 'dim': dim }

    def K_des(self,z):
        xgoal = self.par['xgoal']
        dim = self.par['dim']
        Kp = self.par['Kp']
        x = z[0:dim,:]
        udes = -Kp*(x-xgoal)
        return udes

    def CBF(self, z):
        dim = self.par['dim']
        xO = self.par['xO']
        DO = self.par['DO']
        x = z[0:dim,:]
        # Control Barrier Function
        hk = np.zeros((len(DO[0]),1))
        for kob in range(len(DO[0])):
            xob = np.array([xO[:,kob]]).T
            Dob = DO[0,kob]
            hk[kob] = np.linalg.norm(x-xob) - Dob

        # Use only the closest obstacle
        h = hk.min()
        idx = hk.argmin()
        xob = np.array([xO[:,idx]]).T
        gradh = ((x - xob)/np.linalg.norm(x - xob)).T
        Lfh = 0
        Lgh = gradh
        LghLgh = 1
        return h, Lfh, Lgh, LghLgh

    def K_CBF(self, z):
        alpha = self.par['alpha']
        udes = self.K_des(z)
        #Safety Filter
        if self.par['xO'] is not None:
            h, Lfh, Lgh, LghLgh = self.CBF(z)
            phi = Lfh + Lgh@udes + alpha*h
            u = udes + max(0, -phi)*Lgh.T/LghLgh
        else:
            u = udes
        return u

def sim_controller(goal = np.array([[4,-1]]).T, obstacles = np.array([[1.5,0],[3, -1.5],[10,-3]]).T, steps = 150, dt = 0.3):
    stand_idqp()
    time.sleep(2)
    obstacles = None

    # record values
    u_traj = []
    x_traj = []
    u_des_traj = []
    h_traj = []
    controller = Red_order_controller(goal = goal, obstacles = obstacles)
    for i in range(steps):
        print(i)
        state = get_tlm_data()['q']
        x_traj.append(state[0:2])
        state = np.array([state[0:3]]).T


        udes = controller.K_CBF(state)
        u_traj.append(controller.K_des(state))
        u_des_traj.append(udes[0:2])
        if obstacles is not None:
            h_traj.append(controller.CBF(state)[0])

        walk_mpc_idqp(vx=udes[0], vy=udes[1])
        print(udes)
        time.sleep(dt)
    lie()
    x_traj = np.array(x_traj)
    u_traj = np.squeeze(np.array(u_traj))
    u_des_traj = np.squeeze(np.array(u_des_traj))
    return x_traj, u_traj, u_des_traj, h_traj, controller

if __name__ == "__main__":
    x_traj, u_traj, u_des_traj, h_traj, controller = sim_controller()
    if True:
        h_traj = np.array(h_traj)

        theta = np.linspace(0,2*np.pi + 0.1)
        circ_x = controller.par['DO'][0,0]*np.cos(theta)
        circ_y = controller.par['DO'][0,0]*np.sin(theta)

        plt.figure()
        plt.plot(x_traj[:,0], x_traj[:,1])
        for xob in controller.par['xO'].T:
            plt.plot(xob[0], xob[1], 'r')
            plt.plot(xob[0] + circ_x, xob[1] + circ_y, 'r')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.legend(['state', 'obs1', 'obs2', 'obs3'])

        plt.figure()
        plt.plot(u_traj, 'r--')
        plt.plot(u_des_traj, 'g')
        plt.legend(['$v_{x,des}$', '$v_{y,des}$', '$v_{x,cbf}$', '$v_{y,cby}$'])

        plt.figure()
        plt.plot(h_traj)
        plt.hlines(0, xmin=0, xmax=len(h_traj), linestyles='dashed')
        plt.legend(['CBF'])

        plt.show()
