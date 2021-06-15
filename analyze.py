import sys
import subprocess
import signal
import psutil
import os
import numpy as np
import math
import h5py
import matplotlib
matplotlib.use('TKAgg', force = True)
from matplotlib import colors as mcolors
from matplotlib import rc, font_manager
import matplotlib.pyplot as plt
import pdb
from mpac_cmd import *
from UCB_Bayes import UCBOptimizer
import time

# matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath,amssymb}']
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text',usetex=False)
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

class info_parser:
    def __init__(self):
        pass

    def load_data(self, filename = 'data/latest.h5'):
        print("Starting with file: ", filename)
        self.data_file = h5py.File(filename, "r", swmr=True)
        self.data = self.data_file['time_series'];
        self.attributes = self.data_file['attributes'][()]; #copying into a numpy array for speed

    def plot_data(self, smoother = 500):
        fig, ax = plt.subplots()
        maxt = len(self.data)
        times = [i for i in range(maxt)]
        z_vals = self.data[0:maxt]['q'][:,2]
        x_vals = self.data[0:maxt]['q'][:,0]
        y_vals = self.data[0:maxt]['q'][:,1]
        vx = self.data[0:maxt]['qd'][:,0]
        smoothed_vel = [None for i in range(maxt - smoother)]
        for i in range(maxt - smoother):
            smoothed_vel[i] = sum(vx[i:smoother+i])/smoother
        ax.plot(times, z_vals, color = 'black', linewidth = 2)
        ax.plot(times[smoother:], smoothed_vel, color = 'red', linewidth = 2)
        plt.show()
        pass

    def spawn_process(self, command):
        # Spawns the process associated with the given command and provides
        # the process id for the spawned process (in case it needs to be
        # killed later)
        p = subprocess.Popen([command], shell = True)
        time.sleep(0.25)
        children = psutil.Process(p.pid).children(recursive=True)
        return children[0].pid

    def kill_process(self, pid):
        # Assumes it receives a list of pids for processes that are to be killed
        # then it kills all relevant processes by looping through pids
        for p in pid: os.kill(p, signal.SIGTERM)
        pass

    def setup(self, show_output = False):
        self.process_list = []
        self.process_list.append(self.spawn_process(command = './ctrl'))
        lie()
        time.sleep(0.5)
        self.process_list.append(self.spawn_process(command = './tlm'))
        time.sleep(0.25)
        if show_output:
            self.process_list.append(self.spawn_process(command = 'python3 vis.py'))
            time.sleep(2.5)
        pass

    def gen_gait_data(self, init_state = np.array([[0.2, 0.2, 0.1, 0.1]]), wait_time = 3.5, vis_flag = False):
        self.setup(show_output = vis_flag)
        stand_idqp(h = init_state[0,0], rx = init_state[0,1], ry = init_state[0,2], rz = init_state[0,3])
        time.sleep(1)
        walk_mpc_idqp(h = 0.25, vx = 0.3)
        if vis_flag:
            time.sleep(10)
        else:
            time.sleep(wait_time)
        self.kill_process(self.process_list)
        pass

    def gen_walk_data(self, wait_time = 3.5, vis_flag = False):
        self.setup(show_output = vis_flag)
        walk_mpc_idqp(vx = 0.3)
        if vis_flag:
            time.sleep(5)
        else:
            time.sleep(wait_time)
        walk_mpc_idqp(vx = 0.3, vy = 0.3)
        if vis_flag:
            time.sleep(5)
        self.kill_process(self.process_list)
        pass

    def velocity_robustness(self, visualizer = False):
        # Checks the generated data-file to see if the quadruped satisfied
        # it's specification.  That within 2 seconds, it settles above a
        # prespecified forward velocity, and maintains that forward velocity
        # for 1 second.
        self.load_data()
        if visualizer: plt.rcParams.update({'font.size': 30})
        if visualizer: fig, ax = plt.subplots()
        ctrl_des = self.data[:]['ctrl_mode_des'].tolist()
        init_index = ctrl_des.index(b'walk_mpc_idqp(0.25,0.30,0.00,0.0')
        smoother = 200
        vx = self.data[init_index-smoother:]['qd'][:,0]
        et = self.data[init_index-smoother:]['epoch_time']
        n_samples = len(vx)
        smoothed_vel = [None for i in range(n_samples-smoother)]
        smoothed_time = [None for i in range(n_samples-smoother)]
        offset = sum(et[0:smoother])/smoother
        for i in range(n_samples - smoother):
            smoothed_vel[i] = sum(vx[i:smoother+i])/smoother
            smoothed_time[i] = sum(et[i:smoother+i])/smoother - offset
        two_sec_mark = next(x for x,val in enumerate(smoothed_time) if val>=2)
        three_sec_mark = next(x for x,val in enumerate(smoothed_time) if val>=3)
        first_robustness = max(smoothed_vel[0:two_sec_mark]) - 0.25
        second_robustness = min(smoothed_vel[two_sec_mark:three_sec_mark]) - 0.25
        robustness = min((first_robustness, second_robustness))
        if visualizer:
            ax.plot(smoothed_time, smoothed_vel, color = 'black',lw = 4)
            ax.plot(smoothed_time, 0.25*np.ones(n_samples-smoother,),
                color = 'red', ls = '--', lw = 2)
            ax.vlines(smoothed_time[two_sec_mark], 0, 0.35, color = 'red', lw = 2, ls = '--')
            ax.vlines(smoothed_time[three_sec_mark], 0, 0.35, color = 'red', lw = 2, ls = '--')
            ax.set_xlabel('time (sec)', fontsize = 30)
            ax.set_ylabel('v', rotation = 0, fontsize = 30)
            ax.yaxis.set_label_coords(-0.075,0.45)
            ax.set_title('Tracking Quadruped Forward Velocity', fontsize = 34)
            plt.show()
        return robustness

    def hardware_vel_rob(self, filename):
        self.load_data(filename = filename)
        ctrl_des = self.data[:]['ctrl_mode_des'].tolist()
        init_index = ctrl_des.index(b'walk_mpc_idqp(0.25,0.30,0.00,0.0')
        smoother = 200
        vx = self.data[init_index-smoother:]['qd'][:,0]
        et = self.data[init_index-smoother:]['epoch_time']
        n_samples = len(vx)
        smoothed_vel = [None for i in range(n_samples-smoother)]
        smoothed_time = [None for i in range(n_samples-smoother)]
        offset = sum(et[0:smoother])/smoother
        for i in range(n_samples - smoother):
            smoothed_vel[i] = sum(vx[i:smoother+i])/smoother
            smoothed_time[i] = sum(et[i:smoother+i])/smoother - offset
        two_sec_mark = next(x for x,val in enumerate(smoothed_time) if val>=2)
        three_sec_mark = next(x for x,val in enumerate(smoothed_time) if val>=3)
        first_robustness = max(smoothed_vel[0:two_sec_mark]) - 0.25
        second_robustness = min(smoothed_vel[two_sec_mark:three_sec_mark]) - 0.25
        robustness = min((first_robustness, second_robustness))
        return robustness


    def find_worst_case(self):
        bounds = np.array([[0.2, 0.0, 0.0, 0.0],[0.25, 0.2, 0.2, 0.2]]).transpose()
        def objective(x):
            self.gen_gait_data(init_state = x)
            return -self.velocity_robustness()

        optimizer = UCBOptimizer(objective = objective, bounds = bounds, B = 0.3,
            R = 0.05, delta = 1e-6, tolerance = 0.002, n_init_samples = 1)

        optimizer.initialize()
        optimizer.optimize()
        return optimizer

    def compare_systems(self):
        bounds = np.array([[0.2, 0.0, 0.0, 0.0],[0.25, 0.2, 0.2, 0.2]]).transpose()
        def objective(x):
            self.gen_gait_data(init_state = x)
            print('Would like for the hardware system to initialize:')
            print('height: %.3f'%x[0,0])
            print('rx: %.3f'%x[0,1])
            print('ry: %.3f'%x[0,2])
            print('rz: %.3f'%x[0,3])
            print(' ')
            sim_rob = self.velocity_robustness()
            print('Enter filename (with path) for the hardware data file:')
            hardware_file = input()
            # hardware_file = 'data/latest.h5'
            hard_rob = self.hardware_vel_rob(filename = hardware_file)
            fin_calc = abs(sim_rob - hard_rob)
            print('Calculated difference between robustness measures: %.4f'%fin_calc)
            print(' ')
            return fin_calc

        optimizer = UCBOptimizer(objective = objective, bounds = bounds, B = 0.1,
            R = 0.05, delta = 1e-6, tolerance = 0.005, n_init_samples = 1)

        optimizer.initialize()
        optimizer.optimize()
        return optimizer



if __name__ == '__main__':
    commander = info_parser()
    # optimality = commander.find_worst_case()
    # np.save('optimality/X_sample.npy',optimality.X_sample)
    # np.save('optimality/Y_sample.npy',optimality.Y_sample)
    optimality = commander.compare_systems()
