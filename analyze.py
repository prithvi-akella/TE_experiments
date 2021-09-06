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
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pdb
from reduced_order_barrier import *
from mpac_cmd import *
from UCB_Bayes import UCBOptimizer
import time
import pickle, dill

matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath,amssymb}']
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text',usetex=False)
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
        time.sleep(1.25)
        children = psutil.Process(p.pid).children(recursive=True)
        return children[0].pid

    def kill_process(self, pid):
        # Assumes it receives a list of pids for processes that are to be killed
        # then it kills all relevant processes by looping through pids
        for p in pid:
          try:
            os.kill(p, signal.SIGTERM)
          except:
            os.system("sudo killall ctrl")

    def setup(self, show_output = False, hardware = False):
        self.process_list = []
        if hardware:
          self.process_list.append(self.spawn_process(command = 'sudo chrt 99 ./ctrl hardware'))
        else:
          self.process_list.append(self.spawn_process(command = './ctrl'))
        lie()
        time.sleep(0.5)
        self.process_list.append(self.spawn_process(command = './tlm'))
        time.sleep(0.25)
        if show_output:
            self.process_list.append(self.spawn_process(command = 'python3 analysis/live/vis.py'))
            time.sleep(2.5)
        pass

    def gen_gait_data(self, init_state = np.array([[0.2, 0.2, 0.1, 0.1]]), wait_time = 3.5, vis_flag = False, hardware = False):
        self.setup(show_output = vis_flag, hardware=hardware)
        if vis_flag:
            time.sleep(5)
        stand_idqp(h = init_state[0,0], rx = init_state[0,1], ry = init_state[0,2], rz = init_state[0,3])
        time.sleep(2)
        walk_mpc_idqp(h = 0.25, vx = 0.3)
        if vis_flag:
            time.sleep(10)
        else:
            time.sleep(1)
        lie()
        time.sleep(1)
        self.kill_process(self.process_list)
        pass

    def gen_walk_data(self, wait_time = 2, vis_flag = False, velocity = 0.1, vel_interest = 0, hardware = False):
        self.setup(show_output = vis_flag, hardware=hardware)
        time.sleep(1)
        # if velocity >= 0 and vel_interest == 0:
        #     scale_factor = (velocity/0.3)*0.05
        # elif vel_interest == 0:
        #     scale_factor = (velocity/0.2)*0.075
        velocities = [0,0,0]
        velocities[vel_interest] = velocity
        walk_mpc_idqp(vx = velocities[0], vy = velocities[1], vrz = velocities[2])
        time.sleep(1)
        velocities[vel_interest] = velocity
        walk_mpc_idqp(vx = velocities[0], vy = velocities[1], vrz = velocities[2])
        if vis_flag:
            time.sleep(5)
        else:
            time.sleep(wait_time)
        # lie()
        # time.sleep(1)
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
        init_index = ctrl_des.index(b'walk_mpc_idqp(0.25,0.20,0.00,0.0')
        smoother = 200
        vx = self.data[init_index-smoother:]['qd'][:,0]
        vy = self.data[init_index-smoother:]['qd'][:,1]
        rz = self.data[init_index-smoother:]['q'][:,5]
        vx = vx*np.cos(rz) + vy*np.sin(rz)
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

    def hardware_vel_rob(self, filename, visualizer = False):
        self.load_data(filename = filename)
        ctrl_des = self.data[:]['ctrl_mode_des'].tolist()
        init_index = ctrl_des.index(b'walk_mpc_idqp(0.25,0.30,0.00,0.0')
        smoother = 200
        vx = self.data[init_index-smoother:]['qd'][:,0]
        vy = self.data[init_index-smoother:]['qd'][:,1]
        rz = self.data[init_index-smoother:]['q'][:,5]
        vx = vx*np.cos(rz) + vy*np.sin(rz)
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
        if visualizer: plt.rcParams.update({'font.size': 30})
        if visualizer: fig, ax = plt.subplots()
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

    def walking_robustness(self, des_vel = 0.1, vel_flag = 0, visualizer = False):
        # Checks to see if the quadruped satisfied it's specification.
        # vel_flag determines for which velocity the specification is to be
        # evaluated.
        #     vel_flag[0] = vx, vel_flag[1] = vy, vel_flag[2] = vrz
        # Each different velocity has a different requirement, though in all cases
        # the quadruped is to achieve it's desired velocity within 0.25 seconds of
        # initially commanding the quadruped to move.
        #     1) For vx: From 0.25-2.25 seconds, the quadruped is to maintain this
        #         forward velocity and deviate about the desired velocity by no
        #         more than epsilon = 0.5
        #     2) For vy:  From 0.25-2.25 secods, the quadruped is to maintain this
        #         velocity and deviate by nore more than 0.1
        #     3) For vrz: From 0.25-2.25 seconds, the quadruped is to maintain this
        #         desired velocity and deviate by no more than 0.05.
        self.load_data()
        ctrl_des = self.data[:]['ctrl_mode_curr'].tolist()
        init_index = [idx for idx, s in enumerate(ctrl_des) if b'walk_mpc_idqp(' in s][0]
        smoother = 50
        tube_epsilons = [0.075,0.1,0.05]
        overshoots = [0.3,0.2,0.4]
        settle_times = [0.5, 0.75, 0.35]
        settle_time = settle_times[vel_flag]
        epsilon = tube_epsilons[vel_flag]
        overshoot = overshoots[vel_flag]
        velocity = self.data[init_index-smoother:]['qd'][:,vel_flag]
        et = self.data[init_index-smoother:]['epoch_time']
        n_samples = len(et)
        smoothed_vel = [None for i in range(n_samples-smoother)]
        smoothed_time = [None for i in range(n_samples-smoother)]
        offset = sum(et[0:smoother])/smoother

        for i in range(n_samples - smoother):
            smoothed_vel[i] = sum(velocity[i:smoother+i])/smoother
            smoothed_time[i] = sum(et[i:smoother+i])/smoother - offset

        print('Determining starting/finishing times')
        start_time_mark = self.find_first_index(smoothed_time, lambda x: x>=settle_time)
        end_time_mark = self.find_first_index(smoothed_time, lambda x: x>=settle_time + 1)
        if des_vel >= 0:
            settling_robustness = self.determine_settling(vel = smoothed_vel,
                time = smoothed_time, des_vel = des_vel, overshoot = overshoot,
                req_time = settle_time)
        else:
            shuffled_vel = [-1*value for value in smoothed_vel]
            settling_robustness = self.determine_settling(vel = shuffled_vel,
                time = smoothed_time, des_vel = -1*des_vel, overshoot = overshoot,
                req_time = settle_time)

        maintain_rob1 = min(smoothed_vel[start_time_mark:end_time_mark]) - des_vel + epsilon
        maintain_rob2 = des_vel + epsilon - max(smoothed_vel[start_time_mark:end_time_mark])
        maintain_rob = min((maintain_rob1, maintain_rob2))
        robustness = min((settling_robustness, maintain_rob))

        print('Robustness = %.4f'%robustness)
        print('Setting time robustness: %.4f'%settling_robustness)
        print('Bound Maintenance robustness: %.4f'%maintain_rob)
        if visualizer:
            plt.rcParams.update({'font.size': 30})
            fig, ax = plt.subplots()
            ax.plot(smoothed_time[:end_time_mark], smoothed_vel[:end_time_mark], color = 'black',lw = 4)
            ax.plot(smoothed_time[start_time_mark:end_time_mark], des_vel*np.ones(end_time_mark-start_time_mark,),
                color = 'green', ls = '--', lw = 2)
            ax.hlines(y = des_vel + epsilon, xmin = smoothed_time[start_time_mark], xmax = smoothed_time[end_time_mark],
                ls = '--', lw = 2, color = 'red')
            ax.hlines(y = des_vel - epsilon, xmin = smoothed_time[start_time_mark], xmax = smoothed_time[end_time_mark],
                ls = '--', lw = 2, color = 'red')
            if des_vel >= 0:
                min_bound = min((des_vel - epsilon, des_vel - overshoot))
                max_bound = max((des_vel + overshoot, des_vel + epsilon))
            else:
                min_bound = min((des_vel - overshoot, des_vel - epsilon))
                max_bound = max((des_vel + overshoot, des_vel + epsilon))
            ax.vlines(smoothed_time[start_time_mark], min_bound, max_bound, color = 'blue', lw = 2, ls = '--')
            if des_vel >= 0:
                ax.plot(smoothed_time[:start_time_mark],
                    (des_vel+overshoot)*np.ones(start_time_mark,), color = 'blue', ls = '--', lw = 2)
                ax.plot(smoothed_time[:start_time_mark],
                    (des_vel-overshoot)*np.ones(start_time_mark,), color = 'blue', ls = '--', lw = 2)
            else:
                ax.plot(smoothed_time[:start_time_mark],
                    (des_vel-overshoot)*np.ones(start_time_mark,), color = 'blue', ls = '--', lw = 2)
                ax.plot(smoothed_time[:start_time_mark],
                    (des_vel+overshoot)*np.ones(start_time_mark,), color = 'blue', ls = '--', lw = 2)
            ax.set_xlabel('time (sec)', fontsize = 30)
            if vel_flag == 0:
                ax.set_ylabel(r'$v_x$', rotation = 0, fontsize = 30)
                ax.yaxis.set_label_coords(-0.09,0.45)
            else:
                ax.set_ylabel(r'$v_y$', rotation = 0, fontsize = 30)
                ax.yaxis.set_label_coords(-0.09,0.45)
            ax.set_title('Tracking Quadruped Forward Velocity', fontsize = 34)
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            plt.show()
        return robustness

    def determine_settling(self, vel, time, des_vel, overshoot, req_time):
        req_time_index = self.find_first_index(time, lambda x: x >= req_time)
        # if des_vel >= 0:
        rob2 = lambda v, a, b: min([des_vel + overshoot - value for value in v[a:b]])
        rob3 = lambda v, a, b: min([value - des_vel + overshoot for value in v[a:b]])

        absolute_checker = lambda v, a, b: min([overshoot - np.abs(value - des_vel) for value in v[a:b]])

        print('Entered loop calculating output list')
        # output_list = [min((vel[i] - des_vel,
        #     rob2(vel, i, req_time_index+1), rob3(vel, i, req_time_index+1)))
        #     for i in range(req_time_index-1)]
        output_list = [absolute_checker(vel, i, req_time_index+1) for i in range(req_time_index-1)]
        return max(output_list)

    def find_first_index(self, list, checker):
        # Finds the first index in a list that outputs true when passed by checker
        try:
            output = next(x for x, val in enumerate(list) if checker(val))
            return output
        except:
            return -1

    def find_worst_case(self):
        bounds = np.array([[0.22, -0.2, -0.2, -0.2],[0.28, 0.2, 0.2, 0.2]]).transpose()
        def objective(x):
            self.gen_gait_data(init_state = x)
            return -self.velocity_robustness()

        optimizer = UCBOptimizer(objective = objective, bounds = bounds, B = 0.3,
            R = 0.05, delta = 1e-6, tolerance = 0.002, n_init_samples = 1)

        optimizer.initialize()
        optimizer.optimize()
        return optimizer

    def find_worst_case_velocity(self, vel_interest = 0, scale = 10, B = 1, R = 0.1, tolerance = 0.01):
        bound_list = [np.array([[-0.2*scale],[0.3*scale]]).transpose(), np.array([[-0.15*scale],[0.15*scale]]).transpose()]
        bounds = bound_list[vel_interest]
        def objective(x, vis_flag = False):
            des_vel = x[0,0]/scale
            self.gen_walk_data(velocity = des_vel, vel_interest = vel_interest)
            output_rob = self.walking_robustness(des_vel = des_vel, vel_flag = vel_interest)
            # if output_rob < 0:
            #     self.walking_robustness(des_vel = des_vel, vel_flag = vel_interest, visualizer = True)
            return -output_rob


        optimizer = UCBOptimizer(objective = objective, bounds = bounds, B = B,
            R = R, delta = 1e-6, tolerance = tolerance, n_init_samples = 1,
            length_scale = 1)

        optimizer.initialize()
        optimizer.optimize()
        print('Function minimum value: %.4f'%-optimizer.UCB_val)
        os.system('rm -r data/*')
        return optimizer

    def compare_systems(self):
        bounds = np.array([[0.22, -0.2, -0.2, -0.2],[0.28, 0.2, 0.2, 0.2]]).transpose()
        def objective(x):
            self.gen_gait_data(init_state = x)
            print('Would like for the hardware system to initialize:')
            print('height: %.3f'%x[0,0])
            print('rx: %.3f'%x[0,1])
            print('ry: %.3f'%x[0,2])
            print('rz: %.3f'%x[0,3])
            print(' ')
            sim_rob = self.velocity_robustness()
            nothing = input('Hit enter to start hardware test.')
            self.gen_gait_data(init_state = x, hardware = True)
            hardware_file = "data/latest.h5" #input()
            # hardware_file = 'data/latest.h5'
            hard_rob = self.hardware_vel_rob(filename = hardware_file)
            fin_calc = abs(sim_rob - hard_rob)
            print('Calculated difference between robustness measures: %.4f'%fin_calc)
            print('Hard_rob: %.4f'%hard_rob)
            print(' ')
            return fin_calc

        optimizer = UCBOptimizer(objective = objective, bounds = bounds, B = 0.2,
            R = 0.05, delta = 1e-6, tolerance = 0.005, n_init_samples = 1)

        optimizer.initialize()
        optimizer.optimize()
        return optimizer

    def compare_velocities(self, vel_flag = 0, scale = 10, B = 0.1, R = 0.05, tolerance = 0.001):
        bound_list = [np.array([[-0.2*scale],[0.3*scale]]).transpose(), np.array([[-0.15*scale],[0.15*scale]]).transpose()]
        bounds = bound_list[vel_flag]
        def objective(x, vis_flag = False):
            des_vel = x[0,0]/scale
            self.gen_walk_data(velocity = des_vel, vel_interest = vel_flag)
            sim_rob = self.walking_robustness(des_vel = des_vel, vel_flag = vel_flag)

            nothing = input('Hit enter to start hardware test.')
            self.gen_walk_data(velocity = des_vel, vel_interest = vel_flag, hardware = True)
            hard_rob = self.walking_robustness(des_vel = des_vel, vel_flag = vel_flag)
            fin_calc = np.abs(sim_rob - hard_rob)
            print('Calculated difference between robustness measures: %.4f'%fin_calc)
            print('Hard_rob: %.4f'%hard_rob)
            print(' ')
            return fin_calc

        optimizer = UCBOptimizer(objective = objective, bounds = bounds, B = B,
            R = R, delta = 1e-6, tolerance = tolerance, n_init_samples = 1,
            length_scale = 1)
        optimizer.initialize()
        optimizer.optimize()
        print('Maximum norm difference between robustness measures: %.4f'%optimizer.UCB_val)
        return optimizer

    def animate_data(self, filename = 'data/latest.h5', hardware = False, vel_flag = 0, des_vel = 0):
        self.load_data()
        ctrl_des = self.data[:]['ctrl_mode_curr'].tolist()
        init_index = [idx for idx, s in enumerate(ctrl_des) if b'walk_mpc_idqp(' in s][0]
        smoother = 50
        tube_epsilons = [0.5,0.1,0.05]
        overshoots = [0.3,0.2,0.4]
        settle_times = [0.25, 0.25, 0.35]
        settle_time = settle_times[vel_flag]
        epsilon = tube_epsilons[vel_flag]
        overshoot = overshoots[vel_flag]
        velocity = self.data[init_index-smoother:]['qd'][:,vel_flag]
        et = self.data[init_index-smoother:]['epoch_time']
        n_samples = len(et)
        smoothed_vel = [None for i in range(n_samples-smoother)]
        smoothed_time = [None for i in range(n_samples-smoother)]
        offset = sum(et[0:smoother])/smoother

        for i in range(n_samples - smoother):
            smoothed_vel[i] = sum(velocity[i:smoother+i])/smoother
            smoothed_time[i] = sum(et[i:smoother+i])/smoother - offset

        print('Determining starting/finishing times')
        start_time_mark = self.find_first_index(smoothed_time, lambda x: x>=settle_time)
        end_time_mark = self.find_first_index(smoothed_time, lambda x: x>=settle_time + 1)

        fig, ax = plt.subplots()
        ax.plot(smoothed_time[start_time_mark:end_time_mark], des_vel*np.ones(end_time_mark-start_time_mark,),
            color = 'green', ls = '--', lw = 2)
        ax.hlines(y = des_vel + epsilon, xmin = smoothed_time[start_time_mark], xmax = smoothed_time[end_time_mark],
            ls = '--', lw = 2, color = 'red')
        ax.hlines(y = des_vel - epsilon, xmin = smoothed_time[start_time_mark], xmax = smoothed_time[end_time_mark],
            ls = '--', lw = 2, color = 'red')
        if des_vel >= 0:
            min_bound = des_vel - epsilon
            max_bound = max((des_vel + overshoot, des_vel + epsilon))
        else:
            min_bound = min((des_vel - overshoot, des_vel - epsilon))
            max_bound = des_vel + epsilon
        ax.vlines(smoothed_time[start_time_mark], min_bound, max_bound, color = 'red', lw = 2, ls = '--')
        if des_vel >= 0:
            ax.plot(smoothed_time[:start_time_mark],
                (des_vel+overshoot)*np.ones(start_time_mark,), color = 'blue', ls = '--', lw = 2)
            ax.plot(smoothed_time[:start_time_mark],
                des_vel*np.ones(start_time_mark,), color = 'blue', ls = '--', lw = 2)
        else:
            ax.plot(smoothed_time[:start_time_mark],
                (des_vel-overshoot)*np.ones(start_time_mark,), color = 'blue', ls = '--', lw = 2)
            ax.plot(smoothed_time[:start_time_mark],
                des_vel*np.ones(start_time_mark,), color = 'blue', ls = '--', lw = 2)
        ax.set_xlabel('time (sec)')
        ax.set_ylabel('v', rotation = 0)
        ax.yaxis.set_label_coords(-0.075,0.45)
        if vel_flag == 1: ax.set_ylim(-0.05,0.45)
        ax.set_title('Tracking Quadruped Forward Velocity')

        if hardware:
            ax.plot(smoothed_time, smoothed_vel, lw = 3, color = 'black', label = 'sim')
            self.load_data(filename)
            ctrl_des = self.data[:]['ctrl_mode_des'].tolist()
            init_index = ctrl_des.index(b'walk_mpc_idqp(0.25,0.30,0.00,0.0')
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
            line, = ax.plot([],[],lw = 3, color = 'blue', label = 'hardware')
            ax.legend(loc = 'best')
        else:
            line, = ax.plot([],[], lw = 3, color = 'black')

        def animate(frame_number):
            line.set_data(smoothed_time[:frame_number], smoothed_vel[:frame_number])
            return line,


        anim = animation.FuncAnimation(fig, animate, frames = end_time_mark, interval = 0.01, blit = False)
        writervideo = animation.FFMpegWriter(fps=60)
        if hardware:
            anim.save('comparison.mp4',writer=writervideo)
        elif vel_flag == 0:
            anim.save('Figure/vx_worst_case.mp4',writer=writervideo)
        else:
            anim.save('Figure/vy_worst_case.mp4',writer=writervideo)
        pass

    def make_figures(self, vel_interest = 0, opt_values = None):
        scale = 10
        steps = 100
        if vel_interest == 0 and opt_values is None:
            X_sample = np.load('Figure/vx_X_sample.npy')
            Y_sample = np.load('Figure/vx_Y_sample.npy')
            mean = np.load('Figure/vx_mean.npy').tolist()
            ub = np.load('Figure/vx_ub.npy').tolist()
            lb = np.load('Figure/vx_lb.npy').tolist()
            xspace = np.linspace(-0.2,0.3,steps).tolist()
        elif vel_interest == 1 and opt_values is None:
            X_sample = np.load('Figure/vy_X_sample.npy')
            Y_sample = np.load('Figure/vy_Y_sample.npy')
            mean = np.load('Figure/vy_mean.npy').tolist()
            ub = np.load('Figure/vy_ub.npy').tolist()
            lb = np.load('Figure/vy_lb.npy').tolist()
            xspace = np.linspace(-0.2,0.2,steps).tolist()
        else:
            X_sample = opt_values[0]
            Y_sample = opt_values[1]
            mean = opt_values[2]
            ub = opt_values[3]
            lb = opt_values[4]
            xspace = opt_values[5]
            max_val = opt_values[6]


        # plt.rcParams.update({'font.size': 24})
        fig, ax = plt.subplots()
        ax.scatter(X_sample/scale, Y_sample, marker = 'x', s = 200, linewidth = 4, color = 'red', label = 'samples')
        ax.vlines(X_sample[-1,0]/scale, ymin = min(lb), ymax = max(ub), linewidth = 4, linestyle = '--', color = 'green')
        ax.hlines(max_val, xmin = min(xbase), xmax = max(xbase), linewidth = 4, linestyle = '--', color = 'green')
        ax.plot(xspace, mean, color = 'black', lw = 8, label = r'$\mu$')
        shading = ax.fill_between(xspace, lb, ub, color = colors['gray'], alpha = 0.5, lw = 3)
        if vel_interest == 0:
            ax.set_xlabel(r'$v^{\mathrm{req}}_x$', fontsize = 30)
        else:
            ax.set_xlabel(r'$v^{\mathrm{req}}_y$', fontsize = 30)
        ax.set_ylabel(r'$\Delta \rho$', fontsize = 30, rotation = 0)
        ax.yaxis.set_label_coords(-0.09,0.45)
        ax.set_title(r'$\mathrm{Sim2Real~Gap~Identification}$', fontsize = 34)
        ax.legend(loc = 'best')
        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        figinfo = [X_sample, Y_sample, mean, ub, lb, xspace, vel_interest]
        pickle.dump(figinfo, open('worst_x.pickle', 'wb'))
        plt.show()
        pass

    def worst_case_goal_finding(self):
        bounds = np.array([[-2,-2],[2,2]]).transpose()
        def objective(x):
            self.setup(show_output = False, hardware = False)
            time.sleep(1)
            x_traj, u_traj, u_des_traj, h_traj, controller = sim_controller(goal = x.reshape(2,1), obstacles = None, steps = 100, dt = 0.1)
            self.kill_process(self.process_list)
            r_values = [0.1 - np.linalg.norm(x_traj[i,:].reshape(2,1)) for i in range(100)]
            return -max(r_values)

        optimizer = UCBOptimizer(objective = objective, bounds = bounds, B = 1, R = 0.05, delta = 1e-6, tolerance = 0.002,
            n_init_samples = 1, length_scale = 1)

        optimizer.initialize()
        optimizer.optimize()
        print('Function minimum value: %.4f'%-optimizer.UCB_val)
        os.system('rm -r data/*')
        return optimizer

    def goal_finding_comparison(self):
        bounds = np.array([[-2,-2],[2,2]]).transpose()
        def objective(x):
            self.setup(show_output = False, hardware = False)
            time.sleep(1)
            x_traj, u_traj, u_des_traj, h_traj, controller = sim_controller(goal = x.reshape(2,1), obstacles = None, steps = 100, dt = 0.1)
            self.kill_process(self.process_list)
            sim_r_values = [0.1 - np.linalg.norm(x_traj[i,:].reshape(2,1)) for i in range(100)]
            sim_value = max(sim_r_values)

            self.setup(show_output = False, hardware = True)
            time.sleep(1)
            x_traj, u_traj, u_des_traj, h_traj, controller = sim_controller(goal = x.reshape(2,1), obstacles = None, steps = 100, dt = 0.1)
            self.kill_process(self.process_list)
            hard_r_values = [0.1 - np.linalg.norm(x_traj[i,:].reshape(2,1)) for i in range(100)]
            hard_value = max(hard_r_values)

            difference = np.abs(sim_r_values - hard_r_values)
            return difference

        optimizer = UCBOptimizer(objective = objective, bounds = bounds, B = 1, R = 0.05, delta = 1e-6, tolerance = 0.002,
            n_init_samples = 1, length_scale = 1)

        optimizer.initialize()
        optimizer.optimize()
        print('Function minimum value: %.4f'%-optimizer.UCB_val)
        os.system('rm -r data/*')
        return optimizer




if __name__ == '__main__':
    commander = info_parser()

    # First runs:
    commander.compare_velocities(vel_flag = 0, scale = 1, B = 1, R = 0.05, tolerance = 0.005)
    xbase = [value/scale for value in optimizer.xbase]
    opt_values = [optimizer.X_sample, optimizer.Y_sample, optimizer.mean_values,
        optimizer.ub, optimizer.lb, xbase, -optimizer.UCB_val]
    filename = "saved_info/comparing_velocities.txt"
    pickle.dump(opt_values, open(filename, "wb"))
    print('Finished analysis')

    # Simulation Analysis - Repeatability
    # scale = 1
    # vel_interest = 0
    # B_values = [round(3 + 0.2*i,2) for i in range(6)]
    # print(B_values)
    # for B in B_values:
    #     for n in range(20):
    #         start = time.time()
    #         optimizer = commander.find_worst_case_velocity(vel_interest = vel_interest, scale = scale, B = B, R = 0.05, tolerance = 0.02)
    #         elapsed = time.time() - start
    #         print('Total process took: {} minutes'.format(round(elapsed/60,2)))
    #         xbase = [value/scale for value in optimizer.xbase]
    #         opt_values = [optimizer.X_sample, optimizer.Y_sample, optimizer.mean_values,
    #             optimizer.ub, optimizer.lb, xbase, -optimizer.UCB_val]
    #         filename = "saved_info/{}_{}.txt".format(B, n)
    #         pickle.dump(opt_values,open(filename, "wb"))
    #         print('Finished with B = {} run = {}'.format(B, n))
    #         print(' ')


    # Simulation Analysis - Walking
    optimizer = commander.worst_case_goal_finding()
    opt_values = [optimizer.X_sample, optimizer.Y_sample, optimizer.mean_values,
        optimizer.ub, optimizer.lb, xbase, -optimizer.UCB_val]
    filename = "saved_info/future_analysis.txt"
    pickle.dump(opt_values, open(filename, "wb"))
    print('Finished analysis')
