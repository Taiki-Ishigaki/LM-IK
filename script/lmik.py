#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ARM2DOF(object):
    joint_dof_ = 2
    task_dof_ = 2
    L1 = 1
    L2 = 1

    def __init__(self):
        a = 0
    
    def kinematics(self, q):
        x = np.zeros((self.task_dof_, 1))
        x[0][0] = self.L1*np.cos(q[0][0]) + self.L2*np.cos(q[0][0] + q[1][0])
        x[1][0] = self.L1*np.sin(q[0][0]) + self.L2*np.sin(q[0][0] + q[1][0])
        return x

    def jacobian(self, q):
        jacob = np.zeros((self.task_dof_, self.joint_dof_))
        jacob[0][0] = -self.L1*np.sin(q[0][0]) - self.L2*np.sin(q[0][0] + q[1][0])
        jacob[0][1] = -self.L2*np.sin(q[0][0] + q[1][0])
        jacob[1][0] =  self.L1*np.cos(q[0][0]) + self.L2*np.cos(q[0][0] + q[1][0])
        jacob[1][1] =  self.L2*np.cos(q[0][0] + q[1][0])
        return jacob

    def joint_dof(self):
        return self.joint_dof_

    def task_dof(self):
        return self.task_dof_

    def joint2pos(self, q):
        x = np.zeros((self.task_dof_,1))
        x[0][0] = self.L1*np.cos(q[0][0]) + self.L2*np.cos(q[0][0] + q[1][0])
        x[1][0] = self.L1*np.sin(q[0][0]) + self.L2*np.sin(q[0][0] + q[1][0])
        return x

    def joint1pos(self, q):
        x = np.zeros((self.task_dof_,1))
        x[0][0] = self.L1*np.cos(q[0][0])
        x[1][0] = self.L1*np.sin(q[0][0])
        return x



class LMIK(object):
    EPS = 1E-5
    def __init__(self, robot_):
        self.robot = robot_
        w_E = 1
        daig_w_E = np.full(self.robot.task_dof(), w_E)
        self.W_E = np.diag(daig_w_E)
        w_N_ = 0.001
        daig_w_N_ = np.full(self.robot.joint_dof(), w_N_)
        self.W_N_ = np.diag(daig_w_N_)
    
    def evaluate(self, e):
        value = e.transpose() @ self.W_E @ e / 2 
        return value

    def inverse_kinematics(self, q_, pd):
        q = q_
        e = pd - self.robot.kinematics(q)
        for k in range(100):
            jacob = self.robot.jacobian(q)
            W_N = self.evaluate(e) * np.identity(self.robot.joint_dof()) + self.W_N_
            H = jacob.transpose() @ self.W_E @ jacob + W_N
            g = jacob.transpose() @ self.W_E @ e
            q += np.linalg.inv(H) @ g
            e = pd - self.robot.kinematics(q)
            if self.evaluate(e) < self.EPS:
                break
        return q



def video(robot, q1, q2, t, fig):
    alpha = 0.2
    ax = fig.add_subplot(222, aspect='equal', ylabel='y', xlabel='x', xlim=(-2.0-alpha, 2.0+alpha), ylim=(-2.0-alpha, 2.0+alpha))
    ax.grid()

    time_text = ax.text(0.02, 1.1, 'time = 0.0', transform=ax.transAxes)

    #link
    link1, = ax.plot([], [], '-r', lw=2)
    link2, = ax.plot([], [], '-r', lw=2)

    # joint circle
    joint1, = ax.plot([], [], '-r', lw=2)
    joint2, = ax.plot([], [], '-r', lw=2)
    base, = ax.plot([], [], '-r', lw=2)
    radius = 0.1
    angles = np.arange(0.0, np.pi * 2.0, np.radians(3.0))
    ox = radius * np.cos(angles) 
    oy = radius * np.sin(angles)

    def init():
        link1.set_data([], [])
        link2.set_data([], [])
        joint1.set_data([], [])
        joint2.set_data([], [])
        base.set_data([], [])
        time_text.set_text('')
        return time_text

    def animate(i):
        q = np.array([[q1[i]],[q2[i]]])
        j1 = robot.joint1pos(q)
        j2 = robot.joint2pos(q)
        link1.set_data([0, j1[0][0]], [0, j1[1][0]])
        link2.set_data([j1[0][0], j2[0][0]], [j1[1][0], j2[1][0]])
        joint1.set_data([ox + j1[0][0]],[ oy + j1[1][0]])
        joint2.set_data([ox + j2[0][0]],[ oy + j2[1][0]])
        base.set_data([ox + 0],[ oy + 0])
        time_text.set_text('time = {0:.2f}'.format(i*t))
        return time_text

    ani = animation.FuncAnimation(fig, animate, frames=range(400),
                                  interval=t*100, blit=False, init_func=init)
    #plt.show()
    ani.save("output.gif", writer="imagemagick")

def plan_orbit1(x):
    y = -1/2*(x**2) + 2
    return y

def plan_orbit2(x):
    y = -np.sqrt(4 - x**2)
    return y

if __name__ == '__main__':
    dt = 0.01
    step = 200
    robot = ARM2DOF()
    IK = LMIK(robot)

    x = np.linspace(2, -2, step)
    X = np.array([[x[0]], [plan_orbit1(x[0])]])
    q = np.zeros((2, 1))
    X_history = np.array(X)
    q = IK.inverse_kinematics(q, X)
    q_history = np.array(q)
    for i in range(step-1):
        X = np.array([[x[i]], [plan_orbit1(x[i])]])
        X_history = np.append(X_history, X, axis=1)
        q = IK.inverse_kinematics(q, X)
        # print(q)
        q_history = np.append(q_history, q, axis=1)

    x = np.linspace(-2, 2, step)
    for i in range(step):
        X = np.array([[x[i]], [plan_orbit2(x[i])]])
        X_history = np.append(X_history, X, axis=1)
        q = IK.inverse_kinematics(q, X)
        q_history = np.append(q_history, q, axis=1)

    t = np.arange(0, 2*step*dt, dt)

    fig = plt.figure(tight_layout=True)

    ax1 = fig.add_subplot(221, aspect='equal', title='target endpoint trajectory', ylabel='y', xlabel='x')
    ax1.plot(X_history[0], X_history[1])

    ax2 = fig.add_subplot(212, title='joint trajectory', ylabel='theta', xlabel='t')
    ax2.plot(t, q_history[0], label="joint1")
    ax2.plot(t, q_history[1], label="joint2")
    
    video(robot, q_history[0], q_history[1], dt, fig)
