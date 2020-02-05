#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ARM3DOF(object):
    joint_dof_ = 3
    task_dof_ = 3
    L1 = 3.
    L2 = 3.
    L3 = 3.

    def __init__(self):
        a = 0
    
    def kinematics(self, q):
        x = np.zeros((self.task_dof_, 1))
        x[0][0] = self.L1*np.cos(q[0][0]) + self.L2*np.cos(q[0][0] + q[1][0]) + self.L3*np.cos(q[0][0] + q[1][0] + q[2][0])
        x[1][0] = self.L1*np.sin(q[0][0]) + self.L2*np.sin(q[0][0] + q[1][0]) + self.L3*np.sin(q[0][0] + q[1][0] + q[2][0])
        x[2][0] = q[0][0] + q[1][0] + q[2][0]
        return x

    def jacobian(self, q):
        jacob = np.zeros((self.task_dof_, self.joint_dof_))
        jacob[0][0] = -self.L1*np.sin(q[0][0]) - self.L2*np.sin(q[0][0] + q[1][0]) - self.L3*np.sin(q[0][0] + q[1][0] + q[2][0])
        jacob[0][1] = -self.L2*np.sin(q[0][0] + q[1][0]) - self.L3*np.sin(q[0][0] + q[1][0] + q[2][0])
        jacob[0][2] = -self.L3*np.sin(q[0][0] + q[1][0] + q[2][0])
        jacob[1][0] =  self.L1*np.cos(q[0][0]) + self.L2*np.cos(q[0][0] + q[1][0]) + self.L3*np.cos(q[0][0] + q[1][0] + q[2][0])
        jacob[1][1] =  self.L2*np.cos(q[0][0] + q[1][0]) + self.L3*np.cos(q[0][0] + q[1][0] + q[2][0])
        jacob[1][2] =  self.L3*np.cos(q[0][0] + q[1][0] + q[2][0])
        jacob[1][0] =  1
        jacob[1][1] =  1
        jacob[1][2] =  1
        return jacob

    def joint_dof(self):
        return self.joint_dof_

    def task_dof(self):
        return self.task_dof_
    
    def joint3pos(self, q):
        x = np.zeros((self.task_dof_, 1))
        x[0][0] = self.L1*np.cos(q[0][0]) + self.L2*np.cos(q[0][0] + q[1][0]) + self.L3*np.cos(q[0][0] + q[1][0] + q[2][0])
        x[1][0] = self.L1*np.sin(q[0][0]) + self.L2*np.sin(q[0][0] + q[1][0]) + self.L3*np.sin(q[0][0] + q[1][0] + q[2][0])
        return x

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



def video(robot, q1, q2, q3, t, fig):
    alpha = 0.2
    ax = fig.add_subplot(222, aspect='equal', ylabel='y', xlabel='x', xlim=(-6.0-alpha, 6.0+alpha), ylim=(-6.0-alpha, 6.0+alpha))
    ax.grid()

    time_text = ax.text(0.02, 1.1, 'time = 0.0', transform=ax.transAxes)

    #link
    link1, = ax.plot([], [], '-r', lw=2)
    link2, = ax.plot([], [], '-r', lw=2)
    link3, = ax.plot([], [], '-r', lw=2)    

    # joint circle
    joint1, = ax.plot([], [], '-r', lw=2)
    joint2, = ax.plot([], [], '-r', lw=2)
    joint3, = ax.plot([], [], '-r', lw=2)
    base, = ax.plot([], [], '-r', lw=2)
    radius = 0.1
    angles = np.arange(0.0, np.pi * 2.0, np.radians(3.0))
    ox = radius * np.cos(angles) 
    oy = radius * np.sin(angles)

    def init():
        link1.set_data([], [])
        link2.set_data([], [])
        link3.set_data([], [])
        joint1.set_data([], [])
        joint2.set_data([], [])
        joint3.set_data([], [])
        base.set_data([], [])
        time_text.set_text('')
        return time_text

    def animate(i):
        q = np.array([[q1[i]],[q2[i]],[q3[i]]])
        j1 = robot.joint1pos(q)
        j2 = robot.joint2pos(q)
        j3 = robot.joint3pos(q)
        link1.set_data([0, j1[0][0]], [0, j1[1][0]])
        link2.set_data([j1[0][0], j2[0][0]], [j1[1][0], j2[1][0]])
        link3.set_data([j2[0][0], j3[0][0]], [j2[1][0], j3[1][0]])
        joint1.set_data([ox + j1[0][0]],[ oy + j1[1][0]])
        joint2.set_data([ox + j2[0][0]],[ oy + j2[1][0]])
        joint3.set_data([ox + j3[0][0]],[ oy + j3[1][0]])
        base.set_data([ox + 0],[ oy + 0])
        time_text.set_text('time = {0:.2f}'.format(i*t))
        return time_text

    ani = animation.FuncAnimation(fig, animate, frames=range(200),
                                  interval=t*100, blit=False, init_func=init)
    plt.show()
    #ani.save("output.gif", writer="imagemagick")

def plan_orbit(t):
    if t >= 0 and t < np.pi:
        X = np.array([[3.*np.cos(t)],[3.*np.sin(t)], [0]])
    elif t >= np.pi and t < 3.*np.pi/2.:
        X = np.array([[3.*np.cos(-(t-np.pi)+np.pi/2)-3],[3.*np.sin(-(t-np.pi)+np.pi/2)-3.], [0]])
    else:
        X = np.array([[3.*np.cos(-t-3.*np.pi/2.)+3.],[3.*np.sin(-t-3.*np.pi/2.)-3.], [0]])
    return X

if __name__ == '__main__':
    dt = 0.01
    step = 200
    robot = ARM3DOF()
    IK = LMIK(robot)

    t = np.linspace(0, 2*np.pi, step)
    X = plan_orbit(t[0])
    q = np.zeros((robot.joint_dof(), 1))
    X_history = np.array(X)
    q = IK.inverse_kinematics(q, X)
    q_history = np.array(q)
    for i in range(step-1):
        X = plan_orbit(t[i])
        X_history = np.append(X_history, X, axis=1)
        q = IK.inverse_kinematics(q, X)
        q_history = np.append(q_history, q, axis=1)

    t = np.arange(0, step*dt, dt)

    fig = plt.figure(tight_layout=True)

    ax1 = fig.add_subplot(221, aspect='equal', title='target endpoint trajectory', ylabel='y', xlabel='x')
    ax1.plot(X_history[0], X_history[1])

    ax2 = fig.add_subplot(212, title='joint trajectory', ylabel='theta', xlabel='t')
    ax2.plot(t, q_history[0], label="joint1")
    ax2.plot(t, q_history[1], label="joint2")
    ax2.plot(t, q_history[2], label="joint3")
    ax2.legend()
    
    video(robot, q_history[0], q_history[1], q_history[2], dt, fig)
