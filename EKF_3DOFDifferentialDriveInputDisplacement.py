from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *

class EKF_3DOFDifferentialDriveInputDisplacement(GFLocalization, DR_3DOFDifferentialDrive, EKF):
    """
    This class implements an EKF localization filter for a 3 DOF Diffenteial Drive using an input displacement motion model incorporating
    yaw measurements from the compass sensor.
    It inherits from :class:`GFLocalization.GFLocalization` to implement a localization filter, from the :class:`DR_3DOFDifferentialDrive.DR_3DOFDifferentialDrive` class and, finally, it inherits from
    :class:`EKF.EKF` to use the EKF Gaussian filter implementation for the localization.
    """
    def __init__(self, kSteps, robot, *args):
        """
        Constructor. Creates the list of  :class:`IndexStruct.IndexStruct` instances which is required for the automated plotting of the results.
        Then it defines the inital stawe vecto mean and covariance matrix and initializes the ancestor classes.

        :param kSteps: number of iterations of the localization loop
        :param robot: simulated robot object
        :param args: arguments to be passed to the base class constructor
        """

        self.dt = 0.1  # dt is the sampling time at which we iterate the KF
        x0 = np.zeros((3, 1))  # initial state x0=[x y z psi u v w r]^T
        P0 = np.zeros((3, 3))  # initial covariance

        # this is required for plotting
        index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("z", 2, 0), IndexStruct("yaw", 3, 1)]

        self.t_1 = 0
        self.t = 0
        self.Dt = self.t - self.t_1
        super().__init__(index, kSteps, robot, x0, P0, *args)

    def f(self, xk_1, uk):
        # TODO: To be completed by the student

        #xk_bar = Pose3D(xk_1).oplus(Pose3D(uk))  # predicted state vector
        # but for computing jacobians is more interesting to use the following form of the motion model
        x, y, psi = xk_1[0, 0], xk_1[1, 0], xk_1[2, 0]
        dx, dy, dpsi = uk[0, 0], uk[1, 0], uk[2, 0]

        xk_bar = np.array([
            [x + dx * np.cos(psi) - dy * np.sin(psi)],
            [y + dx * np.sin(psi) + dy * np.cos(psi)],
            [psi + dpsi]
        ])
        return xk_bar

    def Jfx(self, xk_1):
        # TODO: To be completed by the student

        psi =  xk_1[2, 0]
        dx, dy = self.uk[0, 0], self.uk[1, 0]

        J = np.array([
            [1.0, 0.0, -dx * np.sin(psi) - dy * np.cos(psi)],
            [0.0, 1.0, dx * np.cos(psi) - dy * np.sin(psi)],
            [0.0, 0.0, 1.0]
        ])

        return J

    def Jfw(self, xk_1):
        # TODO: To be completed by the student
        
        #the noise is inside the input vector, so the jacobian of the motion model with respect to the noise vector is the same as the jacobian of the motion model with respect to the input vector
        psi =  xk_1[2, 0]
        J = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        return J

    def h(self, xk):  #:hm(self, xk):
        # TODO: To be completed by the student

        h = np.array([[xk[2, 0]]])
        return h  # return the expected observations

    def GetInput(self):
        """

        :return: uk,Qk
        """
        # TODO: To be completed by the student
        zsk, Rsk = self.robot.ReadEncoders()
        n_L = zsk[0, 0]  # left wheel encoder reading
        n_R = zsk[1, 0]  # right wheel encoder reading    
        
        # metros
        kp = (np.pi * self.robot.wheelRadius) / self.robot.pulse_x_wheelTurns
        d_L = 2 * kp * n_L 
        d_R = 2 * kp * n_R
        
        # displacement and change in orientation

        d = (d_R + d_L) / 2.0
        d_psi = (d_R - d_L) / self.robot.wheelBase

        uk = np.array([[d], [0.0],[d_psi]])  # input vector. Y = 0 since we are not considering lateral displacement in this model

        # noise covariance matrix
        # we have the covariance of the encoder readings in pulses. We need to convert it to meters and radians
        # we need to compute the Jacobian of the transformation from pulses to meters and radians
        # the first column will be the derivative of u with respect to n_L, the second column will be the derivative of d_R with respect to n_L and n_R, and the third column will be the derivative of u with respect to n_R

        Ju = np.array([[kp, kp],
                       [0, 0],
                       [-2 * kp / self.robot.wheelBase,
                        2 * kp / self.robot.wheelBase]])

        Qk = Ju @ Rsk @ Ju.T

        return uk, Qk

    def GetMeasurements(self):  # override the observation model
        """

        :return: zk, Rk, Hk, Vk
        """
        # TODO: To be completed by the student
        # we shold manage the case when the compass is not available, in that case we should return None for zk and Rk, and Hk and Vk should be empty matrices
        if self.k % self.robot.yaw_reading_frequency == 0:
            yaw, R_yaw = self.robot.ReadCompass()
            zk = np.array([[yaw]])
            Rk = np.array([[R_yaw]])
            Hk = np.array([[0, 0, 1]])
            Vk = np.array([[1]])
        else:
            zk = None
            Rk = None
            Hk = None
            Vk = None

    
        return zk, Rk, Hk, Vk


if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6,1))  # initial simulated robot pose

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object
    kSteps = 5000

    xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    x0 = np.zeros((3, 1))
    P0 = np.zeros((3, 3))

    dd_robot = EKF_3DOFDifferentialDriveInputDisplacement(kSteps,robot)  # initialize robot and KF
    dd_robot.LocalizationLoop(x0, P0, np.array([[0.5, 0.03]]).T)

    exit(0)