######  PROGRAM MEMANGGIL WINDOWS PYQT5 ##########################

####### memanggil library PyQt5 ##################################
#----------------------------------------------------------------#
from PyQt5.QtCore import * 
from PyQt5.QtGui import * 
from PyQt5.QtQml import * 
from PyQt5.QtWidgets import *
from PyQt5.QtQuick import *  
import sys
import cvxpy as cp
from scipy.linalg import expm
import math
#----------------------------------------------------------------#

import numpy as np
from filterpy.kalman import KalmanFilter


pubdelay = 2 #delay publish to all wind and engine box
counter = 0



start_lat = -6.215861
start_lon = 107.803706

latitude = -6.215861
latitude_dot = 0.00001
longitude = 107.803706
longitude_dot = 0.00001
yaw = 0


delta_lat = 0
delta_lon = 0
sp_lat = -6.215861
sp_lon = 107.803706
sp_yaw = 0

delta_x = 0.00001
delta_y = 0.0000
theta_dot = 1

x_dot = 0
y_dot = 0
eta = np.array([[x_dot], [y_dot], [yaw]])
V = np.array([[latitude_dot], [longitude_dot], [yaw]]) 

x_target = 0
y_target = 0


heading_error = 0

steering1 = 0
steering2 = 0
steering3 = 0
steering4 = 0

gas_throttle1 = 0
gas_throttle2 = 0
gas_throttle3 = 0
gas_throttle4 = 0

def coordinate_conv(x, y, theta):
    j_theta = np.array([[np.cos(theta * float(np.pi/180)), -np.sin(theta * float(np.pi/180)), 0],
              [np.sin(theta * float(np.pi/180)), np.cos(theta* float(np.pi/180)), 0],
              [0, 0, 1]])
    
    result = j_theta @ np.array([[x], [y], [0]])  # Gunakan 0 untuk rotasi biasa

    delta_lat = result[0, 0]  # Mengambil nilai skalar dari array
    delta_lon = result[1, 0]

    return delta_lat, delta_lon



def rotation(x, y, theta):
    j_theta = np.array([[np.cos(theta * float(np.pi/180)), -np.sin(theta * float(np.pi/180)), 0],
              [np.sin(theta * float(np.pi/180)), np.cos(theta* float(np.pi/180)), 0],
              [0, 0, 1]])
    
    result = j_theta @ np.array([[x], [y], [0]])  # Gunakan 0 untuk rotasi biasa

    x_accent = result[0, 0]  # Mengambil nilai skalar dari array
    y_accent = result[1, 0]
    theta = theta

    return x_accent, y_accent, theta

'''
def meter_conversion(lat1, long1, lat2, long2):
    delta_lat = (lat1 - lat2)*111000
    delta_lon = (long1 - long2)*111000
    distance = sqrt(pow(delta_lat, 2) +  pow(delta_lon, 2))
    return distance
'''

def meter_conversion(coord1, coord1_alt, coord2, coord2_alt):
    return (coord2 - coord1) * 111000  # Faktor konversi kasar: 1 derajat ≈ 111.139 km


def shortest_psi(psi_ref, psi_d):
    psi_temp = (psi_ref-psi_d)%360
    psi_shortest = (psi_temp + 360) *-1 %360 
    if (psi_shortest > 180):
        psi_shortest = psi_shortest - 360
    return psi_shortest




import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, lsim
from math import sin, cos, sqrt, atan2, radians, atan


############ Model Kinetik ################################

print("==== Kinetik =======")
xg = 0.5 #posisi x center of gravity
yg = 0.4 #posisi y center of gravity
mass = 1000 #massa kapal
r = 0 #posisi arah surge / kecepatan sudut (psi_dot)
Iz = 0 #momen inersia akibat percepatan sumbu y

x_u = 0.9
y_v = 0.6
n_r = 0.3
y_r = 0.1
n_v = 0.2

x_udot = 0.9
y_vdot = 0.6
y_rdot = 0.7
n_vdot = 0.3
n_rdot = 0.4
u = 0
v = 0
r = 0

j_theta = np.array([[0,0,0],[0,0,0],[0,0,0]])
v = np.array([[0],[0],[0]])
n_error = 0
e_error = 0
error_body_fixed = np.array([[0],[0],[0]])

x_error = 0
y_error = 0

#gaya akibat massa


#dari thesis teguh
'''
m = np.array([[mass, 0, -mass*yg], [0, mass, mass*xg], [-mass*yg, mass*xg, Iz]])
+ np.array([[-x_udot,0,0],[0,-y_vdot,-y_rdot],[0,-n_vdot,-n_rdot]])
print(m)
'''
#dari Thomas P. DeRensis
m = np.array([[mass, -0, 0], [0.0, mass, mass*xg], [-0.00, mass*xg, Iz]])
+ np.array([[-x_udot,-0.00,0.00],[-0.00,y_vdot,y_rdot],[-0.00,n_vdot,n_rdot]])
print(m)



#gaya akibat coriolis
c = np.array([ [0, 0, -mass * (xg*r + float(y_dot))],
               [0, 0, -mass * (yg*r + float(x_dot))],
               [-mass * (xg*r + float(y_dot)),-mass * (yg*r + float(x_dot)), 0]]) + np.array([[0,0,-y_vdot*float(y_dot) - ((y_rdot+n_vdot)/2)*r],
            [0,0,x_udot*float(x_dot)],
            [-y_vdot*float(y_dot) - ((y_rdot+n_vdot)/2)*r,x_udot*float(x_dot),0]])

#gaya akibat drag
d = np.array([[x_u,-0.00,-0.00],
    [-0.00,y_v,y_r],
    [-0.00,n_v,n_r]]) 

print("d",d)



# Matriks State-Space
'''
A = np.block([
    [np.zeros((3, 3)), np.eye(3)],
    [-np.linalg.inv(m) @ d, -np.linalg.inv(m) @ c]
])
'''

A = np.block([
    [np.zeros((3, 3)), np.eye(3)],
    [np.zeros((3, 3)), -np.linalg.inv(m) @ d]
])

B = np.block([
    [np.zeros((3, 3))],
    [np.linalg.inv(m)]
])
C = np.block([
    [np.eye(3), np.zeros((3, 3))]
])
D = np.zeros((3, 3))

x_next = np.array([[0], [0], [0], [0], [0], [0]])
 

x = np.array([[0], [0], [0], [0], [0], [0]]) 
u_optimal = np.array([[0], [0], [0]])
y = np.array([[0], [0.0], [0]])

print("Continous state space : ")
print("A")
print(str(A))

print("B")
print(B)

print("C")
print(C)


print("D")
print(D)

# Mendiskretisasi matriks A dan B
# Matriks identitas
I = np.eye(A.shape[0])
T = 1

# Menghitung Ad dan Bd dengan Tustin
A = np.linalg.inv(I - (T/2) * A) @ (I + (T/2) * A)
B = np.linalg.inv(I - (T/2) * A) @ (T * B)

# Cd dan Dd tetap sama
C = C
D = D


print("Discrete state space : ")
print("A")
print(str(A))

print("B")
print(B)

print("C")
print(C)


print("D")
print(D)


# Matriks T untuk 4 baling-baling
T = np.array([[1, 0, 1, 0, 1, 0, 1, 0],   # Menambah kolom untuk F_x4
              [0, 1, 0, 1, 0, 1, 0, 1],   # Menambah kolom untuk F_y4
              [-0.25, 1, -0.25, -1, 0.25, -1, 0.25, 1]])  # Menyesuaikan gaya kontrol
#-ly1 lx1 -fy2 -fx2 fy3 -fx3 fy4 fx4

T_transpose = T.T
W = np.eye(8)
W_inv = np.linalg.inv(W)
TWT_inv = np.linalg.inv(T @ W_inv @ T_transpose)
T_pseudo_inverse = W_inv @ T_transpose @ TWT_inv
tau_control = np.array([0, 0, 10])

# ===== Parameter MPC =====
N = 10  # Prediction horizon
#Q =  10000
Q = np.diag([10000, 10000, 10000])  # Penalty for output error (adjusted for 3 outputs)
R = np.diag([1, 1, 1])  # Penalty for control effort (adjusted for 3 inputs)
delta_u_penalty = np.diag([10, 10, 10])  # Penalti perubahan kontrol (adjusted for 3 inputs)
u_min, u_max = -50, 50  # Batas kontrol

# ===== Variabel Simulasi =====
x0 = np.array([[0], [0], [yaw], [0], [0], [0]])   # Status awal
print("x0 =", x0)
predicted_states = []
applied_inputs = []
time_steps = []
y = np.array([[0], [0.0], [0]])
y_ref = np.array([0, 0, 10]).reshape(-1, 1)

heading_error = 0

# VARIABEL KALMAN FILTER
dim_x = 6  
dim_z = 3

kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

kf.F = A
kf.B = B
kf.H = C

kf.Q = np.eye(dim_x) * 1e-4  # Noise proses
kf.R = np.diag([0.1, 0.1, 0.1])  # Noise pengukuran
kf.P = np.eye(dim_x) * 1.0  # Kovariansi awal

kf.x = x



print("kalman filter succesfully defined ... ")
########## mengisi class table dengan instruksi pyqt5#############
#----------------------------------------------------------------#
class table(QObject):    
    def __init__(self, parent = None):
        super().__init__(parent)
        self.app = QApplication(sys.argv)
        self.engine = QQmlApplicationEngine(self)
        self.engine.rootContext().setContextProperty("backend", self)    
        self.engine.load(QUrl("main.qml"))
        sys.exit(self.app.exec_())
        
    @pyqtSlot(result=float)
    def latitude(self):return round(latitude,7)
    
    @pyqtSlot(result=float)
    def longitude(self):return round(longitude,7)
    
    @pyqtSlot(result=float)
    def start_lat(self):return round(start_lat,7)
    
    @pyqtSlot(result=float)
    def start_lon(self):return round(start_lon,7)
    
    
    @pyqtSlot(result=float)
    def delta_lat(self):return round(delta_lat)
    
    @pyqtSlot(result=float)
    def delta_lon(self):return round(delta_lon)
    
    
    
    @pyqtSlot(result=float)
    def steering1(self):return round(steering1)
    
    @pyqtSlot(result=float)
    def steering2(self):return round(steering2)
    
    @pyqtSlot(result=float)
    def steering3(self):return round(steering3)
    
    @pyqtSlot(result=float)
    def steering4(self):return round(steering4)
    
    
    
    @pyqtSlot(result=float)
    def gas_throttle1(self):return round(gas_throttle1,3)
    
    @pyqtSlot(result=float)
    def gas_throttle2(self):return round(gas_throttle2,3)
    
    @pyqtSlot(result=float)
    def gas_throttle3(self):return round(gas_throttle3,3)
    
    @pyqtSlot(result=float)
    def gas_throttle4(self):return round(gas_throttle4,3)
    
    
    @pyqtSlot(str, str, str)
    def setpoint(self, message1, message2, message3):
        global sp_lat
        global sp_lon
        global sp_yaw
        global y_ref
        global latitude
        global longitude
        
        global start_lat
        global start_lon
        
        global delta_lat
        global delta_lon
        global yaw
        global y
        global x0
        
        delta_lat = 0
        delta_lon = 0
        
        sp_lat = float(message1)
        sp_lon = float(message2)
        sp_yaw = float(message3)
        
        start_lat = latitude
        start_lon = longitude
        #latitude = start_lat
        #longitude = start_lon
        j_theta = np.array([[np.cos(yaw * float(np.pi/180)), -np.sin(yaw * float(np.pi/180)), 0],
              [np.sin(yaw * float(np.pi/180)), np.cos(yaw* float(np.pi/180)), 0],
              [0, 0, 1]])
                
        try:
            n_error = round(meter_conversion(latitude, 0, float(sp_lat), 0))
            e_error = round(meter_conversion(longitude, 0, float(sp_lon), 0))
        except:
            n_error = 0
            e_error = 0
            

        #error_body_fixed = np.linalg.inv(j_theta) @ np.array([[n_error],[e_error],[yaw]])
        error_body_fixed = j_theta.T @ np.array([[n_error], [e_error], [yaw]])  # Gunakan .T

        x_error = (round(float(error_body_fixed[0]),1))
        y_error = (round(float(error_body_fixed[1]),1))
        
        
        x0[0:2] = 0 
        y_ref = np.array([x_error, y_error, sp_yaw]).reshape(-1, 1)
        
        
        
        print(start_lat, start_lon, sp_yaw)
    
    
    @pyqtSlot(str)
    def animate(self, message):
        global latitude
        global latitude_dot
        global longitude
        global longitude_dot
        global yaw
        global yaw_dot
        global V
        global x
        global x0
        global x_next
        global y
        global y_ref
        
        global steering1
        global steering2
        global steering3
        global steering4
        
        global gas_throttle1
        global gas_throttle2
        global gas_throttle3
        global gas_throttle4
        global x0
        global U
        global heading_error
        global u_optimal
        
        global n_error
        
        global latitude_now 
        global longitude_now
        
        global x_target
        global y_target
        
        global start_lat
        global start_lon
        
        global delta_lat
        global delta_lon
        
        #  # Reshape untuk dimensi (3, 1)

        #kalkulasi setpoint
        '''
        
        j_theta = np.array([[np.cos(yaw * float(np.pi/180)), -np.sin(yaw * float(np.pi/180)), 0],
              [np.sin(yaw * float(np.pi/180)), np.cos(yaw* float(np.pi/180)), 0],
              [0, 0, 1]])
                
        try:
            n_error = round(meter_conversion(latitude, 0, float(sp_lat), 0))
            e_error = round(meter_conversion(longitude, 0, float(sp_lon), 0))
        except:
            n_error = 0
            e_error = 0
            

        #error_body_fixed = np.linalg.inv(j_theta) @ np.array([[n_error],[e_error],[yaw]])
        error_body_fixed = j_theta.T @ np.array([[n_error], [e_error], [yaw]])  # Gunakan .T

        x_error = (round(float(error_body_fixed[0]),1))
        y_error = (round(float(error_body_fixed[1]),1))
        
        y_ref = np.array([x_error, y_error, sp_yaw]).reshape(-1, 1)
        '''
        #print(f"lat: {latitude}, long: {longitude}, sp_lat: {sp_lat}, sp_lon: {sp_lon}")
        #print(f"x : {y[0][0]}, y: {y[1][0]}")
        #print(f"n error: {n_error}, e error: {e_error}; x error: {x_error}, y error: {y_error}")
        
        #y_ref = np.array([10, -20, sp_yaw]).reshape(-1, 1)
        
        # ===== Variabel Optimisasi =====
        x = cp.Variable((A.shape[0], N + 1))  # State variables
        u = cp.Variable((B.shape[1], N))  # Control inputs

        # ===== Fungsi Biaya dan Kendala =====
        cost = 0
        constraints = []

        for k in range(N):
            cost += cp.quad_form(C @ x[:, k] - y_ref.flatten(), Q)  # Penalti error (gunakan y_ref yang sudah direshape)
            cost += cp.quad_form(u[:, k], R)  # Penalti kontrol
            if k > 0:
                cost += cp.quad_form(u[:, k] - u[:, k - 1], delta_u_penalty)  # Penalti perubahan kontrol
            constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k]]
            constraints += [u_min <= u[:, k], u[:, k] <= u_max]

        # Status awal
        constraints += [x[:, 0] == x0.flatten()]

        # Problem MPC
        try:
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve()
        except:
            pass
        # ===== Ambil Kontrol Optimal =====
        if problem.status != 'optimal':
            print(f"Solver failed at step. Status: {problem.status}")
            
        try:
            u_optimal = u.value[:, 0]
        except:
            pass
        # ===== Simulasikan Sistem =====
        x0 = A @ x0 + B @ u_optimal.reshape(-1, 1)
        #x0 = A @ x0 + B @ np.array([[0.0001], [0], [0]])
        y = C @ x0
        print(x0)

        #Kalman filter
        kf.predict(u= u_optimal.reshape(-1, 1))  # Prediksi dengan sinyal kendali
        kf.update(y)  # Update dengan data sensor
        #print(f"Estimasi theta: {kf.x[0, 0]:.2f}, omega: {kf.x[1, 0]:.2f}")

        # Simpan Hasil
        predicted_states.append(y.flatten())
        applied_inputs.append(u_optimal)
        
        
        delta_x = y[0][0]
        delta_y = y[1][0]
        
        delta_lat, delta_lon = coordinate_conv(delta_x, delta_y, yaw) 
        #print(f"d_lat : {delta_lat}, d_lon: {delta_lon}")
        
        latitude = start_lat + (delta_lat/111000)
        longitude = start_lon + (delta_lon/111000)
        yaw = y[2][0]
        
        tau_control = u_optimal
        
        # Gaya yang harus diberikan oleh setiap thruster

        f = T_pseudo_inverse @ tau_control

            #####################
            #==2==         ==1==#
            #                   #
            #                   #
            #                   # -----------> Thruster Orientation
            #                   #
            #                   #
            #                   #
            #==4==         ==3==#
            #####################
            #new vs old
            # 1 > 1
            # 2 > 3
            # 3 > 4
            # 4 > 2
            



        try:
            steering1 = math.atan2(float(f[1]),float(f[0])) * 180/math.pi
        except:
            steering1 = 90
            
        gas_throttle1 = math.sqrt(float(f[1])**2 + float(f[0])**2)


        try:
            steering3 = math.atan2(float(f[3]),float(f[2])) * 180/math.pi
        except:
            steering3 = 90

        gas_throttle3 = math.sqrt(float(f[3])**2 + float(f[2])**2)

        try:
            steering4 = math.atan2(float(f[5]),float(f[4])) * 180/math.pi
        except:
            steering4 = 90
        gas_throttle4 = math.sqrt(float(f[5])**2 + float(f[4])**2)


        try:
            steering2 = math.atan2(float(f[7]),float(f[6])) * 180/math.pi
        except:
            steering2 = 90
        gas_throttle2 = math.sqrt(float(f[7])**2 + float(f[6])**2)
        
        
    
    @pyqtSlot(result=float)
    def yaw(self):return yaw
    
    @pyqtSlot(result=str)
    def A_ss(self):return str(np.round(A, decimals=4))
    
    @pyqtSlot(result=str)
    def B_ss(self):return str(B)
    
    @pyqtSlot(result=str)
    def C_ss(self):return str(C)
    
    @pyqtSlot(result=str)
    def x_ss(self):return str(np.round(x0, decimals=2))
    #np.round(y.flatten(), decimals=2)
    
    @pyqtSlot(result=str)
    def u_ss(self):return str(np.round(u_optimal.reshape(-1, 1), decimals=4))
    
    @pyqtSlot(result=str)
    def y_ss(self):return str(np.round(y, decimals=3))
    
    
    @pyqtSlot(result=str)
    def yref_ss(self):return str(np.round(y_ref, decimals=3))
    
    
#----------------------------------------------------------------#




########## memanggil class table di mainloop######################
#----------------------------------------------------------------#    
if __name__ == "__main__":

    main = table()
    
    
#----------------------------------------------------------------#
    