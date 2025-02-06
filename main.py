######  PROGRAM MEMANGGIL WINDOWS PYQT5 ##########################

####### memanggil library PyQt5 ##################################
#----------------------------------------------------------------#
from PyQt5.QtCore import * 
from PyQt5.QtGui import * 
from PyQt5.QtQml import * 
from PyQt5.QtWidgets import *
from PyQt5.QtQuick import *  
import sys
from scipy.linalg import expm
import math
#----------------------------------------------------------------#

import numpy as np

pubdelay = 2 #delay publish to all wind and engine box
counter = 0


latitude = -6.215861
latitude_dot = 0.00001
longitude = 107.803706
longitude_dot = 0.00001
yaw = 190


x_dot = 0.00001
y_dot = 0.0000
theta_dot = 1

eta = np.array([[x_dot], [y_dot], [yaw]])
V = np.array([[latitude_dot], [longitude_dot], [yaw]]) 


def rotation(x, y, theta):
    theta_rad = np.radians(theta)  # Konversi derajat ke radian
    j_theta = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0],
                        [np.sin(theta_rad),  np.cos(theta_rad), 0],
                        [0, 0, 1]])
    
    result = j_theta @ np.array([[x], [y], [0]])  # Gunakan 0 untuk rotasi biasa

    x_accent = result[0, 0]  # Mengambil nilai skalar dari array
    y_accent = result[1, 0]
    theta = theta

    return x_accent, y_accent, theta


def meter_conversion(lat1, long1, lat2, long2):
    delta_lat = (lat1 - lat2)*111000
    delta_lon = (long1 - long2)*111000
    distance = sqrt(pow(delta_lat, 2) +  pow(delta_lon, 2))
    return distance

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, lsim

# Parameter sistem
'''
m = np.array([[2.0, 0.1, 0.2], [0.1, 2.5, 0.3], [0.2, 0.3, 3.0]])  # matriks massa (3x3)
c = np.array([[0.5, 0.1, 0.0], [0.1, 0.6, 0.2], [0.0, 0.2, 0.7]])  # matriks redaman (3x3)
d = np.array([[0.2, 0.0, 0.1], [0.0, 0.3, 0.0], [0.1, 0.0, 0.4]])  # matriks gesekan (3x3)
'''

############ Model Kinetik ################################

print("==== Kinetik =======")
xg = 0.5 #posisi x center of gravity
yg = 0.4 #posisi y center of gravity
mass = 27 #massa kapal
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

A = np.block([
    [np.zeros((3, 3)), np.eye(3)],
    [np.zeros((3, 3)), -np.linalg.inv(m) @ d]
])
'''
A = np.block([
    [np.zeros((3, 3)), np.eye(3)],
    [-np.linalg.inv(m) @ d, -np.linalg.inv(m) @ c]
])
'''
B = np.block([
    [np.zeros((3, 3))],
    [np.linalg.inv(m)]
])
C = np.block([
    [np.eye(3), np.zeros((3, 3))]
])
D = np.zeros((3, 3))

x_next = np.array([[0], [0], [0], [0], [0], [0]])
x0 = np.array([[0], [0], [0], [0], [0], [0]])  

x = np.array([[0], [0], [0], [0], [0], [0]]) 
U = np.array([[0], [0], [0]])
y = np.array([[0], [0.0], [0]])


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
Ad = np.linalg.inv(I - (T/2) * A) @ (I + (T/2) * A)
Bd = np.linalg.inv(I - (T/2) * A) @ (T * B)

# Cd dan Dd tetap sama
Cd = C
Dd = D

print("Matriks A (diskrit):\n", Ad)
print("Matriks B (diskrit):\n", Bd)

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
    
    @pyqtSlot(str)
    def state_space_run(self, message):
        pass
    
    
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
        global x_next
        global y
        
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
        
        
        if (message == "1"):
            U = np.array([[0], [0.0001], [0]])
            
        if (message == "0"):
            U = np.array([[0], [0], [0]])
        dt = 1
        
        x0 = Ad @ x0 * dt + Bd @ U * dt
        y = Cd @ x0
        
        print(y)
        eta = y
        
        V[0][0], V[1][0], V[2][0] = rotation(eta[0][0], eta[1][0],eta[2][0])
        
        
        latitude = latitude + (V[0][0] / 111000)
        longitude = longitude + (V[1][0] / 111000)
        yaw =yaw + V[2][0]
        
        tau_control = U
        
        # Gaya yang harus diberikan oleh setiap thruster

        f = T_pseudo_inverse @ tau_control
        # Cetak hasil
        print("Gaya yang harus dihasilkan oleh thruster:")
        print("f genap fx, f ganjil fy")
        print(f)


        try:
            steering1 = math.atan2(float(f[1]),float(f[0])) * 180/math.pi
        except:
            steering1 = 90
            
        gas_throttle1 = math.sqrt(float(f[1])**2 + float(f[0])**2)
        print(f"Thruster 1 Allocation : steering 1 {steering1}, throttle 1: {gas_throttle1}")


        try:
            steering2 = math.atan2(float(f[3]),float(f[2])) * 180/math.pi
        except:
            steering2 = 90

        gas_throttle2 = math.sqrt(float(f[3])**2 + float(f[2])**2)
        print(f"Thruster 2 Allocation : steering 2 {steering2}, throttle 2: {gas_throttle2}")


        try:
            steering3 = math.atan2(float(f[5]),float(f[4])) * 180/math.pi
        except:
            steering3 = 90
        gas_throttle3 = math.sqrt(float(f[5])**2 + float(f[4])**2)
        print(f"Thruster 3 Allocation : steering 3 {steering3}, throttle 3: {gas_throttle3}")



        try:
            steering4 = math.atan2(float(f[7]),float(f[6])) * 180/math.pi
        except:
            steering4 = 90
        gas_throttle4 = math.sqrt(float(f[7])**2 + float(f[6])**2)
        print(f"Thruster 4 Allocation : steering 4 {steering4}, throttle 4: {gas_throttle4}")
      
        
        
    
    @pyqtSlot(result=float)
    def yaw(self):return yaw
    
    @pyqtSlot(result=str)
    def A_ss(self):return str(Ad)
    
    @pyqtSlot(result=str)
    def B_ss(self):return str(Bd)
    
    @pyqtSlot(result=str)
    def C_ss(self):return str(Cd)
    
    @pyqtSlot(result=str)
    def x_ss(self):return str(x0)
    
    
    @pyqtSlot(result=str)
    def u_ss(self):return str(U)
    
    @pyqtSlot(result=str)
    def y_ss(self):return str(y)
    
    
#----------------------------------------------------------------#




########## memanggil class table di mainloop######################
#----------------------------------------------------------------#    
if __name__ == "__main__":

    main = table()
    
    
#----------------------------------------------------------------#
    