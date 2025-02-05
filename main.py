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
#----------------------------------------------------------------#

import numpy as np

pubdelay = 2 #delay publish to all wind and engine box
counter = 0


latitude = -6.215861
latitude_dot = 0.00001
longitude = 107.803706
longitude_dot = 0.00001
yaw = 90


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
m = 27 #massa kapal
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

'''
#dari thesis teguh

m = np.array([[m, 0, -m*yg], [0, m, m*xg], [-m*yg, m*xg, Iz]])
+ np.array([[-x_udot,0,0],[0,-y_vdot,-y_rdot],[0,-n_vdot,-n_rdot]])
print(m)
'''
#dari Thomas P. DeRensis
m = np.array([[m, -0.003, 0.001], [0.002, m, m*xg], [-0.005, m*xg, Iz]])
+ np.array([[-x_udot,-0.002,0.003],[-0.003,y_vdot,y_rdot],[-0.001,n_vdot,n_rdot]])
print(m)

#gaya akibat drag
d = np.array([[x_u,-0.004,-0.002],
    [-0.003,y_v,y_r],
    [-0.001,n_v,n_r]]) 

print("d",d)



# Matriks State-Space
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

x = np.array([[0], [0], [0], [0], [0], [0]]) 
U = np.array([[0.0001], [0], [0]])


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
        
        V[0][0], V[1][0], V[2][0] = rotation(eta[0][0], eta[1][0],eta[2][0])
        
        
        latitude = latitude + V[0][0]
        longitude = longitude + V[1][0]
        yaw = V[2][0]
        print(V)
        
    
    @pyqtSlot(result=float)
    def yaw(self):return yaw
    
    @pyqtSlot(result=str)
    def A_ss(self):return str(Ad)
    
    @pyqtSlot(result=str)
    def B_ss(self):return str(Bd)
    
    @pyqtSlot(result=str)
    def C_ss(self):return str(Cd)
    
    
    @pyqtSlot(result=str)
    def x_ss(self):return str(x)
    
    
    @pyqtSlot(result=str)
    def u_ss(self):return str(U)
    
    
#----------------------------------------------------------------#




########## memanggil class table di mainloop######################
#----------------------------------------------------------------#    
if __name__ == "__main__":

    main = table()
    
    
#----------------------------------------------------------------#
    