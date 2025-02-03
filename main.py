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
#broker="123.45.0.10"


pubdelay = 2 #delay publish to all wind and engine box
counter = 0


latitude = -6.215861
latitude_dot = 0.00001
longitude = 107.803706
longitude_dot = 0.00001
yaw = 90
yaw_dot = 1

x_dot = 0
y_dot = 0
theta_dot = 0

def rotation(x, y, theta):
    j_theta = np.array([[np.cos(theta * float(np.pi/180)), -np.sin(theta * float(np.pi/180)), 0],
              [np.sin(theta * float(np.pi/180)), np.cos(theta* float(np.pi/180)), 0],
              [0, 0, 1]])
    result = ((j_theta)@ np.array([[x],[y],[theta]]))
    
    x_accent = result[1]
    y_accent = result[0]

    return x_accent, y_accent


def meter_conversion(lat1, long1, lat2, long2):
    delta_lat = (lat1 - lat2)*111000
    delta_lon = (long1 - long2)*111000
    distance = sqrt(pow(delta_lat, 2) +  pow(delta_lon, 2))
    return distance

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, lsim

# Parameter sistem
m = np.array([[2.0, 0.1, 0.2], [0.1, 2.5, 0.3], [0.2, 0.3, 3.0]])  # matriks massa (3x3)
c = np.array([[0.5, 0.1, 0.0], [0.1, 0.6, 0.2], [0.0, 0.2, 0.7]])  # matriks redaman (3x3)
d = np.array([[0.2, 0.0, 0.1], [0.0, 0.3, 0.0], [0.1, 0.0, 0.4]])  # matriks gesekan (3x3)

# Matriks State-Space
A = np.block([
    [np.zeros((3, 3)), np.eye(3)],
    [-np.linalg.inv(m) @ c, -np.linalg.inv(m) @ d]
])
B = np.block([
    [np.zeros((3, 3))],
    [np.linalg.inv(m)]
])
C = np.block([
    [np.eye(3), np.zeros((3, 3))]
])
D = np.zeros((3, 3))


print("A")
print(A)

print("B")
print(B)

print("C")
print(C)


print("D")
print(D)


# Interval waktu (s)
T_s = 0.1  # contoh interval waktu diskrit

# Mendiskretisasi matriks A dan B
A_d = expm(A * T_s)  # Matriks A diskrit
B_d = np.linalg.inv(A) @ (A_d - np.eye(A.shape[0])) @ B  # Matriks B diskrit
C_d = C  # Matriks C tetap sama untuk diskrit
D_d = D  # Matriks D tetap sama untuk diskrit

print("Matriks A (diskrit):\n", A_d)
print("Matriks B (diskrit):\n", B_d)



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
    def latitude(self):return latitude
    
    @pyqtSlot(result=float)
    def longitude(self):return longitude
    
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
        
        latitude = latitude + latitude_dot
        longitude = longitude + longitude_dot
        yaw = yaw + yaw_dot
        
    
    @pyqtSlot(result=float)
    def yaw(self):return yaw
    
    
#----------------------------------------------------------------#




########## memanggil class table di mainloop######################
#----------------------------------------------------------------#    
if __name__ == "__main__":

    main = table()
    
    
#----------------------------------------------------------------#
    