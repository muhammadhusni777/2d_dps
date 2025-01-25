######  PROGRAM MEMANGGIL WINDOWS PYQT5 ##########################

####### memanggil library PyQt5 ##################################
#----------------------------------------------------------------#
from PyQt5.QtCore import * 
from PyQt5.QtGui import * 
from PyQt5.QtQml import * 
from PyQt5.QtWidgets import *
from PyQt5.QtQuick import *  
import sys
#----------------------------------------------------------------#

import paho.mqtt.client as paho
#broker="123.45.0.10"
broker="127.0.0.1"
port = 1883

pubdelay = 2 #delay publish to all wind and engine box
counter = 0


latitude = -6.215861
longitude = 107.803706
yaw = 90


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
    
    @pyqtSlot(result=float)
    def yaw(self):return yaw
    
    
#----------------------------------------------------------------#

def on_message(client, userdata, message):
        msg = str(message.payload.decode("utf-8"))
        t = str(message.topic)

        if(msg[0] == 'c'):
            val =  1
        else:
            val = (msg)

        if (t == "lat"):
            global latitude
            latitude = float(msg)
            print(latitude)
            
        if (t == "long"):
            global longitude
            longitude = float(msg)
            
        if (t == "yaw"):
            global yaw
            yaw = float(msg)



########## memanggil class table di mainloop######################
#----------------------------------------------------------------#    
if __name__ == "__main__":
    ##Mosquitto Mqtt Configuration
    client= paho.Client("GUI")
    client.on_message=on_message

    print("connecting to broker ",broker)
    client.connect(broker,port)#connect
    print(broker," connected")

    
    client.loop_start()
    print("Subscribing")


    client.subscribe("lat")
    client.subscribe("long")
    client.subscribe("yaw")
    main = table()
    
    
#----------------------------------------------------------------#
    