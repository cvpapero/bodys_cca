#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import math
import json
import numpy

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui  import *

import rospy
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from std_msgs.msg import ColorRGBA

class Correlation(QtGui.QWidget):

    def __init__(self):
        self.jsonInput()

        super(Correlation, self).__init__()
        self.initUI()
        rospy.init_node('correlation', anonymous=True)
        self.mpub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        self.ppub = rospy.Publisher('joint_diff', PointStamped, queue_size=10)

        self.carray = []
        clist = [[1,1,0,1],[0,1,0,1],[1,0,0,1]]
        for c in clist:
            color = ColorRGBA()
            color.r = c[0]
            color.g = c[1]
            color.b = c[2]
            color.a = c[3]
            self.carray.append(color) 

    def initUI(self):
        grid = QtGui.QGridLayout()
        form = QtGui.QFormLayout()

        self.winSizeBox = QtGui.QLineEdit()
        self.winSizeBox.setText('30')
        self.winSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.winSizeBox.setFixedWidth(100)
        form.addRow('window size', self.winSizeBox)

        self.AFrameBox = QtGui.QLineEdit()
        self.AFrameBox.setText('0')
        self.AFrameBox.setAlignment(QtCore.Qt.AlignRight)
        self.AFrameBox.setFixedWidth(100)
        form.addRow('A User Frame', self.AFrameBox)

        self.BFrameBox = QtGui.QLineEdit()
        self.BFrameBox.setText('0')
        self.BFrameBox.setAlignment(QtCore.Qt.AlignRight)
        self.BFrameBox.setFixedWidth(100)
        form.addRow('B User Frame', self.BFrameBox)

        boxCtrl = QtGui.QHBoxLayout()
        btnExec = QtGui.QPushButton('exec')
        btnExec.clicked.connect(self.doExec)
        boxCtrl.addWidget(btnExec)

        grid.addLayout(form,1,0)
        grid.addLayout(boxCtrl,2,0)

        self.setLayout(grid)
        self.resize(400,100)
        
        self.setWindowTitle("joint select window")
        self.show()

    def jsonInput(self):
        f = open('test1014.json', 'r');
        jsonData = json.load(f)
        #print json.dumps(jsonData, sort_keys = True, indent = 4)
        f.close()

        self.data = {}
        self.pdata = {}
        self.jointnames = []
        self.time = []
        self.usrSize = len(jsonData)

        u = 0;
        for user in jsonData:

            #position
            psize = len(user["datas"][0]["jdata"])            
            tsize = len(user["datas"])
            pobj = {}
            for t in range(tsize):
                plist = []
                for p in range(psize):
                    pl = []
                    #print str(p)+","+str(t)
                    #print user["datas"][t]["jdata"][p]
                    pl.append(user["datas"][t]["jdata"][p][0])
                    pl.append(user["datas"][t]["jdata"][p][1])
                    pl.append(user["datas"][t]["jdata"][p][2])
                    plist.append(pl)
                pobj[t] = plist
            self.pdata[u] = pobj
            u+=1
        
        #print "pdata:"+str(len(self.pdata[0][0]))

        for itime in jsonData[0]["datas"]:
            #print itime["time"]
            self.time.append(itime["time"])

        #self.jointSize = len(self.data[0])
        self.dataSize = tsize

        #joint index
        f = open('joint_index.json', 'r');
        jsonIndexData = json.load(f)
        #print json.dumps(jsonData, sort_keys = True, indent = 4)
        f.close()
        self.idata = []

        for index in jsonIndexData:
            ilist = []
            for i in index:
                ilist.append(i)
            self.idata.append(ilist)


    def doExec(self):
        print "exec!"
        self.winSize = int(self.winSizeBox.text())
        self.maxRange = self.dataSize - self.winSize

        Ast = float(self.AFrameBox.text())
        Bst = float(self.BFrameBox.text())

        self.pubViz(Ast, Bst)

        print "end"

    def pubViz(self, ast, bst):

        rate = rospy.Rate(10)

        for i in range(self.winSize):

            msgs = MarkerArray()
            
            #use1について
            msg = Marker()
            #いろんなプロパティ
            msg.header.frame_id = 'camera_link'
            msg.header.stamp = rospy.Time.now()
            msg.ns = 'j1'
            msg.action = 0
            msg.id = 1
            msg.type = 8
            msg.scale.x = 0.1
            msg.scale.y = 0.1
            msg.color = self.carray[2]
            #ジョイントポイントを入れる処理
            for j1 in range(len(self.pdata[0][ast+i])):
                point = Point()
                point.x = self.pdata[0][ast+i][j1][0]
                point.y = self.pdata[0][ast+i][j1][1]
                point.z = self.pdata[0][ast+i][j1][2]
                msg.points.append(point) 
            msg.pose.orientation.w = 1.0
            msgs.markers.append(msg)    
            
            msg = Marker()
            msg.header.frame_id = 'camera_link'
            msg.header.stamp = rospy.Time.now()
            msg.ns = 'j2'
            msg.action = 0
            msg.id = 2
            msg.type = 8
            msg.scale.x = 0.1
            msg.scale.y = 0.1
            msg.color = self.carray[1]

            for j2 in range(len(self.pdata[0][bst+i])):
                point = Point()
                point.x = self.pdata[1][bst+i][j2][0]
                point.y = self.pdata[1][bst+i][j2][1]
                point.z = self.pdata[1][bst+i][j2][2]
                msg.points.append(point) 
            msg.pose.orientation.w = 1.0

            msgs.markers.append(msg)

            self.mpub.publish(msgs)
            rate.sleep()

def main():
    app = QtGui.QApplication(sys.argv)
    corr = Correlation()
    sys.exit(app.exec_())


if __name__=='__main__':
    main()
