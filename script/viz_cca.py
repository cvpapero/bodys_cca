#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
2016.1.29
hdf5に対応
wx_m,wy_mのデータ構造を変更

2016.1.17
ros_body_ccaで作成したデータを可視化する

2016.1.11
決定版をつくる

"""

import sys
import os.path
import math
import json
import time
import h5py

import numpy as np
from numpy import linalg as NLA
import scipy as sp
from scipy import linalg as SLA

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui  import *

#import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl
import matplotlib.animation as animation

import rospy
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from std_msgs.msg import ColorRGBA


class GRAPH(QtGui.QWidget):

    def __init__(self):
        super(GRAPH, self).__init__()

    def std(self, u):
        nu = np.array(u)
        return (nu - np.mean(nu))/np.std(nu)

    def drawStaticCoef(self, row, col, data1, data2, r_m, wx_m, wy_m, wins, dtd):
        
        pl.ion()
        pl.clf()

        #for w in range(wins):
        d1 = np.array(data1)
        d2 = np.array(data2)
        
        u1 = d1[row:row+wins,:]
        u2 = d2[col:col+wins,:]
        #print u1.shape

        #寄与率
        fu1, con1= self.staticCoef(u1, wx_m[row,col])
        gu2, con2= self.staticCoef(u2, wy_m[row,col])
        #con1 = np.mean(fu1**2)
        #con2 = np.mean(gu2**2)
        print "con rate 1:",con1,", 2:",con2

        #冗長率
        gu1, red1 = self.staticCoef(u2, wx_m[row,col])
        fu2, red2 = self.staticCoef(u1, wy_m[row,col])
        #red1 = np.mean(gu1**2)
        #red2 = np.mean(fu2**2)
        print "red rate 1:",red1,", 2:",red2

        #vec bars
        c, = fu1.shape
        xl = np.arange(c)        
        pl.subplot2grid((1,2),(0,0))
        pl.bar(xl, fu1)
        pl.xlim(0,c)
        pl.ylim(-1,1)
        pl.subplot2grid((1,2),(0,1))
        pl.bar(xl, gu2)
        pl.xlim(0,c)
        pl.ylim(-1,1)
        pl.tight_layout()
        pl.draw()

    def staticCoef(self, d, v):        
        f = np.dot(d, v)
        fu = np.corrcoef(np.c_[f,d].T)[0,1:]
        return fu, np.mean(fu**2)


    def drawSCoefMap(self, data1, data2, r_m, wx_m, wy_m, wins, dtmr):

        pl.ion()
        pl.clf()

        #for w in range(wins):
        d1 = np.array(data1)
        d2 = np.array(data2)
        
        maps1 = []
        maps2 = []
        for row in range(dtmr):
            map1 = []
            map2 = []
            for col in range(dtmr):
                #u1 = d1[row:row+wins,:]
                #u2 = d2[col:col+wins,:]
                fu1, con1= self.staticCoef(d1[row:row+wins,:], wx_m[row,col])
                gu2, con2= self.staticCoef(d2[row:row+wins,:], wy_m[row,col])
                map1.append(con1)
                map2.append(con2)
            maps1.append(map1)
            maps2.append(map2)


        maps = (np.array(maps1)+np.array(maps2))/2
        dr, dc = maps.shape
        Y,X = np.mgrid[slice(0, dc+1, 1),slice(0, dr+1, 1)]
        pl.pcolor(X, Y, maps, vmin=0, vmax=1)
        cbar = pl.colorbar()
        fs = 10
        cbar.ax.tick_params(labelsize=fs-1) 
        pl.gray()
        pl.xlim(0,dc)
        pl.ylim(0,dr)

        
        """
        pl.subplot2grid((2,2),(0,0))
        maps1 = np.array(maps1)
        dr, dc = maps1.shape
        Y,X = np.mgrid[slice(0, dc+1, 1),slice(0, dr+1, 1)]
        pl.pcolor(X, Y, maps1, vmin=0, vmax=1)
        cbar = pl.colorbar()
        fs = 10
        cbar.ax.tick_params(labelsize=fs-1) 
        pl.gray()
        pl.xlim(0,dc)
        pl.ylim(0,dr)

        pl.subplot2grid((2,2),(1,0))
        maps2 = np.array(maps2)
        dr, dc = maps2.shape
        Y,X = np.mgrid[slice(0, dc+1, 1),slice(0, dr+1, 1)]
        pl.pcolor(X, Y, maps2, vmin=0, vmax=1)
        cbar = pl.colorbar()
        fs = 10
        cbar.ax.tick_params(labelsize=fs-1) 
        pl.gray()
        pl.xlim(0,dc)
        pl.ylim(0,dr)

        """

        pl.tight_layout()

        pl.draw()

    def drawSigRhoVecPlot(self, row, col, data1, data2, r_m, wx_m, wy_m, wins, dtd, p1, p2):

        pl.ion()
        pl.clf()
        fs = 8

        #構造係数
        d1 = np.array(data1)
        d2 = np.array(data2)
        fu1, con1= self.staticCoef(d1[row:row+wins,:], wx_m[:,row,col])
        gu2, con2= self.staticCoef(d2[col:col+wins,:], wy_m[:,row,col])


        od1 = np.argmax(np.fabs(wx_m[:,row,col]))
        od2 = np.argmax(np.fabs(wy_m[:,row,col]))
        #od1 = np.argmax(np.fabs(fu1))
        #od2 = np.argmax(np.fabs(gu2))

        U1=d1[row:row+wins,od1]
        U2=d2[col:col+wins,od2]

        #signal plot
        pl.subplot2grid((3,3),(0,0),colspan=3,rowspan=2)
        #pl.plot(self.std(U1), color="r",label="User_x j:"+str(od1)+", f:"+str(row)+"-"+str(row+wins))
        #pl.plot(self.std(U2), color="b",label="User_y j:"+str(od2)+", f:"+str(col)+"-"+str(col+wins),linestyle="--")
        pl.plot(U1, color="r",label="User_x j:"+str(od1)+", f:"+str(row)+"-"+str(row+wins),alpha=0.5)
        pl.plot(U2, color="b",label="User_y j:"+str(od2)+", f:"+str(col)+"-"+str(col+wins),linestyle="--",alpha=0.5)
        leg = pl.legend(prop={'size':10})
        leg.get_frame().set_alpha(0.5)
        pl.xlim(0,wins)
        pl.ylim(0,math.pi)
        pl.xticks(fontsize=fs)
        pl.yticks(fontsize=fs)
        #corr = np.corrcoef(self.std(U1),self.std(U2))
        corr = np.corrcoef(U1,U2)
        pl.title("joints cca:"+str(round(r_m[row][col],3))+",rho:"+str(corr[0][1]),fontsize=fs+3)
        


        #for w in range(wins):
        #d1 = np.array(data1)
        #d2 = np.array(data2)

        #寄与率
        #fu1, con1= self.staticCoef(d1[row:row+wins,:], wx_m[row,col])
        #gu2, con2= self.staticCoef(d2[col:col+wins,:], wy_m[row,col])        
        
        
        #vec bars
        xl = np.arange(dtd)        
        
        pl.subplot2grid((3,3),(2,0))
        pl.bar(xl, wx_m[:,row,col])
        pl.xlim(0,dtd)
        pl.ylim(-1,1)
        pl.xticks(fontsize=fs)
        pl.yticks(fontsize=fs)
        pl.title("user_1 vec",fontsize=fs+1)
        
        pl.subplot2grid((3,3),(2,1))
        pl.bar(xl, wy_m[:,row,col])
        pl.xlim(0,dtd)
        pl.ylim(-1,1)
        pl.xticks(fontsize=fs)
        pl.yticks(fontsize=fs)
        pl.title("user_2 vec",fontsize=fs+1)
        
        #pca
        """
        U1=d1[row:row+wins,:]
        Su1=np.cov(U1.T)
        l,v= NLA.eigh(Su1)
        idx = np.argsort(l)[::-1]
        l = l[idx]
        v = v[:,idx]
        pl.subplot2grid((3,3),(2,0))
        pl.bar(xl, v[:,0])
        pl.xlim(0,dtd)
        pl.ylim(-1,1)
        pl.xticks(fontsize=fs)
        pl.yticks(fontsize=fs)
        pl.title("user_1 pca",fontsize=fs+1)

        U2=d2[col:col+wins,:]
        Su2=np.cov(U2.T)
        l,v= NLA.eigh(Su2)
        idx = np.argsort(l)[::-1]
        l = l[idx]
        v = v[:,idx]
        pl.subplot2grid((3,3),(2,1))
        pl.bar(xl, v[:,0])
        pl.xlim(0,dtd)
        pl.ylim(-1,1)
        pl.xticks(fontsize=fs)
        pl.yticks(fontsize=fs)
        pl.title("user_2 pca",fontsize=fs+1)
        """


        """
        pl.subplot2grid((3,3),(1,0))
        pl.bar(xl, fu1)
        pl.xlim(0,dtd)
        pl.ylim(-1,1)
        pl.xticks(fontsize=fs)
        pl.yticks(fontsize=fs)
        pl.title("user_1 st",fontsize=fs+1)

        pl.subplot2grid((3,3),(2,0))
        pl.bar(xl, gu2)
        pl.xlim(0,dtd)
        pl.ylim(-1,1)
        pl.xticks(fontsize=fs)
        pl.yticks(fontsize=fs)
        pl.title("user_2 st",fontsize=fs+1)
        """
        """
        corrs=[]
        for rows in range(len(r_m)):
            corr = []
            for cols in range(len(r_m)):
                fu1, con1= self.staticCoef(d1[rows:rows+wins,:], wx_m[rows,cols])
                gu2, con2= self.staticCoef(d2[cols:cols+wins,:], wy_m[rows,cols])
                od1 = np.argmax(np.fabs(fu1))
                od2 = np.argmax(np.fabs(gu2))
                udata1=d1[rows:rows+wins,od1]
                udata2=d2[cols:cols+wins,od2]
                corr.append(np.corrcoef(self.std(udata1),self.std(udata2))[0][1])     
            corrs.append(corr)     
        corrs=np.array(corrs)
        """

        """
        #rho plot
        
        pl.subplot2grid((3,3),(1,1),colspan=2,rowspan=2)
        #print corrs.shape
        #dr, dc = corrs.shape
        dr, dc = r_m.shape
        Y,X = np.mgrid[slice(0, dc+1, 1),slice(0, dr+1, 1)]
        #pl.pcolor(X, Y, corrs,vmin=-1, vmax=1)
        pl.pcolor(X, Y, r_m, vmin=0, vmax=1)
        cbar = pl.colorbar()
        cbar.ax.tick_params(labelsize=fs-1) 
        pl.gray()
        pl.xlim(0,dc)
        pl.ylim(0,dr)
        pl.xticks(fontsize=fs)
        pl.yticks(fontsize=fs)
        pl.title("rho",fontsize=fs+1)
        """
        
        """
        #vecs plot
        pl.subplot2grid((3,3),(1,1),colspan=2)
        dwx_m = wx_m[row,:,:]
        vr,vc = dwx_m.shape
        Y,X = np.mgrid[slice(0, vc+1, 1),slice(0, vr+1, 1)]
        pl.pcolor(X, Y, dwx_m.T, vmin=-1, vmax=1)
        cbar = pl.colorbar()
        cbar.ax.tick_params(labelsize=fs-1) 
        pl.gray()
        pl.xlim(col,vr)
        pl.ylim(0,vc)
        pl.xticks(fontsize=fs)
        pl.yticks(fontsize=fs)
        pl.title("user_1",fontsize=fs+1)
        

        pl.subplot2grid((3,3),(2,1),colspan=2)
        dwy_m = wy_m[row,:,:]
        vr,vc = dwy_m.shape
        Y,X = np.mgrid[slice(0, vc+1, 1),slice(0, vr+1, 1)]
        pl.pcolor(X, Y, dwy_m.T, vmin=-1, vmax=1)
        cbar = pl.colorbar()
        cbar.ax.tick_params(labelsize=fs-1) 
        pl.gray()
        pl.xlim(col,vr)
        pl.ylim(0,vc)
        pl.xticks(fontsize=fs)
        pl.yticks(fontsize=fs)
        pl.title("user_2",fontsize=fs+1)
        """

        pl.tight_layout()

        pl.draw()


class CCA(QtGui.QWidget):

    def __init__(self):
        super(CCA, self).__init__()
        #UIの初期化
        self.initUI()

        #ROSのパブリッシャなどの初期化
        rospy.init_node('ccaviz', anonymous=True)
        self.mpub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        self.ppub = rospy.Publisher('joint_diff', PointStamped, queue_size=10)

        #rvizのカラー設定(未)
        self.carray = []
        clist = [[1,0,0,1],[0,1,0,1],[1,1,0,1]]
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
        
        #ファイル入力ボックス
        self.txtSepFile = QtGui.QLineEdit()
        btnSepFile = QtGui.QPushButton('...')
        btnSepFile.setMaximumWidth(40)
        btnSepFile.clicked.connect(self.chooseDbFile)
        boxSepFile = QtGui.QHBoxLayout()
        boxSepFile.addWidget(self.txtSepFile)
        boxSepFile.addWidget(btnSepFile)
        form.addRow('input file', boxSepFile)

        #selected pub
        self.selected = QtGui.QRadioButton('selected')
        form.addRow('no publish', self.selected)

        #exec
        boxCtrl = QtGui.QHBoxLayout()
        btnExec = QtGui.QPushButton('visualize')
        btnExec.clicked.connect(self.doExec)
        boxCtrl.addWidget(btnExec)

        #テーブルの初期化
        #horizonはuser2の時間
        self.table = QtGui.QTableWidget(self)
        self.table.setColumnCount(0)
        self.table.setHorizontalHeaderLabels("use_2 time") 
        jItem = QtGui.QTableWidgetItem(str(0))
        self.table.setHorizontalHeaderItem(0, jItem)

        #アイテムがクリックされたらグラフを更新
        self.table.itemClicked.connect(self.updateColorTable)
        self.table.setItem(0, 0, QtGui.QTableWidgetItem(1))

        boxTable = QtGui.QHBoxLayout()
        boxTable.addWidget(self.table)
 
        #配置
        grid.addLayout(form,1,0)
        grid.addLayout(boxCtrl,2,0)
        #grid.addLayout(boxUpDown,3,0)
        grid.addLayout(boxTable,3,0)

        self.setLayout(grid)
        self.resize(400,100)

        self.setWindowTitle("cca window")

        self.show()

    def chooseDbFile(self):
        dialog = QtGui.QFileDialog()
        dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        if dialog.exec_():
            fileNames = dialog.selectedFiles()
            for f in fileNames:
                self.txtSepFile.setText(f)
                return
        return self.txtSepFile.setText('')

    def updateColorTable(self, cItem):
        self.r = cItem.row()
        self.c = cItem.column()
        print "now viz r:",self.r,", c:",self.c
        #print "cca:",self.r_m[r][c]
        
        p1, p2 = self.checkMaxJoint(self.r, self.c, self.data1, self.data2, self.wx_m, self.wy_m, self.wins)

        GRAPH().drawSigRhoVecPlot(self.r, self.c, self.data1, self.data2, self.r_m, self.wx_m, self.wy_m, self.wins, self.dtd, p1, p2)
        #GRAPH().drawStaticCoef(self.r, self.c, self.data1, self.data2, self.r_m, self.wx_m, self.wy_m, self.wins, self.dtd)
        #GRAPH().drawSCoefMap(self.data1, self.data2, self.r_m, self.wx_m, self.wy_m, self.wins, self.dtmr)
 
        hitbtn = self.selected.isChecked()
        if hitbtn == False:
            self.doPub(self.r, self.c, self.r_m[self.r, self.c], self.pos1, self.pos2, self.wins, p1, p2)

    def staticCoef(self, d, v):        
        f = np.dot(d, v)
        fu = np.corrcoef(np.c_[f,d].T)[0,1:]
        return fu, np.mean(fu**2)

    def checkMaxJoint(self, row, col, data1, data2, wx_m, wy_m, wins):

        #for w in range(wins):
        d1 = np.array(data1)
        d2 = np.array(data2)
        
        u1 = d1[row:row+wins,:]
        u2 = d2[col:col+wins,:]
        #print u1.shape

        #寄与率
        #fu1, con1= self.staticCoef(u1, wx_m[:,row,col])
        #gu2, con2= self.staticCoef(u2, wy_m[:,row,col])
  
        fu1 = wx_m[:,row,col]
        gu2 = wy_m[:,row,col]

        p1 = []
        p2 = []
        iv1=[np.argmax(np.fabs(fu1)),fu1[np.argmax(np.fabs(fu1))]]
        iv2=[np.argmax(np.fabs(gu2)),gu2[np.argmax(np.fabs(gu2))]]
        p1.append(iv1)
        p2.append(iv2)


        return p1, p2

    def updateTable(self):

        th = 0#float(self.ThesholdBox.text())
        if(len(self.r_m)==0):
            print "No Corr Data! Push exec button..."
        self.table.clear()
        font = QtGui.QFont()
        font.setFamily(u"DejaVu Sans")
        font.setPointSize(5)
        self.table.horizontalHeader().setFont(font)
        self.table.verticalHeader().setFont(font)
        self.table.setRowCount(len(self.r_m))
        self.table.setColumnCount(len(self.r_m))
        for i in range(len(self.r_m)):
            jItem = QtGui.QTableWidgetItem(str(i))
            self.table.setHorizontalHeaderItem(i, jItem)
        hor = True
        for i in range(len(self.r_m)):
            iItem = QtGui.QTableWidgetItem(str(i))
            self.table.setVerticalHeaderItem(i, iItem)
            self.table.verticalHeaderItem(i).setToolTip(str(i))
            #時間軸にデータを入れるなら↓
            #self.table.verticalHeaderItem(i).setToolTip(str(self.timedata[i]))
            for j in range(len(self.r_m[i])):
                if hor == True:
                    jItem = QtGui.QTableWidgetItem(str(j))
                    self.table.setHorizontalHeaderItem(j, jItem)
                    self.table.horizontalHeaderItem(j).setToolTip(str(j))
                    hot = False
                c = 0
                rho = round(self.r_m[i][j],5)
                rho_data = str(rho)
                if rho > th:
                    c = rho*255
                self.table.setItem(i, j, QtGui.QTableWidgetItem())
                self.table.item(i, j).setBackground(QtGui.QColor(c,c,c))
                self.table.item(i, j).setToolTip(str(rho_data))
        self.table.setVisible(False)
        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()
        self.table.setVisible(True)



    def doExec(self):
        filename = str(self.txtSepFile.text())
        #print "exec!"
        #self.r_m,self.wx_m,self.wy_m, self.wins, self.frms, self.dtd, self.dts, self.sidx = self.jsonInput(filename)
        self.r_m,self.wx_m,self.wy_m, self.wins, self.frms, self.dtd, self.dts, self.sidx = self.load_params(filename)
        self.data1, self.data2, self.pos1, self.pos2 = self.poseInput(filename)
        self.jIdx = self.jIdxInput()

        #dmr:data_max_range, frmr:frame_range, dtr:data_range
        self.dtmr = self.dts - self.wins + 1
        self.frmr = self.dts - self.frms + 1
        self.dtr = self.frms - self.wins + 1

        #print "datas_size:",self.dts
        #print "data_max_range:",self.dtmr
        #print "frame_range:",self.frmr
        #print "data_range:",self.dtr

        self.updateTable()
        print "end"

    def jsonInput(self, filename):
        print "open:",filename
        f = open(filename, 'r')
        js = json.load(f)
        f.close()

        #u[0]=r_m, u[1]=wx, u[2]=wy
        r_m = np.array(js["cca"]["r"])
        wx_m = np.array(js["cca"]["wx"])
        wy_m = np.array(js["cca"]["wy"])
        wins = js["prop"]["wins"]
        frms = js["prop"]["frms"]
        dtd = js["prop"]["dtd"]
        dts = js["prop"]["dts"] 
        sidx = js["prop"]["sidx"]

        return r_m, wx_m, wy_m, wins, frms, dtd, dts, sidx

    def load_params(self, filename):
        print "load"+filename
        with h5py.File(filename) as f:            
            wins = f["/prop/wins"].value
            frms = f["/prop/frms"].value
            dtd = f["/prop/dtd"].value
            dts = f["/prop/dts"].value
            sidx = f["/prop/sidx"].value

            r_m = f["/cca/r/0"].value

            dtmr = len(f["/cca/wx/0/0"].value)

            wx_m=[]
            wy_m=[]     

            for i in range(dtd):
                wx_m.append(f["/cca/wx/0/"+str(i)].value)
                wy_m.append(f["/cca/wy/0/"+str(i)].value)
            
            """
            wx_m=[]
            wy_m=[]
            for i in range(dtmr):
                col1 = []
                col2 = []
                for j in range(dtmr):
                    dt1 = []
                    dt2 = []
                    for d in range(dtd):
                        dt1.append(f["/cca/wx/0/"+str(d)].value[i][j])
                        dt2.append(f["/cca/wy/0/"+str(d)].value[i][j])
                    col1.append(dt1)                    
                    col2.append(dt2)
                wx_m.append(col1)
                wy_m.append(col2)
            """
        return np.array(r_m), np.array(wx_m), np.array(wy_m), wins, frms, dtd, dts, sidx
        

    def poseInput(self, filename):
        """
        f = open(filename, 'r')
        js = json.load(f)
        f.close()

        #fposename = "/home/uema/catkin_ws/src/rqt_cca/data2/"+str(js["prop"]["fname"])
        fposename = str(js["prop"]["fname"])
        """
        with h5py.File(filename) as f:  

            fposename = f["/prop/fname"].value
            
            print "open pose file:",fposename
            fp = open(fposename, 'r')
            jsp = json.load(fp)
            f.close()
            
            datas = []
            for user in jsp:
            #data is joint angle
                data = []
                for j in range(self.dts):
                    data.append(user["datas"][j]["data"])
                datas.append(data)

            poses = []
            for user in jsp:
                pos = []
                psize = len(user["datas"][0]["jdata"])
                for j in range(self.dts):
                    pls = []
                    for p in range(psize):
                        pl = []
                        for xyz in range(3):
                            pl.append(user["datas"][j]["jdata"][p][xyz])
                        pls.append(pl)
                    pos.append(pls)
                poses.append(pos)

        return datas[0], datas[1], poses[0], poses[1]

    def jIdxInput(self):
        f = open('/home/uema/catkin_ws/src/rqt_cca/joint_index.json', 'r')
        jsonIdxDt = json.load(f)
        f.close
        jIdx = []
        #ここで使用するjointを定める？
        """
        for idx in jsonIdxDt:
            #print "j:",j,", idx:",idx
            jl = []
            for i in idx:
                jl.append(i)
            jIdx.append(jl)
        """
        for i in self.sidx:
            jl=[]
            #for j in jsonIdxDt[i]:
            #    jl.append(j)
            jIdx.append(jsonIdxDt[i])
        print jIdx
        return jIdx
    
    def doPub(self, r, c, cor, pos1, pos2, wins, p1, p2):
        print "---play back start---"
        print p1
        print p2
        if r > c:
            self.pubViz(r, c, cor, pos1[c:r+wins], pos2[c:r+wins], wins, p1, p2)
        else:
            self.pubViz(r, c, cor, pos1[r:c+wins], pos2[r:c+wins], wins, p1, p2)
        print "---play back end---"
    
    #データを可視化するだけでok
    def pubViz(self, r, c, cor, pos1, pos2, wins, p1, p2):
        rate = rospy.Rate(10)
        #print "pos1[0]:",pos1[0]
        poses = []
        poses.append(pos1)
        poses.append(pos2)
        ps = []
        ps.append(p1)
        ps.append(p2)

        offset = 0
        if r > c: 
            offset = c
        else:
            offset = r

        for i in range(len(poses[0])):
            sq = i+offset

            print "frame:",sq
            if sq > r and sq < r+wins:
                print "now u1 joint:",p1
            if sq > c and sq < c+wins:
                print "now u2 joint:",p2

            msgs = MarkerArray()

            #frame and corr
            tmsg = Marker()
            tmsg.header.frame_id = 'camera_link'
            tmsg.header.stamp = rospy.Time.now()
            tmsg.ns = 'f'+str(10)
            tmsg.action = 0
            tmsg.id = 10
            tmsg.type = 9
            tjs = 0.1
            tmsg.scale.z = tjs
            tmsg.color = self.carray[0]
            tmsg.pose.position.x = 0
            tmsg.pose.position.y = 0
            tmsg.pose.position.z = 0
            tmsg.pose.orientation.w = 1.0
            tmsg.text = "c:"+str(round(cor, 3))+", f:"+str(sq)
            msgs.markers.append(tmsg) 


            for u, pos in enumerate(poses):
                #points
                pmsg = Marker()
                pmsg.header.frame_id = 'camera_link'
                pmsg.header.stamp = rospy.Time.now()
                pmsg.ns = 'p'+str(u)
                pmsg.action = 0
                pmsg.id = u
                pmsg.type = 7
                js = 0.03
                pmsg.scale.x = js
                pmsg.scale.y = js
                pmsg.scale.z = js
                pmsg.color = self.carray[0]
                for j, p in enumerate(pos[i]):
                    point = Point()
                    point.x = p[0]
                    point.y = p[1]
                    point.z = p[2]
                    pmsg.points.append(point)
                pmsg.pose.orientation.w = 1.0
                msgs.markers.append(pmsg)    

                #lines
                lmsg = Marker()
                lmsg.header.frame_id = 'camera_link'
                lmsg.header.stamp = rospy.Time.now()
                lmsg.ns = 'l'+str(u)
                lmsg.action = 0
                lmsg.id = u
                lmsg.type = 5
                lmsg.scale.x = 0.005
                lmsg.color = self.carray[2]
                for jid in self.jIdx:
                    for pi in range(2):
                        for add in range(2):
                            point = Point()
                            point.x = pos[i][jid[pi+add]][0]
                            point.y = pos[i][jid[pi+add]][1]
                            point.z = pos[i][jid[pi+add]][2]
                            lmsg.points.append(point) 
                lmsg.pose.orientation.w = 1.0
                msgs.markers.append(lmsg)

                #text
                tmsg = Marker()
                tmsg.header.frame_id = 'camera_link'
                tmsg.header.stamp = rospy.Time.now()
                tmsg.ns = 't'+str(u)
                tmsg.action = 0
                tmsg.id = u
                tmsg.type = 9
                tjs = 0.1
                tmsg.scale.z = tjs
                tmsg.color = self.carray[0]
                tmsg.pose.position.x = pos[i][3][0]
                tmsg.pose.position.y = pos[i][3][1]+tjs
                tmsg.pose.position.z = pos[i][3][2]+tjs
                tmsg.pose.orientation.w = 1.0
                tmsg.text = "user_"+str(u+1)
                msgs.markers.append(tmsg) 


                #note point 
                npmsg = Marker()
                npmsg.header.frame_id = 'camera_link'
                npmsg.header.stamp = rospy.Time.now()
                npmsg.ns = 'np'+str(u)
                npmsg.action = 0
                #npmsg.lifetime = rospy.Duration.from_sec(0.1)
                npmsg.id = u
                npmsg.type = 7
                njs = 0.05
                npmsg.scale.x = njs
                npmsg.scale.y = njs
                npmsg.scale.z = njs
                npmsg.color = self.carray[1]
                point = Point()
                point.x = pos[i][self.jIdx[ps[u][0][0]][1]][0]
                point.y = pos[i][self.jIdx[ps[u][0][0]][1]][1]
                point.z = pos[i][self.jIdx[ps[u][0][0]][1]][2]
                npmsg.points.append(point)
                npmsg.pose.orientation.w = 1.0
                msgs.markers.append(npmsg) 


                tmsg = Marker()
                tmsg.header.frame_id = 'camera_link'
                tmsg.header.stamp = rospy.Time.now()
                tmsg.ns = 'tp'+str(u)
                tmsg.action = 0
                #tmsg.lifetime = rospy.Duration.from_sec(0.1)
                tmsg.id = u
                tmsg.type = 9
                tjs = 0.07
                tmsg.scale.z = tjs
                tmsg.color = self.carray[1]
                #point = Point()
                tmsg.pose.position.x = pos[i][self.jIdx[ps[u][0][0]][1]][0]+tjs
                tmsg.pose.position.y = pos[i][self.jIdx[ps[u][0][0]][1]][1]+tjs
                tmsg.pose.position.z = pos[i][self.jIdx[ps[u][0][0]][1]][2]+tjs
                tmsg.pose.orientation.w = 1.0
                tmsg.text = str(round(ps[u][0][1],3))
                msgs.markers.append(tmsg) 

                #note point and line 
                if u == 0 and sq > r and sq < r+wins:                    


                    lmsg = Marker()
                    lmsg.header.frame_id = 'camera_link'
                    lmsg.header.stamp = rospy.Time.now()
                    lmsg.ns = 'nl'+str(u)
                    lmsg.action = 0
                    lmsg.lifetime = rospy.Duration.from_sec(0.1)
                    lmsg.id = u
                    lmsg.type = 5
                    lmsg.scale.x = 0.01
                    lmsg.color = self.carray[1]
                    #for jid in self.jIdx:
                    for pi in range(2):
                        for add in range(2):
                            point = Point()
                            point.x = pos[i][self.jIdx[p1[0][0]][pi+add]][0]
                            point.y = pos[i][self.jIdx[p1[0][0]][pi+add]][1]
                            point.z = pos[i][self.jIdx[p1[0][0]][pi+add]][2]
                            lmsg.points.append(point) 
                    lmsg.pose.orientation.w = 1.0
                    msgs.markers.append(lmsg)




                if u == 1 and sq > c and sq < c+wins:    

                    lmsg = Marker()
                    lmsg.header.frame_id = 'camera_link'
                    lmsg.header.stamp = rospy.Time.now()
                    lmsg.ns = 'nl'+str(u)
                    lmsg.action = 0
                    lmsg.lifetime = rospy.Duration.from_sec(0.1)
                    lmsg.id = u
                    lmsg.type = 5
                    lmsg.scale.x = 0.01
                    lmsg.color = self.carray[1]
                    #for jid in self.jIdx:
                    for pi in range(2):
                        for add in range(2):
                            point = Point()
                            point.x = pos[i][self.jIdx[p2[0][0]][pi+add]][0]
                            point.y = pos[i][self.jIdx[p2[0][0]][pi+add]][1]
                            point.z = pos[i][self.jIdx[p2[0][0]][pi+add]][2]
                            lmsg.points.append(point) 
                    lmsg.pose.orientation.w = 1.0
                    msgs.markers.append(lmsg)

            self.mpub.publish(msgs)
            rate.sleep()

def main():
    app = QtGui.QApplication(sys.argv)
    corr = CCA()
    graph = GRAPH()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()
