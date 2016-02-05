#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
2016.2.1
pcaをするモードを作る


2016.1.29
numpy.arrayでデータを管理
ソースが多少短くなった
次はhdl5を使ってデータ保存する

そのうち、新しいpkgを作るか,
つーかこれrosである必要もあんまりない

2016.1.17
計算するだけ
結果の可視化は別のモジュールにバトンタッチ
可視化はしなくてよい

2016.1.11
決定版をつくる

"""

import sys
import os.path
import math
import json
import time
from datetime import datetime
import h5py
import tqdm

#calc
import numpy as np
from numpy import linalg as NLA
import scipy as sp
from scipy import linalg as SLA

#GUI
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui  import *

#plots
import matplotlib.pyplot as plt

#ROS
import rospy

class CCA(QtGui.QWidget):

    def __init__(self):
        super(CCA, self).__init__()
        #UI
        self.init_ui()
        #ROS
        rospy.init_node('calccca', anonymous=True)


    def init_ui(self):
        grid = QtGui.QGridLayout()
        form = QtGui.QFormLayout()
        
        #ファイル入力ボックス
        self.txtSepFile = QtGui.QLineEdit()
        btnSepFile = QtGui.QPushButton('...')
        btnSepFile.setMaximumWidth(40)
        btnSepFile.clicked.connect(self.choose_db_file)
        boxSepFile = QtGui.QHBoxLayout()
        boxSepFile.addWidget(self.txtSepFile)
        boxSepFile.addWidget(btnSepFile)
        form.addRow('input file', boxSepFile)
        """
        #ファイル出力
        self.txtSepFileOut = QtGui.QLineEdit()
        btnSepFileOut = QtGui.QPushButton('...')
        btnSepFileOut.setMaximumWidth(40)
        btnSepFileOut.clicked.connect(self.chooseOutFile)
        boxSepFileOut = QtGui.QHBoxLayout()
        boxSepFileOut.addWidget(self.txtSepFileOut)
        boxSepFileOut.addWidget(btnSepFileOut)
        form.addRow('output file', boxSepFileOut)    
        """
        #window size
        self.winSizeBox = QtGui.QLineEdit()
        self.winSizeBox.setText('90')
        self.winSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.winSizeBox.setFixedWidth(100)
        form.addRow('window size', self.winSizeBox)

        #frame size
        self.frmSizeBox = QtGui.QLineEdit()
        self.frmSizeBox.setText('110')
        self.frmSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.frmSizeBox.setFixedWidth(100)
        form.addRow('frame size', self.frmSizeBox)

        #pca th
        self.pcaThBox = QtGui.QLineEdit()
        self.pcaThBox.setText('0.17')
        self.pcaThBox.setAlignment(QtCore.Qt.AlignRight)
        self.pcaThBox.setFixedWidth(100)
        form.addRow('pca threshold', self.pcaThBox)

        #selected joints
        self.selected = QtGui.QRadioButton('selected')
        form.addRow('dimension', self.selected)

        #selected joints
        self.allSlt = QtGui.QRadioButton('selected')
        form.addRow('all frame', self.allSlt)

        #output file
        boxFile = QtGui.QHBoxLayout()
        btnOutput = QtGui.QPushButton('output')
        btnOutput.clicked.connect(self.save_params)
        boxFile.addWidget(btnOutput)
        #form.addWidget(btnOutput)

        #exec
        boxCtrl = QtGui.QHBoxLayout()
        btnExec = QtGui.QPushButton('exec')
        btnExec.clicked.connect(self.do_exec)
        #btnExec.clicked.connect(self.manyFileExec)
        boxCtrl.addWidget(btnExec)
 
        #配置
        grid.addLayout(form,1,0)
        grid.addLayout(boxCtrl,2,0)
        grid.addLayout(boxFile,3,0)

        self.setLayout(grid)
        self.resize(400,100)

        self.setWindowTitle("cca window")
        self.show()

    def choose_db_file(self):
        dialog = QtGui.QFileDialog()
        dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        if dialog.exec_():
            fileNames = dialog.selectedFiles()
            for f in fileNames:
                self.txtSepFile.setText(f)
                return
        return self.txtSepFile.setText('')
    """
    def chooseOutFile(self):
        dialog = QtGui.QFileDialog()
        dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        if dialog.exec_():
            fileNames = dialog.selectedFiles()
            for f in fileNames:
                self.txtSepFileOut.setText(f)
                return
        return self.txtSepFileOut.setText('')
    """
    def updateColorTable(self, cItem):
        self.r = cItem.row()
        self.c = cItem.column()
        print "now viz r:",self.r,", c:",self.c

    def do_exec(self):
        print "exec start:",datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        #input file
        self.fname = str(self.txtSepFile.text())
        self.data1,self.data2 = self.json_input(self.fname)
        #print self.data1.shape
        #select joints
        hitbtn = self.selected.isChecked()
        self.data1, self.data2 = self.select_input(self.data1, self.data2, hitbtn)
        #if data is big then...
        self.data1, self.data2 = self.cut_datas(self.data1, self.data2, 300)
        self.dts, self.dtd = self.data1.shape

        #ws:window_size, fs:frame_size 
        self.wins = int(self.winSizeBox.text())
        self.frms = int(self.frmSizeBox.text())
        if self.allSlt.isChecked() == True:
            print "frame all"
            self.frms = self.dts

        #dmr:data_max_range, frmr:frame_range, dtr:data_range
        self.dtmr = self.dts - self.wins + 1
        self.frmr = self.dts - self.frms + 1
        self.dtr = self.frms - self.wins + 1

        print "datas_size:",self.dts
        print "frame_size:",self.frms
        print "data_max_range:",self.dtmr
        print "frame_range:",self.frmr
        print "data_range:",self.dtr

        #rho_m:rho_matrix[dmr, dmr, datadimen] is corrs
        #wx_m and wy_m is vectors
        self.r_m, self.wx_m, self.wy_m = self.cca_exec(self.data1, self.data2)
        
        #graph
        self.rhoplot()

        print "end:",datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    def rhoplot(self):
        #rho plot
        
        #pl.subplot2grid((3,3),(1,1),colspan=2,rowspan=2)
        #print corrs.shape
        #dr, dc = corrs.shape
        fs=10
        dr, dc = self.r_m[:,:,0].shape
        Y,X = np.mgrid[slice(0, dc+1, 1),slice(0, dr+1, 1)]
        plt.pcolor(X, Y, self.r_m[:,:,0], vmin=0, vmax=1)
        #plt.pcolor(X, Y, self.r_m[:,:,0])

        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=fs-1) 
        plt.gray()
        plt.xlim(0,dc)
        plt.ylim(0,dr)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.title("rho",fontsize=fs+1)
        plt.show()
        


    def json_input(self, filename):
        f = open(filename, 'r')
        jsonData = json.load(f)
        f.close()
        #angle
        datas = []
        for user in jsonData:
            #data is joint angle
            data = []
            #dts:data_size, dtd:data_dimension
            self.dts = len(user["datas"])
            self.dtd = len(user["datas"][0]["data"])
            for j in range(self.dts):
                data.append(user["datas"][j]["data"])
            datas.append(data)
        return np.array(datas[0]), np.array(datas[1])

    def output(self):
        savefile = "save_w"+str(self.wins)+"_f"+str(self.frms) +"_d"+str(self.dtd)+ "_"+ self.fname.lstrip("/home/uema/catkin_ws/src/rqt_cca/data2/") 

        print "filename:",savefile
        f = open(savefile ,'w')
        prop = {"wins":self.wins, "frms":self.frms, "dtd":self.dtd, "dts":self.dts, "fname":self.fname, "sidx":self.sIdx}
        cca = {"r":self.r_m[:,:,0].tolist(),"wx":self.wx_m[:,:,:,0].tolist(), "wy":self.wy_m[:,:,:,0].tolist()}
        #cca = {"r":self.r_m.tolist(),"wx":self.wx_m.tolist(), "wy":self.wy_m.tolist()}
        js = {"prop":prop, "cca":cca}
        jsons = json.dumps(js)
        f.write(jsons)
        f.close()
        print "save end:",datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    def save_params(self):
        savefile = "save_w"+str(self.wins)+"_f"+str(self.frms) +"_d"+str(self.dtd)+ "_"+ self.fname.lstrip("/home/uema/catkin_ws/src/rqt_cca/data2/") 
        savefile = savefile.rstrip(".json")
        filepath = savefile+".h5"
        print filepath+" is save"
        with h5py.File(filepath, 'w') as f:
            p_grp=f.create_group("prop")
            p_grp.create_dataset("wins",data=self.wins)
            p_grp.create_dataset("frms",data=self.frms)
            p_grp.create_dataset("dtd",data=self.dtd)
            p_grp.create_dataset("dts",data=self.dts) 
            p_grp.create_dataset("fname",data=self.fname)
            p_grp.create_dataset("sidx",data=self.sIdx)

            c_grp=f.create_group("cca")
            r_grp=c_grp.create_group("r")
            wx_grp=c_grp.create_group("wx")
            wy_grp=c_grp.create_group("wy")

            #c_grp.create_dataset("r",data=self.r_m[:,:,0])
            for i in range(self.dtd):
                r_grp.create_dataset(str(i),data=self.r_m[:,:,i])
                wx_v_grp = wx_grp.create_group(str(i))
                wy_v_grp = wy_grp.create_group(str(i))
                for j in range(self.dtd):
                    wx_v_grp.create_dataset(str(j),data=self.wx_m[:,:,j,i])
                    wy_v_grp.create_dataset(str(j),data=self.wy_m[:,:,j,i])

            #d_grp=f.create_group("data")

            f.flush()
        print "save end:",datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    def select_input(self, data1, data2, hit):
        if hit == False:
            self.sIdx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
            return data1, data2
        else:
            #idx = [3, 4, 5, 6, 11, 12, 23, 24, 25, 26, 27, 28]
            self.sIdx = [0,1,2,3,4,5,6,11,12,17,18,19,20,21,22,23,24,25,26,27,28]
            return data1[:,self.sIdx],  data2[:,self.sIdx]

    def cut_datas(self, data1, data2, th): 
        if self.dts < th:
            return data1, data2
        else:
            return data1[0:th,:], data2[0:th,:]

    def cca_exec(self, data1, data2):
        #rho_m:rho_matrix[dmr, dmr, datadimen] is corrs
        #wx_m and wy_m is vectors

        r_m = np.zeros([self.dtmr, self.dtmr, self.dtd])
        wx_m = np.zeros([self.dtmr, self.dtmr, self.dtd, self.dtd])
        wy_m = np.zeros([self.dtmr, self.dtmr, self.dtd, self.dtd])

        for f in tqdm.tqdm(range(self.frmr)):
            #print "f: ",f,"/",self.frmr-1
            if f == 0:
                for t1 in tqdm.tqdm(range(self.dtr)):
                    for t2 in range(self.dtr):
                        u1 = data1[f+t1:f+t1+self.wins,:]
                        u2 = data2[f+t2:f+t2+self.wins,:]
                        u1,u2,id1,id2 = self.pca(u1,u2) 
                        #r_m[f+t1][f+t2]  = self.cca(u1, u2, id1, id2)
                        r_m[f+t1][f+t2], wx_m[f+t1][f+t2], wy_m[f+t1][f+t2] = self.cca(u1, u2, id1, id2)
            else:
                od = f+self.dtr-1
                for t1 in range(self.dtr-1):
                    u1 = data1[f+t1:f+t1+self.wins,:]
                    u2 = data2[od:od+self.wins,:]
                    u1,u2,id1,id2 = self.pca(u1,u2) 
                    #r_m[f+t1][od] = self.cca(u1, u2, id1, id2)
                    r_m[f+t1][od], wx_m[f+t1][od], wy_m[f+t1][od] = self.cca(u1, u2, id1, id2)
                for t2 in range(self.dtr):
                    u1 = data1[od:od+self.wins,:]
                    u2 = data2[f+t2:f+t2+self.wins,:]
                    u1,u2,id1,id2 = self.pca(u1,u2) 
                    #r_m[od][f+t2] = self.cca(u1, u2, id1, id2)
                    r_m[od][f+t2], wx_m[od][f+t2], wy_m[od][f+t2] = self.cca(u1, u2, id1, id2)

        return r_m, wx_m, wy_m


    def pca(self, X, Y):
        th = float(self.pcaThBox.text())
        #th = 0.017
        D = [X, Y]
        idx = []
        out = []
        for d in D:
            s=np.cov(d.T)
            l,v=NLA.eigh(s)
            i = np.argsort(l)[::-1]
            l=l[i]
            v=v[:,i]
            c=v[:,0]
            ci = np.where(np.fabs(c)>th)
            #print "ci",ci
            #print "d s",d.shape
            #print "d[]",d[:,ci]
            idx.append(ci)
            out.append(d[:,ci])
        #print "out s",out[0].shape
        return out[0][:,0,:], out[1][:,0,:], idx[0], idx[1] 



    def cca(self, X, Y, id1, id2):
        '''
        正準相関分析
        http://en.wikipedia.org/wiki/Canonical_correlation
        '''    
        #X = np.array(X)
        #Y = np.array(Y)
        #print X.shape
        n, p = X.shape
        n, q = Y.shape
        
        # zero mean
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        
        # covariances
        S = np.cov(X.T, Y.T, bias=1)
        
        # S = np.corrcoef(X.T, Y.T)
        SXX = S[:p,:p]
        SYY = S[p:,p:]
        SXY = S[:p,p:]
        #SYX = S[p:,:p]
        
        #正則化?
        #Rg = np.diag(np.ones(p)*0.001)
        #SXX = SXX + Rg
        #SYY = SYY + Rg
        #
        sqx = SLA.sqrtm(SLA.inv(SXX)) # SXX^(-1/2)
        sqy = SLA.sqrtm(SLA.inv(SYY)) # SYY^(-1/2)
        M = np.dot(np.dot(sqx, SXY), sqy.T) # SXX^(-1/2) * SXY * SYY^(-T/2)
        A, s, Bh = SLA.svd(M, full_matrices=False)
        B = Bh.T      
        
        """
        vecs = []
        ids = [id1, id2]
        for idx in ids:
            wx = []

            for n in range(self.dtd):
                print "i",i,"n",n
                if n == idx[n]:
                    wx[n] = A[:,n]
                else:
                    wx[n] = np.zeros(self.dtd)
            vecs.append(np.array(wx).T)
        """
        #print np.dot(np.dot(A[:,0].T,SXX),A[:,0])
        z = p if p < q else q
        r = np.zeros([self.dtd])
        wx = np.zeros([self.dtd,self.dtd])
        wy = np.zeros([self.dtd,self.dtd])
        r[:z]=s
        wx[id1,:z] = A
        wy[id2,:z] = B
        return r,wx,wy


def main():
    app = QtGui.QApplication(sys.argv)
    corr = CCA()
    #graph = GRAPH()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()
