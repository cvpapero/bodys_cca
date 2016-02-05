#!/usr/bin/python
# -*- coding: utf-8 -*-

#ランダムでjsonデータを作る
"""
[{"datas":[{"data":[1,2,3]},{}]}, {}]
関節データの範囲は0.1から3.14(rad)
random.uniform(0.1, 3.14)
"""

import random
import json
import sys
import math

class Dummy():

    def __init__(self):
        self.jsonCreate()

    def jsonCreate(self):

        usize = 2
        nsize = 15#300
        psize = 4#29
        
        users = []
        for u in range(usize): 
            ds_array = []
            for n in range(nsize):
                rjs = []
                for p in range(psize):
                    rj = random.uniform(0.1, 3.14)
                    rjs.append(round(rj,3))
                ds = {"data":rjs}
                ds_array.append(ds)
        
            datas = {"datas":ds_array}
            users.append(datas)
        #jsonString = json.dumps(users, indent=2)
        #print users
        f = open('testdata_ran.json','w') 
        jString = json.dumps(users)
        f.write(jString)
        f.close()


def main():
    dy = Dummy()
        

if __name__=='__main__':
    main()
