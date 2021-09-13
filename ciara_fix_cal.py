#! /usr/bin/env python
import numpy as np
import lofar.parmdb as pb
from argparse import ArgumentParser
import sys
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
parser = ArgumentParser("Simple parm filter")
parser.add_argument('parmdb',help='input parmdb',metavar='parmdb')
#parser.add_argument('-o','--out',help='output parmdb')
def filter_values(values):
    return np.ones_like(values)*np.average(values)
def change_format_of_array(array):    
    replace=[]
    for jj in array:
        replace.append([jj])
  #  print('same format to put into table',len(replace), np.array(replace))
    return(replace)
def make_rows_cols(Num):
    rows,columns=[],[]
    for i in range(Num):
        n=np.full((1,Num),i)[0]
        rows.append(n)
        n=np.arange(0,Num,1)
        columns.append(n)
    columns=np.concatenate(columns)
    rows=np.concatenate(rows)
    return(rows,columns)
def smoothing_remote(time,array):
    #remove section between 19650 to 19800 # fo
    st,end=19650,19800  # time in seconds where burst starts and ends - find out using parmdbplot.py
    time_res=0.16777216
    region=150
    x_new=time[st-region:end+region]
    xs=time[st-region:st]
    ys=array[st-region:st]
    array[st:end]=ys
    replace=change_format_of_array(array)    
    return(np.array(replace),array)
def smoothing(time,array):
    #remove section between 19650 to 19800 # fo
    st,end=1762, 1763  # time in seconds where burst starts and ends - find out using parmdbplot.py
    time_res=0.16777216
    region=200
    x_new=time[st-region:end+region]
    ys,xs=[],[]
    ys.append((array[st-region:st]))
    ys.append((array[end:end+region]))
    ys=np.concatenate(ys)
    xs.append(time[st-region:st])
    xs.append(time[end:end+region])
    xs=np.concatenate(xs)
    z = np.polyfit(xs, ys, 2)
    f = np.poly1d(z)
    array[st-region:end+region]=f(x_new)
    replace=change_format_of_array(array)    
    return(np.array(replace),array)
def make_subplot(parm,time,or_array,fixed_array,axsp):
        axsp.plot(time,or_array,'-',color='red',linewidth=1.5,alpha=.2) # original
        axsp.plot(time,fixed_array,'-',color='black',linewidth=.8) # fixed
        axsp.set_title(parm,fontsize=6)
        axsp.set_xlim(18e3,20e3)
        axsp.set_ylim(-500,500)
#def main(argv):
print('---------------------')
print('loading table')
#print(argv)
argv = sys.argv[1:] 
print('---------------------')
args=parser.parse_args(argv)
#args=parser.parse_args(['L401011_SB403_uv.dppp.MS/instrument/'])
# newpb=pb.parmdb(args.parmdb+"NEW",create=True)
newparms={} #dictionary with a copy of eveerything of the old parms, but new filtered values
parms=pb.parmdb(args.parmdb).getValuesGrid("*")
Nparms = len(parms)
Num = int(np.ceil(np.sqrt(Nparms))) # number of plots needed - each station has phase and amp - then each pahse and amp has real and imag
# fp, axp = plt.subplots(Num, Num,sharex=True, sharey=True, figsize=(20,20))
#make the subplot notation correct
# rows,columns=make_rows_cols(Num)
print('---------------------')
print('resolving corrupted calibration data')
print('---------------------')
cnt=1
for parm in sorted(parms.keys()):
    newparm=parms[parm].copy()
   # print(parms[parm])       
    val=parms[parm]['values']
  #  print('old',len(val), val)
    original_array=np.concatenate(val)
    time = np.linspace(0,len(original_array), len(original_array))
    # if str(parm)[14:16]=='RS':
    #     newparm['values'],fixed_array = smoothing_remote(time,original_array) 
    # else:
    #     newparm['values'],fixed_array = smoothing(time,original_array)
    val=parms[parm]['values']
    original_array=np.concatenate(val) # to get it into the right format
    # axiss=axp[rows[cnt],columns[cnt]]
    # make_subplot(parm,time,original_array,fixed_array,axiss)
    cnt=cnt+1
    newparms[parm]=newparm #update new array
# fp.subplots_adjust(wspace=0.8, hspace=0.8)  
# fp.savefig("sb347_smooth_differ_solutions.png",dpi=500) 
# newpb.addValues(newparms) #save to tables
#if __name__ == '__main__':
#    main(sys.argv[1:])    


