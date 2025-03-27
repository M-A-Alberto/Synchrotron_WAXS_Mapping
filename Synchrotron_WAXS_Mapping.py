# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:12:53 2024

@author: amart
"""



import numpy as np

import pyFAI

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl
import os 

import fabio

from scipy.integrate import simpson
from numpy import trapz
from sklearn.metrics import auc

from scipy.optimize import curve_fit

sns.set_theme()
sns.set_style("ticks")
sns.set_context("paper",font_scale=1.5)
sns.set_palette("tab10")

os.chdir(r"XXXXXXXXXXX") #Input working directory


def recta(x,m,a):
    
    y = m*x+a
    
    return y




z = {}

x = {}

i = 0
j = 0

for file in os.listdir("raw"):
    if file == "Mesh_snapdmesh_3967":
        
        for data_file in os.listdir(f"./raw/{file}/data_00/"):
                
            data_dir = f"./raw/{file}/data_00/{data_file}"
            
            
            img = fabio.open(data_dir)
            
            header = img.getheader()

            z_pos = round(float(header["stz"]),2)
            x_pos = round(float(header["sx"]),2)
            
            if str(z_pos) not in z.keys():
                
                z[str(z_pos)] = i
                
                i+=1
            
            if  str(x_pos) not in x.keys():
                
                x[str(x_pos)] = j
                j+=1

    
bkg = pd.read_table(r"Processed/LaVue1D/rayonix_NPs_empty_000_0000.dat",skiprows = 23,header=None,sep = "\s+",names = ["q","I","sigma"])

bkg.I = bkg.I/np.max(bkg.I)

ROI_bkg = bkg.loc[(bkg.q > 24.7) & (bkg.q<25.4),:].reset_index()

t = 0

for mesh in os.listdir("./Processed/"):
    
    if "Mesh" in mesh:
        os.makedirs(r"maps/WAXS/Curves",exist_ok=True)
        os.makedirs(r"maps/WAXS/maps",exist_ok=True)
        map_data = np.zeros((len(z),len(x)))
        map_abs = np.zeros((len(z),len(x)))
        #fig, ax = plt.subplots(len(z),len(x))
        
        nrows = len(z)
        
        for file in os.listdir(f"./Processed/{mesh}/data/LaVue1D/"):
        
            if "rayonix" in file:
                name = file[:-4]     
                processed_file = f"./Processed/{mesh}/data/LaVue1D/{name}.dat"
                raw_file = f"./raw/{mesh}/data_00/{name}.edf"
                
                raw = fabio.open(raw_file)
                
                header = raw.getheader()
        
                z_pos = round(float(header["stz"]),2)
                x_pos = round(float(header["sx"]),2)
                
                row = z[str(z_pos)]
                col = x[str(x_pos)]
                
                
                I = float(header["Photo"]) / float(header["Monitor"])
                
                processed = pd.read_table(processed_file,skiprows = 23,header=None,sep = "\s+",names = ["q","I","sigma"])
                
                processed.I = processed.I/np.max(processed.I)
                
                ROI = processed.loc[(processed.q > 24.7) & (processed.q<25.4),:].reset_index()
                
                ROI.I = ROI.I - ROI_bkg.I
                
                
                y1 = ROI.iloc[0,2]
                y2 = ROI.iloc[-1,2]
                
                x1 = ROI.iloc[0,1]
                x2 = ROI.iloc[-1,1]
                
                m = (y2-y1)/(x2-x1)
                
                a = y2-m*x2
                
                """
                ax[row,col].plot(ROI.q,ROI.I,label="Profile")
                
                ax[row,col].plot(ROI.q,recta(ROI.q,m,a),label="Background")
                ax[row,col].set_yticks([])
                ax[row,col].set_xticks([])
                
                #ax[row,col].set_xscale("log")
                #ax[row,col].set_yscale("log")
                #ax[row,col].set_xlim([24.5,25.5])
                """
                
                
           
                area = auc(ROI.q,ROI.I-recta(ROI.q,m,a))
            
                
                
                
                map_data[row,col] = area
                map_abs[row,col] = I
                #plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        
        
        plt.suptitle(f"Time = {t} mins")
        plt.savefig(f"maps/WAXS/Curves/Time_{t}_mins.tif",dpi=300,bbox_inches="tight")
        plt.show()
        plt.close()     
        
        #map_data = map_data/np.max(map_data)
        
        plt.imshow(map_data,vmin=0,cmap = "gnuplot2",interpolation=None)
        plt.colorbar()
        plt.xticks(np.arange(0,len(x.keys()),1),[int(float(i)*1000+111)for i in x.keys()]) #
        plt.yticks(np.arange(0,len(z.keys()),1),[int(np.abs(float(i))*1000-27588) for i in z.keys()]) #
        plt.title(f"t = {t} mins")
        plt.xlabel("Relative x position to (0,0) ($\mu$m)")
        plt.ylabel("Relative z position to (0,0) ($\mu$m)")
        plt.savefig(f"maps/WAXS/maps/Time_{t}_mins.tif",dpi=300,bbox_inches="tight")
        plt.show()
        plt.close()
        
        
        map_abs = map_abs/np.max(map_abs)
        
        plt.imshow(map_abs,vmin=np.min(map_abs))
        plt.colorbar()
        plt.xticks(np.arange(0,len(x.keys()),1),[int(float(i)*1000+111)for i in x.keys()]) #
        plt.yticks(np.arange(0,len(z.keys()),1),[int(np.abs(float(i))*1000-27588) for i in z.keys()]) #
        plt.title(f"Normalized Transmittance Map t = {t} mins")
        plt.xlabel("Relative x position to (0,0) ($\mu$m)")
        plt.ylabel("Relative z position to (0,0) ($\mu$m)")
        plt.savefig(f"maps/WAXS/maps/Time_{t}_mins_transmittance.tif",dpi=300,bbox_inches="tight")
        plt.show()
        plt.close()
        t+=35