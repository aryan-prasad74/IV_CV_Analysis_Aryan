# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:27:04 2019
This module outputs the CV curve, Doping Profile, and Inverse squared capacitance plots for a given inputed text file. All the user specific information can be parsed using optional parsers. 

@author: sneha
"""
import Functions as fun
import Constants as con
from optparse import OptionParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from astropy.modeling import models, fitting



"""
Optional Parsing
"""

parser = OptionParser()
parser.add_option("--delim", type = "string", help = "What kind of delimiter is used?", dest = "delim")
parser.add_option("--fileName", type="string",
                  help="Path of file",
                  dest="fileName", default="Datasets\CV\EXX28995-WNo4-Type3-1\LG1-SE5-0.txt")

parser.add_option("--impathCV",type = "string", help = "Path to store CV plot", dest = "impathCV", default = "CV_plots/a1.png")
parser.add_option("--impathDPW",type = "string", help = "Path to store Doping Profile - Width plot", dest = "impathDPW", default = "DPW_plots/a2.png")
parser.add_option("--impathCVinv",type = "string", help = "Path to store inverse CV plot", dest = "impathCVinv", default = "DPW_plots/b1.png")
parser.add_option("--impathDPWcut",type = "string", help = "Path to store DPW plots with log scale", dest = "impathDPWcut", default = "DPW_plots/a.png")
parser.add_option("--xlpath", type = "string", help = "Path of excel file to store results", dest = "xlpath", default = 'Testdata/CV31.xlsx')
parser.add_option("--saveExcel", type = "int", help = "Set to 1 if you want to save the output in an excel file, 0 otherwise", default = 0)
parser.add_option("--rowName", type="string", help = "A unique string that identifies the row before the data starts", default = "BEGIN")
parser.add_option("--voltageColumn", type="int", help = "The data column where the voltage is stored", default = 0)
parser.add_option("--capacitanceColumn", type="int", help = "The data column where the capacitance is stored", default = 1)


options, arguments = parser.parse_args()

impathCV = options.impathCV
impathDPW = options.impathDPW
impathDPWcut = options.impathDPWcut
impathCVinv = options.impathCVinv
delim = options.delim
xlpath = options.xlpath

"""
Rest of the Program
"""

fun.setPlotStyle() #Setting format of the graph

if options.delim:
    df = fun.simpletextread(options.fileName, delim)
    x = df.columns[0]
    y = df.columns[1]
    
else:
    df = fun.storedataCV(options.fileName, options.rowName)
    x = df.columns[0]
    y = df.columns[2]
    
#CV
df = fun.cleanupCV(df, x, y)
fun.dataplot(df, x, y, impathCV, 'nolog', 'Voltage [V]','Capacitance [F]') 

#Doping Profile (DPW)
CV = df.loc[:, [x, y]]
 
CV.columns = [options.voltageColumn, options.capacitanceColumn]

#Inverse Capacitance
CV['inverse capacitance'] = CV.apply(lambda row: row[options.capacitanceColumn]**-2, axis = 1)


outlr = fun.isoutlierCV(CV, 3, 0.8, 'inverse capacitance', 'lr')

dflr_out = fun.splitdata(outlr)[0]
dflr_nout = fun.splitdata(outlr)[1]


gain_deplvol = fun.reg_intersect(dflr_out, dflr_nout, options.voltageColumn, "inverse capacitance", 60)[0] # Calculating depletion voltage of the gain layer

fun.dataplot(CV,options.voltageColumn,'inverse capacitance', impathCVinv, 'nolog',  'Voltage [V]', r'$C^{-2}$ [$F^{-2}$]', 'invCV', gain_deplvol, 0)


DPW = fun.get_doping_profile(CV,con.area, options.voltageColumn, options.capacitanceColumn)
n = DPW['profile'].shape[0]


x = DPW["width"].tolist()[0:n]
y = DPW["profile"].tolist()[0:n]


maxy = max(y)
g_init = models.Gaussian1D(amplitude=maxy, mean=0 , stddev=max(x))
#g_init = models.Gaussian1D(amplitude=max(y), mean=0 , stddev=1.)
fit_g = fitting.LevMarLSQFitter()
g = fit_g(g_init, x, y)
gainpeak = max(g(x))
idx = np.where(list(g(x)) == gainpeak)[0][0]
gainpeakwidth = x[idx]

FWHM = fun.FWHM(x, g(x))

fun.dataplotDPW(DPW,'width','profile', impathDPW, 'log',  r'Width [$\mu m$]', r'Doping Concentration [$cm^{-3}$]', gainpeakwidth, gainpeak, FWHM = [FWHM, x, g])



#Storing Data in Excel file 
if options.saveExcel:
  #fun.tablebreakdownVol(xlpath, gain_deplvol,'C','CV')
  #fun.tablebreakdownVol(xlpath, tot_deplvol,'D','CV')
  fun.tablebreakdownVol(xlpath,gainpeakwidth,'E','CV')
  fun.tablebreakdownVol(xlpath,gainpeak,'F','CV' )
  fun.tablebreakdownVol(xlpath,FWHM,'G','CV')



