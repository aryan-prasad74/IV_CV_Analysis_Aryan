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
parser.add_option("--impathCVinv",type = "string", help = "Path to store inverse CV plot", dest = "impathCVinv", default = "CVinv_plots/b1.png")
parser.add_option("--impathDPWcut",type = "string", help = "Path to store DPW plots with log scale", dest = "impathDPWcut", default = "DPW_plots/a.png")
parser.add_option("--xlpath", type = "string", help = "Path of excel file to store results", dest = "xlpath", default = 'Testdata/CV31.xlsx')
parser.add_option("--saveExcel", type = "int", help = "Set to 1 if you want to save the output in an excel file, 0 otherwise", default = 0)
parser.add_option("--rowName", type="string", help = "A unique string that identifies the row before the data starts", default = "BEGIN")
parser.add_option("--voltageColumn", type="int", help = "The data column where the voltage is stored", default = 0)
parser.add_option("--capacitanceColumn", type="int", help = "The data column where the current is stored", default = 1)
parser.add_option("--currentColumn", type="int", help = "The data column where the current is stored", default = 4)


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
fun.dataplot(df, x, y, impathCV, 'nolog', 'Voltage [V]','Capacitance [F]', iden = "invCV", gaindeplvol = 0, totdeplvol = 0) 

#Doping Profile (DPW)
CV = df.loc[:, [x, y]]
 
CV.columns = [options.voltageColumn, options.capacitanceColumn]
DPW = fun.get_doping_profile(CV,con.area, options.voltageColumn, options.capacitanceColumn, options.currentColumn)
n = DPW['profile'].shape[0]
#gainpeakwidth = DPW.loc[DPW['profile'].loc[:0.5*n].idxmax()][0]
#gainpeak = DPW.loc[DPW['profile'].loc[:0.5*n].idxmax()][1]

x = DPW["width"].tolist()[0:n]
y = DPW["profile"].tolist()[0:n]


g_init = models.Gaussian1D(amplitude=10**16, mean=0 , stddev=1.)
fit_g = fitting.LevMarLSQFitter()
g = fit_g(g_init, x, y)
gainpeak = max(g(x))
idx = np.where(list(g(x)) == gainpeak)[0][0]
gainpeakwidth = x[idx]

FWHM = fun.FWHM(x, g(x))

fun.dataplotDPW(DPW,'width','profile', impathDPW, 'log',  r'Width [$\mu m$]', r'Doping Concentration [$cm^{-3}$]', gainpeakwidth, gainpeak, FWHM = [FWHM, x, g])
#fun.dataplot(DPW,'width','profile', impathDPWcut, 'log',  r'Width [$\mu m$]', r'Doping Concentration [$cm^{-3}$]', 'DPWcut')

#Inverse Capacitance
CV['inverse capacitance'] = CV.apply(lambda row: row[options.capacitanceColumn]**-2, axis = 1)

N = CV.shape[0]


# I do not understand where these values come from...
# Like, what is the 0.53 about?

#datalr = CV.sort_values('inverse capacitance').head(int(0.2*N)) #0.53
#datarl = CV.sort_values('inverse capacitance',ascending=False).head(int((0.9*N))) #0.52
#
#outlr = fun.isoutlierCV(datalr, 3, 0.7, 'inverse capacitance', 'lr') #0.8
#outrl = fun.isoutlierCV(datarl, 3 , 0.8,'inverse capacitance','rl') #0.8

datalr = CV.sort_values('inverse capacitance').head(int(0.53*N))
datarl = CV.sort_values('inverse capacitance',ascending=False).head(int((0.53*N)))


outlr = fun.isoutlierCV(datalr, 3, 0.8, 'inverse capacitance', 'lr')
outrl = fun.isoutlierCV(datarl, 3 , 0.8,'inverse capacitance','rl')


dflr_out = fun.splitdata(outlr)[0]
dflr_nout = fun.splitdata(outlr)[1]
dfrl_out = fun.splitdata(outrl)[0]
dfrl_nout = fun.splitdata(outrl)[1]

#print "test", dflr_out
#print "test2", dflr_nout 
 
tot_deplvol = fun.reg_intersect(dfrl_out, dfrl_nout, options.voltageColumn, "inverse capacitance", 60)[0] # Calculating depletion voltage of the gain layer
gain_deplvol = fun.reg_intersect(dflr_out, dflr_nout, options.voltageColumn, "inverse capacitance", 60)[0] # Calculating depletion voltage of the gain layer

print(gain_deplvol)
print(tot_deplvol)

fun.dataplot(CV,'voltage','inverse capacitance', impathCVinv, 'nolog',  'Voltage [V]', r'$C^{-2}$ [$F^{-2}$]', 'invCV', gain_deplvol, tot_deplvol)

#Storing Data in Excel file 

if options.saveExcel:
  #fun.tablebreakdownVol(xlpath, gain_deplvol,'C','CV')
  #fun.tablebreakdownVol(xlpath, tot_deplvol,'D','CV')
  fun.tablebreakdownVol(xlpath,gainpeakwidth,'E','CV')
  fun.tablebreakdownVol(xlpath,gainpeak,'F','CV' )
  fun.tablebreakdownVol(xlpath,FWHM,'G','CV')



