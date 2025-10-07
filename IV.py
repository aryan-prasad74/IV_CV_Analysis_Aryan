# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:11:15 2019

This is a module for plotting IV curves and determining the breakdown voltage. All user specific values (ex. file path, image path etc.) can be specified using optional parsers.  
@author: Sneha Ramshanker 

"""

import Functions as fun
import Constants as con
from optparse import OptionParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re


"""
Optional Parsing
"""

parser = OptionParser()
parser.add_option("--fileName", type="string", help="Path of file with IV data", dest="fileName", default=None)
parser.add_option("--fileList", type="string", help="Path of file with list of files/names", dest="fileList", default=None)
parser.add_option("--outfile",type = "string", help = "Path to store IV plot", default = "test.png")
parser.add_option("--outdir",type = "string", help = "Directory where plots should be stored", default = "IVplots")
parser.add_option("--xlrow", type = "int", help = "What row of the excel file should this data be stored in", dest = "xlrow", default = 2)
parser.add_option("--xlpath", type = "string", help = "Path of excel file to store results", dest = "xlpath", default = 'Testdata/Breakdown.xlsx')
parser.add_option("--saveExcel", type = "int", help = "Set to 1 if you want to save the output in an excel file, 0 otherwise", default = 0)
parser.add_option("--rowName", type="string", help = "A unique string that identifies the row before the data starts", default = "BEGIN")
parser.add_option("--voltageColumn", type="int", help = "The data column where the voltage is stored", default = 0)
parser.add_option("--currentColumn", type="int", help = "The data column where the current is stored", default = 2)
parser.add_option("--separatorString", type="string", help = "The string between data columns to separate them", default = "\t")

options, arguments = parser.parse_args()

fun.setPlotStyle() #Setting format of the graph


files = []
fileTitles = []

if options.fileName:
  files = [options.fileName]
  # Use only the filename (no directories) and remove the extension
  fileTitles = [os.path.splitext(os.path.basename(options.fileName))[0]]
elif options.fileList:
  dfFiles = pd.read_csv(options.fileList, sep=" ", header=None)
  # Prepend the folder path to each filename
  files = ["Measurement_CSVs/" + f for f in dfFiles[0]]
  fileTitles = dfFiles[1]
else:
  print("No input files")

try:
  os.mkdir(options.outdir)
except:
  "Directory already exists"

# Helper function to extract sensor code like W3082 from filename
def get_sensor_code(filename):
  m = re.search(r'(W\d{4})', filename)
  if m:
    return m.group(1)
  else:
    return "Unknown"

# for cfile, ctitle in zip(files, fileTitles):
#   df = fun.storedataIV(cfile, options.rowName, separator=options.separatorString)
#   df = fun.cleanupIV(df, options.voltageColumn, options.currentColumn)
#   breakdownVol = fun.breakdownVol(df, options.voltageColumn, options.currentColumn)

#   outfileName = os.path.join(options.outdir, ctitle + options.outfile)
#   fun.dataplot(df, options.voltageColumn, options.currentColumn, outfileName, 'ylog', 'Voltage [V]', 'Current [A]', iden = "IV", breakdownVol = breakdownVol)

# Group files by sensor code (e.g. W3082)
sensor_groups = {}
for cfile, ctitle in zip(files, fileTitles):
  sensor = get_sensor_code(cfile)
  if sensor not in sensor_groups:
    sensor_groups[sensor] = []
  sensor_groups[sensor].append((cfile, ctitle))

# Helper function to extract sensor code and sub-ID like W3082_05-18
def get_sensor_subid(filename):
    m = re.search(r'(W\d{4})_(\d{2}-\d{2})', filename)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    else:
        return "Unknown"

# Group files by sensor and sub-ID
sensor_groups = {}
for cfile, ctitle in zip(files, fileTitles):
    sensor_subid = get_sensor_subid(cfile)
    sensor = sensor_subid.split('_')[0]  # W3082
    if sensor not in sensor_groups:
        sensor_groups[sensor] = {}
    if sensor_subid not in sensor_groups[sensor]:
        sensor_groups[sensor][sensor_subid] = []
    sensor_groups[sensor][sensor_subid].append(cfile)

# Loop through each sensor and plot averaged curves for each sub-ID
for sensor, subid_dict in sensor_groups.items():
    plt.figure(figsize=(8,6))
    for subid, file_list in subid_dict.items():
        # Load all CSVs for this sub-ID
        dfs = [fun.storedataIV(f, options.rowName, separator=options.separatorString) for f in file_list]
        # Compute average curve
        V_avg, I_avg = fun.average_IV_csvs(dfs, options.voltageColumn, options.currentColumn)
        # Compute breakdown voltage
        df_avg = pd.DataFrame({options.voltageColumn: V_avg, options.currentColumn: I_avg})
        bd_vol = fun.breakdownVol(df_avg, options.voltageColumn, options.currentColumn)
        # Plot
        plt.plot(V_avg, I_avg, label=f"{subid}_avg")
        plt.axvline(bd_vol, linestyle='--', alpha=0.5, label=f"{subid}_BDV: {bd_vol:.1f} V")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [A]")
    plt.yscale("log")
    plt.title(f"IV Curves for Sensor {sensor}")
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    outfileName = os.path.join(options.outdir, f"{sensor}_IV_avg.png")
    plt.savefig(outfileName)
    plt.close()



#Storing data in an excel file 
if options.saveExcel:
  xlrow = options.xlrow 
  xlpath = options.xlpath
  fun.tablebreakdownVol(xlpath, breakdownVol ,'C', 'Breakdown Voltage')


