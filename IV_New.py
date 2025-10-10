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
from collections import defaultdict



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
  fileTitles = [options.fileName[:options.fileName.rfind(".")]]
elif options.fileList:
  dfFiles = pd.read_csv(options.fileList, sep=" ", header = None) #Store data into dataframe 
  files = dfFiles[0]
  fileTitles = dfFiles[1]
else:
  print("No input files")

try:
  os.mkdir(options.outdir)
except:
  "Directory already exists"

sensor_files = {}
breakdownVolts = {}

for cfile, ctitle in zip(files, fileTitles):
    # Add Subdirectory for all filepaths
    cfile = os.path.join("Measurement_CSVs", cfile)

    # Extract sensor ID, e.g., "W3108_12-14"
    m = re.search(r'(W\d{4}_\d{2}-\d{2})', cfile)
    if m:
        sensor_id = m.group(1)
    else:
        sensor_id = "Unknown"

    # Process the IV data
    df = fun.storedataIV(cfile, options.rowName, separator=options.separatorString)
    df = fun.cleanupIV(df, options.voltageColumn, options.currentColumn)
    breakdownVol = fun.safe_breakdownVol(df, options.voltageColumn, options.currentColumn)

    # Store breakdown voltage
    if sensor_id not in breakdownVolts:
        breakdownVolts[sensor_id] = []
    breakdownVolts[sensor_id].append(breakdownVol)
    #print(f"File: {ctitle}, Sensor: {sensor_id}, Breakdown Voltage: {breakdownVol:.2f} V")

    # Store filenames for that sensor
    if sensor_id not in sensor_files:
        sensor_files[sensor_id] = []
    sensor_files[sensor_id].append(cfile)
    #print(f"{cfile} added to sensor {sensor_id}")


# Calculating Average Breakdown Voltages for each sensor type
avg_breakdown = {}
for sensor, vbd_list in breakdownVolts.items():
    if vbd_list:  # avoid empty lists
        avg_breakdown[sensor] = np.mean(vbd_list)
        print(f"Sensor: {sensor}, Average Breakdown Voltage: {avg_breakdown[sensor]:.2f} V")
    else:
        avg_breakdown[sensor] = np.nan


# Averaging IV curves for each sensor
# Store cleaned DataFrames for each sensor for plotting/averaging
sensor_dfs = {}
for sensor, file_list in sensor_files.items():
    df_list = []
    for f in file_list:
        df = fun.storedataIV(f, options.rowName, separator=options.separatorString)
        df = fun.cleanupIV(df, options.voltageColumn, options.currentColumn)
        df_list.append(df)
    sensor_dfs[sensor] = df_list


# Group sensors by family (prefix before underscore) using sensor_files
sensor_families = {}
for sensor in sensor_dfs.keys():
    family = sensor.split('_')[0]
    if family not in sensor_families:
        sensor_families[family] = []
    sensor_families[family].append(sensor)



# Optional debugging output
# print(sensor_families)
# print(breakdownVolts)
# print(sensor_files) 


########## PLOTTING ##########


# Plot averaged IV curves per sensor family
for family, sensors in sensor_families.items():
    # Collect all DataFrames for sensors in this family
    family_dfs = {sensor: sensor_dfs[sensor] for sensor in sensors if sensor in sensor_dfs}
    
    # Collect corresponding breakdown voltages for each sensor
    family_breakdowns = {sensor: avg_breakdown[sensor] for sensor in sensors if sensor in avg_breakdown}
    
    # Output path
    save_path = os.path.join(options.outdir, f"Averaged_{family}_IV_TotCurr.png")
    
    fun.plot_family_avg_IV(family_name=family, sensor_dfs=family_dfs, voltage_col=options.voltageColumn,
                           current_col=options.currentColumn, avg_breakdowns=family_breakdowns, save_path=save_path)


    # # Call plotting function
    # fun.plot_family_avg_IV(
    #     family_name=family,
    #     sensor_dfs=family_dfs,
    #     voltage_col=options.voltageColumn,
    #     current_col=options.currentColumn,
    #     avg_breakdowns=family_breakdowns,
    #     save_path=save_path
    # )





# #Storing data in an excel file 
# if options.saveExcel:
#   xlrow = options.xlrow 
#   xlpath = options.xlpath
#   fun.tablebreakdownVol(xlpath, breakdownVol ,'C', 'Breakdown Voltage')