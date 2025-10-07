# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:11:15 2019

This module plots IV curves and determines breakdown voltage.
One plot per sensor type, with averaged curves per sensor ID.
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

# ------------------ Optional Parsing ------------------
parser = OptionParser()
parser.add_option("--fileName", type="string", help="Path of file with IV data", dest="fileName", default=None)
parser.add_option("--fileList", type="string", help="Path of file with list of files/names", dest="fileList", default=None)
parser.add_option("--outfile",type = "string", help = "Path to store IV plot", default = ".png")
parser.add_option("--outdir",type = "string", help = "Directory where plots should be stored", default = "IVplots")
parser.add_option("--xlrow", type = "int", help = "Excel row to store breakdown", dest = "xlrow", default = 2)
parser.add_option("--xlpath", type = "string", help = "Path of excel file to store results", dest = "xlpath", default = 'Testdata/Breakdown.xlsx')
parser.add_option("--saveExcel", type = "int", help = "Set to 1 to save output in Excel, 0 otherwise", default = 0)
parser.add_option("--rowName", type="string", help = "Unique string identifying row before data starts", default = "BEGIN")
parser.add_option("--voltageColumn", type="int", help = "Data column where voltage is stored", default = 0)
parser.add_option("--currentColumn", type="int", help = "Data column where current is stored", default = 2)
parser.add_option("--separatorString", type="string", help = "String between data columns", default = "\t")

options, arguments = parser.parse_args()

fun.setPlotStyle()  # Set format of the graph

# ------------------ Load Files ------------------
files, fileTitles = [], []

if options.fileName:
    files = [options.fileName]
    fileTitles = [os.path.splitext(os.path.basename(options.fileName))[0]]
elif options.fileList:
    dfFiles = pd.read_csv(options.fileList, sep=" ", header=None)
    files = ["Measurement_CSVs/" + f for f in dfFiles[0]]
    fileTitles = dfFiles[1]
else:
    print("No input files")

try:
    os.mkdir(options.outdir)
except:
    pass

# ------------------ Helper Functions ------------------
def get_sensor_info(filename):
    m = re.search(r'(W\d{4})_(\d{2}-\d{2})', filename)
    if m:
        return m.group(1), m.group(2)
    else:
        return "Unknown", "Unknown"

# Group files by sensor type and sensor ID
sensor_groups = {}
for cfile, ctitle in zip(files, fileTitles):
    sensor_type, sensor_id = get_sensor_info(cfile)
    if sensor_type not in sensor_groups:
        sensor_groups[sensor_type] = {}
    if sensor_id not in sensor_groups[sensor_type]:
        sensor_groups[sensor_type][sensor_id] = []
    sensor_groups[sensor_type][sensor_id].append(cfile)

# ------------------ Plotting ------------------
for sensor_type, id_group in sensor_groups.items():
    plt.figure(figsize=(8,6))
    
    # Only difference: maroon and purple colors for curves & matching dashed lines
# Only difference: maroon and bluer purple colors for curves & matching dashed lines
    curve_colors = ["#384860", "#298c8c93"]

    for i, (sensor_id, file_list) in enumerate(id_group.items()):
        color = curve_colors[i % len(curve_colors)]

        try:
            df_avg = fun.average_IV_csvs(file_list, 
                                         voltage_col=options.voltageColumn, 
                                         current_col=options.currentColumn, 
                                         rowName=options.rowName, 
                                         separator=options.separatorString)

            breakdownVol = fun.breakdownVol(df_avg, options.voltageColumn, options.currentColumn)

        except Exception as e:
            print(f"Error averaging or calculating Vbd for {sensor_type}_{sensor_id}: {e}")
            df_avg = fun.storedataIV(file_list[0], options.rowName, separator=options.separatorString)
            df_avg = fun.cleanupIV(df_avg, options.voltageColumn, options.currentColumn)
            breakdownVol = fun.breakdownVol(df_avg, options.voltageColumn, options.currentColumn)

        plt.plot(df_avg[options.voltageColumn], df_avg[options.currentColumn], 
                 color=color, label=f"{sensor_type}_{sensor_id}")
        
        plt.axvline(breakdownVol, linestyle='--', color=color, alpha=0.7,
                    label=f"{sensor_type}_{sensor_id} Vbd: {round(breakdownVol,1)} V")

    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [A]")
    plt.yscale("log")
    plt.title(f"IV Curves for Sensor Type {sensor_type}")
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()

    outfileName = os.path.join(options.outdir, f"{sensor_type}_IV.png")
    plt.savefig(outfileName)
    plt.close()

# ------------------ Save Excel ------------------
if options.saveExcel:
    xlrow = options.xlrow
    xlpath = options.xlpath
    fun.tablebreakdownVol(xlpath, breakdownVol ,'C', 'Breakdown Voltage')
