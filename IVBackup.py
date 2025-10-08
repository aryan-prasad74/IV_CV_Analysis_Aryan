# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:11:15 2019

Module for plotting IV curves and determining the breakdown voltage.
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

# --- Optional Parsing ---
parser = OptionParser()
parser.add_option("--fileName", type="string", help="Path of file with IV data", dest="fileName", default=None)
parser.add_option("--fileList", type="string", help="Path of file with list of files/names", dest="fileList", default=None)
parser.add_option("--outfile", type="string", help="Path to store IV plot", default=".png")
parser.add_option("--outdir", type="string", help="Directory where plots should be stored", default="IVplots")
parser.add_option("--xlrow", type="int", help="Excel row to store results", dest="xlrow", default=2)
parser.add_option("--xlpath", type="string", help="Excel path to store results", dest="xlpath", default='Testdata/Breakdown.xlsx')
parser.add_option("--saveExcel", type="int", help="1 to save to excel, 0 otherwise", default=0)
parser.add_option("--rowName", type="string", help="Unique string identifying row before data", default="BEGIN")
parser.add_option("--voltageColumn", type="int", help="Column index for voltage", default=0)
parser.add_option("--currentColumn", type="int", help="Column index for current", default=2)
parser.add_option("--separatorString", type="string", help="String separating columns", default="\t")

options, arguments = parser.parse_args()
fun.setPlotStyle()

# --- Load files ---
files = []
fileTitles = []

if options.fileName:
    files = [options.fileName]
    fileTitles = [os.path.splitext(os.path.basename(options.fileName))[0]]
elif options.fileList:
    dfFiles = pd.read_csv(options.fileList, sep=" ", header=None)
    files = ["Measurement_CSVs/" + f for f in dfFiles[0]]
    fileTitles = dfFiles[1]
else:
    print("No input files")
    exit()

# --- Make output directory ---
os.makedirs(options.outdir, exist_ok=True)

# --- Helper functions ---
def get_sensor_info(filename):
    m = re.search(r'(W\d{4})_(\d{2}-\d{2})', filename)
    if m:
        return m.group(1), m.group(2)
    return "Unknown", "Unknown"

# --- Group files by sensor type and ID ---
sensor_groups = {}
for cfile, ctitle in zip(files, fileTitles):
    sensor_type, sensor_id = get_sensor_info(cfile)
    sensor_groups.setdefault(sensor_type, {}).setdefault(sensor_id, []).append(cfile)

# --- Plotting ---
for sensor_type, id_group in sensor_groups.items():
    plt.figure(figsize=(8,6))

    for sensor_id, file_list in id_group.items():
        try:
            df_avg = fun.average_IV_csvs(file_list,
                                         voltage_col=options.voltageColumn,
                                         current_col=options.currentColumn,
                                         rowName=options.rowName,
                                         separator=options.separatorString)
            breakdownVol = fun.breakdownVol(df_avg, options.voltageColumn, options.currentColumn)
        except Exception as e:
            print(f"Warning: Averaging failed for {sensor_type}_{sensor_id}, using first file only. Error: {e}")
            df_avg = fun.storedataIV(file_list[0], options.rowName, separator=options.separatorString)
            df_avg = fun.cleanupIV(df_avg, options.voltageColumn, options.currentColumn)
            breakdownVol = fun.breakdownVol(df_avg, options.voltageColumn, options.currentColumn)

        # Plot averaged/first curve
        plt.plot(df_avg[options.voltageColumn], df_avg[options.currentColumn],
                 label=f"{sensor_type}_{sensor_id}")

        # Plot breakdown voltage
        plt.axvline(breakdownVol, linestyle='--', color='k', alpha=0.5,
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

# --- Save to Excel ---
if options.saveExcel:
    fun.tablebreakdownVol(options.xlpath, breakdownVol, 'C', 'Breakdown Voltage')
