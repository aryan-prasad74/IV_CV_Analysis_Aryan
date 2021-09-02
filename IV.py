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
  print "No input files"

try:
  os.mkdir(options.outdir)
except:
  "Directory already exists"


for cfile, ctitle in zip(files, fileTitles):
  df = fun.storedataIV(cfile, options.rowName, separator=options.separatorString)
  df = fun.cleanupIV(df, options.voltageColumn, options.currentColumn)
  breakdownVol = fun.breakdownVol(df, options.voltageColumn, options.currentColumn)

  outfileName = options.outdir + "/" + ctitle + options.outfile
  fun.dataplot(df, options.voltageColumn, options.currentColumn, outfileName, 'ylog', 'Voltage [V]', 'Current [A]', iden = "IV", breakdownVol = breakdownVol)

#Storing data in an excel file 
if options.saveExcel:
  xlrow = options.xlrow 
  xlpath = options.xlpath
  fun.tablebreakdownVol(xlpath, breakdownVol ,'C', 'Breakdown Voltage')


