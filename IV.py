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


"""
Optional Parsing
"""

parser = OptionParser()
parser.add_option("--fileName", type="string",
                  help="Path of file",
                  dest="fileName", default="Datasets\IV\EDX30329-WNo10\IV\LG1-SE5-3.txt")
parser.add_option("--impathIV",type = "string", help = "Path to store IV plot", dest = "impathIV", default = "IV_plots/test.png")
parser.add_option("--xlrow", type = "int", help = "What row of the excel file should this data be stored in", dest = "xlrow", default = 2)
parser.add_option("--xlpath", type = "string", help = "Path of excel file to store results", dest = "xlpath", default = 'Testdata/Breakdown.xlsx')
parser.add_option("--saveExcel", type = "int", help = "Set to 1 if you want to save the output in an excel file, 0 otherwise", default = 0)
parser.add_option("--rowName", type="string", help = "A unique string that identifies the row before the data starts", default = "BEGIN")
parser.add_option("--voltageColumn", type="int", help = "The data column where the voltage is stored", default = 0)
parser.add_option("--currentColumn", type="int", help = "The data column where the current is stored", default = 2)

options, arguments = parser.parse_args()

fun.setPlotStyle() #Setting format of the graph

df = fun.storedataIV(options.fileName, options.rowName)
df = fun.cleanupIV(df, options.voltageColumn, options.currentColumn)
breakdownVol = fun.breakdownVol(df, options.voltageColumn, options.currentColumn)
fun.dataplot(df, options.voltageColumn, options.currentColumn, options.impathIV, 'ylog', 'Voltage [V]', 'Current [A]', iden = "IV", breakdownVol = breakdownVol)

#Storing data in an excel file 
if options.saveExcel:
  xlrow = options.xlrow 
  xlpath = options.xlpath
  fun.tablebreakdownVol(xlpath, breakdownVol ,'C', 'Breakdown Voltage')


"""
Determining Uncertainities - threshold 

# If i = 1, columns [0, 1] 2(1)-1
# If i = 2, columns [2, 3] 2(2) - 1
# If i = 3, columns [4, 5] 2(3) - 1
for i in range (1, 7):
    column_name = ['D', 'E', 'F', 'G', 'H', 'I']
    breakdownVol = fun.breakdownVol(df, "Sweep Voltage", "pad", thresh = 0.9995-0.005*i)
    fun.dataplot(df, 'Sweep Voltage', 'pad', options.impathIV[:-4]+str(i)+".png", 'ylog', 'Voltage [V]', 'Current [A]', iden = "IV", breakdownVol = breakdownVol)
    fun.tablebreakdownVol(xlpath, breakdownVol ,column_name[i-1], 'Breakdown Voltage')
"""
