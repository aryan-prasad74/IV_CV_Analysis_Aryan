# IV/CV Analysis 
This program performs analysis on IV data and CV data. There are 6 modules for this program: 
1. **IV.py** : Plots and saves the IV curves and determines the breakdown voltage for a single datafile 
*Required Local Modules*: Functions.py, Constants.py 
2. **IVmultiple.py**: Steering the macro for saving IV curves and breakdown voltages for many datafiles. This macro calls IV.py for each input datafile. 
*Required Local Modules*: Functions.py, Constants.py, IV.py 
3. **CV.py**: Plots and saves the CV curves, Inverse Squared CV curves (InvCV), and Doping Profile vs width curves (DPW) for a single datafile 
*Required Local Modules*: Functions.py, Constants.py 
4. **CVmultiple.py**: Steering the macro for saving CV curves, Inverse Squared CV curves (InvCV), and Doping Profile vs width curves (DPW) for multiple datafiles. This macro calls CV.py for each input datafile.
*Required Local Modules*: Functions.py, Constants.py, CV.py 
5. Functions.py: Main functions library that contains all the functions used by IV.py and CV.py 
6. Constants.py: Contains values of all constants used by IV.py and CV.py (example: vacuum permittivity, charge of an electron, area of the detector etc.)
*Required Local Modules*: Constants.py 


Note: It is recommended to save all these modules in the same directory 

### Functions Library (Functions.py)
**Required python libraries**: numpy, pandas, matplotlib.pyplot, csv, scipy (stats, optimize.fsolve, interpolate.UnivariateSpline), seaborn, openpyxl

To run on lxplus, you will want to set up a virtual environment
```
virtualenv -p `which python` venv
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
```

Most of these libraries should come built-in with python 

```
pip install openpyxl
pip install seaborn
pip install scipy
pip install matplotlib
pip install pandas
```

After the virtual environment has been created, it can be reactivated in a new terminal with

```
source venv/bin/activate
```

This is the functions library that contains all the functions used by IV.py, IVmultiple.py, CV.py, and CVmultiple.py. The functions are listed in alphabetical order and descriptions of the function and all the input parameters, optional input paramaters, return parameters are provided in the module. 

Example 1: This is a function found in Functions.py to determine the breakdown voltage 

```python 
def breakdownvol(df, x, y, thresh = 0.9995):
    """
    Parameters:
        df: Pandas dataframe 
        x: String - name of x column in df (voltage)
        y: String - name of y column in df (capacitance)
    Optional Parameters: 
        thresh: Float - r squared threshold value
    Returns:
        breakdownvol: Float - Breakdown Voltage 
    Note: 
        This function determines the breakdown by calculation the point where the relationship between the first and second derivative becomes extremely linear. 
    """
    frst_der = manual_der(df[x], df[y])
    scnd_der = manual_der(df[x], frst_der)
    
    r_value = 1
    N = 2
    while round(r_value,4) >= thresh: #Threshold for linearity
        X = np.log(frst_der[-N:])
        Y = np.log(scnd_der[-N:])
        slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
        N = N+1
          
    voltage = df[x].as_matrix()
    breakdownvol = voltage[-N+1]
    return breakdownvol
```

As seen by the comments in the function, this function takes a dataframe, and two strings (x and y column names) as inputs and returns the breakdown voltage as a float. It has an optional parameter that can be left to its default value of 0.9995 or changed by passing an input. For example, if we want to change the threshold voltage to 0.8, the function can be called in the following way:
```python 
breakdownvol(df, "Sweep Voltage", "pad", thresh = 0.8 )
```

### Constants library (Constants.py)

**Constants listed in this module**: vacuum permittivity (eps0), silicon permittivity factor (eps_si), electric charge of an electron (q), several scaling parameters, area of detector 


### IV Analysis 
**Required python libraries**: optparse, numpy, pandas,  seaborn, matplotlib.pyplot 

#### IV.py
This module performs IV analysis for a single datafile. It plots the IV curves, determines the breakdown voltage and has the capability of storing the breakdown voltage in an excel file. 

For simplicity, an example file has been included in example.iv.
As can be seen in this file, there are three columns of data: bias [V], Total Current [A], and Pad Current [A],
and the data starts the line after the line starting with BEGIN, and end the line before END.


```text 
bias [V]  Total Current [A] Pad Current [A]^M
BEGIN^M
-0.000000E+0  2.470264E-11  -5.962937E-13^M
END
```


The data points of interest are the voltage in the 0th column, and the pad current in the 2nd column.
While it might be hard to tell in the file, the data are separated by tabs (\t).


This is all the information we need to run the script, which can be run using


```
python IV.py --fileName example.iv --outdir ivPlots  --outfile _IV.png --rowName BEGIN --voltageColumn 0 --currentColumn 2 --separatorString \\t
```


Alternatively, you can feed a file of file names and titles, with an example in ivFileList.txt.

```
python IV.py --fileList ivFileList.txt  --outdir ivPlots  --outfile _IV.png --rowName BEGIN --voltageColumn 0 --currentColumn 2 --separatorString \\t
```

Currently, the excel functionality is broken, but I will fix this soon.


### CV Analysis
**Required python libraries**: optparse, numpy, pandas,  seaborn, matplotlib.pyplot , astropy.modeling

#### CV.py

This module performs CV analysis for a single datafile. It plots the CV curves, plots inverse capacitance vs voltage, and plots the Doping profile vs width (determined using the CV data). All these plots are saved as .png images. The program also determines the following parameters:
1. *Depletion Voltage of the Gain Layer*: Determined by fitting two lines and determining the point of intersection for the invCV graph 
2. *Total Depletion Voltage* :  Determined by fitting two lines and determining the point of intersection for the invCV graph
3. *Gain Doping Peak*: Determining by finding the maximum value in a segment of the data (segment is arbitrary selected)
4. *Width(FWHM) of the gain peak*: Determined by fitting a gaussian on the doping profile and finding the full width half maximum.  

All of these parameters can be stored in an excel file using the function tablebreakdownvol (described in Functions.py). At the moment, the first three parameters are being saved in the excel file. 

For simplicity, an example file has been included in example.cv.
As can be seen in this file, there are five columns of data: V_detector [V],  Capacitance [F], Conductivity [S], Bias [V],  Current Power Supply [A],
and the data starts the line after the line starting with BEGIN, and end the line before END

```text 
V_detector [V]  Capacitance [F] Conductivity [S]  Bias [V]  Current Power Supply [A]^M
BEGIN^M
-0.000000E+0  2.448583E-10  6.843721E-5 -0.000000E+0  1.616175E-6^M
END
```

The data points of interest are the voltage in the 0th column, and the capacitance in the 1st column.
While it might be hard to tell in the file, the data are separated by tabs (\t).


```
python CV.py --fileName example.cv --outdir cvPlots --rowName BEGIN --voltageColumn 0 --capacitanceColumn 1 --separatorString \\t
python CV.py --fileList cvFileList.txt --outdir cvPlots --rowName BEGIN --voltageColumn 0 --capacitanceColumn 1 --separatorString \\t
```






