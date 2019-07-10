# ToEtools
# A non-parametric method to calculate the emergence of climate signals
#
## Workshop initiative by IPSL-Labex with contributions from:
 - Eric Pohl
 - Christophe Grenier
 - Marco Gaetani
 - Goulven Laruelle
 - Vincent Thieu
 - Mathieu Vrac
 - Masa Kageyama

and various other inputs through workshops

## Introduction
Tool collection to calculate the emergence of climate signals from CRU-NCEP and CMIP5 data. The base method to define
the emergence is the hellinger distance that describes the dissimilarity of two probability density functions (PDFs)
and can be understood as their geometrical overlap. A distance of 0 correponds to a full overlap (= same PDFs) and a
distance of 1 to no overlap (= fully emerged PDFs).

The workflow is divided into 7 main steps outlined below. The corresponding scripts are in the folder scripts but need
to be copied into the main directory ("ToE_tools") as they will read and write from/into the hardcoded folders "data"
and "figures"

A working example with a data sample is provided to go through the key functions of the toolbox.
The example will start at point (3) where CRUNCEP and CMIP5 model simulations are in a pandas dataframe format. See
below on how to run the example.



The 7 step workflow is:

1) extract a subset (lat/lon bounding box) of CRUNCEP (monthly) data for a single variable
2) extract a subset (lat/lon bounding box) of CMIP5 model collection (monthly) data for a single variable
3) CRU-NCEP: calculate the emergence of a signal with respect to its reference time period:
    the two meta-parameters 'window width' (ww) and 'split year' (split_year) need to be defined:
    - window width defines the period for which to calculate the PDF of the target periods
    - split_year defines the end of the reference period (the start point is fixed to 1901)
    - (a sensitivity analysis concerning ww and split_year are included and have to turned off if not needed)
4) CMIP5: as (3) but for all CMIP5 model simulations
5) Calculate the year when the signal passes and stays above an emergence level (ToE) for CMIP5 simulations and CRUNCEP
    - For CMIP5 this step allows to include the results from the next step to run the analysis with a subset of best performing models. The code section can be uncommented to do so

6) subset CMIP5 models that show the closest match in probability evolution like the probabilities derived from
    CRUNCEP. Criterions are either R2 or Nash-Sutcliffe-Efficiency.
7) Mapping of the results. ToE (year), and variability (standard deviation, value range, min, max) of derived ToE for
    individual models. Models can be subsetted otionally with the derived best perfroming models from the previous step.
    Currently hardcoded (gis.py) for the case of the Lena catchment

## ToE Metrics:
- Hellinger Distance:
    ```
        def hellinger(p, q, bins):
        """
        :param p: probabilities of target distrbution
        :param q: probabilities of reference distrbution
        :param bins: bins / x-values associated to both distributions
        :return: the hellinger distance and the sign of change (-1 or 1)
        """

        sp = np.sum(bins * p)
        sq = np.sum(bins * q)
        sign = np.sign(sp - sq)
        helld = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2
        return helld, sign
    ```

- There are already some versions included that allow for using histograms instead of calculating the KDE-PDF which might save a lot of time

## Dependencies:
Tested on both python 2.7 and 3.5. Use of 2.7 is encouraged for plotting purposes.
Install all needed python packages with Anaconda (https://www.anaconda.com/download/) in an environment:

```conda create -n ToE_tools python=2.7 gdal netcdf4 proj4 tqdm pandas matplotlib basemap basemap-data-hires pillow scipy```

## Issues
All but the map plotting functions (using Basemap) work on both python 2.7.x and 3.5.x
Basemap has some issues and some manual adjustments of the code might be needed to get it running.
In the installation path of Basemap you can find the `__init__.py`:

... /anaconda/envs/<environment-name>/lib/python2.7/site-packages/mpl_toolkits/basemap/

It needs to have all instances of `ax.get_axis_bgcolor()` replaced by `ax.get_fc()`
Furthermore, there will be a flood of warning messages coming at you.
One particular issue is the 'proj' library. An issue is fixed by setting the environmental variable `$PROJ_LIB`
within python to the shared library folder of your environment. The code for this can be found in the header of the
file 'gis.py' and should be according to:
```
env_path = '<path_to_your>/anaconda/envs/ToE_tools/share'
proj_lib = os.path.join(env_path, 'proj')
os.environ["PROJ_LIB"] = proj_lib
```
This is hopefully fixing the issues and allows to use basemap.

## How to:
A simple example in the subdirectory "example/" is provided for an example climate model simulation file in text format with 4 columns (=4 pixels)
Simply execute line by line in the script or run it as a whole. It will calculate the Hellinger distance for a given set of window widths and split year that you can change.

In order to use the full functionality of the toolbox, copy the scripts within the folder "scripts/" into the main directory and follow the numberingas explained in the 7 spte workflow.

Always take care that in the aggregation process (from monthly to annual) of intensive and extensive variables different aggreation methods are needed (sum vs. mean)

One should make sure the output is as expected!

In order to make the directory structure as easy as possible there is an extra script in the actual toolbox folder. In "toe_tools/paths.py" you will find the names of all needed folders and an example configuration. Below the list of folders you will see a commented section that you can run to create the folder structure that is defined in the file. Different folder names are certainly possible but must remain constant during the workflow.

#
Figure output path is hardcoded to be in your ```figures/``` subfolder
In any case, if you want to plot the results you will have to change several parameters in the ```gis.py``` functions.
The current plots are with some shapefiles that are located in the ```data/``` folder and should help you figure out how to adjust the scripts to your needs.
The current projection is set to geographical to allow for plotting any region of the world. 
Figures from the manuscript are plotted in Lambert-Conform-Conical projection; the plot commands to do so are in the "gis.py" file and can be re-enabled.
Some playing with Basemap might help you to adapt the functions to your need. You can do it!!!

# 
A manuscript describing the method in detail and showcasing the application is open-access available under:
xxx
