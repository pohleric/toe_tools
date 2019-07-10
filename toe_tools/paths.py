# This file contains the definitions for PATHS and parameters

# PATHS definitions for the different data sources #
# the toolbox uses:
#  one set of CMIP5 model simulations (=ESD)
#  one database of CRUNCEP v7 data (=CRUNCEP)
# The PATHS need to be defined and it has to be made sure the required sub-folders exist #
CRU_main = "/Users/pohl3/tmp/CRUNCEP/"
CRU_NCDF = CRU_main + 'v7/twodeg/'
CRU_TXT = CRU_main + 'txt_input/'
CRU_HD_pr = CRU_main + 'HD_and_sens_pr/fullSeries/'
CRU_HD_Tair = CRU_main + 'HD_and_sens_Tair/fullSeries/'
CRU_ToE = CRU_main + 'ToE_value_time/'
# # - you can run the following 3 lines of code to generate the folder structure - #
# import os
# CRU_dirs = [CRU_main, CRU_NCDF, CRU_TXT, CRU_HD_pr, CRU_HD_Tair, CRU_ToE]
# [os.makedirs(dir_i) for dir_i in CRU_dirs if not os.path.exists(dir_i)]


ESD_main = "/Users/pohl3/tmp/ESD/"
ESD_NCDF = ESD_main + 'ncdf/'
ESD_TXT = ESD_main + 'txt_input/'
ESD_HD_pr = ESD_main + 'HD_and_sens_pr/fullSeries/'
ESD_HD_tas = ESD_main + 'HD_and_sens_tas/fullSeries/'
ESD_ToE = ESD_main + 'ToE_value_time/'
# # - you can run the following 3 lines of code to generate the folder structure - #
# import os
# ESD_dirs = [ESD_main, ESD_NCDF, ESD_TXT, ESD_HD_pr, ESD_HD_tas, ESD_ToE]
# [os.makedirs(dir_i) for dir_i in ESD_dirs if not os.path.exists(dir_i)]

# other paramters
WW = 21  # the standard moving window width used
precision = 200  # the standard number of points to evaluate kde-PDF
XSPLIT_year = 1921  # the standard year to split the time series
XSPLIT = str(XSPLIT_year) + '-01-01'  # in a YYYY-MM-DD format for pandas
XMIN_year = 1901  # the standard year to start the analysis
XMIN = str(XMIN_year) + '-01-01'  # the standard to start the analysis
XMAX_year = 2100  # the standard year to stop the time series
XMAX = str(XMAX_year) + '-12-01'  # the standard to stop the time series

CRU_var_name_P = 'pr'
CRU_var_name_T = 'Tair'
CRU_var_name_Snow = 'Snowf'
CRU_var_name_Rain = 'Rainf'
ESD_var_name_P = 'pr'
ESD_var_name_T = 'tas'

# the bounding box to extract the data
xmin = 102.0
xmax = 142.0
ymax = 74.0
ymin = 52.0
# if no subset is wanted, leave the subset option in the ncdf2pd.nc_read_cruncep function blank
subset = [ymax, xmin, ymin, xmax]

# the number of models to be used in the sub-selection procedure
n_models = 10

# the confidence or emergence levels for which to identify the ToE
confidence_levels = [0.3, 0.4, 0.5]


# the aggregate function to make seasonal and annual files for precipitation or temperature
def agg_fun(str):
    if str == 'pr':
        retfun = 'sum'
    elif str == 'tas':
        retfun = 'mean'
    elif str == 'Tair':
        retfun = 'mean'
    elif str == 'Snowf':
        retfun = 'sum'
    elif str == 'Rainf':
        retfun = 'sum'
    else:
        print('unknow variable or unknown way how to aggregate values! - Add definition in "paths.py" please ...')
        return False
    print('aggregate function = %s' % retfun)
    return retfun


# plotting parameters
plot_var = 'hellinger'
cex_axis_text = 12
cex_main_text = 10
label_distance = 15
bm10 = '10best'  # variable short for the data source identification text within the figure
