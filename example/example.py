import matplotlib.pyplot as plt

from toe_tools.toe_calc import *

# #### PARAMETERS
season = 'annual'
WW = 21  # the standard moving window width used
precision = 200  # the standard number of points to evaluate kde-PDF
XSPLIT_year = 1921  # the standard year to split the time series
XSPLIT = str(XSPLIT_year) + '-01-01'  # in a YYYY-MM-DD format for pandas
XMIN = '1901-01-01'  # the standard to start the analysis
XMAX = '2100-12-01'  # the standard to stop the time series

# an example with only one CMIP5 model simulation for 4 pixels
file_name = 'example/tas_ESD_002_1861-2100_Siberia_df_monthly-orig.csv'

# make seasonal file, i.e. either mean annual, mean winter, or mean summer temperature
ds_ref = pd_hydroYear(file_name=file_name, season=season)

################################################
# calc overlap

OUT_overlap_pd, OUT_overlap_sign = calc_overlap(pd_dataframe=ds_ref, ww=WW, precision=precision, time_start=XMIN,
                                                time_split=XSPLIT, time_end=XMAX)

# plot
fig, axes = plt.subplots(2, 1)

# plot the HD
OUT_overlap_pd.plot(ax=axes[0])
OUT_overlap_pd.mean(axis=1).plot(ax=axes[0], color='black', linewidth=2, linestyle='--')

# plot the HD sign
OUT_overlap_sign.plot(ax=axes[1])
OUT_overlap_sign.mean(axis=1).plot(ax=axes[1], color='black', linewidth=2, linestyle='--')

################################################
# write output
name_csv = 'example/tas_ESD_hellinger_002_1861-2100_Siberia_df_monthly-orig.csv'
name_csv_sign = 'example/tas_ESD_sign_002_1861-2100_Siberia_df_monthly-orig.csv'
OUT_overlap_pd.to_csv(name_csv)
OUT_overlap_sign.to_csv(name_csv_sign)
