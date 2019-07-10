from toe_tools.ncdf2pd import *
from toe_tools.paths import *

var_name = 'Rainf'
lon_name = 'nav_lon'
lat_name = 'nav_lat'
time_name = 'time'
_aggregate = agg_fun(var_name)

# check the paths.py file to make a subset !
# if no subset is wanted, leave the subset option in the ncdf2pd.nc_read_cruncep function blank
subset = [ymax, xmin, ymin, xmax]

files = [f for f in listdir(CRU_NCDF) if re.match(r'[aA-zZ]+.*\.nc$', f)]

##############################################################################################
# order
files.sort()

# get reference (first year)
f0 = files[0]
d0 = nc_read_cruncep(nc_file=CRU_NCDF + f0, var_name=var_name, subset=subset, header_str_len=6)

# one mean value every x hours; the mean is in kg/m2/s
# create the factor to multiply the series with and create the sum/x hours
d0_tstep = d0.index.to_series().diff().dt.seconds.div(60 * 60, fill_value=0)[1]
d0_fac = 60. * 60. * d0_tstep
d0 = d0 * d0_fac

df_mon_pd = d0.resample('MS').apply(_aggregate)
df_ann_pd = d0.resample('A').apply(_aggregate)

for file_i in files[1:]:
    print(file_i)

    di = nc_read_cruncep(nc_file=CRU_NCDF + file_i, var_name=var_name, subset=subset, header_str_len=6)
    di = di * d0_fac
    dfi_mon_pd = di.resample('MS').apply(_aggregate)
    dfi_ann_pd = di.resample('A').apply(_aggregate)

    # concat adds the individual years into one data frame
    df_mon_pd = [df_mon_pd, dfi_mon_pd]
    df_mon_pd = pd.concat(df_mon_pd)
    df_ann_pd = [df_ann_pd, dfi_ann_pd]
    df_ann_pd = pd.concat(df_ann_pd)

tx_min = df_ann_pd.index.min().year
tx_max = df_ann_pd.index.max().year

df_mon_pd.to_csv(CRU_TXT + "%s_CRU-NCEP_%s-%s_Siberia_df_monthly-orig.csv" % (var_name, tx_min, tx_max))
df_ann_pd.to_csv(CRU_TXT + "%s_CRU-NCEP_%s-%s_Siberia_df_annual-orig.csv" % (var_name, tx_min, tx_max))
