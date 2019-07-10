from toe_tools.ncdf2pd import *
from toe_tools.paths import *

var_name = 'pr'
lon_name = 'lon'
lat_name = 'lat'
time_name = 'time'
_aggregate = agg_fun(var_name)

# check the paths.py file to make a subset !
# if no subset is wanted, leave the subset option in the ncdf2pd.nc_read_cruncep function blank
subset = [ymax, xmin, ymin, xmax]

files = [f for f in listdir(ESD_NCDF) if re.match(r'^%s+.*\.nc$' % var_name, f)]

##############################################################################################
# order
files.sort()

# get factor to transform precipitation into mm/day
f0 = files[0]
d0 = nc_read_cmip5_ESD(nc_file=ESD_NCDF + f0, var_name=var_name, subset=subset, header_str_len=6)

dim = d0.index.to_series().dt.daysinmonth
d0_fac = 60. * 60. * 24. * dim
# d0 = d0.multiply(d0_fac, axis='index', level=None, fill_value=None)

adj_models_list = dict()
cnt = 0
for file_i in files:
    print(file_i)
    di = nc_read_cmip5_ESD(nc_file=ESD_NCDF + file_i, var_name=var_name, subset=subset, header_str_len=6,
                           fixed_start='1861-01-01')
    di = di.multiply(d0_fac, axis='index', level=None, fill_value=None)
    dfi_mon_pd = di.resample('MS').apply(_aggregate)
    dfi_ann_pd = di.resample('A').apply(_aggregate)
    adj_models_list[cnt] = dfi_mon_pd

    model_id = file_i.split('_')[-1].split('.')[0]
    tx_min = dfi_ann_pd.index.min().year
    tx_max = dfi_ann_pd.index.max().year

    dfi_ann_pd.to_csv(ESD_TXT + "%s_ESD_%s_%s-%s_Siberia_df_annual-orig.csv" % (var_name, model_id, tx_min, tx_max))
    dfi_mon_pd.to_csv(ESD_TXT + "%s_ESD_%s_%s-%s_Siberia_df_monthly-orig.csv" % (var_name, model_id, tx_min, tx_max))
    cnt += 1

# ensemble mean
df_conc2 = pd.concat([adj_models_list[key] for key in adj_models_list.keys()])
df_mean_mon = df_conc2.groupby(df_conc2.index).mean()
tx_min = df_mean_mon.index.min().year
tx_max = df_mean_mon.index.max().year
df_mean_mon.to_csv(ESD_TXT + "%s_ESD_mean_%s-%s_Siberia_df_monthly-orig.csv" % (var_name, tx_min, tx_max))
