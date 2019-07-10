# V6 is using data from ../data/ESD/txt_input/bias_corrected_annualOffset/
# these data are corrected(bias) by calculating the average annual offset between the CMIP5 models and CRUNCEP
# V5 is based on offset caluclation based on mean monthly data ... difficult to justify .. still all is
# using the ESD CMIP5 collection ........
from toe_tools.paths import *
from toe_tools.toe_calc import *

# #### PARAMETERS
var_name = 'Tair'
_aggregate = 'mean'

for season in ['summer', 'winter']:
    # monthly
    f0 = CRU_TXT + "%s_CRU-NCEP_1901-2016_Siberia_df_monthly-orig.csv" % var_name

    ds_ref = pd_hydroYear(file_name=f0, season=season, aggregate=_aggregate)

    # write output
    f_base = f0.split('.')[0]
    # now the data is annual/seasonal (in any case one value per year/index)
    f_base = re.sub('monthly', season, f_base)
    name_csv = f_base + ".csv"
    ds_ref.to_csv(name_csv)
