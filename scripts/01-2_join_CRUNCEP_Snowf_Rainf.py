from toe_tools.ncdf2pd import *
from toe_tools.paths import *


def read_csv(filename):
    x = pd.read_csv(filename, index_col=0)
    x.index = pd.to_datetime(x.index)
    return x


# Merge snowfall and rainfall to precipitation

var_name1 = 'Snowf'
var_name2 = 'Rainf'
var_name_out = 'pr'

# PATH_CRUNCEP = "/home/hydrogeol/epohl/data/CRU-NCEP/txt_input/"

pd_snow = read_csv(CRU_TXT + "%s_CRU-NCEP_1901-2016_Siberia_df_monthly-orig.csv" % var_name1)
pd_rain = read_csv(CRU_TXT + "%s_CRU-NCEP_1901-2016_Siberia_df_monthly-orig.csv" % var_name2)

pd_prec = pd_rain.add(pd_snow)
pd_prec.to_csv(CRU_TXT + "%s_CRU-NCEP_1901-2016_Siberia_df_monthly-orig.csv" % var_name_out)

pd_snow = read_csv(CRU_TXT + "%s_CRU-NCEP_1901-2016_Siberia_df_annual-orig.csv" % var_name1)
pd_rain = read_csv(CRU_TXT + "%s_CRU-NCEP_1901-2016_Siberia_df_annual-orig.csv" % var_name2)

pd_prec = pd_rain + pd_snow
pd_prec.to_csv(CRU_TXT + "%s_CRU-NCEP_1901-2016_Siberia_df_annual-orig.csv" % var_name_out)
