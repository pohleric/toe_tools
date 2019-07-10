from os import listdir
from tqdm import tqdm
from toe_tools.paths import *
from toe_tools.toe_calc import *

# #### PARAMETERS
_aggregate = agg_fun(ESD_var_name_T)

for season in ['summer', 'winter']:

    # monthly

    files = [f for f in listdir(ESD_TXT) if
             re.match(r'^%s_ESD_[0-9]{3}_1861-2100_Siberia_df_monthly' % ESD_var_name_T, f)]
    files.sort()

    for f0 in tqdm(files):
        file_name = ESD_TXT + f0
        ds_ref = pd_hydroYear(file_name=file_name, season=season, aggregate=_aggregate)
        # write output
        f_base = f0.split('.')[0]
        # now the data is annual/seasonal (in any case one value per year/index)
        f_base = re.sub('monthly', season, f_base)
        name_csv = ESD_TXT + f_base + ".csv"
        ds_ref.to_csv(name_csv)

    f0 = [f for f in listdir(ESD_TXT) if re.match(r'^%s_ESD_mean_1861-2100_Siberia_df_monthly' % ESD_var_name_T, f)][0]
    file_name = ESD_TXT + f0
    ds_ref = pd_hydroYear(file_name=file_name, season=season)
    # write output
    f_base = f0.split('.')[0]
    # now the data is annual/seasonal (in any case one value per year/index)
    f_base = re.sub('monthly', season, f_base)
    name_csv = ESD_TXT + f_base + ".csv"
    ds_ref.to_csv(name_csv)
