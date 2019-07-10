# sensitivity analysis of CMIP5 with respect to window width and split year.
# note hat this takes significant amounts of time to run
# as the script will test 64 combinations of meta-parameters for each of the 65 models (64*65 = 4225)

from os import listdir
from tqdm import tqdm
from toe_tools.paths import *
from toe_tools.toe_calc import *

# #### PARAMETERS
PATH_ESD = ESD_TXT
PATH_SENS_OUT = ESD_HD_tas
season = 'winter'  # not looped because one full analysis run already takes up to some days

# #### CMIP5 datasets // subset temporal to the same extent as CRUNCEP
# monthly

files = [f for f in listdir(PATH_ESD) if re.match(r'^%s_ESD_[0-9]{3}_1861-2100_Siberia_df_monthly' % ESD_var_name_T, f)]
files.sort()

# LOOP1
for XSPLIT_year in tqdm(np.arange(1915, 1929.1, 2).astype('int')):
    # changing the window width
    for WW in np.arange(17, 29.1, 2).astype('int'):
        # for WW in [15, 29]:

        print(str(XSPLIT_year) + '   ' + str(WW))
        XSPLIT = str(XSPLIT_year) + '-01-01'

        for f0 in tqdm(files):
            file_name = PATH_ESD + f0
            ds_ref = pd_hydroYear(file_name=file_name, season=season)

            ################################################
            # calc overlap

            OUT_overlap_pd, OUT_overlap_sign = calc_overlap(pd_dataframe=ds_ref, ww=WW, precision=precision,
                                                            time_start=XMIN,
                                                            time_split=XSPLIT, time_end=XMAX)

            ################################################
            # write output
            f_base = f0.split('.')[0]
            # now the data is annual/seasonal (in any case one value per year/index)
            f_base = re.sub('monthly', season, f_base)

            name_pickle = PATH_SENS_OUT + f_base + "_" + str(XSPLIT_year) + "_" + str(WW) + "_.pickle"
            name_pickle_sign = PATH_SENS_OUT + f_base + "_" + str(XSPLIT_year) + "_" + str(WW) + "_sign.pickle"

            OUT_overlap_pd.to_pickle(name_pickle, protocol=2)
            OUT_overlap_sign.to_pickle(name_pickle_sign, protocol=2)

            name_csv = PATH_SENS_OUT + f_base + "_" + str(XSPLIT_year) + "_" + str(WW) + "_.csv"
            name_csv_sign = PATH_SENS_OUT + f_base + "_" + str(XSPLIT_year) + "_" + str(WW) + "_sign.csv"
            OUT_overlap_pd.to_csv(name_csv)
            OUT_overlap_sign.to_csv(name_csv_sign)

        # ENSEMBLE MEAN

        # file_ens = PATH_ESD + "%s_ESD_mean_1861-2100_Siberia_df_monthly-adjusted.csv" % var_name
        file_ens = PATH_ESD + "%s_ESD_mean_1861-2100_Siberia_df_monthly-orig.csv" % ESD_var_name_T
        ens_pd = pd_hydroYear(file_name=file_ens)
        ens_overlap, ens_overlap_sign = calc_overlap(pd_dataframe=ens_pd, ww=WW, precision=precision, time_start=XMIN,
                                                     time_split=XSPLIT, time_end=XMAX)

        file_ens_base = file_ens.split('/')[-1].split('.')[0]
        file_ens_base = re.sub('monthly', season, file_ens_base)

        name_pickle = PATH_SENS_OUT + file_ens_base + "_" + str(XSPLIT_year) + "_" + str(WW) + "_.pickle"
        name_pickle_sign = PATH_SENS_OUT + file_ens_base + "_" + str(XSPLIT_year) + "_" + str(WW) + "_sign.pickle"

        ens_overlap.to_pickle(name_pickle, protocol=2)
        ens_overlap_sign.to_pickle(name_pickle_sign, protocol=2)

        name_csv = PATH_SENS_OUT + file_ens_base + "_" + str(XSPLIT_year) + "_" + str(WW) + "_.csv"
        name_csv_sign = PATH_SENS_OUT + file_ens_base + "_" + str(XSPLIT_year) + "_" + str(WW) + "_sign.csv"

        ens_overlap.to_csv(name_csv)
        ens_overlap_sign.to_csv(name_csv_sign)
