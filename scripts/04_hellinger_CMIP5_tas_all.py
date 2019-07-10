from os import listdir
from tqdm import tqdm
from toe_tools.paths import *
from toe_tools.toe_calc import *

# #### PARAMETERS
_aggregate = agg_fun(ESD_var_name_T)

for season in ['annual', 'summer', 'winter']:

    print(str(XSPLIT_year) + '   ' + str(WW))
    # monthly

    files = [f for f in listdir(ESD_TXT) if
             re.match(r'^%s_ESD_[0-9]{3}_1861-2100_Siberia_df_monthly' % ESD_var_name_T, f)]
    files.sort()
    OUT_overlap_pd_list = []
    OUT_overlap_sign_list = []
    for f0 in tqdm(files):
        file_name = ESD_TXT + f0
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

        name_pickle = ESD_HD_tas + f_base + ".pickle"
        name_pickle_sign = ESD_HD_tas + f_base + "_sign.pickle"
        OUT_overlap_pd.to_pickle(name_pickle, protocol=2)
        OUT_overlap_sign.to_pickle(name_pickle_sign, protocol=2)
        name_csv = ESD_HD_tas + f_base + ".csv"
        name_csv_sign = ESD_HD_tas + f_base + "_sign.csv"
        OUT_overlap_pd.to_csv(name_csv)
        OUT_overlap_sign.to_csv(name_csv_sign)

        OUT_overlap_pd_list.append(OUT_overlap_pd)
        OUT_overlap_sign_list.append(OUT_overlap_sign)

    # ENSEMBLE MEAN

    # file_ens = PATH_ESD + "%s_ESD_mean_1861-2100_Siberia_df_monthly-adjusted.csv" % var_name
    file_ens = ESD_TXT + "%s_ESD_mean_1861-2100_Siberia_df_monthly-orig.csv" % ESD_var_name_T
    ens_pd = pd_hydroYear(file_name=file_ens, season=season)
    ens_overlap, ens_overlap_sign = calc_overlap(pd_dataframe=ens_pd, ww=WW, precision=precision, time_start=XMIN,
                                                 time_split=XSPLIT, time_end=XMAX)

    file_ens_base = file_ens.split('/')[-1].split('.')[0]
    file_ens_base = re.sub('monthly', season, file_ens_base)

    name_pickle = ESD_HD_tas + file_ens_base + ".pickle"
    name_pickle_sign = ESD_HD_tas + file_ens_base + "_sign.pickle"
    ens_overlap.to_pickle(name_pickle, protocol=2)
    ens_overlap_sign.to_pickle(name_pickle_sign, protocol=2)
    name_csv = ESD_HD_tas + file_ens_base + ".csv"
    name_csv_sign = ESD_HD_tas + file_ens_base + "_sign.csv"
    ens_overlap.to_csv(name_csv)
    ens_overlap_sign.to_csv(name_csv_sign)

    # HELLINGER MEAN
    # OUT_overlap_pd_list
    # OUT_overlap_sign_list

    df_conc = pd.concat([OUT_overlap_pd_list_i for OUT_overlap_pd_list_i in OUT_overlap_pd_list])
    df_conc_sign = pd.concat([OUT_overlap_pd_list_i for OUT_overlap_pd_list_i in OUT_overlap_sign_list])
    # like this every column key (=pixel) is there 64 times
    # go through column keys and average per column key

    df_mean_mon = df_conc.groupby(df_conc.index).mean()
    # df_mean_mon.shape
    df_sign_mean_mon = df_conc_sign.groupby(df_conc_sign.index).mean()
    # df_mean_mon.shape

    # filename for the first would have been better as ensemble mean,
    # and for the second simple as mean - but that's the way it is now
    file_ens_base = re.sub('ESD_mean', 'ESD_average', file_ens_base)

    name_pickle = ESD_HD_tas + file_ens_base + ".pickle"
    name_pickle_sign = ESD_HD_tas + file_ens_base + "_sign.pickle"
    df_mean_mon.to_pickle(name_pickle, protocol=2)
    df_sign_mean_mon.to_pickle(name_pickle_sign, protocol=2)
    name_csv = ESD_HD_tas + file_ens_base + ".csv"
    name_csv_sign = ESD_HD_tas + file_ens_base + "_sign.csv"
    df_mean_mon.to_csv(name_csv)
    df_sign_mean_mon.to_csv(name_csv_sign)
