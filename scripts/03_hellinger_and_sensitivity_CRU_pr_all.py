# Sensitivity analysis of CRUNCEP regarding window widths and time series split point
from tqdm import tqdm
from toe_tools.paths import *
from toe_tools.toe_calc import *

# #### PARAMETERS
_aggregate = agg_fun(CRU_var_name_P)

# END PARAMETERS _________________________________
for season in ['annual', 'summer', 'winter']:
    pd_df = pd_hydroYear(CRU_TXT + "%s_CRU-NCEP_1901-2016_Siberia_df_monthly-orig.csv" % CRU_var_name_P, season=season,
                         aggregate=_aggregate)

    RowCols = pd_df.keys()
    OUT_overlap_pd_list = []
    OUT_overlap_sign_list = []
    # LOOP1
    # changing the split year
    for XSPLIT_year in tqdm(np.arange(1915, 1931, 2)):
        # changing the window width
        for WW in np.arange(15, 31, 2):
            print(str(XSPLIT_year) + '   ' + str(WW))
            XSPLIT = str(XSPLIT_year) + '-01-01'

            OUT_overlap_pd, OUT_overlap_pd_sign = calc_overlap(pd_dataframe=pd_df, ww=WW, precision=precision,
                                                               time_start=XMIN, time_split=XSPLIT, time_end=XMAX)

            # write output as pickle (binary) and csv
            name_pickle = CRU_HD_pr + "%s_CRU-NCEP_1901-2016_Siberia_hellinger_%s_split%s_WW%s_.pickle" \
                          % (CRU_var_name_P, season, XSPLIT_year, WW)
            name_pickle_sign = CRU_HD_pr + "%s_CRU-NCEP_1901-2016_Siberia_hellinger_%s_split%s_WW%s__sign.pickle" \
                               % (CRU_var_name_P, season, XSPLIT_year, WW)
            OUT_overlap_pd.to_pickle(name_pickle, protocol=2)
            OUT_overlap_pd_sign.to_pickle(name_pickle_sign, protocol=2)
            name_csv = CRU_HD_pr + "%s_CRU-NCEP_1901-2016_Siberia_hellinger_%s_split%s_WW%s_.csv" \
                       % (CRU_var_name_P, season, XSPLIT_year, WW)
            name_csv_sign = CRU_HD_pr + "%s_CRU-NCEP_1901-2016_Siberia_hellinger_%s_split%s_WW%s__sign.csv" \
                            % (CRU_var_name_P, season, XSPLIT_year, WW)
            OUT_overlap_pd.to_csv(name_csv)
            OUT_overlap_pd_sign.to_csv(name_csv_sign)

            OUT_overlap_pd_list.append(OUT_overlap_pd)
            OUT_overlap_sign_list.append(OUT_overlap_pd_sign)

    # the first try was with window width 20 and split year 1921:
    XSPLIT_year = 1921
    WW = 21

    print(str(XSPLIT_year) + '   ' + str(WW))
    XSPLIT = str(XSPLIT_year) + '-01-01'

    OUT_overlap_pd, OUT_overlap_pd_sign = calc_overlap(pd_dataframe=pd_df, ww=WW, precision=precision, time_start=XMIN,
                                                       time_split=XSPLIT, time_end=XMAX)

    # write output as pickle (binary) and csv
    name_pickle = CRU_HD_pr + "%s_CRU-NCEP_1901-2016_Siberia_hellinger_%s_split%s_WW%s_.pickle" % (
        CRU_var_name_P, season, XSPLIT_year, WW)
    name_pickle_sign = CRU_HD_pr + "%s_CRU-NCEP_1901-2016_Siberia_hellinger_%s_split%s_WW%s__sign.pickle" % (
        CRU_var_name_P, season, XSPLIT_year, WW)
    OUT_overlap_pd.to_pickle(name_pickle, protocol=2)
    OUT_overlap_pd_sign.to_pickle(name_pickle_sign, protocol=2)
    name_csv = CRU_HD_pr + "%s_CRU-NCEP_1901-2016_Siberia_hellinger_%s_split%s_WW%s_.csv" % (
        CRU_var_name_P, season, XSPLIT_year, WW)
    name_csv_sign = CRU_HD_pr + "%s_CRU-NCEP_1901-2016_Siberia_hellinger_%s_split%s_WW%s__sign.csv" % (
        CRU_var_name_P, season, XSPLIT_year, WW)
    OUT_overlap_pd.to_csv(name_csv)
    OUT_overlap_pd_sign.to_csv(name_csv_sign)

    # HELLINGER MEAN
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
    file_ens_base = "%s_CRU-NCEP_average_1901-2016_Siberia_df_%s-orig" % (CRU_var_name_P, season)

    name_pickle = CRU_HD_pr + file_ens_base + ".pickle"
    name_pickle_sign = CRU_HD_pr + file_ens_base + "_sign.pickle"
    df_mean_mon.to_pickle(name_pickle, protocol=2)
    df_sign_mean_mon.to_pickle(name_pickle_sign, protocol=2)
    name_csv = CRU_HD_pr + file_ens_base + ".csv"
    name_csv_sign = CRU_HD_pr + file_ens_base + "_sign.csv"
    df_mean_mon.to_csv(name_csv)
    df_sign_mean_mon.to_csv(name_csv_sign)
