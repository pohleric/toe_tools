"""
Calculate ToE (year) based on a defined confidence level (CI; e.g. 95%) and obtain the value (>= CI) to also have
the value if CI criterium is not matched
"""
from toe_tools.paths import *
from toe_tools.toe_calc import *
from os import listdir

# Parameters
set_noConverge_to_NaN = True

years = np.arange(1922, 2005, 1)
for season in ['annual', 'summer', 'winter']:
    for confidence_level in confidence_levels:
        ###################################################################
        # Reference with WW=21 and Split_Year = 1921
        ref_pd = pd.read_pickle(str(CRU_HD_pr) + "%s_CRU-NCEP_1901-2016_Siberia_hellinger_%s_split%s_WW%s_.pickle" % (
            CRU_var_name_P, season, XSPLIT_year, WW))
        ref_pd_sign = pd.read_pickle(
            str(CRU_HD_pr) + "%s_CRU-NCEP_1901-2016_Siberia_hellinger_%s_split%s_WW%s__sign.pickle" % (
                CRU_var_name_P, season, XSPLIT_year, WW))
        datetime_reference = ref_pd.index

        ###################################################################
        # get baseline statistics and indices
        RowCols = ref_pd.keys()

        val_at_years = pd.DataFrame(columns=years)
        sign_at_years = pd.DataFrame(columns=years)
        for year_i in years:
            val_at_year, sign_at_year = find_toe_at_year(ref_pd, ref_pd_sign, year_i)
            val_at_years[year_i] = val_at_year
            sign_at_years[year_i] = sign_at_year

        name_csv = CRU_ToE + "%s_CRU-NCEP_ToE_Sensitivity-valatyears_REF_1901-2016_Siberia_df_%s_overlap_.csv" % (
            CRU_var_name_P, season)
        name_csv_sign = CRU_ToE + "%s_CRU-NCEP_ToE_Sensitivity-signatyears_REF_1901-2016_Siberia_df_%s_overlap_.csv" % (
            CRU_var_name_P, season)
        val_at_years.to_csv(name_csv)
        sign_at_years.to_csv(name_csv_sign)

        val_1st_bool, lastyear = find_timemax(ref_pd, confidence_level)

        src_timemax = get_timemax(val_1st_bool, ref_pd)
        src_valmax, src_valmax_sign = get_valmax(df_timemax=src_timemax, df_raw=ref_pd, bool_mat=val_1st_bool,
                                                 df_raw_sign=ref_pd_sign)

        name_csv = CRU_ToE + "%s_CRU-NCEP_ToE_Sensitivity-valmax_%s_REF_1901-2016_Siberia_df_%s_overlap_.csv" % (
            CRU_var_name_P, confidence_level, season)
        name_csv_sign = CRU_ToE + "%s_CRU-NCEP_ToE_Sensitivity-valmax_%s_REF_1901-2016_Siberia_df_%s_overlap_sign.csv" % (
            CRU_var_name_P, confidence_level, season)
        src_valmax.to_csv(name_csv)
        src_valmax_sign.to_csv(name_csv_sign)
        name_csv = CRU_ToE + "%s_CRU-NCEP_ToE_Sensitivity-timemax_%s_REF_1901-2016_Siberia_df_%s_overlap_.csv" % (
            CRU_var_name_P, confidence_level, season)
        src_timemax.to_csv(name_csv)

        ########################
        # the individual models
        # Tair_CRU-NCEP_1901-2016_Siberia_hellinger_annual_split1920_WW29_
        files = [f for f in listdir(CRU_HD_pr) if
                 re.match(r'^%s_CRU-NCEP_.+_Siberia_hellinger_%s.+split.+_.pickle$' % (CRU_var_name_P, season), f)]
        files.sort()

        model_str = [x.split('split')[1] for x in files]
        model_years = [x.split('_')[0] for x in model_str]
        model_ww = [x.split('_')[1] for x in model_str]
        model_ww = [re.sub('WW', '', x) for x in model_ww]
        model_ids = [(x + '-' + y) for x, y in zip(model_years, model_ww)]
        col_names = model_ids[:]

        # write output into this
        sensi_timemax_pd = pd.DataFrame(index=src_timemax.index, columns=col_names)
        sensi_valmax_pd = pd.DataFrame(index=src_valmax.index, columns=col_names)
        sensi_valmax_pd_sign = pd.DataFrame(index=src_valmax.index, columns=col_names)
        # sens_df.shape

        cnt_mod_i = 0
        for model_i in files:
            print(model_i)
            name_pickle = CRU_HD_pr + model_i
            name_pickle_sign = re.sub('_.pickle', '__sign.pickle', name_pickle)

            # read
            tmp_pd = pd.read_pickle(name_pickle)
            tmp_pd_sign = pd.read_pickle(name_pickle_sign)

            # val_1st_bool = find_timemax(ref_pd)
            tmp_val_1st_bool, lastyear = find_timemax(tmp_pd, confidence_level)

            targ_timemax = get_timemax(tmp_val_1st_bool, tmp_pd)

            # or alternatively get really the last value
            targ_valmax = tmp_pd.loc[lastyear]

            # instead get the last sign value of the time series
            targ_valmax_sign = tmp_pd_sign.loc[lastyear]

            # some models do not converge -> no ToE -> set NaN
            if set_noConverge_to_NaN:
                t, v = nan_no_converge(targ_timemax, targ_valmax, confidence_level)

            tmp_col = model_ids[cnt_mod_i]
            sensi_timemax_pd[tmp_col] = targ_timemax
            sensi_valmax_pd[tmp_col] = targ_valmax
            sensi_valmax_pd_sign[tmp_col] = targ_valmax_sign

            cnt_mod_i += 1

        name_pickle = CRU_ToE + "%s_CRU-NCEP_ToE_Sensitivity-timemax_%s_1901-2016_Siberia_df_%s_overlap_pProt2.pickle" % (
            CRU_var_name_P, confidence_level, season)
        sensi_timemax_pd.to_pickle(name_pickle, protocol=2)
        name_csv = CRU_ToE + "%s_CRU-NCEP_ToE_Sensitivity-timemax_%s_1901-2016_Siberia_df_%s_overlap_.csv" % (
            CRU_var_name_P, confidence_level, season)
        sensi_timemax_pd.to_csv(name_csv)

        name_pickle = CRU_ToE + "%s_CRU-NCEP_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_%s_overlap_pProt2.pickle" % (
            CRU_var_name_P, confidence_level, season)
        name_pickle_sign = CRU_ToE + "%s_CRU-NCEP_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_%s_overlap_sign_pProt2.pickle" % (
            CRU_var_name_P, confidence_level, season)
        sensi_valmax_pd.to_pickle(name_pickle, protocol=2)
        sensi_valmax_pd_sign.to_pickle(name_pickle_sign, protocol=2)
        name_csv = CRU_ToE + "%s_CRU-NCEP_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_%s_overlap_.csv" % (
            CRU_var_name_P, confidence_level, season)
        name_csv_sign = CRU_ToE + "%s_CRU-NCEP_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_%s_overlap_sign.csv" % (
            CRU_var_name_P, confidence_level, season)
        sensi_valmax_pd.to_csv(name_csv)
        sensi_valmax_pd_sign.to_csv(name_csv_sign)
