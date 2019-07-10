"""
Calculate ToE (year) based on a defined confidence level (CI; e.g. 95%) and obtain the value (>= CI) to also have
the value if CI criterium is not matched
"""
from os import listdir
from toe_tools.paths import *
from toe_tools.toe_calc import *

best_models = [format('%03d' % m) for m in np.loadtxt('data/tas_best_10_models_all_seasons_NSE__LongStations.txt')]
best_model = best_models[0]
bm10 = '10best'

# Parameters
set_noConverge_to_NaN = True

years = np.arange(1922, 2088, 1)
for season in ['annual', 'summer', 'winter']:
    for confidence_level in confidence_levels:
        PATH_IN = ESD_HD_tas
        PATH_OUT = ESD_ToE

        ###################################################################
        # ensemble mean
        ref_pd = pd.read_pickle(
            str(PATH_IN) + "%s_ESD_average_1861-2100_Siberia_df_%s-orig.pickle" % (ESD_var_name_T, season))
        datetime_reference = ref_pd.index
        ref_pd_sign = pd.read_pickle(
            str(PATH_IN) + "%s_ESD_average_1861-2100_Siberia_df_%s-orig_sign.pickle" % (ESD_var_name_T, season))

        ###################################################################
        # get baseline statistics and indices
        RowCols = ref_pd.keys()

        val_at_years = pd.DataFrame(columns=years)
        sign_at_years = pd.DataFrame(columns=years)

        # using all the models
        for year_i in years:
            val_at_year, sign_at_year = find_toe_at_year(ref_pd, ref_pd_sign, year_i)
            val_at_years[year_i] = val_at_year
            sign_at_years[year_i] = sign_at_year
            # y = ref_pd['0055.00103.0'].values
            # x = ref_pd['0055.00103.0'].index.values
            # plt.scatter(x, y)

        name_csv = PATH_OUT + "%s_ESD_ToE_Sensitivity-valatyears_REF_1901-2016_Siberia_df_%s_overlap_.csv" % (
            ESD_var_name_T, season)
        name_csv_sign = PATH_OUT + "%s_ESD_ToE_Sensitivity-signatyears_REF_1901-2016_Siberia_df_%s_overlap_.csv" % (
            ESD_var_name_T, season)
        val_at_years.to_csv(name_csv)
        sign_at_years.to_csv(name_csv_sign)

        # ----------------------------------------------------------------------
        # this is with the results produced from 06_select best models
        # either with the best overall model:

        # -----------
        # # the best model
        # ref_pd = pd.read_pickle(str(PATH_IN)+"%s_ESD_%s_1861-2100_Siberia_df_%s-orig.pickle" % (ESD_var_name_T, best_model,season))
        # datetime_reference = ref_pd.index
        # ref_pd_sign = pd.read_pickle(str(PATH_IN)+"%s_ESD_%s_1861-2100_Siberia_df_%s-orig_sign.pickle" % (ESD_var_name_T, best_model, season))
        # # years = [1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080]
        # val_at_years = pd.DataFrame(columns=years)
        # sign_at_years= pd.DataFrame(columns=years)
        #
        # # using all the models
        # for year_i in years:
        #     val_at_year, sign_at_year = find_toe_at_year(ref_pd, ref_pd_sign, year_i)
        #     val_at_years[year_i] = val_at_year
        #     sign_at_years[year_i] = sign_at_year
        #     # y = ref_pd['0055.00103.0'].values
        #     # x = ref_pd['0055.00103.0'].index.values
        #     # plt.scatter(x, y)
        # name_csv = PATH_OUT+"%s_ESD_ToE_Sensitivity-valatyears_%s_1901-2016_Siberia_df_%s_overlap_.csv" % (ESD_var_name_T, best_model, season)
        # name_csv_sign = PATH_OUT+"%s_ESD_ToE_Sensitivity-signatyears_%s_1901-2016_Siberia_df_%s_overlap_.csv" % (ESD_var_name_T, best_model, season)
        # val_at_years.to_csv(name_csv)
        # sign_at_years.to_csv(name_csv_sign)
        # # --------

        # or with the 10 best models:
        # --------
        cnt_model = 0
        t_pd_dict = {}

        for best_model in best_models:
            ref_pd = pd.read_pickle(
                str(PATH_IN) + "%s_ESD_%s_1861-2100_Siberia_df_%s-orig.pickle" % (ESD_var_name_T, best_model, season))
            datetime_reference = ref_pd.index
            ref_pd_sign = pd.read_pickle(
                str(PATH_IN) + "%s_ESD_%s_1861-2100_Siberia_df_%s-orig_sign.pickle" % (ESD_var_name_T, best_model, season))
            # years = [1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080]
            val_at_years = pd.DataFrame(columns=years)
            sign_at_years = pd.DataFrame(columns=years)

            # using all the models
            for year_i in years:
                val_at_year, sign_at_year = find_toe_at_year(ref_pd, ref_pd_sign, year_i)
                val_at_years[year_i] = val_at_year
                sign_at_years[year_i] = sign_at_year
                # y = ref_pd['0055.00103.0'].values
                # x = ref_pd['0055.00103.0'].index.values
                # plt.scatter(x, y)
            name_csv = PATH_OUT + "%s_ESD_ToE_Sensitivity-valatyears_%s_1901-2016_Siberia_df_%s_overlap_.csv" % (
                ESD_var_name_T, best_model, season)
            name_csv_sign = PATH_OUT + "%s_ESD_ToE_Sensitivity-signatyears_%s_1901-2016_Siberia_df_%s_overlap_.csv" % (
                ESD_var_name_T, best_model, season)
            val_at_years.to_csv(name_csv)
            sign_at_years.to_csv(name_csv_sign)
            t_pd_dict[cnt_model] = {'dat': val_at_years, 'sign': sign_at_years}
            cnt_model += 1

        t_keys = t_pd_dict.keys()
        # t2_pd=0
        t2_pd = pd.concat(t_pd_dict[t_key]['dat'] for t_key in t_keys)
        t2_pd_sign = pd.concat(t_pd_dict[t_key]['sign'] for t_key in t_keys)

        t2_pd_mean = t2_pd.groupby(t2_pd.index).mean()
        t2_pd_mean_sign = t2_pd_sign.groupby(t2_pd_sign.index).mean()
        name_csv = PATH_OUT + "%s_ESD_ToE_Sensitivity-valatyears_%s_1901-2016_Siberia_df_%s_overlap_.csv" % (
            ESD_var_name_T, bm10, season)
        name_csv_sign = PATH_OUT + "%s_ESD_ToE_Sensitivity-signatyears_%s_1901-2016_Siberia_df_%s_overlap_.csv" % (
            ESD_var_name_T, bm10, season)
        t2_pd_mean.to_csv(name_csv)
        t2_pd_mean_sign.to_csv(name_csv_sign)
        # ----------------------------------------------------------------------

        val_1st_bool, lastyear = find_timemax(ref_pd, confidence_level)

        # src_timemax = (val_1st_bool.cumsum(axis=0) == 1).idxmax()
        src_timemax = get_timemax(val_1st_bool, ref_pd)
        src_valmax, src_valmax_sign = get_valmax(df_timemax=src_timemax, df_raw=ref_pd, bool_mat=val_1st_bool,
                                                 df_raw_sign=ref_pd_sign, posneg=True)

        name_csv = PATH_OUT + "%s_ESD_ToE_Sensitivity-valmax_%s_REF_1901-2016_Siberia_df_%s_overlap_.csv" % (
            ESD_var_name_T, confidence_level, season)
        name_csv_sign = PATH_OUT + "%s_ESD_ToE_Sensitivity-valmax_%s_REF_1901-2016_Siberia_df_%s_overlap__sign.csv" % (
            ESD_var_name_T, confidence_level, season)
        src_valmax.to_csv(name_csv)
        src_valmax_sign.to_csv(name_csv_sign)
        name_csv = PATH_OUT + "%s_ESD_ToE_Sensitivity-timemax_%s_REF_1901-2016_Siberia_df_%s_overlap_.csv" % (
            ESD_var_name_T, confidence_level, season)
        src_timemax.to_csv(name_csv)

        ########################
        # the individual models
        files = [f for f in listdir(PATH_IN) if
                 re.match(r'^%s_ESD_[0-9]{3}_1861-2100_Siberia_df_%s.+orig.pickle$' % (ESD_var_name_T, season), f)]
        files.sort()

        model_ids = [x.split('_')[2] for x in files]
        col_names = model_ids[:]

        # write output into this
        sensi_timemax_pd = pd.DataFrame(index=src_timemax.index, columns=col_names)
        sensi_valmax_pd = pd.DataFrame(index=src_valmax.index, columns=col_names)
        sensi_valmax_pd_sign = pd.DataFrame(index=src_valmax.index, columns=col_names)
        # sens_df.shape

        cnt_mod_i = 0
        for model_i in files:
            print(model_i)
            # model_i = 'tas_ESD_028_1861-2100_Siberia_df_annual-orig.pickle'
            # key = '0053.00141.0'
            name_pickle = PATH_IN + model_i
            name_pickle_sign = re.sub('orig.pickle', 'orig_sign.pickle', name_pickle)

            tmp_pd = pd.read_pickle(name_pickle)
            tmp_pd_sign = pd.read_pickle(name_pickle_sign)

            # # val_1st_bool = find_timemax(ref_pd)
            # tmp_val_1st_bool, index_max = find_timemax(tmp_pd, confidence_level)
            #
            # targ_timemax = get_timemax(tmp_val_1st_bool, tmp_pd)
            # targ_valmax, targ_valmax_sign = get_valmax(df_timemax=targ_timemax, df_raw=tmp_pd, bool_mat=tmp_val_1st_bool,df_raw_sign=tmp_pd_sign)

            # val_1st_bool = find_timemax(ref_pd)
            tmp_val_1st_bool, lastyear = find_timemax(tmp_pd, confidence_level)

            targ_timemax = get_timemax(tmp_val_1st_bool, tmp_pd)
            # targ_valmax, _ = get_valmax(df_timemax=targ_timemax, df_raw=tmp_pd, bool_mat=tmp_val_1st_bool, df_raw_sign=tmp_pd_sign)
            # or alternatively get really the last value
            targ_valmax = tmp_pd.loc[lastyear]

            # instead get the last sign value of the time series
            # targ_valmax, _ = get_valmax(df_timemax=targ_timemax, df_raw=tmp_pd, bool_mat=tmp_val_1st_bool, df_raw_sign=tmp_pd_sign)
            # targ_valmax_sign = tmp_pd_sign.loc[index_max]
            targ_valmax_sign = tmp_pd_sign.loc[lastyear]

            # some models do not converge -> no ToE -> set NaN
            if set_noConverge_to_NaN:
                t, v = nan_no_converge(targ_timemax, targ_valmax, confidence_level)

            tmp_col = model_ids[cnt_mod_i]
            sensi_timemax_pd[tmp_col] = targ_timemax

            sensi_valmax_pd[tmp_col] = targ_valmax
            sensi_valmax_pd_sign[tmp_col] = targ_valmax_sign

            cnt_mod_i += 1

        # tas_ESD_ToE_Sensitivity-valmax_REF_1901-2016_Siberia_df_annual_overlap_

        name_pickle = PATH_OUT + "%s_ESD_ToE_Sensitivity-timemax_%s_1901-2016_Siberia_df_%s_overlap_pProt2.pickle" % (
            ESD_var_name_T, confidence_level, season)
        sensi_timemax_pd.to_pickle(name_pickle, protocol=2)
        name_csv = PATH_OUT + "%s_ESD_ToE_Sensitivity-timemax_%s_1901-2016_Siberia_df_%s_overlap_.csv" % (
            ESD_var_name_T, confidence_level, season)
        sensi_timemax_pd.to_csv(name_csv)

        name_pickle = PATH_OUT + "%s_ESD_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_%s_overlap_pProt2.pickle" % (
            ESD_var_name_T, confidence_level, season)
        name_pickle_sign = PATH_OUT + "%s_ESD_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_%s_overlap_sign_pProt2.pickle" % (
            ESD_var_name_T, confidence_level, season)
        sensi_valmax_pd.to_pickle(name_pickle, protocol=2)
        sensi_valmax_pd_sign.to_pickle(name_pickle_sign, protocol=2)
        name_csv = PATH_OUT + "%s_ESD_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_%s_overlap_.csv" % (
            ESD_var_name_T, confidence_level, season)
        name_csv_sign = PATH_OUT + "%s_ESD_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_%s_overlap_sign.csv" % (
            ESD_var_name_T, confidence_level, season)
        sensi_valmax_pd.to_csv(name_csv)
        sensi_valmax_pd_sign.to_csv(name_csv_sign)
