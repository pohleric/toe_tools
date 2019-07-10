"""
Calculate corresponding temperature/precipitation change at various significance levels
"""
from os.path import isfile
from os import listdir
from toe_tools.gis import *
from toe_tools.paths import *
from toe_tools.toe_calc import *

# Parameters
set_noConverge_to_NaN = True
model_real = 'CRUNCEP'

for Varname_CRU in [CRU_var_name_T, CRU_var_name_P]:
    for season in ['annual', 'summer', 'winter']:
        for confidence_level in confidence_levels:

            # only needed for the to_geo_array function to store the data in a more convinient way
            # has no impact on other calculations :
            model = 'CMIP5'

            if Varname_CRU == CRU_var_name_T:
                PATH_IN = CRU_HD_Tair
            elif Varname_CRU == CRU_var_name_P:
                PATH_IN = CRU_HD_pr

            PATH_OUT = CRU_ToE
            PATH_CRU_raw = CRU_TXT

            # the individual models
            files = [f for f in listdir(PATH_CRU_raw) if
                     re.match(r'^%s_CRU-NCEP_1901-2016_Siberia_df_%s.+orig.csv$' % (Varname_CRU, season), f)]
            files_toe = '%s_CRU-NCEP_1901-2016_Siberia_hellinger_%s_split1921_WW21_.pickle' % (Varname_CRU, season)

            col_names = ['CRU']

            # write output into this
            ref0_pd = pd.read_pickle(str(PATH_IN) + files_toe)
            ref0_val_1st_bool, lastyear = find_timemax(ref0_pd, confidence_level)

            src_timemax = get_timemax(ref0_val_1st_bool, ref0_pd)
            val_at_conflvl_pd = pd.DataFrame(index=src_timemax.index, columns=col_names)
            range_at_conflvl_pd = pd.DataFrame(index=src_timemax.index, columns=col_names)
            std_at_conflvl_pd = pd.DataFrame(index=src_timemax.index, columns=col_names)

            filename_raw = PATH_CRU_raw + files[0]
            ds_ref = read_csv_ann(filename_raw)

            # get the mean of the reference period
            if season == 'annual':
                ds_sub_ref = ds_ref.loc[str(XMIN_year - 1) + '-12-31':str(XSPLIT_year) + '-12-31']
            else:
                ds_sub_ref = ds_ref.loc[1900:1921]
            ds_sub_mean_ref = ds_sub_ref.mean(axis=0)

            # get the mean of the ww of the year where we have ToE
            ind_at_confLvl = ref0_val_1st_bool.apply(lambda x: ref0_val_1st_bool.index[np.where(x == 1)[0]],
                                                     axis=0).values

            # test :
            # ind_at_confLvl[0,20] = lastyear
            ds_sub_mean_tar = copy(ds_sub_mean_ref)
            # ind_at_confLvl = np.where(ind_at_confLvl == lastyear, np.nan, ind_at_confLvl).astype('int').flatten()
            ind_at_confLvl = np.where(ind_at_confLvl == lastyear, np.nan, ind_at_confLvl).flatten()

            ri_cnt = 0
            for ri in ind_at_confLvl:
                if season == 'annual':
                    str_start = (str(ri - int(WW / 2)) + '-12-31')
                    str_end = (str(ri + int(WW / 2)) + '-12-31')
                else:
                    str_start = (ri - int(WW / 2))
                    str_end = (ri + int(WW / 2))
                ds_sub_tar_mean_i = ds_ref.ix[:, ri_cnt].loc[str_start:str_end].mean()
                ds_sub_mean_tar.ix[ri_cnt] = ds_sub_tar_mean_i
                ri_cnt += 1

            t_anom = np.abs(ds_sub_mean_tar - ds_sub_mean_ref)
            t_anom[t_anom == 0] = 0.0001  # some issue with np.sign function on a zero
            ds_sub_std_ref = ds_sub_ref.std(axis=0)
            ds_sub_diff_ref = ds_sub_ref.max(axis=0) - ds_sub_ref.min(axis=0)

            # t_anom.reshape((-1,1))
            ################################################
            tmp_col = col_names
            val_at_conflvl_pd[tmp_col] = t_anom.values.reshape((-1, 1))
            std_at_conflvl_pd[tmp_col] = ds_sub_std_ref.values.reshape((-1, 1))
            range_at_conflvl_pd[tmp_col] = ds_sub_diff_ref.values.reshape((-1, 1))
            val_mean = val_at_conflvl_pd.mean(axis=1)
            val_mean[val_mean == 0] = 0.0001
            std_mean = std_at_conflvl_pd.mean(axis=1)
            std_mean[std_mean == 0] = 0.0001
            range_mean = range_at_conflvl_pd.mean(axis=1)
            range_mean[range_mean == 0] = 0.0001
            # # assign column name according to confidence level
            # # type(val_mean )
            std_mean = std_mean.to_frame()
            std_mean.columns = ['reference_period']
            range_mean = range_mean.to_frame()
            range_mean.columns = ['reference_period']
            val_mean = val_mean.to_frame()
            val_mean.columns = ['%s' % confidence_level]

            if np.sum(~np.isnan(val_mean.values)) <= 1:
                continue

            # --------------------------------------------------------------------------------------------------------------------------------
            # VALUE DIFFERENCE BETWEEN REFERENCE AND TARGET PERIOD
            tmp_out_path = CRU_ToE
            name_csv = tmp_out_path + "%s_CRU-NCEP_ToE_valAtConflvl_%s_1901-2016_Siberia_df_%s_.csv" % (
                Varname_CRU, confidence_level, season)
            val_mean.to_csv(name_csv)

            _dname = '%s_%s_valAtSignifLvl_%s_%s' % (Varname_CRU, model_real, confidence_level, season)
            ######################
            if not isfile(name_csv):
                continue
            t_pd = pd_toe_to_geoarray(input_array=name_csv, nan_mask=name_csv, model_IDs=None, sign=name_csv)

            # get variability in obtained outcome
            t_pd = calc_array_stats(t_pd, model)
            if Varname_CRU == CRU_var_name_T:
                if confidence_level <= 0.5:
                    # Z_RANGE = [0.3, 5]
                    # Z_RANGE = [0.0, 4.6]
                    Z_RANGE = [0.0, 5.9]
                else:
                    # Z_RANGE = [2.5, 15]
                    Z_RANGE = [0.0, 13.3]
            elif Varname_CRU == CRU_var_name_P:

                if confidence_level <= 0.5:
                    Z_RANGE = [0.0, 139]
                else:
                    Z_RANGE = [0.0, 313]
            tmp_max = np.nanmax(t_pd['array'])
            print(_dname + '    ' + str(tmp_max))
            plot_map_negpos(t_pd, key_name='ref', dataset_name=_dname, cmap_name2='gnuplot2_r', cmap_name1='cubehelix',
                            save=True, n_colors=30,
                            z_range=Z_RANGE, skip_rounding=True, omit_title=True, omit_cbar=False)

            # STD -------------------------------------------------
            tmp_out_path = CRU_ToE
            name_csv = tmp_out_path + "%s_CRUNCEP_ToE_STD_refPeriod_1901-2016_Siberia_df_%s_.csv" % (
                Varname_CRU, season)
            std_mean.to_csv(name_csv)
            _dname = '%s_%s_STD_refPeriod_%s' % (Varname_CRU, model_real, season)

            t_pd = pd_toe_to_geoarray(input_array=name_csv, nan_mask=name_csv, model_IDs=None, sign=name_csv)

            # get variability in obtained outcome
            t_pd = calc_array_stats(t_pd, model)
            # t_pd['sign']
            # t_pd['array_pn'].shape
            # np.mean(t_pd['array_pn'],axis=0).shape
            if Varname_CRU == CRU_var_name_T:
                Z_RANGE = [0.0, 2.3]

            elif Varname_CRU == CRU_var_name_P:
                Z_RANGE = [0.0, 115]

            tmp_max = np.nanmax(t_pd['array'])
            print(_dname + '    ' + str(tmp_max))
            # Z_RANGE = [0, tmp_max]

            plot_map_negpos(t_pd, key_name='ref', dataset_name=_dname, cmap_name1='gnuplot2',
                            save=True, n_colors=30,
                            z_range=Z_RANGE, skip_rounding=True, omit_title=True, omit_cbar=False)

            # RANGE -------------------------------------------------
            tmp_out_path = CRU_ToE
            name_csv = tmp_out_path + "%s_CRUNCEP_ToE_RANGE_refPeriod_1901-2016_Siberia_df_%s_.csv" % (
                Varname_CRU, season)
            range_mean.to_csv(name_csv)
            if not isfile(name_csv):
                continue
            # make map

            _dname = '%s_%s_RANGE_refPeriod_%s' % (Varname_CRU, model_real, season)
            ######################
            # if not isfile(name_csv):
            #     continue
            t_pd = pd_toe_to_geoarray(input_array=name_csv, nan_mask=name_csv, model_IDs=None, sign=name_csv)

            # get variability in obtained outcome
            t_pd = calc_array_stats(t_pd, model)
            # t_pd['sign']
            # t_pd['array_pn'].shape
            # np.mean(t_pd['array_pn'],axis=0).shape
            if Varname_CRU == CRU_var_name_T:
                Z_RANGE = [0.0, 9.6]

            elif Varname_CRU == CRU_var_name_P:
                Z_RANGE = [0.0, 469]

            tmp_max = np.nanmax(t_pd['array'])
            print(_dname + '    ' + str(tmp_max))
            # Z_RANGE = [0, tmp_max]

            plot_map_negpos(t_pd, key_name='ref', dataset_name=_dname, cmap_name1='gnuplot2',
                            save=True, n_colors=30,
                            z_range=Z_RANGE, skip_rounding=True, omit_title=True, omit_cbar=False)
