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
model = 'CMIP5'

for Varname_ESD in [ESD_var_name_T, ESD_var_name_P]:
    # Varname_ESD = 'tas'
    for season in ['annual', 'summer', 'winter']:
        for confidence_level in confidence_levels:
            # season = 'summer'
            if Varname_ESD == ESD_var_name_T:
                PATH_IN = ESD_HD_tas
            else:
                PATH_IN = ESD_HD_pr

            # the individual models
            files = [f for f in listdir(ESD_TXT) if
                     re.match(r'^%s_ESD_[0-9]{3}_1861-2100_Siberia_df_%s.+orig.csv$' % (Varname_ESD, season), f)]
            files.sort()

            model_ids = [x.split('_')[2] for x in files]
            col_names = model_ids[:]

            # write output into this
            ref0_pd = pd.read_pickle(
                str(PATH_IN) + "%s_ESD_average_1861-2100_Siberia_df_%s-orig.pickle" % (Varname_ESD, season))
            ref0_val_1st_bool, lastyear = find_timemax(ref0_pd, confidence_level)

            src_timemax = get_timemax(ref0_val_1st_bool, ref0_pd)
            val_at_conflvl_pd = pd.DataFrame(index=src_timemax.index, columns=col_names)
            range_at_conflvl_pd = pd.DataFrame(index=src_timemax.index, columns=col_names)
            std_at_conflvl_pd = pd.DataFrame(index=src_timemax.index, columns=col_names)

            filename_raw = ESD_TXT + files[0]
            ds_ref = read_csv_ann(filename_raw)

            # get the mean of the reference period
            if season == 'annual':
                ds_sub_ref = ds_ref.loc[str(XMIN_year - 1) + '-12-31':str(XSPLIT_year) + '-12-31']
            else:
                ds_sub_ref = ds_ref.loc[XMIN_year - 1:XSPLIT_year]
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

            ################################################
            # write output

            tmp_out_path = ESD_ToE
            name_csv = tmp_out_path + "%s_ESD_ToE_valAtConflvl_%s_1901-2016_Siberia_df_%s_.csv" % (
                Varname_ESD, confidence_level, season)
            val_at_conflvl_pd.to_csv(name_csv)

            # make map
            _dname = '%s_%s_valAtSignifLvl_%s_%s' % (Varname_ESD, model, confidence_level, season)

            if not isfile(name_csv):
                continue
            t_pd = pd_toe_to_geoarray(input_array=name_csv, nan_mask=name_csv, model_IDs=None, sign=name_csv)

            # get variability in obtained outcome
            t_pd = calc_array_stats(t_pd, model)

            if Varname_ESD == ESD_var_name_T:
                if confidence_level <= 0.5:
                    Z_RANGE = [0.0, 3.6]
                else:
                    Z_RANGE = [0.0, 8.3]
            elif Varname_ESD == ESD_var_name_P:

                if confidence_level <= 0.5:
                    Z_RANGE = [0.0, 172]
                else:
                    Z_RANGE = [0.0, 313]

            tmp_max = np.nanmax(t_pd['array'])
            tmp_min = np.nanmin(t_pd['array'])
            tmp_mean = np.nanmean(t_pd['array'])
            tmp_median = np.nanmedian(t_pd['array'])
            print(_dname + '    max: ' + str(tmp_max))
            print(_dname + '    min: ' + str(tmp_min))
            print(_dname + '    mean: ' + str(tmp_mean))
            print(_dname + '    median: ' + str(tmp_median))

            plot_map_negpos(t_pd, key_name='ref', dataset_name=_dname, cmap_name2='gnuplot2_r', cmap_name1='cubehelix',
                            save=True, n_colors=30,
                            z_range=Z_RANGE, skip_rounding=True, omit_title=True, omit_cbar=False)
