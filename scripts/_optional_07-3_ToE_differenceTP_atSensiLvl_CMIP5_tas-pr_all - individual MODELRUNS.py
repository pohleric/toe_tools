"""
Calculate corresponding temperature/precipitation change at various significance levels
"""
from toe_tools.toe_calc import *
from toe_tools.gis import *
from copy import copy
from os.path import isfile

ww= 21

# Parameters
set_noConverge_to_NaN = False
set_noConverge_to_NaN = True
# confidence_level = 0.999  # confidence level at which we assume ToE
confidence_levels = [0.3, 0.4, 0.5]
# confidence_levels = [0.3, 0.4, 0.5, 0.9]
Varname_ESD = 'pr'
model = 'CMIP5'
for Varname_ESD in ['tas']:
    Varname_ESD2 = 'pr'
    for season in ['annual', 'summer', 'winter']:
        for confidence_level in confidence_levels:
            # confidence_level = 0.95  # confidence level at which we assume ToE
            # confidence_level = 0.50  # confidence level at which we assume ToE
            # confidence_level = 0.40  # confidence level at which we assume ToE
            # confidence_level = 0.30  # confidence level at which we assume ToE
            # Varname_ESD = 'tas'
            # season = 'annual'
            PATH_ESD = "/home/hydrogeol/epohl/data/ESD/"
            if Varname_ESD == 'tas':
                PATH_IN = PATH_ESD + "sensitivity_Lena_hellinger/fullSeries/"
            else:
                PATH_IN = PATH_ESD + "sensitivity_Lena_hellinger/fullSeries_pr/"
            # PATH_OUT = PATH_ESD + "overlap_Lena_hellinger/fullSeries/"
            PATH_ESD_raw = "/home/hydrogeol/epohl/data/ESD/txt_input/"
            PATH_IN2 = PATH_ESD + "sensitivity_Lena_hellinger/fullSeries_pr/"

            # the individual models
            files = [f for f in listdir(PATH_ESD_raw) if re.match(r'^%s_ESD_[0-9]{3}_1861-2100_Siberia_df_%s.+orig.csv$' % (Varname_ESD, season), f)]
            files.sort()

            files2 = [f for f in listdir(PATH_ESD_raw) if re.match(r'^%s_ESD_[0-9]{3}_1861-2100_Siberia_df_%s.+orig.csv$' % (Varname_ESD2, season), f)]
            files2.sort()

            model_ids = [x.split('_')[2] for x in files]
            col_names = model_ids[:]

            # write output into this
            ref0_pd = pd.read_pickle(str(PATH_IN) + "%s_ESD_average_1861-2100_Siberia_df_%s-orig.pickle" % (Varname_ESD, season))
            ref0_val_1st_bool, lastyear = find_timemax(ref0_pd, confidence_level)

            src_timemax = get_timemax(ref0_val_1st_bool, ref0_pd)
            val_at_conflvl_pd = pd.DataFrame(index=src_timemax.index, columns=col_names)

            src_timemax2 = get_timemax(ref0_val_1st_bool, ref0_pd)
            val_at_conflvl_pd2 = pd.DataFrame(index=src_timemax.index, columns=col_names)

            # cnt_mod_i = 0
            # for model_i in files:
            #     print(model_i)
            #     # model_i = 'tas_ESD_028_1861-2100_Siberia_df_annual-orig.pickle'
            #     # key = '0053.00141.0'
            #
            #     filename_toe = PATH_IN + model_i
            #     ds_toe = read_csv_ann(filename_toe)
            #     val_1st_bool, lastyear = find_timemax(ds_toe, confidence_level)
            #
            #     # filename_raw = PATH_ESD_raw + model_i
            #     # ds_ref = read_csv_ann(filename_raw)
            #
            #     # # get the mean of the reference period
            #     # if season == 'annual':
            #     #     ds_sub_ref = ds_ref.loc['1900-12-31':'1921-12-31']
            #     # else:
            #     #     ds_sub_ref = ds_ref.loc[1900:1921]
            #     # ds_sub_mean_ref = ds_sub_ref.mean(axis=0)
            #
            #     # get the mean of the ww of the year where we have ToE
            #     ind_at_confLvl = val_1st_bool.apply(lambda x: val_1st_bool.index[np.where(x == 1)[0]], axis=0).values
            #
            #     # test :
            #     # ind_at_confLvl[0,20] = lastyear
            #     # ds_sub_mean_tar = copy(ds_sub_mean_ref)
            #     ind_at_confLvl = np.where(ind_at_confLvl == lastyear, np.nan, ind_at_confLvl).flatten()
            #     # ri_cnt = 0
            #     # for ri in ind_at_confLvl:
            #     #     if season == 'annual':
            #     #         str_start = (str(ri - int(ww / 2)) + '-12-31')
            #     #         str_end = (str(ri + int(ww / 2)) + '-12-31')
            #     #     else:
            #     #         str_start = (ri - int(ww / 2))
            #     #         str_end = (ri + int(ww / 2))
            #     #     ds_sub_tar_mean_i = ds_ref.ix[:, ri_cnt].loc[str_start:str_end].mean()
            #     #     ds_sub_mean_tar.ix[ri_cnt] = ds_sub_tar_mean_i
            #     #     ri_cnt += 1
            #
            #     # t_anom = ds_sub_mean_tar - ds_sub_mean_ref
            #     t_anom = ind_at_confLvl
            #     ################################################
            #     tmp_col = model_ids[cnt_mod_i]
            #     val_at_conflvl_pd[tmp_col] = t_anom
            #
            #     cnt_mod_i += 1
            #
            # val_mean = val_at_conflvl_pd.mean(axis=1)
            # # assign column name according to confidence level
            # # type(val_mean )
            # val_mean = val_mean .to_frame()
            # val_mean.columns = ['%s' % confidence_level]
            # if np.sum(~np.isnan(val_mean.values)) <= 1:
            #     continue

            ################################################
            # write output

            # val_mean.to_csv(name_csv)
            tmp_out_path = PATH_ESD + 'value_of_emergence/'
            name_csv = tmp_out_path + "%s_ESD_ToE_timeAtConflvl_%s_1901-2016_Siberia_df_%s_.csv" % (
                Varname_ESD, confidence_level, season)
            name_csv2 = tmp_out_path + "%s_ESD_ToE_timeAtConflvl_%s_1901-2016_Siberia_df_%s_.csv" % (
                Varname_ESD2, confidence_level, season)

            # make map

            _dname = '%s_%s_ToEdiff_AtSignifLvl_%s_%s' % (Varname_ESD, model, confidence_level, season)
            ######################
            if not isfile(name_csv):
                continue
            t_pd = pd_toe_to_geoarray(input_array=name_csv, nan_mask=name_csv, model_IDs=None, sign=name_csv)
            t_pd2 = pd_toe_to_geoarray(input_array=name_csv2, nan_mask=name_csv2, model_IDs=None, sign=name_csv2)
            t_diff = copy(t_pd)
            t_diff['array'] = t_pd2['array'] - t_pd['array']
            # t_diff['array'] = t_pd['array'] - t_pd2['array']
            # t_pd.keys()

            # # get variability in obtained outcome
            # t_pd = calc_array_stats(t_pd, model)

            # get variability in obtained outcome
            t_pd = calc_array_stats(t_diff, model)

            # t_pd['sign']
            # t_pd['array_pn'].shape
            # np.mean(t_pd['array_pn'],axis=0).shape
            # if Varname_ESD == 'tas':

            Z_RANGE = [16, 83]
            # elif Varname_ESD == 'pr':
            #     # if season == 'winter':
            #     #     if confidence_level <= 0.5:
            #     #         Z_RANGE = [0, 20]
            #     #     else:
            #     #         Z_RANGE = [0, 60]
            #     # else:
            #     #     if confidence_level <= 0.5:
            #     #         Z_RANGE = [0, 60]
            #     #     else:
            #     #         Z_RANGE = [0, 60]
            #     # Z_RANGE = [1.8, 23]
            #     if confidence_level <= 0.5:
            #         # Z_RANGE = [0.3, 5]
            #         # Z_RANGE = [0.0, 4.6]
            #         Z_RANGE = [0.0, 139]
            #     else:
            #         # Z_RANGE = [2.5, 15]
            #         Z_RANGE = [0.0, 313]
            # tmp_max = np.nanmax(t_anom.values) * 1.1
            # tmp_max = np.nanmax(t_anom.values)
            tmp_max = np.nanmax(t_pd['array'])
            tmp_min = np.nanmin(t_pd['array'])
            tmp_mean = np.nanmean(t_pd['array'])
            tmp_median = np.nanmedian(t_pd['array'])
            print(_dname + '    max: ' + str(tmp_max))
            print(_dname + '    min: ' + str(tmp_min))
            print(_dname + '    mean: ' + str(tmp_mean))
            print(_dname + '    median: ' + str(tmp_median))
            # Z_RANGE = [0, tmp_max]

            # plot_map_negpos(t_pd, key_name='ref', dataset_name=_dname, cmap_name1='gnuplot2', save=True, n_colors=20, z_range=Z_RANGE)
            # plot_map_negpos(t_pd, key_name='ref', dataset_name=_dname, cmap_name1='cubehelix', save=True, n_colors=20, z_range=Z_RANGE,skip_rounding=True)
            plot_map_negpos(t_pd, key_name='ref', dataset_name=_dname, cmap_name1='cubehelix', save=True, n_colors=30,
                            z_range=Z_RANGE, skip_rounding=False, omit_title=True, omit_cbar=False)
