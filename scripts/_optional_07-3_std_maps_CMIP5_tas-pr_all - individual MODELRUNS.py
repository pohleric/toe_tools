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
for Varname_ESD in ['tas', 'pr']:
    for season in ['annual', 'summer', 'winter']:
        # for confidence_level in confidence_levels:
        # confidence_level = 0.30  # confidence level at which we assume ToE
        if Varname_ESD == 'tas':
            PATH_IN = ESD_HD_tas
        else:
            PATH_IN = ESD_HD_pr
        # PATH_OUT = PATH_ESD + "overlap_Lena_hellinger/fullSeries/"
        PATH_ESD_raw = ESD_TXT

        # the individual models
        files = [f for f in listdir(PATH_ESD_raw) if
                 re.match(r'^%s_ESD_[0-9]{3}_1861-2100_Siberia_df_%s.+orig.csv$' % (Varname_ESD, season), f)]
        files.sort()

        model_ids = [x.split('_')[2] for x in files]
        col_names = model_ids[:]

        # write output into this
        ref0_pd = pd.read_pickle(
            str(PATH_IN) + "%s_ESD_average_1861-2100_Siberia_df_%s-orig.pickle" % (Varname_ESD, season))
        ref0_val_1st_bool, lastyear = find_timemax(ref0_pd, 0.9)

        src_timemax = get_timemax(ref0_val_1st_bool, ref0_pd)
        range_at_conflvl_pd = pd.DataFrame(index=src_timemax.index, columns=col_names)
        std_at_conflvl_pd = pd.DataFrame(index=src_timemax.index, columns=col_names)

        # cnt_mod_i = 0
        # for model_i in files:
        #     print(model_i)
        #     # model_i = 'tas_ESD_028_1861-2100_Siberia_df_annual-orig.pickle'
        #     # key = '0053.00141.0'
        #
        #     # filename_toe = PATH_IN + model_i
        #     # ds_toe = read_csv_ann(filename_toe)
        #     # val_1st_bool, lastyear = find_timemax(ds_toe, confidence_level)
        #
        #     filename_raw = PATH_ESD_raw + model_i
        #     ds_ref = read_csv_ann(filename_raw)
        #
        #     # get the mean of the reference period
        #     if season == 'annual':
        #         ds_sub_ref = ds_ref.loc['1900-12-31':'1921-12-31']
        #     else:
        #         ds_sub_ref = ds_ref.loc[1900:1921]
        #
        #     ds_sub_std_ref = ds_sub_ref.std(axis=0)
        #     ds_sub_diff_ref = ds_sub_ref.max(axis=0) - ds_sub_ref.min(axis=0)
        #
        #     tmp_col = model_ids[cnt_mod_i]
        #     std_at_conflvl_pd[tmp_col] = ds_sub_std_ref
        #     range_at_conflvl_pd[tmp_col] = ds_sub_diff_ref
        #
        #     cnt_mod_i += 1
        #
        # std_mean = std_at_conflvl_pd.mean(axis=1)
        # range_mean = range_at_conflvl_pd.mean(axis=1)
        # # # assign column name according to confidence level
        # # # type(val_mean )
        # std_mean = std_mean.to_frame()
        # std_mean.columns = ['reference_period']
        #
        # range_mean = range_mean.to_frame()
        # range_mean.columns = ['reference_period']
        #
        # if np.sum(~np.isnan(std_mean.values)) <= 1:
        #     continue

        ################################################
        # write output

        # STD -------------------------------------------------
        tmp_out_path = ESD_ToE
        name_csv = tmp_out_path + "%s_ESD_ToE_STD_refPeriod_1901-2016_Siberia_df_%s_.csv" % (
            Varname_ESD, season)
        # std_mean.to_csv(name_csv)
        if not isfile(name_csv):
            continue
        # make map

        _dname = '%s_%s_STD_refPeriod_%s' % (Varname_ESD, model, season)
        ######################
        # if not isfile(name_csv):
        #     continue
        t_pd = pd_toe_to_geoarray(input_array=name_csv, nan_mask=name_csv, model_IDs=None, sign=name_csv)

        # get variability in obtained outcome
        t_pd = calc_array_stats(t_pd, model)
        # t_pd['sign']
        # t_pd['array_pn'].shape
        # np.mean(t_pd['array_pn'],axis=0).shape
        if Varname_ESD == 'tas':
            Z_RANGE = [0.0, 2.3]

        elif Varname_ESD == 'pr':
            Z_RANGE = [0.0, 115]

        tmp_max = np.nanmax(t_pd['array'])
        print(_dname + '    ' + str(tmp_max))

        plot_map_negpos(t_pd, key_name='ref', dataset_name=_dname, cmap_name1='gnuplot2', save=True,
                        n_colors=30, z_range=Z_RANGE, skip_rounding=True, omit_title=True, omit_cbar=False)

        # RANGE -------------------------------------------------
        tmp_out_path = ESD_ToE
        name_csv = tmp_out_path + "%s_ESD_ToE_RANGE_refPeriod_1901-2016_Siberia_df_%s_.csv" % (
            Varname_ESD, season)
        # range_mean.to_csv(name_csv)
        if not isfile(name_csv):
            continue
        # make map

        _dname = '%s_%s_RANGE_refPeriod_%s' % (Varname_ESD, model, season)
        ######################
        # if not isfile(name_csv):
        #     continue
        t_pd = pd_toe_to_geoarray(input_array=name_csv, nan_mask=name_csv, model_IDs=None, sign=name_csv)

        # get variability in obtained outcome
        t_pd = calc_array_stats(t_pd, model)

        if Varname_ESD == 'tas':
            Z_RANGE = [0.0, 8.5]

        elif Varname_ESD == 'pr':
            Z_RANGE = [0.0, 415]

        tmp_max = np.nanmax(t_pd['array'])
        print(_dname + '    ' + str(tmp_max))

        plot_map_negpos(t_pd, key_name='ref', dataset_name=_dname, cmap_name1='gnuplot2', save=True, n_colors=30,
                        z_range=Z_RANGE, skip_rounding=True, omit_title=True, omit_cbar=False)
