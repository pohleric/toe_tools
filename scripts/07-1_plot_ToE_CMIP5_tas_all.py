"""
Plot the obtained ToE and showcase the variability based on the sensitivity analysis (window width, time choice to
split the time-series), and the best n performing models of the CMIP5 collection
"""
from toe_tools.gis import *
from toe_tools.paths import *

# PATH_ESD = '/home/hydrogeol/epohl/data/ESD/overlap_Lena_hellinger/fullSeries'

# Parameters
set_noConverge_to_NaN = True

model_IDs = [format('%03d' % m) for m in np.loadtxt('data/tas_best_10_models_all_seasons_NSE__LongStations.txt')]
best_model_0 = model_IDs[0]

model = 'CMIP5'
Z_RANGE = [1960, 2080]
Z_RANGE_STD = [0, 20]
Z_RANGE_HD = [0.2, 1.001]
COL_HD = 'viridis'

years = np.arange(1922, 2088, 1)
for season in ['annual', 'summer', 'winter']:
    for confidence_level in confidence_levels:
        filename_toe = ESD_ToE + '/%s_ESD_ToE_Sensitivity-timemax_%s_1901-2016_Siberia_df_%s_overlap_.csv' % (
            ESD_var_name_T, confidence_level, season)
        filename_valmax = ESD_ToE + '/%s_ESD_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_%s_overlap_.csv' % (
            ESD_var_name_T, confidence_level, season)
        filename_sign = ESD_ToE + '/%s_ESD_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_%s_overlap_sign.csv' % (
            ESD_var_name_T, confidence_level, season)

        ##################################
        # best 10 models
        # MAX value
        _dname = '%s_val_%s_10best_%s_%s' % (ESD_var_name_T, model, confidence_level, season)
        ######################
        t_pd = pd_toe_to_geoarray(input_array=filename_valmax, nan_mask=filename_valmax, model_IDs=model_IDs,
                                  sign=filename_sign)

        # get variability in obtained outcome
        t_pd = calc_array_stats(t_pd, model)

        plot_map_negpos(t_pd, key_name='ref', dataset_name=_dname, cmap_name1='viridis', save=True, z_range=Z_RANGE_HD)
        plot_map_negpos(t_pd, key_name='median', dataset_name=_dname, cmap_name1='viridis', save=True,
                        z_range=Z_RANGE_HD)
        # plot_map_negpos(t_pd, key_name='mean',  dataset_name=_dname , cmap_name1='viridis')
        plot_map_negpos(t_pd, key_name='std', dataset_name=_dname, cmap_name1='viridis')

        # TOE
        _dname = '%s_ToE_%s_10best_%s_%s' % (ESD_var_name_T, model, confidence_level, season)
        ######################
        t_pd = pd_toe_to_geoarray(input_array=filename_toe, nan_mask=filename_valmax, model_IDs=model_IDs,
                                  sign=filename_sign)
        # get variability in obtained outcome
        t_pd = calc_array_stats(t_pd, model)

        plot_map_negpos(t_pd, key_name='median', dataset_name=_dname, cmap_name1='cubehelix', z_range=Z_RANGE,
                        n_colors=10, save=True)
        plot_map_negpos(t_pd, key_name='min', dataset_name=_dname, cmap_name1='cubehelix', z_range=Z_RANGE, n_colors=10,
                        save=True)
        plot_map_negpos(t_pd, key_name='max', dataset_name=_dname, cmap_name1='cubehelix', z_range=Z_RANGE, n_colors=10,
                        save=True)
        plot_map_negpos(t_pd, key_name='std', dataset_name=_dname, cmap_name1='magma', z_range=Z_RANGE_STD, n_colors=10,
                        save=True)

        # ##################################
        # # ALL models
        # # MAX value
        # _dname = '%s_val_%s_%s_%s'%(ESD_var_name_T, model, confidence_level, season)
        # ######################
        # t_pd = pd_toe_to_geoarray(input_array=filename_valmax, nan_mask=filename_valmax,
        # model_IDs=None, sign=filename_sign)
        #
        # # get variability in obtained outcome
        # t_pd = calc_array_stats(t_pd, model)
        #
        # plot_map_negpos(t_pd, key_name='median',  dataset_name=_dname, cmap_name1='viridis')
        # # plot_map_negpos(t_pd, key_name='mean',  dataset_name=_dname , cmap_name1='viridis')
        # # plot_map_negpos(t_pd, key_name='std',  dataset_name=_dname,cmap_name1='viridis')

        # TOE
        _dname = '%s_ToE_%s_%s_%s' % (ESD_var_name_T, model, confidence_level, season)
        ######################
        t_pd = pd_toe_to_geoarray(input_array=filename_toe, nan_mask=filename_valmax, model_IDs=None,
                                  sign=filename_sign)

        # get variability in obtained outcome
        t_pd = calc_array_stats(t_pd, model)

        plot_map_negpos(t_pd, key_name='median', dataset_name=_dname, cmap_name1='cubehelix', z_range=Z_RANGE,
                        n_colors=10, save=True)
        plot_map_negpos(t_pd, key_name='min', dataset_name=_dname, cmap_name1='cubehelix', z_range=Z_RANGE, n_colors=10,
                        save=True)
        plot_map_negpos(t_pd, key_name='max', dataset_name=_dname, cmap_name1='cubehelix', z_range=Z_RANGE, n_colors=10,
                        save=True)
        plot_map_negpos(t_pd, key_name='std', dataset_name=_dname, cmap_name1='magma', z_range=Z_RANGE_STD, n_colors=10,
                        save=True)

    # -----------------------------------------------------------------------------------------------------------
    # individual years

    # average HD
    name_csv = ESD_ToE + "/%s_ESD_ToE_Sensitivity-valatyears_REF_1901-2016_Siberia_df_%s_overlap_.csv" % (
        ESD_var_name_T, season)
    name_csv_sign = ESD_ToE + "/%s_ESD_ToE_Sensitivity-signatyears_REF_1901-2016_Siberia_df_%s_overlap_.csv" % (
        ESD_var_name_T, season)

    cnt = 0
    for year_i in years:
        # year_i = 2087
        # cnt = 165
        _dname = '%s_val_avg-HD_mean_%s_%s_' % (ESD_var_name_T, model, season)
        ######################
        t_pd = pd_toe_to_geoarray(input_array=name_csv, nan_mask=name_csv, model_IDs=None, sign=name_csv_sign)
        t_pd = calc_array_stats(t_pd, model, annualoverview=year_i)
        t_pd[year_i] = t_pd['array'][cnt, :, :]
        plot_map_negpos(t_pd, key_name=year_i, dataset_name=_dname, z_range=Z_RANGE_HD, cmap_name1=COL_HD, n_colors=10,
                        save=True)
        cnt += 1
    # -----------------------------------------------------------------------------------------------------------

    # individual years
    # best overall model
    name_csv = ESD_ToE + "/%s_ESD_ToE_Sensitivity-valatyears_%s_1901-2016_Siberia_df_%s_overlap_.csv" % (
        ESD_var_name_T, best_model_0, season)
    name_csv_sign = ESD_ToE + "/%s_ESD_ToE_Sensitivity-signatyears_%s_1901-2016_Siberia_df_%s_overlap_.csv" % (
        ESD_var_name_T, best_model_0, season)

    cnt = 0
    for year_i in years:
        _dname = '%s_val_best_model(%s)_%s_%s_' % (ESD_var_name_T, best_model_0, model, season)
        ######################
        t_pd = pd_toe_to_geoarray(input_array=name_csv, nan_mask=name_csv, model_IDs=None, sign=name_csv_sign)
        t_pd = calc_array_stats(t_pd, model, annualoverview=year_i)
        t_pd[year_i] = t_pd['array'][cnt, :, :]
        plot_map_negpos(t_pd, key_name=year_i, dataset_name=_dname, z_range=Z_RANGE_HD, cmap_name1=COL_HD, n_colors=10,
                        save=True)
        cnt += 1
    # -----------------------------------------------------------------------------------------------------------
    # individual years
    # 10 best models
    name_csv = ESD_ToE + "/%s_ESD_ToE_Sensitivity-valatyears_%s_1901-2016_Siberia_df_%s_overlap_.csv" % (
        ESD_var_name_T, bm10, season)
    name_csv_sign = ESD_ToE + "/%s_ESD_ToE_Sensitivity-signatyears_%s_1901-2016_Siberia_df_%s_overlap_.csv" % (
        ESD_var_name_T, bm10, season)

    cnt = 0
    for year_i in years:
        _dname = '%s_val_best_model(%s)_%s_%s_' % (ESD_var_name_T, bm10, model, season)
        ######################
        t_pd = pd_toe_to_geoarray(input_array=name_csv, nan_mask=name_csv, model_IDs=None, sign=name_csv_sign)
        t_pd = calc_array_stats(t_pd, model, annualoverview=year_i)
        t_pd[year_i] = t_pd['array'][cnt, :, :]
        plot_map_negpos(t_pd, key_name=year_i, dataset_name=_dname, z_range=Z_RANGE_HD, cmap_name1=COL_HD, n_colors=10,
                        save=True)
        cnt += 1
