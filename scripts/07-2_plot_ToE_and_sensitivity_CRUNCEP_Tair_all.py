"""
Plot the obtained ToE and showcase the variability based on the sensitivity analysis (window width, time choice to
split the time-series), and the best n performing models of the CMIP5 collection
"""
from toe_tools.gis import *
from toe_tools.paths import *

# Parameters
set_noConverge_to_NaN = True

Z_RANGE = [1960, 2080]
Z_RANGE_STD = [0, 20]
Z_RANGE_HD = [0.2, 1.001]
COL_HD = 'viridis'
model = 'CRUNCEP'

years = np.arange(1922, 2088, 1)
for season in ['annual', 'summer', 'winter']:

    for confidence_level in confidence_levels:
        filename_toe = CRU_ToE + '%s_CRU-NCEP_ToE_Sensitivity-timemax_%s_1901-2016_Siberia_df_%s_overlap_.csv' % (
            CRU_var_name_T, confidence_level, season)
        filename_valmax = CRU_ToE + '%s_CRU-NCEP_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_%s_overlap_.csv' % (
            CRU_var_name_T, confidence_level, season)
        filename_sign = CRU_ToE + '%s_CRU-NCEP_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_%s_overlap_sign.csv' % (
            CRU_var_name_T, confidence_level, season)

        ##################################
        # ALL models
        # MAX value
        _dname = '%s_val_%s_%s_%s' % (CRU_var_name_T, model, confidence_level, season)
        ######################
        t_pd = pd_toe_to_geoarray(input_array=filename_valmax, nan_mask=filename_valmax, model_IDs=None,
                                  sign=filename_sign)

        # get variability in obtained outcome
        t_pd = calc_array_stats(t_pd, model)
        plot_map_negpos(t_pd, key_name='ref', dataset_name=_dname, cmap_name1='viridis', save=True, pickled=True)
        plot_map_negpos(t_pd, key_name='std', dataset_name=_dname, cmap_name1='magma', save=True)

        # TOE
        _dname = '%s_ToE_%s_%s_%s' % (CRU_var_name_T, model, confidence_level, season)
        ######################
        t_pd = pd_toe_to_geoarray(input_array=filename_toe, nan_mask=filename_valmax, model_IDs=None,
                                  sign=filename_sign)
        t_pd = calc_array_stats(t_pd, model)

        plot_map_negpos(t_pd, key_name='median', dataset_name=_dname, cmap_name1='cubehelix', z_range=Z_RANGE,
                        n_colors=10, save=True)
        plot_map_negpos(t_pd, key_name='ref', dataset_name=_dname, cmap_name1='cubehelix', z_range=Z_RANGE, n_colors=10,
                        save=True)
        plot_map_negpos(t_pd, key_name='posneg_na', dataset_name=_dname, cmap_name1='magma', z_range=[0, 1],
                        n_colors=10, save=True)

    # individual years
    name_csv = CRU_ToE + "%s_CRU-NCEP_ToE_Sensitivity-valatyears_REF_1901-2016_Siberia_df_%s_overlap_.csv" % (
        CRU_var_name_T, season)
    name_csv_sign = CRU_ToE + "%s_CRU-NCEP_ToE_Sensitivity-signatyears_REF_1901-2016_Siberia_df_%s_overlap_.csv" % (
        CRU_var_name_T, season)

    cnt = 0
    for year_i in years:
        # year_i= 2000
        _dname = '%s_val_avg-HD_mean_%s_%s_' % (CRU_var_name_T, model, season)
        ######################
        if year_i <= 2004:
            t_pd = pd_toe_to_geoarray(input_array=name_csv, nan_mask=name_csv, model_IDs=None, sign=name_csv_sign)
            t_pd = calc_array_stats(t_pd, model, annualoverview=year_i)
            t_pd[year_i] = t_pd['array'][cnt, :, :]
            plot_map_negpos(t_pd, key_name=year_i, dataset_name=_dname, z_range=Z_RANGE_HD, cmap_name1=COL_HD,
                            n_colors=10,
                            save=True)

        else:
            # t_pd.keys()
            t_pd['array'].fill(2)
            t_pd['array_pn'].fill(1)
            t_pd = calc_array_stats(t_pd, model, annualoverview=2000)
            t_pd[year_i] = t_pd['array'][0, :, :]

            plot_map_negpos(t_pd, key_name=year_i, dataset_name=_dname, z_range=Z_RANGE_HD, cmap_name1=COL_HD,
                            n_colors=10,
                            save=True)

        cnt += 1
