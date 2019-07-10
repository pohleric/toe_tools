from collections import Counter
from os import listdir
from tqdm import tqdm
from toe_tools.paths import *
from toe_tools.toe_calc import *

stats_list_NSE = pd.DataFrame()
stats_list_R2 = pd.DataFrame()

for season in ['annual', 'summer', 'winter']:

    cru = read_csv_ann(
        CRU_HD_pr + '%s_CRU-NCEP_1901-2016_Siberia_hellinger_%s_split%s_WW%s_.csv' % (
            CRU_var_name_P, season, XSPLIT_year, WW))
    cru_sign = read_csv_ann(
        CRU_HD_pr + '%s_CRU-NCEP_1901-2016_Siberia_hellinger_%s_split%s_WW%s__sign.csv' % (
            CRU_var_name_P, season, XSPLIT_year, WW))
    cru_posneg = cru * cru_sign

    df_CRU_pixel_ll = cru.keys()
    cru_dates = cru.index

    # Use the meteorological stations in the Lena Catchment to check between CRU and CMIP5 hellinger distances
    stations_pd = pd.read_csv('data/LenaStationsLong.txt', sep=' ')
    stations_pd_unq = get_unique_stations(stations_pd, df_CRU_pixel_ll)
    locations = stations_pd_unq['pixel_ID']
    stations = stations_pd_unq['name']
    # locations = ['0063.00129.0']
    # stations = ['Yakutsk']

    files = [f for f in listdir(ESD_HD_pr) if
             re.match(r'^%s_ESD_[0-9]{3}_1861-2100_Siberia_df_%s.+orig.csv$' % (ESD_var_name_P, season), f)]
    files.sort()

    fname = ESD_HD_pr + files[0]
    fname_sign = re.sub('orig.csv', 'orig_sign.csv', fname)
    esd_0 = read_csv_ann(fname)
    esd_0_sign = read_csv_ann(fname_sign)
    esd_posneg_0 = esd_0 * esd_0_sign
    df_ESD_pixel_ll = esd_0.keys()
    esd_dates = esd_0.index

    ###############################################################################
    best_models = {}
    for loc_i, station in zip(locations, stations):
        loc_i = [loc_i]
        station = [station]
        model_IDs = [x.split('_')[2] for x in files]
        cormat = pd.DataFrame(index=model_IDs, columns=['NSE', 'R2'])
        np.zeros((model_IDs.__len__(), 3))
        esds = {}
        for i in tqdm(np.arange(files.__len__())):
            file_name = ESD_HD_pr + files[i]
            file_name_sign = re.sub('orig.csv', 'orig_sign.csv', file_name)
            esd_i = read_csv_ann(file_name)
            esd_sign_i = read_csv_ann(file_name_sign)
            esd_i_posneg = esd_i * esd_sign_i
            esds[model_IDs[i]] = {'hellinger': esd_i[loc_i], 'sign': esd_sign_i[loc_i], 'posneg': esd_i_posneg[loc_i]}

            # with positive/negative changes taken into account
            ab = pd.merge(cru_posneg[loc_i], esd_i_posneg[loc_i], left_index=True, right_index=True).dropna()
            nash = nashsutcliffe(ab)
            corr = ab.corr().values[0, 1]
            cormat.loc[model_IDs[i]] = [nash, corr]

        # save order and reverse so that the lowest get plotted first and the best last
        ord_nse = cormat.sort_values('NSE', ascending=False).index[0:n_models][::-1]
        ord_r2 = cormat.sort_values('R2', ascending=False).index[0:n_models][::-1]

        cor_order = ord_nse
        # cormat.loc[ord_nse, 'NSE']
        best_models[loc_i[0]] = {'NSE': {'IDs': ord_nse, 'name': station, 'NSE': cormat.loc[ord_nse, 'NSE']},
                                 'R2': {'IDs': ord_r2, 'name': station, 'R2': cormat.loc[ord_r2, 'R2']}}

        # #__________________________________________________________________
        # # PLOT
        # ####--- this is only for plotting the curves of CRUNCEP vs CMIP5 models : ---#
        # import matplotlib.pyplot as plt
        # color = plt.get_cmap('Greys')(np.linspace(0.2, 1, n_models))
        #
        # # change font
        # import matplotlib as mpl
        #
        # mpl.rcParams['backend'] = 'TkAgg'
        # mpl.rcParams['font.family'] = ['sans-serif']
        # mpl.rcParams['font.sans-serif'] = ['Helvetica']
        # # fig = plt.figure(figsize=(4.5, 6))
        # fig = plt.figure(figsize=(4, 4))
        # c_s = '#0033DD77'
        # x = esd_dates
        #
        # # NSE
        # ax1 = plt.subplot(111)
        # for i, c in zip(range(n_models), color):
        #     t_ind = ord_nse[i]
        #     y = esds[t_ind]['hellinger']
        #     y_sign = esds[t_ind]['posneg']
        #     y_s = y[y_sign[loc_i] < 0].dropna()
        #     x_s = y_sign[y_sign[loc_i] < 0].dropna().index
        #     ax1.plot(x, y, c=c, label=t_ind)
        #     # ax1.plot(x_s, y_s, c=c_s, linestyle='None', marker='o', markersize=2, markerfacecolor='none', label='negative')
        #     ax1.plot(x_s, y_s, c=c_s, linestyle='None', marker='o', markersize=2, markerfacecolor='none',
        #              label='negative')
        #
        # ax1.plot(cru_dates, cru[loc_i], c='#00FF00', label=station[0])
        # # ax1.axhline(y=0.95, c='k', linewidth=.5, linestyle='--')
        #
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = OrderedDict(zip(labels, handles))
        # # negative as keyword ?
        # if 'negative' in labels:
        #     by_label = move_element(by_label, 'negative', 0)
        # ax1.legend(by_label.values(), by_label.keys(), prop={'size': 7})
        #
        # ax1.set_title(r'%s - %s (NSE) @ %s' % (ESD_var_name_P, season, station[0]), fontsize=cex_main_text)
        # ax1.set_ylabel('Hellinger distance [-]')
        # for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
        #              ax1.get_xticklabels() + ax1.get_yticklabels()):
        #     item.set_fontsize(cex_axis_text)
        # ax1.yaxis.labelpad = label_distance
        # x1, x2, y1, y2 = ax1.axis()
        # ax1.axis((1900., 2100., 0., 1.05))
        # tick_spacing = 2
        # xticks = ax1.get_xticks()
        # xlabels = ax1.get_xticklabels()
        #
        # ax1.grid(color='grey', linestyle='--', linewidth=.5)
        #
        # ax1.set_xticks(xticks[::tick_spacing])
        # ax1.set_xticklabels(xlabels[::tick_spacing])
        #
        # ax2 = plt.subplot(211)
        # for i, c in zip(range(n_models), color):
        #     t_ind = ord_r2[i]
        #     y = esds[t_ind]['hellinger']
        #     y_sign = esds[t_ind]['posneg']
        #     y_s = y[y_sign[loc_i] < 0].dropna()
        #     x_s = y_sign[y_sign[loc_i] < 0].dropna().index
        #     ax2.plot(x, y, c=c, label=t_ind)
        #     # ax2.plot(x_s, y_s, c=c_s, linestyle='None', marker='o', markersize=2, markerfacecolor='none', label='negative')
        #     ax2.plot(x_s, y_s, c=c_s, linestyle='None', marker='o', markersize=2, markerfacecolor='none',
        #              label='negative')
        #
        # ax2.plot(cru_dates, cru[loc_i], c='#00FF00', label=station[0])
        # ax2.axhline(y=0.95, c='k', linewidth=.5, linestyle='--')
        #
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = OrderedDict(zip(labels, handles))
        # if 'negative' in labels:
        #     by_label = move_element(by_label, 'negative', 0)
        # ax2.legend(by_label.values(), by_label.keys(), prop={'size': 6})
        #
        # # ax2.legend(prop={'size': 7})
        # ax2.set_title(r'best models (R$^2$) @ %s' % station[0], fontsize=cex_main_text)
        # ax2.set_ylabel('Hellinger distance [-]')
        # for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
        #              ax2.get_xticklabels() + ax2.get_yticklabels()):
        #     item.set_fontsize(cex_axis_text)
        # ax2.yaxis.labelpad = label_distance
        #
        # x1, x2, y1, y2 = ax2.axis()
        # ax2.axis((1900., 2100., 0., 1.05))
        # tick_spacing = 2
        # xticks = ax2.get_xticks()
        # xlabels = ax2.get_xticklabels()
        # ax2.set_xticks(xticks[::tick_spacing])
        # # ax2.set_xticklabels(xlabels[::tick_spacing])
        #
        # fig.tight_layout()
        #
        # fig.savefig('figures/%s_best_%s_models_%s_based_on_hellinger__%s_LongStations.pdf' % (
        # ESD_var_name_P, n_models, season, station[0]))
        # plt.close()
        #
        #
        # #### --- this is to write the results for each individual model and their performance with respect to the CRUNCEP ---#
        # # write best models into txt
        # with open('data/%s_best_%s_models_%s_NSE_%s_LongStations.txt' % (ESD_var_name_P, n_models, season, station[0]),
        #           'w') as outtxt:
        #     for item in ord_nse:
        #         # print(item)
        #         # outtxt.write("%s\n" % item)
        #         outtxt.write("%s %s\n" % (item, cormat.loc[item, 'NSE']))
        #         # outtxt.write("%s\n" % item)
        #
        # # write best models into txt
        # with open('data/%s_best_%s_models_%s_R2_%s_LongStations.txt' % (ESD_var_name_P, n_models, season, station[0]),
        #           'w') as outtxt:
        #     for item in ord_r2:
        #         # print(item)
        #         # outtxt.write("%s\n" % item)
        #         outtxt.write("%s %s\n" % (item, cormat.loc[item, 'R2']))

    # get the corresponding NSE for each station with mean, std, and min/max
    models_nse_stats = {}
    key_list = copy(best_models.keys())
    for key in key_list:
        models_nse_stats[key] = {  # 'IDs': np.array(copy(best_models[key]['NSE']['IDs'])),
            'Name': np.array(copy(best_models[key]['NSE']['name'][0])),
            'NSE_mean': np.array(copy(best_models[key]['NSE']['NSE'])).mean(),
            'NSE_std': np.array(copy(best_models[key]['NSE']['NSE'])).std(),
            'NSE_min': np.array(copy(best_models[key]['NSE']['NSE'])).min(),
            'NSE_max': np.array(copy(best_models[key]['NSE']['NSE'])).max()}
    models_nse_stats_pd = pd.DataFrame(models_nse_stats)
    models_nse_stats_pd.to_csv(
        'data/%s_best_%s_models_%s_NSE_stats_LongStations.txt' % (ESD_var_name_P, n_models, season))

    # NSE
    models_nse = []
    key_list = copy(best_models.keys())
    for key in key_list:
        models_nse.append(np.array(copy(best_models[key]['NSE']['IDs'])))
    models_nse = [y for x in models_nse for y in x]
    models_counts = Counter(models_nse)
    df = pd.DataFrame.from_dict(models_counts, orient='index')
    df_sort = df.sort_values(by=0, ascending=False, axis=0)

    str_str = df_sort.iloc[0:n_models, 0].index.values
    pd.DataFrame(str_str).to_csv('data/%s_best_%s_models_%s_NSE_LongStations.txt' % (ESD_var_name_P, n_models, season),
                                 header=False, index=False)
    df_sort.to_csv('data/%s_all_models_%s_NSE_LongStations.txt' % (ESD_var_name_P, season), header=False, index=True)

    # write into stats_list
    stats_list_NSE = stats_list_NSE.append(df_sort.iloc[0:n_models, 0])

    # R2
    models_r2 = []
    for key in best_models.keys():
        models_r2.append(np.array(best_models[key]['R2']['IDs']))
    models_r2 = [y for x in models_r2 for y in x]
    models_counts = Counter(models_r2)
    df = pd.DataFrame.from_dict(models_counts, orient='index')
    df_sort = df.sort_values(by=0, ascending=False, axis=0)

    # write best models into txt
    str_str = df_sort.iloc[0:n_models, 0].index.values
    pd.DataFrame(str_str).to_csv('data/%s_best_%s_models_%s_R2_LongStations.txt' % (ESD_var_name_P, n_models, season),
                                 header=False, index=False)
    df_sort.to_csv('data/%s_all_models_%s_R2_LongStations.txt' % (ESD_var_name_P, season), header=False, index=True)

    # write into stats_list
    stats_list_R2 = stats_list_R2.append(df_sort.iloc[0:n_models, 0])

# write the table of combined annual, winter, and summer NSE into one output file
all_table_NSE = stats_list_NSE.sum(0).sort_values(0, ascending=False)[0:n_models].index
pd.DataFrame(all_table_NSE).to_csv(
    'data/%s_best_%s_models_all_seasons_NSE__LongStations.txt' % (ESD_var_name_P, n_models), header=False, index=False)
all_table_R2 = stats_list_R2.sum(0).sort_values(0, ascending=False)[0:n_models].index
pd.DataFrame(all_table_R2).to_csv(
    'data/%s_best_%s_models_all_seasons_R2__LongStations.txt' % (ESD_var_name_P, n_models), header=False, index=False)
