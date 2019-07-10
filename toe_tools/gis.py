# collection to organize the output panda dataframes into netcdf files for plotting
# from netCDF4 import Dataset, num2date, date2num, date2index
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# # this is a tweak to fix basemap if it doesn't find a needed path
# # the following might fix it - try the below import of Basemap first
# # if it works you do not need the fix
# # for the fix you have to find the 'epsg' file that contains the projecion information needed by Basemap
# env_path = '../anaconda2/envs/ToE_tools/share/'
# proj_lib = os.path.join(env_path, 'proj')
# os.environ["PROJ_LIB"] = proj_lib

# try again if you can import Basemap without getting an error:


def read_csv(filename):
    x = pd.read_csv(filename, index_col=0)
    # x.index = pd.to_datetime(x.index)
    return x


def quick_plot(pd_df, layer_n=0, header_str_len=6):
    """
    plot lons vs lats and values from first or layer_n layer (index, e.g. 2005)
    :param pd_df: any of the pd_df that has the lat_lon column headers
    :param layer_n: 0 or any other layer index
    :param header_str_len: half size of column names
    :return:
    """
    ll = pd_df.keys()
    ind = pd_df.index
    lats = [float(li[0:header_str_len]) for li in ll]
    lons = [float(li[header_str_len:]) for li in ll]
    v0 = pd_df.ix[ind[layer_n]].values
    plt.scatter(lons, lats, c=v0)


def pd_series_to_geoarray(filename):
    """
    reads csv of indovidual file and outputs a disct with: nd-array and associated infos on lats, lons
    :param filename:
    :return: dict
    """

    a = read_csv(filename)
    # a.replace('nan', np.NaN)
    header = a.keys()
    # indices = a.index

    nchar = header[0].__len__()
    # nch_half = int(nchar / 2)

    lats = [float(s[0:int(nchar / 2)]) for s in header]
    lons = [float(s[int(nchar / 2) + 1:]) for s in header]

    # lats_str = [(s[0:int(nchar/2)]) for s in header]
    # lons_str = [(s[int(nchar/2)+1:]) for s in header]

    lats_unq = np.sort(np.unique(lats))
    lons_unq = np.sort(np.unique(lons))

    # output array
    zxy = np.zeros((a.index.__len__(), lats_unq.__len__(), lons_unq.__len__()))
    # xy = np.zeros((lats_unq.__len__(), lons_unq.__len__()))

    # cnt_h = 0
    for h in np.arange(header.__len__()):
        rc = header[h]
        # h_lat = lats_str[h]
        # h_lon = lons_str[h]

        h_latv = lats[h]
        h_lonv = lons[h]

        # r_m = lats_unq[(lats_unq == h_latv)]
        # c_m = lons_unq[(lons_unq == h_lonv)]
        r_m = np.where(lats_unq == h_latv)[0]
        c_m = np.where(lons_unq == h_lonv)[0]

        zxy[:, r_m, c_m] = a[rc].values.reshape((-1, 1))
    out = {'array': zxy, 'lats': lats, 'lons': lons, 'lats_unq': lats_unq, 'lons_unq': lons_unq, 'header': header}
    return out


def pd_toe_to_geoarray(input_array, nan_mask, model_IDs=None, sign=None):
    """
    reads csv of hellinger distances and outputs a disct with: nd-array and associated infos on lats, lons
    :param input_array: the derived time of emergence using a certain confidence level
    :param nan_mask: (e.g. valmax) the derived value >= Confidence Level OR the maximum value in case the CI was not reached
    :param model_IDs: list of model IDs that shall be selected. Their names are the column names
    :param sign: the sign dataframe that shows 1 or -1 , depending on whether Hellinger has moved to the left or right
    :return: dict
    """
    # varname = 'Tair'
    # season = 'winter'
    # PATH_CRU = '/home/hydrogeol/epohl/data/CRU-NCEP/%s_overlap_Lena_hellinger/fullSeries/' % varname
    # confidence_level = 0.95  # confidence level at which we assume ToE
    # # confidence_level = 0.50  # confidence level ToE
    # confidence_level = 0.40  # confidence level ToE
    # filename_toe = PATH_CRU + '%s_CRU-NCEP_ToE_Sensitivity-timemax_%s_1901-2016_Siberia_df_annual_overlap_.csv' % (
    # varname, confidence_level)
    # filename_valmax = PATH_CRU + '%s_CRU-NCEP_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_annual_overlap_.csv' % (
    # varname, confidence_level)
    # filename_sign = PATH_CRU + '%s_CRU-NCEP_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_annual_overlap_sign.csv' % (
    # varname, confidence_level)
    #
    # input_array = filename_toe
    # nan_mask = filename_valmax
    # sign = filename_sign

    global asig

    def read_csv_(filename):
        x = pd.read_csv(filename, index_col=0)
        # x.index = pd.to_datetime(x.index)
        return x

    atoe = read_csv_(input_array)
    avmx = read_csv_(nan_mask)

    if sign:
        asig = read_csv(sign)
        # atoe.mul(asig)['041']
        atoe_pn = atoe.mul(asig)
        avmx_pn = avmx.mul(asig)
    else:
        atoe_pn = atoe
        avmx_pn = avmx

    # subset atoe and avmax if model IDs for subsetting are supplied
    if model_IDs:
        model_IDs = np.array(model_IDs)
        atoe = atoe[model_IDs]
        atoe_pn = atoe_pn[model_IDs]
        avmx = avmx[model_IDs]
        avmx_pn = avmx_pn[model_IDs]

    # set ToE to nan if valmax is nan (ToE values result from copy of existing dataset but they need to be removed)
    nan_index = avmx.index[avmx.isnull().all(1)]
    atoe.loc[nan_index, :] = np.nan
    atoe_pn.loc[nan_index, :] = np.nan

    # create lists with model ID and lat/lon (header)
    models = atoe.keys()
    header = atoe.index

    nchar = header[0].__len__()

    lats = [float(s[0:int(nchar / 2)]) for s in header]
    lons = [float(s[int(nchar / 2) + 1:]) for s in header]

    lats_unq = np.sort(np.unique(lats))
    lons_unq = np.sort(np.unique(lons))

    # output array
    zxy = np.zeros((models.__len__(), lats_unq.__len__(), lons_unq.__len__()))
    zxy[:] = np.nan
    zxy2 = copy.copy(zxy)
    # xy = np.zeros((lats_unq.__len__(), lons_unq.__len__()))

    # cnt_h = 0
    for h in np.arange(header.__len__()):
        rc = header[h]
        # h_lat = lats_str[h]
        # h_lon = lons_str[h]

        h_latv = lats[h]
        h_lonv = lons[h]

        # r_m = lats_unq[(lats_unq == h_latv)]
        # c_m = lons_unq[(lons_unq == h_lonv)]
        r_m = np.where(lats_unq == h_latv)[0]
        c_m = np.where(lons_unq == h_lonv)[0]

        zxy[:, r_m, c_m] = atoe.ix[rc].values.reshape((-1, 1))
        zxy2[:, r_m, c_m] = atoe_pn.ix[rc].values.reshape((-1, 1))
    out = {'array': zxy, 'array_pn': zxy2, 'sign': asig,
           'lats': lats, 'lons': lons, 'lats_unq': lats_unq,
           'lons_unq': lons_unq, 'header': header, 'models': models}
    return out


def calc_toe_variability(toe_dict):
    """
    calculate variability in terms of:
    -standard deviation
    -mean
    -median
    -total value range
    :param toe_dict: output from either 'pd_series_to_geoarray' or 'pd_toe_to_geoarray'
    :return: array (x,y)
    """
    # array.keys()
    # a = t_pd['array']
    # acp = copy.copy(t_pd)
    acp = copy.copy(toe_dict)

    acp['std'] = np.std(acp['array'], axis=0)
    acp['mean'] = np.mean(acp['array'], axis=0)
    acp['median'] = np.median(acp['array'], axis=0)
    acp['min'] = np.min(acp['array'], axis=0)
    acp['max'] = np.max(acp['array'], axis=0)
    acp['range'] = (acp['max'] - acp['min'])
    return acp


def calc_width_heigth(latmin, latmax, lonmin, lonmax):
    from math import sin, cos, sqrt, atan2, radians

    # earth radius
    r = 6373.0
    lat1 = radians(latmin)
    lon1 = radians(lonmax)
    lat2 = radians(latmax)
    lon2 = radians(lonmin)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    w_low = cos(lat1) ** 2 * sin(dlon / 2) ** 2
    w_hig = cos(lat2) ** 2 * sin(dlon / 2) ** 2
    h_lef = sin(dlat / 2) ** 2

    c_w_low = 2 * atan2(sqrt(w_low), sqrt(1 - w_low))
    c_w_hig = 2 * atan2(sqrt(w_hig), sqrt(1 - w_hig))
    c_h_lef = 2 * atan2(sqrt(h_lef), sqrt(1 - h_lef))

    distance_w_low = r * c_w_low
    distance_w_hig = r * c_w_hig
    distance_h_lef = r * c_h_lef
    width_max = int(np.max((distance_w_low, distance_w_hig)) * 1000)
    heigth = int(distance_h_lef * 1000)
    return width_max, heigth


def plot_map_negpos(toe_dict, key_name, dataset_name='CMIP5', z_range=None, lon_inc=8, lat_inc=4,
                    cmap_name1='jet', cmap_name2=None, layer=None, n_colors=10, save=True, pickled=True,
                    skip_rounding=False, omit_title=False, omit_cbar=False):
    """
    same as plot_map but using two colormaps to plot negative AND positive values with two different colorscales
    -
    Use either of the dictionaries and plot the array in the key slots. If an array with several layers is handed,
    one can specify which layer shall be selected (default = 0)
    :param toe_dict:  e.g. 'calc_toe_variability'
    :param key_name: e.g. 'mean'
    :param dataset_name: to be included in the filename
    :param layer: 0 or any other layer (1st dimension of the array)
    :param z_range: min and max
    # :param vmin:
    # :param vmax:
    :param lon_inc:
    :param lat_inc:
    :param save: T/F save image ?
    :param n_colors: n colors for classification
    :param pickled: read from pickled basemap instance
    :return: plot of map
    """
    # toe_dict = t_pd
    # # key_name = 'array'
    # dataset_name = 'asdasdasdCMIP5'
    # save = True
    # key_name = 'mean'
    # # layer = 2
    # layer = None
    # # cmap_name = 'jet'
    # # cmap_name1 = 'inferno'
    # # cmap_name1 = 'BrBG'
    #
    # cmap_name1 = 'PRGn'
    # cmap_name1 = 'magma'
    # cmap_name2 = None
    # # cmap_name1 = 'GnBu_r'
    # # cmap_name1 = 'BuPu_r'
    # # cmap_name2 = 'RdPu'
    # # cmap_name2 = 'YlOrRd'
    # # vmin, vmax = z_range
    # z_range = [2000,2100]
    # # vmin = 10
    # # vmax = 18
    # lon_inc = 8
    # lat_inc = 4
    # n_colors = 10
    # pickled = True

    from mpl_toolkits.basemap import Basemap
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.patheffects as pe
    import matplotlib.colors as mcolors
    import pickle
    # import os

    def plot_point_wtext(shpname, text, zorder):
        p = m.readshapefile(shpname, '')
        p1_x = p[2][0]
        p1_y = p[2][1]
        x, y = m(p1_x, p1_y)
        plt.text(x + 0.01 * width_max, y + 0.01 * heigth, text,
                 path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                 fontsize=8, zorder=zorder)
        m.plot(x, y, 'ko', markersize=2, zorder=zorder - 1)
        m.plot(x, y, 'wo', markersize=4, zorder=zorder - 2)

    def plot_point_wtext_latlon(shpname, text, zorder):
        p = m.readshapefile(shpname, '')
        p1_x = p[2][0]
        p1_y = p[2][1]
        x, y = m(p1_x, p1_y)
        plt.text(x + 0.01, y + 0.01, text, path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                 fontsize=8, zorder=zorder)
        m.plot(x, y, 'ko', markersize=2, zorder=zorder - 1)
        m.plot(x, y, 'wo', markersize=4, zorder=zorder - 2)

    if layer:
        z = toe_dict[key_name][layer, :, :]
    else:
        z = toe_dict[key_name]
    # z.shape

    if z_range:
        vmin, vmax = z_range
    elif cmap_name2:
        vmin, vmax = [-1 * np.max(np.abs([np.nanmin(z), np.nanmax(z)])), np.max(np.abs([np.nanmin(z), np.nanmax(z)]))]
    else:
        vmin, vmax = np.abs([np.nanmin(z), np.nanmax(z)])

    # make mesh with one mor lon and lat for colormesh (in between points)
    lat_unq = toe_dict['lats_unq']
    lat_d = lat_unq[1] - lat_unq[0]
    lon_unq = toe_dict['lons_unq']
    lon_d = lon_unq[1] - lon_unq[0]
    # center to bb coordinates
    lons = [(ll - (0.5 * lon_d)) for ll in lon_unq]
    lons.append(lon_unq[-1] + (0.5 * lon_d))
    lats = [(ll - (0.5 * lat_d)) for ll in lat_unq]
    lats.append(lat_unq[-1] + (0.5 * lat_d))

    toe_dict['lon_mesh'], toe_dict['lat_mesh'] = np.meshgrid(lons, lats)
    toe_dict['lon_mesh_gc'], toe_dict['lat_mesh_gc'] = np.meshgrid(toe_dict['lons_unq'], toe_dict['lats_unq'])
    # lons.__len__()

    lonmin, lonmax, latmin, latmax = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]
    lonmean = np.mean((lonmin, lonmax))
    latmean = np.mean((latmin, latmax))
    l0 = latmin
    l1 = latmean
    l2 = latmax

    # get width and height approximation for lcc projection
    width_max, heigth = calc_width_heigth(latmin, latmax, lonmin, lonmax)
    plot_w = width_max + 0.2 * width_max
    plot_h = heigth + 0.2 * heigth

    col_water = '#aabbff'

    # PLOT
    # colorbars and set color for NANs
    if cmap_name1 and cmap_name2:
        colors1 = plt.cm.get_cmap(cmap_name1)(np.linspace(0., 1, n_colors / 2))
        colors2 = plt.cm.get_cmap(cmap_name2)(np.linspace(0., 1, n_colors / 2))

        # combine them and build a new colormap
        colors = np.vstack((colors1, colors2))
    else:
        colors = plt.cm.get_cmap(cmap_name1)(np.linspace(0., 1, n_colors))
    cmap_magma = mcolors.LinearSegmentedColormap.from_list('cmap_magma', colors, N=n_colors)
    cmap_magma.set_bad(color='grey', alpha=0)

    # bounds and norms for colorbars
    bounds_mean = np.linspace(vmin, vmax, n_colors + 1)
    norm_mean = mpl.colors.BoundaryNorm(bounds_mean, ncolors=n_colors)

    fig = plt.figure(figsize=(5, 4))
    if not pickled:

        # # suitable for high latitudes and medium/large-sized area ; but not circumpolar
        # m = Basemap(projection='lcc', lon_0=lonmean, width=plot_w, height=plot_h, resolution='i',
        #             lat_0=l1, lat_1=l2)

        # simple lat lon
        m = Basemap(llcrnrlat=latmin, urcrnrlat=latmax, llcrnrlon=lonmin, urcrnrlon=lonmax, resolution='i')

        pickle.dump(m, open('map.pickle', 'wb'), -1)  # pickle it
    else:
        m = pickle.load(open('map.pickle', 'rb'))  # load here the above pickle

    m.drawparallels(np.arange(latmin, latmax + latmax * 0.01, lat_inc), labels=[1, 0, 0, 0], linewidth=0.5, zorder=100)
    m.drawmeridians(np.arange(lonmin, lonmax + lonmax * 0.01, lon_inc), labels=[0, 0, 0, 1], linewidth=0.5, zorder=99)

    x, y = m(toe_dict['lon_mesh_gc'], toe_dict['lat_mesh_gc'])
    # cs = m.contour(x, y, toe_dict['posneg'] * 100, 4, linewidths=1.5, cmap=plt.cm.get_cmap('RdBu_r'), labels='cont',
    #                zorder=97)
    # cs = m.contour(x, y, toe_dict['posneg'] * 100, np.array([0, .25, .50, .75, 1]), linewidths=1.5,
    #                cmap=plt.cm.get_cmap('RdBu_r'), labels='cont', zorder=97)

    # instead give the pixels a symbol based on their percentage
    # posneg_perc = (toe_dict['posneg_na'] + 1.) / 2.
    # posneg_perc[posneg_perc == 1.] = np.nan

    # take the ref signs instead
    posneg_perc = toe_dict['posneg_ref'] * 1.
    posneg_perc[posneg_perc == 1.] = np.nan
    colors = plt.cm.get_cmap('RdBu_r')(posneg_perc.reshape((-1,)))
    colors[:, :] = [0, 0, 1, 1]
    colors[np.isnan(posneg_perc.reshape((-1,)))] = [0, 0, 0, 0]
    # m.scatter(x,y, marker='o',  s=posneg_perc.reshape((-1,))*100, facecolors=[0,0,0,0],
    # edgecolor=colors, zorder=97, linewidths=1.5)
    # m.scatter(x, y, marker='o', s=2, facecolors=[0, 0, 0, 0], edgecolor=colors, zorder=97, linewidths=1.5)
    m.scatter(x, y, marker='o', s=2, facecolors=[0, 0, 0, 0], edgecolor=colors, zorder=97, linewidths=1.5)
    # plt.clabel(cs, inline=1, fontsize=7, fmt='%3d', inline_spacing=0.5, zorder=98)

    shp_name = 'data/yakutsk'
    # plot_point_wtext(shp_name, 'Yakutsk', zorder=96)
    plot_point_wtext_latlon(shp_name, 'Yakutsk', zorder=101)

    # Use the meteorological stations in the Lena Catchment to check between CRU and CMIP5 hellinger distances
    m.readshapefile('data/lena', 'lena', linewidth=.5, color='#222222', zorder=95)

    m.drawcoastlines(linewidth=0.5, color='#999999', zorder=93)
    m.drawcountries(linewidth=0.5, color='#999999', zorder=92)
    # m.shadedrelief(scale=0.1, alpha=.3, zorder=91 , )

    cmap_magma.set_over(color='black', alpha=0)
    m.pcolormesh(toe_dict['lon_mesh'], toe_dict['lat_mesh'],
                 z, latlon=True, cmap=cmap_magma, norm=norm_mean, rasterized=False,
                 zorder=-1)  # zorder=p_cnt.__next__())

    # m.drawmapboundary(linewidth=0.5, color='grey',fill_color=col_water, zorder=p_cnt.__next__())
    m.drawmapboundary(linewidth=0.5, color='grey', fill_color=col_water, zorder=-2)
    # m.fillcontinents(color='#aaaaaa', lake_color=col_water, zorder=p_cnt.__next__()-10)
    m.fillcontinents(color='#aaaaaa', lake_color=col_water, zorder=-3)

    # decoration
    if not omit_title:
        plt.title(key_name, fontsize=14)
    plt.annotate(dataset_name, xy=(0.01, 0.95), xycoords='axes fraction', zorder=110)
    if not omit_cbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cbar = plt.colorbar(cax=cax)
        # cbar = plt.colorbar(shrink=.5)
        vinc = (vmax - vmin) / 10.
        cbar.set_ticks(np.arange(vmin, vmax + vinc, vinc))

        # built in option to skip the rounding
        if not skip_rounding:
            if (vmax - vmin) > 5:
                # cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in np.arange(vmin, vmax + vinc, vinc)])
                tlabs = ['{:.0f}'.format(x) for x in cbar.get_ticks()]
                if np.unique(tlabs).__len__() < tlabs.__len__():
                    tlabs = ['{:.1f}'.format(x) for x in cbar.get_ticks()]
                cbar.ax.set_yticklabels(tlabs)
            else:
                # cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in np.arange(vmin, vmax + vinc, vinc)])
                cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in cbar.get_ticks()])
        else:
            cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in cbar.get_ticks()])

    if save:
        pfile = 'figures/toe_%s_%s.pdf' % (dataset_name, key_name)
        fig.savefig(pfile, dip=200, bbox_inches='tight')
        # os.system('pdfcrop ' + pfile + ' ' + pfile)
        plt.close()
    #


def calc_array_stats(array, model, ref_year_or_model=u'1921-21', annualoverview=None):
    """
    calc the stats in one step for more convinience
    :param array: t_pd input array - from read_geo_array
    :param model: 'CRUNCEP' or 'CMIP5'
    :param ref_year_or_model: the split-year/ww or model identifiers
    :param annualoverview: to produce annual HD maps
    :return: all calculated stats
    """
    # array = t_pd
    # ref_year_or_model = u'1921-21'

    array['std'] = np.nanstd(array['array'], axis=0)
    array['median'] = np.nanmedian(array['array'], axis=0)
    array['mean'] = np.nanmean(array['array'], axis=0)
    array['min'] = np.nanmin(array['array'], axis=0)
    array['max'] = np.nanmax(array['array'], axis=0)
    array['posneg'] = np.mean(np.sign(array['array_pn']), axis=0)
    array['posneg_na'] = np.nanmean(np.sign(array['array_pn']), axis=0)

    # get the standard version WW=21, SPLIT YEAR= 1921
    if (model == 'CRUNCEP') and (annualoverview is None):
        ind_ref_model = np.where(array['models'] == ref_year_or_model)[0]
        array['ref'] = array['array'][ind_ref_model, :, :].reshape(array['median'].shape)
        array['posneg_ref'] = np.sign(array['array_pn'][ind_ref_model, :, :].reshape(array['median'].shape))
    elif (model == 'CMIP5') and (annualoverview is None):
        array['ref'] = array['median']  # for CMIP5 take the models median
        array['posneg_ref'] = np.sign(array['median']).reshape(array['median'].shape)

    # in case of annual overview maps: use the same procedure as for CRUNCEP but take the HD overlap as main input
    # and the same time slice sign as signature
    # annualoverview = 1960
    if annualoverview:
        ind_ref_model = np.where(array['models'] == str(annualoverview))[0]
        array['ref'] = array['array'][ind_ref_model, :, :].reshape(array['median'].shape)
        array['posneg_ref'] = np.sign(array['array_pn'][ind_ref_model, :, :].reshape(array['median'].shape))

    return array
