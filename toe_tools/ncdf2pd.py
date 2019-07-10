import re
from os import listdir

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from pandas.tseries.offsets import *


def max_char_ll(lats, lons):
    # max length of lat/lon strings to make column names for pandas data frame
    nchr_len_max_lat = max([str(i) for i in lats[:]], key=len).__len__()
    nchr_len_max_lon = max([str(i) for i in lons[:]], key=len).__len__()
    nchr_len_max = max(nchr_len_max_lat, nchr_len_max_lon)
    return (nchr_len_max)


def matrix_coords(lats, lons, lon_min, lon_max, lat_min, lat_max, upsidedown=False):
    """

    :param lats: nc lats of pixel centers
    :param lons: nc lons of pixel centers
    :param lon_min:
    :param lon_max:
    :param lat_min:
    :param lat_max:
    :param upsidedown: cruncep has latitudes from high to low !
    :return: latli - lower index; latui - upper; lonli - lon lower; lonui - lon upper
    """
    # lats = ytar
    # lons = xtar
    # lon_min = 100.
    # lon_max = 160.
    # lat_min = 54.
    # lat_max = 70.
    # # #

    # coordinates are pixel center coords. If in the subset procedure a subset coordinate lies in between to center
    # coordinates (exactly on the edge of two adjuscent pixels), chose the first (lower left corner) or the last
    # (upper right corner) match

    # latitude lower and upper index
    if upsidedown:
        latli = (lats.__len__() - np.argmin(np.abs(lats[::-1] - lat_max))) - 1
        latui = np.argmin(np.abs(lats - lat_min))
    else:
        latli = (lats.__len__() - np.argmin(np.abs(lats[::-1] - lat_min))) - 1
        latui = np.argmin(np.abs(lats - lat_max))

    # longitude lower and upper index
    lonli = (lons.__len__() - np.argmin(np.abs(lons[::-1] - lon_min))) - 1
    lonui = np.argmin(np.abs(lons - lon_max))
    # lons[lonui]
    return latli, latui, lonli, lonui


def nc_read_cruncep(nc_file, var_name, subset=None, lon_name='nav_lon', lat_name='nav_lat', time_name='time',
                    header_str_len=False):
    """
    Will need specification for CMIP5 monthly, CRU-NCEP, and other data due to their slightly different
    file structure and variable names.
    CRU-NCEP annual files.

    :param nc_file: filename with path
    :param var_name: 't2m' or any other ncdf variable
    :param subset: list with 4 coordinates: upper left; lower right
    :param lon_name: nc variable name for lons
    :param lat_name: nc variable name for lats
    :param time_name: nc variable name for time
    :param header_str_len: define the column header strring width; in case it should be the same length as for CRUNCEP
    and CMIP5 ... None/or Integer (half width:lat/lon with sign)
    :return:
    """
    # nc_file = '/home/orchideeshare/igcmg/IGCM/SRF/METEO/CRU-NCEP/v7.2/twodeg/cruncep_twodeg_1901.nc'
    # nc_file = '/home/orchideeshare/igcmg/IGCM/SRF/METEO/CRU-NCEP/v7.2/twodeg/cruncep_twodeg_1901.nc'
    # var_name = 'Rainf'
    # lon_name = 'nav_lon'
    # lat_name = 'nav_lat'
    # time_name = 'time'
    #
    # xmin = 100.0
    # xmax = 160.0
    # ymax = 74.0
    # # ymax = 75.0
    # ymin = 54.0
    # subset = [ymax,xmin,ymin, xmax]

    nc = Dataset(nc_file, mode='r')

    ytar = nc.variables[lat_name][:, 0]
    xtar = nc.variables[lon_name][0, :]

    # latitude lower and upper index
    if subset:
        ymax, xmin, ymin, xmax = subset
    else:
        # ymax, xmin, ymin, xmax = [0., 0., 0., 0.]
        ymax, xmin, ymin, xmax = [90., 0., -180., 360.]

    # get matrix coordinates
    latli, latui, lonli, lonui = matrix_coords(lats=ytar, lons=xtar, lon_min=xmin, lon_max=xmax, lat_min=ymin,
                                               lat_max=ymax, upsidedown=True)

    d_time = nc.variables[time_name]

    # write out lats and lons for later use
    # but make sure to account for the reverse order of latitudes AND values
    # ytar[np.arange(latui, (latli-1), -1)] <-- reverse
    # array([55., 57., 59., 61., 63., 65., 67., 69., 71., 73.], dtype=float32)

    latslons_list = [xtar[lonli:(lonui + 1)], ytar[np.arange(latli, latui + 1, 1)]]

    # create needed output for pandas
    # t_step_H = d_time.tstep_sec / 60. / 60.
    # ts_len = d_time.__len__()
    ts_start = d_time.units
    ts_dates = num2date(d_time[:], ts_start)
    dates = pd.to_datetime(ts_dates)

    # pd.datetime(ts_dates[0])
    # pd.date_range('1901-01-01', periods=ts_len, freq='6H')
    data_ss = nc.variables[var_name][:, latli:(latui + 1), lonli:(lonui + 1)]
    # plt.imshow(data_ss[0,:,:])

    # collapse to 2D for pandas
    data_ss_2d = data_ss.swapaxes(2, 0).reshape((data_ss.shape[1] * data_ss.shape[2], data_ss.shape[0])).T

    # create column names based on lat lon values
    if not header_str_len:
        str_len_max = max_char_ll(ytar, xtar)
    else:
        str_len_max = header_str_len

    tmp_str1 = np.tile(latslons_list[1], latslons_list[0].__len__())
    tmp_str2 = np.repeat(latslons_list[0], latslons_list[1].__len__())

    # there might be a more elegant solution to this...
    pixel_ids = [str(format(str1, '%.2d' % str_len_max)) + str(format(str2, '%.2d' % str_len_max)) for str1, str2 in
                 zip(tmp_str1, tmp_str2)]

    pd_dataframe = pd.DataFrame(data_ss_2d, index=dates, columns=pixel_ids)
    nc.close()

    return pd_dataframe


def nc_read_cmip5_ESD(nc_file, var_name, subset=None, lon_name='lon', lat_name='lat', time_name='time',
                      header_str_len=False, fixed_start='1861-01-01'):
    """
    Will need specification for CMIP5 monthly, CRU-NCEP, and other data due to their slightly different
    file structure and variable names.
    CMIP5 differences in folder structure between ESD and CICLAD

    :param nc_file: filename with path
    :param var_name: 't2m' or any other ncdf variable
    :param subset: list with 4 coordinates: upper left; lower right
    :param lon_name: nc variable name for lons
    :param lat_name: nc variable name for lats
    :param time_name: nc variable name for time
    :param header_str_len: define the column header strring width; in case it should be the same length as for CRUNCEP
    and CMIP5 ... None/or Integer (half width:lat/lon with sign)
    :return:
    """
    #
    # PATH_IN = "/home/hydrogeol/epohl/data/ESD/data/siberia_ss/rcp85/"
    # # files = [f for f in listdir(PATH_IN) if re.match(r'[aA-zZ]+.*\.nc$', f)]
    # files = [f for f in listdir(PATH_IN) if re.match(r'^tas+.*\.nc$', f)]
    # # order
    # files.sort()
    #
    # nc_file = '/home/hydrogeol/epohl/data/ESD/data/siberia_ss/rcp85/tas_Amon_ens_rcp85_000.nc'
    # nc_file = mod_i['path_hist'] + file__hist_i[0]
    # var_name = 'tas'
    # lon_name = 'lon'
    # lat_name = 'lat'
    # time_name = 'time'
    # header_str_len = 6
    #
    # xmin = 100.0
    # xmax = 160.0
    # ymax = 74.0
    # ymax = 75.0
    # ymin = 52.0
    # #
    # # xmin = 100.0
    # # xmax = 160.0
    # # ymax = 74.0
    # # ymin = 52.0
    # subset = [ymax,xmin,ymin, xmax]

    nc = Dataset(nc_file, mode='r')

    ytar = nc.variables[lat_name][:]
    xtar = nc.variables[lon_name][:]

    # latitude lower and upper index
    if subset:
        ymax, xmin, ymin, xmax = subset
    else:
        ymax, xmin, ymin, xmax = [0., 0., 0., 0.]

    # get matrix coordinates
    latli, latui, lonli, lonui = matrix_coords(lats=ytar, lons=xtar, lon_min=xmin, lon_max=xmax, lat_min=ymin,
                                               lat_max=ymax)

    d_time = nc.variables[time_name]

    # write out lats and lons for later use
    # xy_minmax = [xtar[:][lonli], xtar[:][lonui], ytar[:][latli], ytar[:][latui]]
    latslons_list = [xtar[lonli:(lonui + 1)], ytar[latli:(latui + 1)]]

    # if fixed_start is set, assume that all models have the same start date and same time increment
    if fixed_start:
        ts_dates = pd.date_range(fixed_start, periods=len(d_time), freq='1MS')

    else:

        # create needed output for pandas
        # t_step_H = d_time.tstep_sec / 60. / 60.
        # ts_len = d_time.__len__()
        ts_start = d_time.units

        if ts_start.find('days') >= 0:
            ts_dates = num2date(d_time[:], ts_start)
        else:
            match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', ts_start)
            ts_dates = pd.date_range(match[0], periods=len(d_time), freq='1MS') + DateOffset(months=(d_time[0] - 1))

    dates = pd.to_datetime(ts_dates)

    # pd.datetime(ts_dates[0])
    # pd.date_range('1901-01-01', periods=ts_len, freq='6H')
    data_ss = nc.variables[var_name][:, latli:(latui + 1), lonli:(lonui + 1)]
    # data_ss = nc.variables[var_name][:, latui:latli, lonli:lonui]
    # plt.imshow(data_ss[0,:,:])

    # collapse to 2D for pandas
    data_ss_2d = data_ss.swapaxes(2, 0).reshape((data_ss.shape[1] * data_ss.shape[2], data_ss.shape[0])).T

    # create column names based on lat lon values
    if not header_str_len:
        str_len_max = max_char_ll(ytar, xtar)
    else:
        str_len_max = header_str_len

    tmp_str1 = np.tile(latslons_list[1], latslons_list[0].__len__())
    tmp_str2 = np.repeat(latslons_list[0], latslons_list[1].__len__())

    # there might be a more elegant solution to this...
    pixel_ids = [str(format(str1, '%.2d' % str_len_max)) + str(format(str2, '%.2d' % str_len_max)) for str1, str2 in
                 zip(tmp_str1, tmp_str2)]

    pd_dataframe = pd.DataFrame(data_ss_2d, index=dates, columns=pixel_ids)
    nc.close()

    return pd_dataframe


def merge_hist_fut_cmip5(var_name, mainpath, merge_path, subset, header_str_len=6):
    # var_name = 'tas'
    # mainpath = PATH_IN
    # merge_path = PATH_CICLAD
    # header_str_len = 6
    import os
    model_dirs = os.walk(mainpath).next()[1]
    historical_dir = [mainpath + i + '/historical/' for i in model_dirs]
    future_dir = [mainpath + i + '/rcp85/' for i in model_dirs]
    pd_models = pd.DataFrame(np.array([model_dirs, historical_dir, future_dir]).T,
                             columns=['model', 'path_hist', 'path_fut'])
    pd_models.to_csv(mainpath + 'CMIP5_models.csv')

    # and select all models that are complete to generate a model mean
    all_models_list = dict()
    # files_hist = [listdir(f) for f in pd_models['path_hist'].values if re.match(r'^%s+.*\.nc$'% var_name, f)]
    for i in pd_models.index.values:
        mod_i = pd_models.loc[i, :]
        file__hist_i = [f for f in listdir(mod_i['path_hist']) if re.match(r'^%s+.*\.nc$' % var_name, f)]
        file__fut_i = [f for f in listdir(mod_i['path_fut']) if re.match(r'^%s+.*\.nc$' % var_name, f)]
        if (file__hist_i.__len__() > 0) & (file__fut_i.__len__() > 0):
            print ('found simulation with historical and future data: %s  %s' % (file__hist_i, file__fut_i))
            mod_hist = nc_read_cmip5_ESD(mod_i['path_hist'] + file__hist_i[0], var_name=var_name, subset=subset,
                                         fixed_start=False, header_str_len=header_str_len)

            #                             fixed_start='2006-01-01', header_str_len= header_str_len)
            mod_fut = nc_read_cmip5_ESD(mod_i['path_fut'] + file__fut_i[0], var_name=var_name, subset=subset,
                                        fixed_start=False, header_str_len=header_str_len)

            mod_full = mod_hist.append(mod_fut)

            dfi_mon_pd = mod_full.resample('MS').apply(np.mean)
            dfi_ann_pd = mod_full.resample('A').apply(np.mean)

            model_id = format(i, '03d')
            tx_min = dfi_ann_pd.index.min().year
            tx_max = dfi_ann_pd.index.max().year

            dfi_ann_pd.to_csv(
                merge_path + "%s_CICLAD_%s_%s-%s_Siberia_df_annual-orig.csv" % (var_name, model_id, tx_min, tx_max))
            dfi_mon_pd.to_csv(
                merge_path + "%s_CICLAD_%s_%s-%s_Siberia_df_monthly-orig.csv" % (var_name, model_id, tx_min, tx_max))

            all_models_list[model_id] = dfi_mon_pd
    return all_models_list
