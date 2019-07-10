# V6 is using data from ../data/ESD/txt_input/bias_corrected_annualOffset/
# these data are corrected(bias) by calculating the average annual offset between the CMIP5 models and CRUNCEP
# V5 is based on offset caluclation based on mean monthly data ... difficult to justify .. still all is
# using the ESD CMIP5 collection ........
import re
from copy import copy

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


# START Functions

def get_unique_stations(pd_stations_dataframe, reference_pixels):
    lats = pd_stations_dataframe['lat']
    lons = pd_stations_dataframe['lon']
    # find closest pixels mathcing the center coordinates in CRUNCEP
    header = reference_pixels
    nchar = header[0].__len__()
    cr_lats = [float(s[0:int(nchar / 2)]) for s in header]
    cr_lons = [float(s[int(nchar / 2) + 1:]) for s in header]
    lats_close = []
    lons_close = []
    for l in lats:
        lats_close.append(min(cr_lats, key=lambda x: abs(x - l)))
    for l in lons:
        lons_close.append(min(cr_lons, key=lambda x: abs(x - l)))
    # format
    pixel_ids = [str(format(str1, '%.2d' % int(nchar / 2))) + str(format(str2, '%.2d' % int(nchar / 2))) for str1, str2
                 in zip(lats_close, lons_close)]
    pd_stations_dataframe['pixel_ID'] = pixel_ids
    pd_stations_dataframe = pd_stations_dataframe[~pd_stations_dataframe['pixel_ID'].duplicated()]
    return pd_stations_dataframe


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def nashsutcliffe(pd_df):
    o = pd_df.iloc[:, 0]
    s = pd_df.iloc[:, 1]
    ns = 1 - sum((s - o) ** 2) / sum((o - np.mean(o)) ** 2)
    return ns


def move_element(odict, thekey, newpos):
    odict[thekey] = odict.pop(thekey)
    i = 0
    for key, value in odict.items():
        if key != thekey and i >= newpos:
            odict[key] = odict.pop(key)
        i += 1
    return odict


# version2 with half window sizes at the end
def window_pad(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable
     with extension of half ww to left and right
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   """
    x_seq_or = seq[:]
    x_max = x_seq_or[-1]
    x_min = 0

    for j in x_seq_or:
        # print(j)
        x0 = x_seq_or[j] - int(n / 2)
        x1 = x_seq_or[j] + int(n / 2)
        if x0 < x_min:
            x0 = x_min
        if x1 > x_max:
            x1 = x_max
        seq_j = np.arange(x0, x1 + 1, 1)
        yield tuple(seq_j)


# version2 but for Pandas Series with half window sizes at the end
def window_pad_pd(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable
     with extension of half ww to left and right
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   """

    # for x in window_pad(time_seq_num, ww):
    #     # tmp_KDEpdf_target = gaussian_kde(values[[x]]).pdf(tmp_xseq)
    #     tmp_KDEpdf_target = np.histogram(y_tar_ww[x])

    # seq = time_seq_num
    # n = ww

    x_seq_or = seq[:]
    x_max = x_seq_or[-1]
    x_min = x_seq_or[0]

    for j in x_seq_or:
        # print(j)
        x0 = j - int(n / 2)
        x1 = j + int(n / 2)
        if x0 < x_min:
            x0 = x_min
        if x1 > x_max:
            x1 = x_max
        seq_j = np.arange(x0, x1 + 1, 1)
        yield tuple(seq_j)


# get value range of an array or list and return: min, max, value spread
def x_range(x, stretch=1):
    '''
    :param x: array
    :param stretch: to de(in)crease the min(max) by x times the value difference of the two
    :return:
    '''
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    xrange = xmax - xmin
    xlw = xmin - (stretch * xrange)
    xup = xmax + (stretch * xrange)
    return [xlw, xup, (xup - xlw)]


def read_csv(filename):
    """ for reading monthly output pd csv files with format:
    YYYY-MM-DD
    """
    x = pd.read_csv(filename, index_col=0)
    x.index = pd.to_datetime(x.index)
    return x


def read_csv_ann(filename):
    """ for reading monthly output pd csv files with format:
    YYYY-MM-DD
    """
    x = pd.read_csv(filename, index_col=0)
    # x.index = pd.to_datetime(x.index)
    return x


def calc_run_hellinger(time_seq, values, ref_values, ww, tmp_xseq):
    '''
    :param values: temeprature values
    :param ww: window width
    :param ref_values: the values of the reference period
    :param tmp_xseq: the series of x-values for which the PDF is evaluted for
    :return: PDF for each time step used
    '''
    # make time_seq indexed to be used with window width
    time_seq_num = np.arange(time_seq.shape[0])

    _SQRT2 = np.sqrt(2)

    def hellinger(p, q):
        return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2

    if ww % 2 == 0:
        ww = ww + 1

    local_f = list()

    for x in window_pad(time_seq_num, ww):
        tmp_KDEpdf_target = gaussian_kde(values[[x]]).pdf(tmp_xseq)
        tmp_pdf_target = tmp_KDEpdf_target / np.sum(tmp_KDEpdf_target)
        local_f.append(hellinger(p=tmp_pdf_target, q=ref_values))
    return local_f


def calc_run_hellinger_posneg(time_seq, values, ref_values, ww, tmp_xseq):
    '''
    :param values: temeprature values
    :param ww: window width
    :param ref_values: the values of the reference period
    :param tmp_xseq: the series of x-values for which the PDF is evaluted for
    :return: PDF for each time step used
    '''
    # make time_seq indexed to be used with window width
    time_seq_num = np.arange(time_seq.shape[0])

    _SQRT2 = np.sqrt(2)

    # def hellinger(p, q):
    #     return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2

    def hellinger(p, q, bins):
        """

        :param p: probabilities of target distrbution
        :param q: probabilities of reference distrbution
        :param bins: bins / x-values associated to both distributions
        :return:
        """
        # bins = x1
        # p = p1
        # q = p2
        sp = np.sum(bins * p)
        sq = np.sum(bins * q)
        sign = np.sign(sp - sq)
        helld = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2
        return helld, sign

    if ww % 2 == 0:
        ww = ww + 1

    local_f = list()
    local_sign = list()

    for x in window_pad(time_seq_num, ww):
        tmp_KDEpdf_target = gaussian_kde(values[[x]]).pdf(tmp_xseq)
        tmp_pdf_target = tmp_KDEpdf_target / np.sum(tmp_KDEpdf_target)
        tmp_h, tmp_s = hellinger(p=tmp_pdf_target, q=ref_values, bins=tmp_xseq)
        local_f.append(tmp_h)
        local_sign.append(tmp_s)

    return local_f, local_sign


def calc_run_hellinger_posneg_hist(time_seq, tar_values, ref_values, ww, tmp_xseq, bin_centers):
    """
    :param time_seq: indices (pd format) of pd dataframe
    :param tar_values: temeprature values of target
    :param ww: window width
    :param ref_values: the values of the reference period
    :param tmp_xseq: the series of x-values for which the PDF is evaluted for
    :param bin_centers: as tmp_xseq (the limits of bins) but for the centers (n-1)
    :return: PDF for each time step used
    """

    # time_seq = y_times_tar_ww
    # tar_values = y_tar_ww
    # ref_values = ref_pdf_list
    # ww = ww
    # tmp_xseq = tmp_xseq
    # bin_centers = tmp_xseq_centers

    # make time_seq indexed to be used with window width
    # time_seq_num = np.arange(time_seq.shape[0])
    time_seq_num = time_seq

    _SQRT2 = np.sqrt(2)

    # def hellinger(p, q):
    #     return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2

    def hellinger_hist(p, q, bins):
        """

        :param p: probabilities of target distrbution
        :param q: probabilities of reference distrbution
        :param bins: bins center locations / x-values associated with both distributions
        :return:
        """
        # bins = x1
        # p = p1
        # q = p2
        sp = np.sum(bins * p)
        sq = np.sum(bins * q)
        sign = np.sign(sp - sq)
        helld = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2
        return helld, sign

    if ww % 2 == 0:
        ww = ww + 1

    local_f = list()
    local_sign = list()

    for x in window_pad_pd(time_seq_num, ww):
        tar_h_freq, tar_h_div = np.histogram(tar_values.loc[list(x)], bins=tmp_xseq)
        tmp_pdf_list = tar_h_freq / (np.sum(tar_h_freq) * 1.0)
        tmp_h, tmp_s = hellinger_hist(p=tmp_pdf_list, q=ref_values, bins=bin_centers)
        local_f.append(tmp_h)
        local_sign.append(tmp_s)

    return local_f, local_sign


def calc_run_hellinger_hist(time_seq, tar_values, ref_values, ww, tmp_xseq, bin_centers):
    """
    :param tar_values: temeprature values of target
    :param ww: window width
    :param ref_values: the values of the reference period
    :param tmp_xseq: the series of x-values for which the PDF is evaluted for
    :param: bin_centers: as tmp_xseq (the limits of bins) but for the centers (n-1)
    :param bin_centers: as tmp_xseq (the limits of bins) but for the centers (n-1)
    :return: PDF for each time step used
    """
    # time_seq = y_times_tar_ww
    # tar_values = y_tar_ww
    # ref_values = ref_pdf_list
    # ww = ww
    # tmp_xseq = tmp_xseq
    # bin_centers = tmp_xseq_centers

    # make time_seq indexed to be used with window width
    # time_seq_num = np.arange(time_seq.shape[0])
    time_seq_num = time_seq

    _SQRT2 = np.sqrt(2)

    # def hellinger(p, q):
    #     return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2

    def hellinger_hist(p, q, bins):
        """
        :param p: probabilities of target distrbution
        :param q: probabilities of reference distrbution
        :param bins: bins center locations / x-values associated with both distributions
        :return:
        """
        # bins = x1
        # p = p1
        # q = p2
        helld = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2
        return helld

    if ww % 2 == 0:
        ww = ww + 1

    local_f = list()
    local_sign = list()

    for x in window_pad_pd(time_seq_num, ww):
        tar_h_freq, tar_h_div = np.histogram(tar_values.loc[list(x)], bins=tmp_xseq)
        tmp_pdf_list = tar_h_freq / (np.sum(tar_h_freq) * 1.0)
        tmp_h = hellinger_hist(p=tmp_pdf_list, q=ref_values, bins=bin_centers)
        local_f.append(tmp_h)
    return local_f, local_sign


def pd_hydroYear(file_name, season='annual', aggregate='mean'):
    """
    Shifts the months by 9 months so that the start of the year is in September to be centered for the hydrological
    year. the first and last year and consequently not complete and are thus truncated.

    :param file_name:
    :param season: standar = 'annual' -> mean annual values; winter and summer can be specified
    :param aggregate: aggregate time series by mean or sum
    :return: shifted and truncated time series
    """
    tm_2d_ann_pd = read_csv(file_name)

    # SHIFT YEAR INDEX FOR ALL OF THEM TO HAVE HYDROLOGICAL YEAR
    # start with october as it is the month where precipitation starts to fall as snow
    # remove/shift 9 months
    tm_2d_ann_pd.index = tm_2d_ann_pd.index.shift(-9, freq='MS')

    if season != 'annual':
        # shifted winter and summer months April = 1st month
        # winter_months = [2, 3, 4, 5, 6, 7]
        # summer_months = [9, 10, 11, 12]
        # summer_months = [2, 3, 4, 5, 6]  # - changed 2018-12-17
        # winter_months = [8, 9, 10, 11, 12]
        # ao_months = [1, 7]  # April and October the transition months

        winter_months = [2, 3, 4, 5, 6]  # - changed 2018-12-18 -- 2=Nov, 3=Dec
        summer_months = [8, 9, 10, 11, 12]  # 8=May, 12=Sep
        ao_months = [1, 7]  # April and October the transition months

        if season == 'winter':
            season_months = winter_months
        elif season == 'summer':
            season_months = summer_months
        elif season == 'AO':
            season_months = ao_months

        ts_2d_ann_pd = tm_2d_ann_pd[tm_2d_ann_pd.index.map(lambda t: t.month in season_months)]
        if aggregate == 'mean':
            ts_2d_ann_pd = ts_2d_ann_pd.groupby([lambda x: x.year]).mean()
        elif aggregate == 'sum':
            # the pandas .sum does not keep the NaNs --> use apply instead
            # ts_2d_ann_pd = ts_2d_ann_pd.groupby([lambda x: x.year]).sum()
            ts_2d_ann_pd = ts_2d_ann_pd.groupby([lambda x: x.year]).apply(pd.DataFrame.sum, skipna=False)
        else:
            raise Exception('provide an aggreagte function: mean or sum')
    else:
        if aggregate == 'mean':
            ts_2d_ann_pd = tm_2d_ann_pd.groupby([lambda x: x.year]).mean()
        elif aggregate == 'sum':
            # the pandas .sum does not keep the NaNs --> use apply instead
            # ts_2d_ann_pd = tm_2d_ann_pd.groupby([lambda x: x.year]).sum()
            ts_2d_ann_pd = tm_2d_ann_pd.groupby([lambda x: x.year]).apply(pd.DataFrame.sum, skipna=False)
        else:
            raise Exception('provide an aggreagte function: mean or sum')

    # drop the first and last entries because of the shift () because it is incomplete. Jan - October
    ind_min = ts_2d_ann_pd.index.min() + 1
    ind_max = ts_2d_ann_pd.index.max() - 1
    ts_2d_ann_pd = ts_2d_ann_pd[ts_2d_ann_pd.index > ind_min]
    ts_2d_ann_pd = ts_2d_ann_pd[ts_2d_ann_pd.index < ind_max]

    ################################################
    # SET THE TARGET SEASON OR ANNUAL
    #
    return ts_2d_ann_pd
    ################################################


def calc_overlap(pd_dataframe, ww, precision, time_start, time_split, time_end, posneg=True, crop=True):
    """
    Calculate the Hellinger distance between reference (divided at time_split) and target period

    :param pd_dataframe: input pandas dataframe that is split into target and reference parts
    :param ww: window width
    :param precision: the number of points to evaluate the PDF
    :param time_start: start of series
    :param time_split: where to split the series
    :param time_end: where to stop
    :param posneg: returns the signal of the hellinger shift as well
    :param crop: set values at half window size at the beginning and end of the time series to NaN
    :return: the hellinger distances of each target series time step with respect to the reference, and the sign
    """
    #
    # # pd_dataframe = ds_ref
    # pd_dataframe= pd_df
    # ww = WW
    # precision = precision
    # time_start = XMIN
    # time_split = XSPLIT
    # time_end = XMAX
    # posneg = True
    # crop = True

    # make copy .. some issues before without ...
    ts_index = pd_dataframe.index
    t_2d_ann_pd = copy(pd_dataframe)
    OUT_overlap_pd = copy(pd.DataFrame(t_2d_ann_pd))
    OUT_overlap_pd[:] = np.NaN
    OUT_overlap_pd_sign = copy(pd.DataFrame(t_2d_ann_pd))
    OUT_overlap_pd_sign[:] = np.NaN
    # get they column keys (lat/lon)
    RowCols = t_2d_ann_pd.keys()

    # RowCol loop; go through the keys of pandas dataframe that are the row and column indices
    for RowCol in RowCols:
        # RowCol = u'0073.00145.0'
        # all
        y_all = t_2d_ann_pd[RowCol].values

        # check if there is data; some pixels are only NA
        if np.isnan(y_all).all():
            continue

        # check if every data record is 0
        if (y_all == 0).all():
            continue

        # reference
        y_ref = t_2d_ann_pd[RowCol].ix[time_start:time_split].values
        if np.isnan(y_ref).all():
            continue

        # target
        y_tar = t_2d_ann_pd[RowCol].ix[time_split:time_end].values

        # target - half window
        y_tar_ww = t_2d_ann_pd[RowCol].ix[str(int(time_split.split('-')[0]) - int(ww / 2)) + '-01-01':time_end].values
        y_times_tar_ww = t_2d_ann_pd[RowCol].ix[
                         str(int(time_split.split('-')[0]) - int(ww / 2)) + '-01-01':time_end].index
        if np.isnan(y_tar).all():
            continue

        # get the values for evaluation
        ybnds = x_range(y_all, stretch=1)
        tmp_xseq = np.arange(ybnds[0], ybnds[1], (ybnds[2] / precision))

        # REFERENCE PDF
        ref_KDEpdf_list = gaussian_kde(y_ref).pdf(tmp_xseq)

        ref_pdf_list = ref_KDEpdf_list / np.sum(ref_KDEpdf_list)
        y_ref_bs_arr = np.array(ref_pdf_list)

        if posneg:
            dist_hellinger, dist_hellinger_sign = calc_run_hellinger_posneg(time_seq=y_times_tar_ww, values=y_tar_ww,
                                                                            ref_values=y_ref_bs_arr, ww=ww,
                                                                            tmp_xseq=tmp_xseq)
            OUT_overlap_pd_sign[RowCol] = pd.DataFrame(dist_hellinger_sign, index=y_times_tar_ww)

        else:
            dist_hellinger = calc_run_hellinger(time_seq=y_times_tar_ww, values=y_tar_ww, ref_values=y_ref_bs_arr,
                                                ww=ww, tmp_xseq=tmp_xseq)

        # and finally write into complete PD data frame
        OUT_overlap_pd[RowCol] = pd.DataFrame(dist_hellinger, index=y_times_tar_ww)

        if crop:
            # set values at (index <= XSPLIT and index > time_end - (ww/2)) to NaN
            time_split_year = re.split('-', time_split)[0]
            time_end_year = ts_index.max()
            ind_crop_lw = OUT_overlap_pd_sign.index <= int(time_split_year)
            ind_crop_up = OUT_overlap_pd_sign.index > int(time_end_year - int(ww / 2))
            OUT_overlap_pd_sign.ix[(ind_crop_lw | ind_crop_up)] = np.NaN
            OUT_overlap_pd.ix[(ind_crop_lw | ind_crop_up)] = np.NaN

    return OUT_overlap_pd, OUT_overlap_pd_sign


def calc_overlap_hist(pd_dataframe, ww, precision, time_start, time_split, time_end, posneg=True, crop=True):
    """
    Calculate the Hellinger distance between reference (divided at time_split) and target period

    :param pd_dataframe: input pandas dataframe that is split into target and reference parts
    :param ww: window width
    :param precision: the number of points to evaluate the PDF
    :param time_start: start of series
    :param time_split: where to split the series
    :param time_end: where to stop
    :param posneg: returns the signal of the hellinger shift as well
    :param crop: set values at half window size at the beginning and end of the time series to NaN
    :return: the hellinger distances of each target series time step with respect to the reference, and the sign
    """
    #
    # # pd_dataframe = ds_ref
    # pd_dataframe= pd_df
    # ww = WW
    # precision = precision
    # time_start = XMIN
    # time_split = XSPLIT
    # time_end = XMAX
    # posneg = True
    # crop = True

    # make copy .. some issues before without ...
    ts_index = pd_dataframe.index
    t_2d_ann_pd = copy(pd_dataframe)
    OUT_overlap_pd = copy(pd.DataFrame(t_2d_ann_pd))
    OUT_overlap_pd[:] = np.NaN
    OUT_overlap_pd_sign = copy(pd.DataFrame(t_2d_ann_pd))
    OUT_overlap_pd_sign[:] = np.NaN
    # get they column keys (lat/lon)
    RowCols = t_2d_ann_pd.keys()

    # RowCol loop; go through the keys of pandas dataframe that are the row and column indices
    for RowCol in RowCols:
        # RowCol = '0073.00145.0'
        # RowCol = RowCols[10]
        # all
        y_all = t_2d_ann_pd[RowCol].values

        # check if there is data; some pixels are only NA
        if np.isnan(y_all).all():
            continue

        # check if every data record is 0
        if (y_all == 0).all():
            continue

        # reference
        y_ref = t_2d_ann_pd[RowCol].ix[time_start:time_split]
        if np.isnan(y_ref).values.all():
            continue

        # target
        y_tar = t_2d_ann_pd[RowCol].ix[time_split:time_end]

        # target - half window
        y_tar_ww = t_2d_ann_pd[RowCol].ix[
                   str(int(time_split.split('-')[0]) - int(ww / 2)) + '-01-01':time_end]  # removed .values
        y_times_tar_ww = t_2d_ann_pd[RowCol].ix[
                         str(int(time_split.split('-')[0]) - int(ww / 2)) + '-01-01':time_end].index
        if np.isnan(y_tar).all():
            continue

        # get the values for evaluation
        ybnds = x_range(y_all, stretch=1)
        tmp_xseq = np.linspace(ybnds[0], ybnds[1], precision)
        tmp_xseq_centers = tmp_xseq[0:-1] + ((ybnds[1] - ybnds[0]) / precision)

        # changes here to test use of Histogram function instead of KDE
        #
        #
        # ___ new version 2018-11-07

        # ref_KDEpdf_list = gaussian_kde(y_ref).pdf(tmp_xseq)
        # these are a lot of bins. check to test if this works:
        ref_h_freq, _ = np.histogram(y_ref, bins=tmp_xseq)
        ref_pdf_list = ref_h_freq / (np.sum(ref_h_freq) * 1.0)
        # new calc_run_hellinger with Histogram instead of KDE.PDF()

        if posneg:
            dist_hellinger, dist_hellinger_sign = calc_run_hellinger_posneg_hist(time_seq=y_times_tar_ww,
                                                                                 tar_values=y_tar_ww,
                                                                                 ref_values=ref_pdf_list, ww=ww,
                                                                                 tmp_xseq=tmp_xseq,
                                                                                 bin_centers=tmp_xseq_centers)

            OUT_overlap_pd_sign[RowCol] = pd.DataFrame(dist_hellinger_sign, index=y_times_tar_ww)
        else:
            dist_hellinger = None

        # else:
        #     dist_hellinger = calc_run_hellinger(time_seq=y_times_tar_ww, values=y_tar_ww, ref_values=y_ref_bs_arr,
        #                                         ww=ww, tmp_xseq=tmp_xseq)

        # and finally write into complete PD data frame
        OUT_overlap_pd[RowCol] = pd.DataFrame(dist_hellinger, index=y_times_tar_ww)

        if crop:
            # set values at (index <= XSPLIT and index > time_end - (ww/2)) to NaN
            time_split_year = re.split('-', time_split)[0]
            time_end_year = ts_index.max()
            ind_crop_lw = OUT_overlap_pd_sign.index <= int(time_split_year)
            ind_crop_up = OUT_overlap_pd_sign.index > int(time_end_year - int(ww / 2))
            OUT_overlap_pd_sign.ix[(ind_crop_lw | ind_crop_up)] = np.NaN
            OUT_overlap_pd.ix[(ind_crop_lw | ind_crop_up)] = np.NaN

    return OUT_overlap_pd, OUT_overlap_pd_sign


def find_timemax(x, confidence_level):
    # # different approach
    # # go through each column and drop NANs and find the last occurence where the cumsum was increasing
    # x = copy(tmp_pd)
    # x_sign = copy(tmp_pd_sign)
    # # confidence_level =confidence_level
    # # key = '0059.00119.0'
    # # # key = '0069.00147.0'
    # key = "0053.00141.0"
    # key = "0073.00103.0"

    # x[:] = 0.8
    index0 = x.index[0]
    indexlast = x.index[x.notna().any(1)][-1]
    index_not_nan = x.notna().any()

    # the last valid data entry:
    index_last_row_not_na = x.index[x.notnull().any(axis=1)][-1]
    x_out = copy(x)
    x_out[:] = np.NaN

    # get the indices where values greater/equal and smaller threshold
    x_ge = x >= confidence_level
    x_sm = x < confidence_level
    # build cumsum
    x_sm_cs = x_sm.cumsum()
    x_sm_max = x_sm_cs.max()
    x_sm_cs_max = x_sm_cs == x_sm_max

    # where is:
    # x_sm_cs_max True      AND    x_ge True
    bool_list = (x_sm_cs_max & x_ge).idxmax()

    # if the hellinger distance drops below the CI at the end, bool_list will determine index[0] as
    # timemax ! --> take instead the last date
    # we don't have to remember where this happens because we have the absolute value later on as well ..
    # non_converged = np.where((bool_list == index0) & index_not_nan)[0]
    bool_list[((bool_list == index0) & index_not_nan)] = indexlast

    # get bool_list into an array where only the bool_list entries are True
    keys = x_out.keys()
    for key_i in np.arange(0, len(keys)):
        tmp_index = bool_list[keys[key_i]]
        x_out.loc[tmp_index, keys[key_i]] = 1.0

    # non_converged -- not neccessary. get the last year value as index to take the sign.
    # reject the value if the threshold was not reached
    # --> instead return last year
    return x_out, index_last_row_not_na


def get_timemax(x, pd_raw):
    # get_timemax -> in case the confidence level is not reached, which is the case for most models before 2015
    # there will be a False statement in the output of find_timemax. in this case use the last value
    # pd_raw = tmp_pd
    # x = tmp_val_1st_bool
    # key = "0053.00141.0"
    #
    # x= val_1st_bool
    # pd_raw = ref_pd
    # x = val_1st_bool
    # pd_raw = ref_pd

    tmp_max = (x.cumsum(axis=0) == 1).idxmax()

    # x = tmp_val_1st_bool
    for key in x:
        # print(key)
        # if all values were below the threshold, all bools stay False. In this case take the last value

        if not x[key].any():
            print(key)
            # use the last year where the raw data set is NOT NA
            tmp_x = pd_raw[key].dropna()
            if tmp_x.isnull().all():
                tmp_max[key] = np.NaN
            else:
                yr_last = tmp_x.index[-1]
                # yr_last = x.index[-1]
                tmp_max[key] = yr_last
    return tmp_max


def get_valmax(df_timemax, df_raw, bool_mat, df_raw_sign, posneg=True):
    #####
    # df_timemax = src_timemax
    # df_raw = ref_pd
    # df_raw_sign = ref_pd_sign
    # bool_mat = val_1st_bool

    # df_timemax = targ_timemax
    # df_raw = tmp_pd
    # bool_mat = tmp_val_1st_bool
    # df_raw_sign = tmp_pd_sign

    #####
    x_out = copy(df_timemax)
    x_out_sign = copy(df_timemax)
    tmp = df_raw[(bool_mat.cumsum(axis=0) == 1)]
    tmp_sign = df_raw_sign[(bool_mat.cumsum(axis=0) == 1)]

    for cc in tmp:
        # print(cc)
        # cc= '061111'
        # case: all bool_mat values are False (below CI) but values exist
        if df_raw[cc].any() and tmp[cc].isna().all():
            tmp_x = df_raw[cc].dropna()
            tmp_x_sign = df_raw_sign[cc].dropna()
            tmp_fill = tmp_x.values[-1]
            tmp_fill_sign = tmp_x_sign.values[-1]
        else:
            ctmp = tmp[cc].dropna()
            ctmp_sign = tmp_sign[cc].dropna()
            if ctmp.values.__len__() == 0:
                tmp_fill = np.NaN
                tmp_fill_sign = np.NaN
            else:
                tmp_fill = ctmp.values
                tmp_fill_sign = ctmp_sign.values
        x_out.loc[cc] = tmp_fill
        x_out_sign.loc[cc] = tmp_fill_sign
    return x_out, x_out_sign


def nan_no_converge(timemax, valmax, confidence_level):
    tmp_index = (valmax < confidence_level)
    timemax[tmp_index] = np.NaN
    valmax[tmp_index] = np.NaN
    return timemax, valmax


def find_toe_at_year(x, xs, year):
    # x = copy(tmp_pd)
    # x_sign = copy(tmp_pd_sign)
    # key = "0053.00141.0"
    # key = "0073.00103.0"

    # x[:] = 0.8
    val_at_year = x.loc[year]
    sign_at_year = xs.loc[year]
    return val_at_year, sign_at_year
