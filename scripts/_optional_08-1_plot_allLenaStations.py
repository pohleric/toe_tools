"""xmin = 100.0
xmax = 160.0
ymax = 74.0
ymin = 52.0
Plot the obtained ToE and showcase the variability based on the sensitivity analysis (window width, time choice to
split the time-series), and the best n performing models of the CMIP5 collection
"""

from toe_tools.gis import *

varname = 'tas'
PATH_ESD = '/home/hydrogeol/epohl/data/ESD/overlap_Lena_hellinger/fullSeries'
confidence_level = 0.95  # confidence level at which we assume ToE
model = 'CMIP5_AllLenaStations'

filename_toe = PATH_ESD + '/%s_ESD_ToE_Sensitivity-timemax_%s_1901-2016_Siberia_df_annual_overlap_.csv' % (
    varname, confidence_level)
filename_valmax = PATH_ESD + '/%s_ESD_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_annual_overlap_.csv' % (
    varname, confidence_level)
filename_sign = PATH_ESD + '/%s_ESD_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_annual_overlap_sign.csv' % (
    varname, confidence_level)
# model_IDs = ["000", "002", "015", "020", "021" ,"024", "042" ,"056" ,"058" ,"063"]
model_IDs = [format('%03d' % m) for m in np.loadtxt('data/tas_best_10_models_NSE_AllStations.txt')]

##################################
# best 10 models
# MAX value
_dname = '%s_val_%s_10best_%s' % (varname, model, confidence_level)
######################
t_pd = pd_toe_to_geoarray(input_array=filename_valmax, nan_mask=filename_valmax, model_IDs=model_IDs,
                          sign=filename_sign)

# get variability in obtained outcome
t_pd['std'] = np.nanstd(t_pd['array'], axis=0)

filename = '/home/hydrogeol/epohl/data/DEM/ntopo_siberia_big.tif'
filename = '/homel/epohl/data/DEM/ETOPO5_Ice_resampled_from1.tif'


def read_dem(filename):
    # filename = '/home/hydrogeol/epohl/data/DEM/ntopo_siberia.tif'
    import gdal
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    GT = ds.GetGeoTransform()
    [cols, rows] = arr.shape
    lats = np.linspace(start=GT[3], stop=GT[3] + (cols * GT[5]), num=cols + 1)
    lats_unq = np.linspace(start=GT[3] + (0.5 * GT[5]), stop=GT[3] + (cols * GT[5]) - (0.5 * GT[5]), num=cols)
    # lats_unq = lats_unq[::-1]
    # lats_unq.shape
    lons = np.linspace(start=GT[0], stop=GT[0] + (rows * GT[1]), num=rows + 1)
    lons_unq = np.linspace(start=GT[0] + (0.5 * GT[1]), stop=GT[0] + (rows * GT[1]) - (0.5 * GT[1]), num=rows)
    # lons_unq.shape
    arr = arr
    out = {'array': arr, 'lats_unq': lats_unq, 'lons_unq': lons_unq, 'lats': lats, 'lons': lons, 'GT': GT}
    ds = None
    return out


dem = read_dem(filename)

toe_dict = t_pd
dem = dem
lon_inc = 8
lat_inc = 4
z_range = [-500, 2500]
cmap_name1 = 'terrain'
cmap_name2 = None
layer = None
n_colors = 12
save = True

#
# # set Font type to Helvetica
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl

mpl.rcParams['backend'] = 'TkAgg'
mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.sans-serif'] = ['Helvetica']
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon


def draw_screen_poly(lats, lons, m):
    x, y = m(lons, lats)
    xy = zip(x, y)
    poly = Polygon(xy, facecolor='red', alpha=0.4)
    plt.gca().add_patch(poly)


def plot_point_wtext(shpname, text, zorder):
    p = m.readshapefile(shpname, '')
    p1_x = p[2][0]
    p1_y = p[2][1]
    x, y = m(p1_x, p1_y)
    plt.text(x + 0.01 * width_max, y + 0.0 * heigth, text,
             path_effects=[pe.withStroke(linewidth=2, foreground="white")],
             fontsize=8, zorder=zorder)
    m.plot(x, y, 'ko', markersize=2, zorder=zorder - 1)
    m.plot(x, y, 'wo', markersize=4, zorder=zorder - 2)


vmin, vmax = z_range
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

# dem mesh
dem_dict = dem
dem_dict['lon_mesh_gc'], dem_dict['lat_mesh_gc'] = np.meshgrid(dem['lons_unq'], dem['lats_unq'])
# dem_dict['lon_mesh'], dem_dict['lat_mesh'] = np.meshgrid(dem_dict['lats'], dem_dict['lons'])
dem_dict['lon_mesh'], dem_dict['lat_mesh'] = np.meshgrid(dem_dict['lons'], dem_dict['lats'])
# dem_dict['lon_mesh'], dem_dict['lat_mesh'] = np.meshgrid(dem_dict['lats'],dem_dict['lons']) # pcolormesh is STUPID
# plt.imshow(dem['array'][::-1,:], cmap='terrain')
lonmin, lonmax, latmin, latmax = [np.min(lons), np.max(lons), np.min(lats), np.max(lats)]
lonmean = np.mean((lonmin, lonmax))
latmean = np.mean((latmin, latmax))
l0 = latmin
l1 = latmean
l2 = latmax

# get width and height approximation
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

# cmap_magma = plt.get_cmap(cmap_name1)
cmap_magma.set_bad(color='grey', alpha=0)

# bounds and norms for colorbars
fig = plt.figure(figsize=(5, 4))
m = Basemap(projection='lcc', lon_0=lonmean, width=plot_w, height=plot_h, resolution='i',
            lat_0=l1, lat_1=l2)

m.drawparallels(np.arange(latmin, latmax + latmax * 0.01, lat_inc), labels=[1, 0, 0, 0], linewidth=0.5, zorder=100)
m.drawmeridians(np.arange(lonmin, lonmax + lonmax * 0.01, lon_inc), labels=[0, 0, 0, 1], linewidth=0.5, zorder=99)

x, y = m(dem_dict['lon_mesh'], dem_dict['lat_mesh'])
m.pcolormesh(x, y, dem_dict['array'], latlon=False, vmin=z_range[0], vmax=z_range[1], cmap=cmap_magma,
             rasterized=True)  # zorder=p_cnt.__next__())
#
shp_name = '/homel/epohl/PycharmProjects/ToEtools/data/yakutsk'
plot_point_wtext(shp_name, 'Yakutsk', zorder=101)
#

# # # Use the meteorological stations in the Lena Catchment to check between CRU and CMIP5 hellinger distances
# stations_pd = pd.read_csv('data/Stations_subset.txt', sep=' ')
# stations_lon = stations_pd['lat']
# stations_lat = stations_pd['lon']
# station_names = stations_pd['name'].tolist()
# # np.array(station_names).reshape(-1,1).shape
# ps = np.hstack((np.array([m(xi, yi) for xi, yi in zip(stations_lon, stations_lat)]), np.array(station_names).reshape(-1,1)))
# m.plot(ps[:, 0].astype('float'), ps[:, 1].astype('float'), 'bo', markersize=4, zorder=95)
# # m.plot(ps[:, 0].astype('float'), ps[:, 1].astype('float'), 'ko', markersize=2, zorder=96)


# # Use the meteorological stations in the Lena Catchment to check between CRU and CMIP5 hellinger distances
stations_pd = pd.read_csv('data/LenaStations.txt', sep=' ')
stations_lon = stations_pd['lon']
stations_lat = stations_pd['lat']
station_names = stations_pd['name'].tolist()
# np.array(station_names).reshape(-1,1).shape
ps = np.hstack(
    (np.array([m(xi, yi) for xi, yi in zip(stations_lon, stations_lat)]), np.array(station_names).reshape(-1, 1)))
m.plot(ps[:, 0].astype('float'), ps[:, 1].astype('float'), 'wo', markersize=4, zorder=96)
m.plot(ps[:, 0].astype('float'), ps[:, 1].astype('float'), 'ko', markersize=2, zorder=96)

stations_pd = pd.read_csv('data/LenaStationsLong.txt', sep=' ')
stations_lon = stations_pd['lon']
stations_lat = stations_pd['lat']
station_names = stations_pd['name'].tolist()
# np.array(station_names).reshape(-1,1).shape
ps = np.hstack(
    (np.array([m(xi, yi) for xi, yi in zip(stations_lon, stations_lat)]), np.array(station_names).reshape(-1, 1)))
m.plot(ps[:, 0].astype('float'), ps[:, 1].astype('float'), 'wo', markersize=4, zorder=102)
m.plot(ps[:, 0].astype('float'), ps[:, 1].astype('float'), 'ro', markersize=2, zorder=102)

# p1_x = ps[:, 0].astype('float')
# p1_y = ps[:, 1].astype('float')
# # x, y = m(p1_x, p1_y)
# for i in np.arange(p1_y.__len__()):
#     plt.text(p1_x[i] + 0.01 * width_max, p1_y[i] + 0.01 * heigth, station_names[i],fontsize=8, zorder=96)
#              # path_effects=[pe.withStroke(linewidth=2, foreground="white")],


m.readshapefile('/homel/epohl/PycharmProjects/ToEtools/data/lena', 'lena', linewidth=.5, color='#222222', zorder=94)
#
m.drawcoastlines(linewidth=0.5, color='#999999', zorder=93)
m.drawcountries(linewidth=0.5, color='#999999', zorder=92)
m.shadedrelief(scale=1, alpha=.3, zorder=91)

# make bounding box in RED to show the extend of the data
from toe_tools.gis import *

varname = 'pr'
season = 'annual'
PATH_CRU = '/home/hydrogeol/epohl/data/CRU-NCEP/%s_overlap_Lena_hellinger/fullSeries/' % varname
filename_sign = PATH_CRU + '%s_CRU-NCEP_ToE_Sensitivity-valmax_%s_1901-2016_Siberia_df_%s_overlap_sign.csv' % (
    varname, confidence_level, season)
######################
t_pd = pd_toe_to_geoarray(input_array=filename_sign, nan_mask=filename_sign, model_IDs=None, sign=filename_sign)
# t_pd.keys()
# latss = t_pd['lats_unq']
# lonss = t_pd['lons_unq']
# lats = [latss.min(), latss.max(), latss.max(), latss.min()]
# # lats = [-30, 30, 30, -30 ]
# lons = [lonss.min(), lonss.min(), lonss.max(), lonss.max()]
# # lons = [-50, -50, 50, 50 ]
#

# draw_screen_poly( lats, lons, m )
lons = toe_dict['lon_mesh'][-1:, ]
lats = toe_dict['lat_mesh'][:, -1]
la1 = np.repeat(lats.min(), lons.shape[1])
la2 = np.repeat(lats.max(), lons.shape[1])
lo1 = lons.flatten()
lo2 = lo1[::-1]
# lons = np.array([lo1, lo2]).flatten()
# lats = np.array([la1, la2]).flatten()

lo1 = np.ndarray.tolist(lo1)
lo2 = np.ndarray.tolist(lo2)
la1 = np.ndarray.tolist(la1)
la2 = np.ndarray.tolist(la2)
lons = lo1 + lo2 + [lo1[0]]
lats = la1 + la2 + [la1[0]]
#####

x, y = m(lons, lats)
m.plot(x, y, '-', markersize=5, linewidth=2, color='red', zorder=102)
#

# # decoration
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.05)
cbar = plt.colorbar(cax=cax)

if save:
    pfile = '/homel/epohl/Desktop/stations_overview.pdf'
    fig.savefig(pfile, dip=200, bbox_inches='tight')
    # os.system('pdfcrop ' + pfile + ' ' + pfile)
    plt.close()
# plot_map_stations(t_pd, dem, cmap_name1='terrain', z_range=[0,2500], n_colors=25, save=False)
