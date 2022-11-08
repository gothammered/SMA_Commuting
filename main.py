import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pysal.viz import splot
from splot import esda as esdaplot
# import contextily
from esda.moran import Moran, Moran_Local
from pysal.lib import weights


from tqdm import tqdm

"""
기본 설정 파트
"""
# 열 출력 제한 수 조정
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 한글은 matplotlib 출력을 위해서 따로 설정이 필요하므로 font매니저 import
import matplotlib.font_manager

# 한글 출력을 위하여 font 설정
font_name = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/HANDotum.ttf').get_name()
matplotlib.rc('font', family=font_name)

def calculateDistance(t_year):
    if t_year not in ['2006', '2010', '2016']:
        raise Exception('Not a vaild year')

    delimiter = lambda x: ',' if t_year == '2016' else '\t'
    encoding = lambda x: 'cp949' if t_year == '2016' else 'utf-8'

    # Read data
    df = pd.read_csv('./data/transit_survey_SMA_{0}.csv'.format(t_year))
    gdf = gpd.read_file('./data/SMA_mainland_{0}.gpkg'.format(t_year))

    # Sort by DISTRICT_I
    gdf = gdf.sort_values(by=['DISTRICT_I'])

    # Get district list from DISTRICT_I
    district_list = gdf['DISTRICT_I'].to_list()

    # Set DISTRICT_I as index
    gdf = gdf.set_index('DISTRICT_I', drop=True)

    # Get centroid for districts
    gdf['centroid'] = gdf['geometry'].centroid

    # Create an empty DataFrame to calculate distance
    distance_df = pd.DataFrame()

    # Calculate distance matrix
    for d in tqdm(district_list):
        # Use concat instead of at for better performance
        distance_df_tp = pd.DataFrame()
        distance_df_tp[d] = gdf['centroid'].distance(gdf.at[d, 'centroid'])
        distance_df = pd.concat([distance_df, distance_df_tp], axis=1)

    # Save the result
    distance_df.to_csv('./data/SMA_mainland_distance_{0}.csv'.format(t_year))

def calcAvgCommDist(t_year):
    if t_year not in ['2006', '2010', '2016']:
        raise Exception('Not a vaild year')

    house_length = {'2006': 10, '2010': 12}

    # Load distance info
    distance_df = pd.read_csv('./data/SMA_mainland_distance_{0}.csv'.format(t_year), dtype={'DISTRICT_I': 'string'})
    distance_df = distance_df.set_index('DISTRICT_I', drop=True)

    # Load OD info
    od_df = pd.read_csv('./data/transit_survey_SMA_{0}.csv'.format(t_year), dtype={'Unnamed: 0': 'string', 'ADM_CD_O': 'string', 'ADM_CD_D': 'string'})
    od_df = od_df.rename(columns={'Unnamed: 0': 'FID'})

    # Read data
    gdf = gpd.read_file('./data/SMA_mainland_{0}.gpkg'.format(t_year))
    # Sort by DISTRICT_I
    gdf = gdf.sort_values(by=['DISTRICT_I'])
    # Get district list from DISTRICT_I
    district_list = gdf['DISTRICT_I'].to_list()
    # Set DISTRICT_I as index
    gdf = gdf.set_index('DISTRICT_I', drop=True)
    # Get centroid for districts
    gdf['centroid'] = gdf['geometry'].centroid

    # Create house info if t_year == 2006 or 2010
    if t_year == '2006' or t_year == '2010':
        od_df['house'] = od_df['FID'].str.slice(0, house_length[t_year])
    else:
        od_df = od_df.rename(columns={'IDX': 'house'})

    # Calculate how many family members commute in a household
    household_commute = pd.pivot_table(od_df, values='FID', index='house', aggfunc='count')
    household_commute = household_commute.rename(columns={'FID': 'commutingCount'})
    household_commute = household_commute.reset_index()
    od_df = od_df.merge(household_commute, on='house', how='left')

    # Calculate distance for every commuter
    def get_distance(o, d):
        return distance_df.at[o, d]

    od_df['distance'] = od_df.apply(lambda x: get_distance(x['ADM_CD_O'], x['ADM_CD_D']), axis=1)

    # Calculate difference between number of commuting family members
    commutingCount_pivot = pd.pivot_table(od_df, values='distance', index='commutingCount', aggfunc='mean')
    commutingStd = pd.pivot_table(od_df, values='distance', index='house', aggfunc=np.std)
    commutingStd = commutingStd.rename(columns={'distance': 'distanceStd'})
    commutingStd = commutingStd.reset_index()

    od_df = od_df.merge(commutingStd, on='house', how='left')

    # Plot the difference
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9, 9))
    plt.subplots_adjust(hspace=0.3)
    vp1 = list()
    vp2 = list()
    for i in range(1, 7):
        vp1.append(od_df[od_df['commutingCount'] == i]['distance'])
        vp2.append(od_df[od_df['commutingCount'] == i]['distanceStd'])
        ax1.text(i, -5000, 'Avg={0}'.format(round(commutingCount_pivot.at[i, 'distance'], 2)), ha='center', fontsize='x-small')

    ax1.violinplot(vp1)
    ax2.violinplot(vp2)
    ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax2.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax1.set_title('Average commuting distance by number of family members who commute ({0})'.format(t_year))
    ax2.set_title('St.D of commuting distance by number of family members who commute({0})'.format(t_year))
    ax1.set_xlabel('Number of family members who commute')
    ax2.set_xlabel('Number of family members who commute')
    ax1.set_ylabel('Average commuting distance (m)')
    ax2.set_ylabel('St.D of commuting distance inside a household')
    # ax.scatter(x=od_df['commutingCount'], y=od_df['distance'])

    fig.savefig('./result/fig/commDistByMemberCount_{0}.png'.format(t_year), dpi=300)
    plt.show()

    # Calculate and plot standard deviation
    commuting_dist_std = pd.pivot_table(od_df[od_df['commutingCount'] == 2], values='distance', index='house', aggfunc='mean')
    commuting_dist_std = commuting_dist_std.join(pd.pivot_table(od_df[od_df['commutingCount'] == 2], values='distance', index='house',
                                        aggfunc=np.std), how='left', lsuffix='_mean', rsuffix='_std')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(commuting_dist_std['distance_mean'], commuting_dist_std['distance_std'])
    plt.show()

    # Calculate average of commuting distance std, double-income households
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))
    fig.suptitle('Average of commuting distance std, double-income households ({0})'.format(t_year), fontsize=16, y=0.95)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    commuting_dist_std = commuting_dist_std.reset_index()
    commuting_dist_std = commuting_dist_std.merge(od_df, on='house', how='left')
    commuting_dist_std = pd.pivot_table(commuting_dist_std, values='distance', index='ADM_CD_O', aggfunc='mean')
    ax1.set_title('Value')
    gdf = gdf.join(commuting_dist_std['distance'], how='left')

    # exclude whose value is larger than 25000 for better readability
    gdf = gdf[gdf['distance'] < 25000]

    gdf.plot('distance', ax=ax1, legend=True, legend_kwds={'shrink': 0.5})
    ax1.axis('off')

    # Calculate Moran's I
    gdf = gdf[gdf['distance'].notna()]

    # Use Kernel weight
    w = weights.distance.Kernel.from_dataframe(gdf)
    w.transform = 'R'

    moran = Moran(gdf['distance'], w)
    print(moran.I)
    print(moran.p_sim)

    lisa = Moran_Local(gdf['distance'], w, permutations=99)
    # gdf['lm'] = lm.q
    # gdf['lm_p_sim'] = lm.p_sim
    # gdf['lm_p_sim'] = gdf['lm_p_sim'].apply(lambda x: 1 if x < 0.05 else 0)
    # gdf['lm'] = gdf['lm'] * gdf['lm_p_sim']

    esdaplot.lisa_cluster(lisa, gdf, p=0.05, ax=ax2)
    ax2.set_title("Local Moran's I")
    fig.text(0.9, 0.1, "Global Moran's I = {0}***".format(round(moran.I, 3)), ha='right', va='bottom')
    fig.savefig('./result/fig/localMoranI_{0}.png'.format(t_year), dpi=300)
    plt.show()


def calcCommDiff(t_year):
    if t_year not in ['2006', '2010', '2016']:
        raise Exception('Not a vaild year')

    house_length = {'2006': 10, '2010': 12}

    # Load distance info
    distance_df = pd.read_csv('./data/SMA_mainland_distance_{0}.csv'.format(t_year), dtype={'DISTRICT_I': 'string'})
    distance_df = distance_df.set_index('DISTRICT_I', drop=True)

    # Load OD info
    od_df = pd.read_csv('./data/transit_survey_SMA_{0}.csv'.format(t_year), dtype={'Unnamed: 0': 'string', 'ADM_CD_O': 'string', 'ADM_CD_D': 'string'})
    od_df = od_df.rename(columns={'Unnamed: 0': 'FID'})

    # Read data
    gdf = gpd.read_file('./data/SMA_mainland_{0}.gpkg'.format(t_year))
    # Sort by DISTRICT_I
    gdf = gdf.sort_values(by=['DISTRICT_I'])
    # Get district list from DISTRICT_I
    district_list = gdf['DISTRICT_I'].to_list()
    # Set DISTRICT_I as index
    gdf = gdf.set_index('DISTRICT_I', drop=True)
    # Get centroid for districts
    gdf['centroid'] = gdf['geometry'].centroid

    # Create house info if t_year == 2006 or 2010
    if t_year == '2006' or t_year == '2010':
        od_df['house'] = od_df['FID'].str.slice(0, house_length[t_year])
    else:
        od_df = od_df.rename(columns={'IDX': 'house'})

    # Calculate how many family members commute in a household
    household_commute = pd.pivot_table(od_df, values='FID', index='house', aggfunc='count')
    household_commute = household_commute.rename(columns={'FID': 'commutingCount'})
    household_commute = household_commute.reset_index()
    od_df = od_df.merge(household_commute, on='house', how='left')

    # Calculate distance for every commuter
    def get_distance(o, d):
        return distance_df.at[o, d]

    od_df['distance'] = od_df.apply(lambda x: get_distance(x['ADM_CD_O'], x['ADM_CD_D']), axis=1)

    # get only whose commutingCount is 2
    od_df = od_df[od_df['commutingCount'] == 2]
    print(od_df.head(10))

    # calculate commuting distance (minimum, actual sum, and difference) for each household
    house_list = od_df['house'].tolist()
    minCommDist = pd.DataFrame()
    for h in tqdm(house_list):
        od_df_tp = od_df[od_df['house'] == h]
        od_df_tp = od_df_tp.reset_index(drop=True)
        a, b = od_df_tp.at[0, 'ADM_CD_D'], od_df_tp.at[1, 'ADM_CD_D']
        minCommDist.at[h, 'minDist'] = get_distance(a, b)
        minCommDist.at[h, 'sumDist'] = od_df_tp.at[0, 'distance'] + od_df_tp.at[1, 'distance']
        minCommDist.at[h, 'ADM_CD_O'] = od_df_tp.at[0, 'ADM_CD_O']

    minCommDist['diffDist'] = minCommDist['sumDist'] - minCommDist['minDist']

    # Can't divide by 0
    minCommDist = minCommDist[minCommDist['minDist'].notna()]
    minCommDist = minCommDist[minCommDist['minDist'] != 0]
    minCommDist['diffDist_p'] = minCommDist['diffDist'] / minCommDist['minDist']

    # Plot the difference
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    plt.subplots_adjust(hspace=0.3)
    ax1.violinplot(minCommDist['diffDist'])
    ax2.violinplot(minCommDist['diffDist_p'])
    fig.suptitle('Average difference between total commuting distance and optimal commuting distance\nCase of double-income households({0})'.format(t_year))
    ax1.set_xlabel('Value\n(avg: {0})'.format(round(minCommDist['diffDist'].mean(), 3)))
    ax1.set_ylim(0, 50000)
    ax2.set_xlabel('Proportion\n(Difference / Sum)\n(avg: {0})'.format(round(minCommDist['diffDist_p'].mean(), 3)))
    ax2.set_ylim(0, 20)

    fig.savefig('./result/fig/commDistDiff_{0}.png'.format(t_year), dpi=300)
    plt.show()

    minCommDist = pd.pivot_table(minCommDist, index='ADM_CD_O', values='diffDist_p', aggfunc='mean')

    # Moran's I and plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9))
    fig.suptitle(
        'Average difference between total commuting distance and optimal commuting distance\nCase of double-income households ({0})'.format(
            t_year), fontsize=16, y=0.95)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    gdf = gdf.join(minCommDist, how='left')
    gdf.plot('diffDist_p', ax=ax1, legend=True, legend_kwds={'shrink': 0.5})
    ax1.axis('off')

    # Calculate Moran's I
    gdf = gdf[gdf['diffDist_p'].notna()]

    # Use Kernel weight
    w = weights.distance.Kernel.from_dataframe(gdf)
    w.transform = 'R'

    moran = Moran(gdf['diffDist_p'], w)
    print(moran.I)
    print(moran.p_sim)

    lisa = Moran_Local(gdf['diffDist_p'], w, permutations=99)

    esdaplot.lisa_cluster(lisa, gdf, p=0.05, ax=ax2)
    ax2.set_title("Local Moran's I")
    fig.text(0.9, 0.1, "Global Moran's I = {0}, p = {1}".format(round(moran.I, 3), round(moran.p_sim, 3)), ha='right',
             va='bottom')
    fig.savefig('./result/fig/commDiff_p_{0}.png'.format(t_year), dpi=300)
    plt.show()





for t_year in tqdm(['2006', '2010', '2016']):
    calcCommDiff(t_year)
