import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

"""
기본 설정 파트
"""
# 열 출력 제한 수 조정
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# 한글은 matplotlib 출력을 위해서 따로 설정이 필요하므로 font매니저 import
import matplotlib.font_manager

# 한글 출력을 위하여 font 설정
font_name = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/HANDotum.ttf').get_name()
matplotlib.rc('font', family=font_name)


# 섬 제외시키기 (결과: 1135중 1111)
# 2006년은 오류나서 QGIS에서 따로 해야됨
def get_SMA_mainland(t_year):
    t_file = './data/EA001G_{0}기준.shp'.format(t_year)

    # 행정구역 shp 로드
    SMA_gdf = gpd.read_file(t_file, encoding='cp949')

    # crs 변경
    SMA_gdf = SMA_gdf.to_crs(5179)

    # 읍면동 단위 남기기
    SMA_gdf = SMA_gdf[SMA_gdf['DISTRICT_T'] == '4']

    # 서울 대도시권만 남기기
    SMA_gdf['SIDO'] = SMA_gdf['DISTRICT_I'].str.slice(0, 2)
    SMA_gdf = SMA_gdf[(SMA_gdf['SIDO'] == '11') | (SMA_gdf['SIDO'] == '23') | (SMA_gdf['SIDO'] == '31')]

    # 대부도 제외(disslove explode할 때 문제됨)
    SMA_gdf = SMA_gdf.loc[SMA_gdf.DISTRICT_I != '3109272']

    # Dissolve then Explode로 본토 남기기
    SMA_gdf_mainland = SMA_gdf.dissolve()
    SMA_gdf_mainland = SMA_gdf_mainland.explode()

    # 본토만 남기고(DISTRICT_I == '1101053') 지우기
    SMA_mainland_identifier = SMA_gdf.loc[SMA_gdf.DISTRICT_I == '1101053'].reset_index()
    SMA_mainland_identifier = SMA_mainland_identifier.at[0, 'geometry']
    SMA_gdf_mainland['is_mainland'] = SMA_gdf_mainland['geometry'].intersects(SMA_mainland_identifier)
    SMA_gdf_mainland = SMA_gdf_mainland[SMA_gdf_mainland['is_mainland']].reset_index()
    SMA_gdf_mainland = SMA_gdf_mainland.at[0, 'geometry']

    SMA_gdf['is_mainland'] = SMA_gdf['geometry'].intersects(SMA_gdf_mainland)
    SMA_gdf = SMA_gdf[SMA_gdf['is_mainland']]
    SMA_gdf.to_file('./data/SMA_mainland_{0}.gpkg'.format(t_year), encoding='cp949')

    return


def transit_survey_SMA(t_year):
    # encoding dic 선언
    encoding_dic = {2016: 'cp949', 2010: 'utf-8', 2006: 'utf-8'}
    dtype_dic = {2016: {'목적통행_출발지_행정동코드': 'string', '목적통행_도착지_행정동코드': 'string'}, 2010: {'start_zcode': 'string', 'end_zcode': 'string', 'sheet_code': 'string', 'seq': 'string'}, 2006: {'ocode': 'string', 'dcode': 'string'}}
    delimeter_dic = {2016: ',', 2010: '\t', 2006: '\t'}

    # 가통자료 load
    transit_survey_df = pd.read_csv('./data/transit_survey_{0}.txt'.format(t_year), encoding=encoding_dic[t_year], dtype=dtype_dic[t_year], delimiter=delimeter_dic[t_year])

    # 통행목적 '출근'만 추출
    if t_year == 2016:
        transit_survey_df = transit_survey_df[transit_survey_df['통행목적'] == 4]
    elif t_year == 2010:
        transit_survey_df = transit_survey_df[transit_survey_df['tr_mokjek'] == 3]
    elif t_year == 2006:
        transit_survey_df = transit_survey_df[transit_survey_df['pur'] == 3]


    # 행안부 행정동코드 -> 통계청 행정동코드 변환
    ADM_CD_info = pd.read_excel('./data/SMA_ADM_CD_info.xlsx', sheet_name='Y{0}'.format(t_year), dtype={'행정동코드(10자리)': 'string', 'ADM_CD': 'string'})
    ADM_CD_info = ADM_CD_info[['행정동코드(10자리)', 'ADM_CD']]
    ADM_CD_info = ADM_CD_info.drop_duplicates()

    O_left_on_dic = {2016: '목적통행_출발지_행정동코드', 2010: 'start_zcode', 2006: 'ocode'}
    D_left_on_dic = {2016: '목적통행_도착지_행정동코드', 2010: 'end_zcode', 2006: 'dcode'}

    transit_survey_df = transit_survey_df.merge(ADM_CD_info, left_on=O_left_on_dic[t_year], right_on='행정동코드(10자리)', how='left')
    transit_survey_df = transit_survey_df.drop(columns=['행정동코드(10자리)'])
    transit_survey_df = transit_survey_df.rename(columns={'ADM_CD': 'ADM_CD_O'})
    transit_survey_df = transit_survey_df.merge(ADM_CD_info, left_on=D_left_on_dic[t_year], right_on='행정동코드(10자리)', how='left')
    transit_survey_df = transit_survey_df.drop(columns=['행정동코드(10자리)'])
    transit_survey_df = transit_survey_df.rename(columns={'ADM_CD': 'ADM_CD_D'})
    transit_survey_df.info()

    # 2010년 2006년은 수단통행 기준 정렬이므로 목적통행 기준으로 변환
    if t_year == 2010:
        transit_survey_df['pno'] = transit_survey_df['sheet_code'].str.cat(transit_survey_df['seq'], sep='-')

    if t_year == 2010 or t_year == 2006:
        pno_list = transit_survey_df['pno'].unique()

        transit_survey_df_rearranged = pd.DataFrame()
        sort_value_dic = {2010: 'tr_seq', 2006: 'order'}

        for pno in tqdm(pno_list, desc='수단통행에서 목적통행으로 변환중...'):
            tp_df = transit_survey_df[transit_survey_df['pno'] == pno]
            tp_df = tp_df.sort_values(by=sort_value_dic[t_year])
            # 처음 나오는 출발지가 목적통행 출발지
            transit_survey_df_rearranged.at[pno, 'ADM_CD_O'] = tp_df.iloc[0].at['ADM_CD_O']
            # 마지막에 나오는 도착지가 목적통행 도착지
            transit_survey_df_rearranged.at[pno, 'ADM_CD_D'] = tp_df.iloc[len(tp_df) - 1].at['ADM_CD_D']

        transit_survey_df = transit_survey_df_rearranged

    # 공간정보 load
    SMA_gdf = gpd.read_file('./data/SMA_mainland_{0}.gpkg'.format(t_year))
    SMA_gdf.plot()
    plt.show()
    SMA_gdf.info()

    # 서울 대도시권 본토 읍면동코드 리스트 불러오기
    SMA_dong_CD_list = SMA_gdf['DISTRICT_I'].tolist()

    # 가통자료 중, 출발과 도착이 서울 대도시권 본토인 자료들만 추출
    transit_survey_df = transit_survey_df[transit_survey_df['ADM_CD_O'].isin(SMA_dong_CD_list)]
    transit_survey_df = transit_survey_df[transit_survey_df['ADM_CD_D'].isin(SMA_dong_CD_list)]

    transit_survey_df.info()
    transit_survey_df.to_csv('./data/transit_survey_SMA_{0}.csv'.format(t_year))

    return

# get_SMA_mainland(2006)
# transit_survey_SMA(2006)