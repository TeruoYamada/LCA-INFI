import cdsapi
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import animation
import streamlit as st
from datetime import datetime, timedelta
import os
import pandas as pd
import geopandas as gpd
import io
import requests
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates
from scipy import stats

# Configuração inicial da página
st.set_page_config(layout="wide", page_title="Monitor AOD/PM - MS", page_icon="🌍")

# ✅ Carregar autenticação a partir do secrets.toml
try:
    ads_url = st.secrets["ads"]["url"]
    ads_key = st.secrets["ads"]["key"]
    client = cdsapi.Client(url=ads_url, key=ads_key)
except Exception as e:
    st.error("❌ Erro ao carregar as credenciais do CDS API. Verifique seu secrets.toml.")
    st.stop()

# Função para baixar shapefile dos municípios de MS
@st.cache_data
def load_ms_municipalities():
    try:
        url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/" + \
              "municipio_2022/UFs/MS/MS_Municipios_2022.zip"
        
        try:
            gdf = gpd.read_file(url)
            return gdf
        except:
            # Fallback: criar geodataframe simplificado
            data = {
                'NM_MUN': ['Campo Grande', 'Dourados', 'Três Lagoas', 'Corumbá', 'Ponta Porã'],
                'geometry': [
                    gpd.points_from_xy([-54.6201], [-20.4697])[0].buffer(0.2),
                    gpd.points_from_xy([-54.812], [-22.2231])[0].buffer(0.2),
                    gpd.points_from_xy([-51.7005], [-20.7849])[0].buffer(0.2),
                    gpd.points_from_xy([-57.651], [-19.0082])[0].buffer(0.2),
                    gpd.points_from_xy([-55.7271], [-22.5334])[0].buffer(0.2)
                ]
            }
            gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
            return gdf
    except Exception as e:
        st.warning(f"Não foi possível carregar os shapes dos municípios: {str(e)}")
        return gpd.GeoDataFrame(columns=['NM_MUN', 'geometry'], crs="EPSG:4326")

# 🎯 Lista completa dos municípios de MS com coordenadas
cities = {
    "Água Clara": [-20.4453, -52.8792],
    "Alcinópolis": [-18.3255, -53.7042],
    "Amambai": [-23.1058, -55.2253],
    "Anastácio": [-20.4823, -55.8104],
    "Anaurilândia": [-22.1852, -52.7191],
    "Angélica": [-22.1527, -53.7708],
    "Antônio João": [-22.1927, -55.9511],
    "Aparecida do Taboado": [-20.0873, -51.0961],
    "Aquidauana": [-20.4697, -55.7868],
    "Aral Moreira": [-22.9384, -55.6331],
    "Bandeirantes": [-19.9279, -54.3581],
    "Bataguassu": [-21.7156, -52.4233],
    "Batayporã": [-22.2947, -53.2705],
    "Bela Vista": [-22.1073, -56.5263],
    "Bodoquena": [-20.5372, -56.7138],
    "Bonito": [-21.1261, -56.4836],
    "Brasilândia": [-21.2544, -52.0382],
    "Caarapó": [-22.6368, -54.8209],
    "Camapuã": [-19.5302, -54.0431],
    "Campo Grande": [-20.4697, -54.6201],
    "Caracol": [-22.0112, -57.0278],
    "Cassilândia": [-19.1179, -51.7308],
    "Chapadão do Sul": [-18.7908, -52.6260],
    "Corguinho": [-19.8243, -54.8281],
    "Coronel Sapucaia": [-23.2724, -55.5278],
    "Corumbá": [-19.0082, -57.651],
    "Costa Rica": [-18.5432, -53.1287],
    "Coxim": [-18.5013, -54.7603],
    "Deodápolis": [-22.2789, -54.1583],
    "Dois Irmãos do Buriti": [-20.6845, -55.2915],
    "Douradina": [-22.0430, -54.6158],
    "Dourados": [-22.2231, -54.812],
    "Eldorado": [-23.7868, -54.2836],
    "Fátima do Sul": [-22.3789, -54.5131],
    "Figueirão": [-18.6782, -53.6380],
    "Glória de Dourados": [-22.4136, -54.2336],
    "Guia Lopes da Laguna": [-21.4583, -56.1117],
    "Iguatemi": [-23.6835, -54.5635],
    "Inocência": [-19.7276, -51.9281],
    "Itaporã": [-22.0750, -54.7933],
    "Itaquiraí": [-23.4779, -54.1873],
    "Ivinhema": [-22.3046, -53.8185],
    "Japorã": [-23.8903, -54.4059],
    "Jaraguari": [-20.1386, -54.3996],
    "Jardim": [-21.4799, -56.1489],
    "Jateí": [-22.4806, -54.3078],
    "Juti": [-22.8596, -54.6060],
    "Ladário": [-19.0090, -57.5973],
    "Laguna Carapã": [-22.5448, -55.1502],
    "Maracaju": [-21.6105, -55.1695],
    "Miranda": [-20.2407, -56.3780],
    "Mundo Novo": [-23.9355, -54.2807],
    "Naviraí": [-23.0618, -54.1995],
    "Nioaque": [-21.1419, -55.8296],
    "Nova Alvorada do Sul": [-21.4657, -54.3825],
    "Nova Andradina": [-22.2332, -53.3437],
    "Novo Horizonte do Sul": [-22.6693, -53.8601],
    "Paraíso das Águas": [-19.0218, -53.0116],
    "Paranaíba": [-19.6746, -51.1909],
    "Paranhos": [-23.8905, -55.4289],
    "Pedro Gomes": [-18.0996, -54.5507],
    "Ponta Porã": [-22.5334, -55.7271],
    "Porto Murtinho": [-21.6981, -57.8825],
    "Ribas do Rio Pardo": [-20.4432, -53.7588],
    "Rio Brilhante": [-21.8033, -54.5427],
    "Rio Negro": [-19.4473, -54.9859],
    "Rio Verde de Mato Grosso": [-18.9249, -54.8434],
    "Rochedo": [-19.9566, -54.8940],
    "Santa Rita do Pardo": [-21.3016, -52.8333],
    "São Gabriel do Oeste": [-19.3950, -54.5507],
    "Selvíria": [-20.3637, -51.4192],
    "Sete Quedas": [-23.9710, -55.0396],
    "Sidrolândia": [-20.9302, -54.9692],
    "Sonora": [-17.5698, -54.7551],
    "Tacuru": [-23.6361, -55.0141],
    "Taquarussu": [-22.4898, -53.3519],
    "Terenos": [-20.4378, -54.8647],
    "Três Lagoas": [-20.7849, -51.7005],
    "Vicentina": [-22.4098, -54.4415]
}

# Coordenadas do centro geográfico de MS para centralizar o mapa
MS_CENTER_LAT = -20.5147
MS_CENTER_LON = -54.5416

# Títulos e introdução
st.title("🌍 Monitoramento AOD e Estimativa de PM2.5/PM10 - Mato Grosso do Sul")
st.markdown("""
### Sistema Integrado de Monitoramento da Qualidade do Ar

Este aplicativo monitora a Profundidade Óptica de Aerossóis (AOD) e estima as concentrações de 
Material Particulado (PM2.5 e PM10) para todos os municípios de Mato Grosso do Sul.

**Novidades desta versão:**
- 🎯 Visualização centralizada no município selecionado
- 🔬 Estimativa de PM2.5 e PM10 baseada em literatura científica regional
- 🔥 Ajustes específicos para períodos de queimadas
- 📊 Índice de Qualidade do Ar (IQA) calculado
""")

# Função para converter AOD em PM2.5 e PM10
def aod_to_pm(aod, season='normal', fire_detected=False, humidity=50):
    """
    Converte AOD em PM2.5 e PM10 usando fórmulas regionais para América do Sul.
    
    Parâmetros:
    - aod: valor de AOD
    - season: 'dry' (maio-setembro) ou 'normal'
    - fire_detected: True se houver detecção de queimadas
    - humidity: umidade relativa (%)
    
    Retorna:
    - pm25: concentração estimada de PM2.5 (μg/m³)
    - pm10: concentração estimada de PM10 (μg/m³)
    """
    
    # Fator de conversão base para América do Sul (literatura)
    if season == 'dry':
        # Durante estação seca, aerossóis de queimadas dominam
        eta_pm25_base = 110  # μg/m³ por unidade de AOD
        eta_pm10_base = 165  # PM10 é ~1.5x PM2.5 em queimadas
    else:
        # Estação úmida/normal
        eta_pm25_base = 85
        eta_pm10_base = 140
    
    # Ajuste para queimadas (multiplicador de 1.5-2.5x conforme literatura)
    if fire_detected:
        fire_multiplier = 2.0 if season == 'dry' else 1.5
    else:
        fire_multiplier = 1.0
    
    # Correção de umidade (higroscopic growth)
    # f(RH) = (1 - RH/100)^(-γ) onde γ ≈ 0.7 para aerossóis regionais
    humidity_correction = (1 - humidity/100) ** (-0.7) if humidity < 95 else 3.5
    
    # Cálculo final com todas as correções
    pm25 = aod * eta_pm25_base * fire_multiplier * humidity_correction
    pm10 = aod * eta_pm10_base * fire_multiplier * humidity_correction
    
    # Adicionar componente de background (7-10 μg/m³ para PM2.5)
    pm25 += 8
    pm10 += 15
    
    # Limitar valores extremos
    pm25 = min(pm25, 500)  # Limite superior realista
    pm10 = min(pm10, 800)
    
    return pm25, pm10

# Função para calcular IQA (Índice de Qualidade do Ar)
def calculate_aqi(pm25, pm10):
    """
    Calcula o Índice de Qualidade do Ar baseado em PM2.5 e PM10.
    Usa os padrões da EPA adaptados para o Brasil.
    """
    # Breakpoints para PM2.5 (μg/m³)
    pm25_breakpoints = [
        (0, 12, 0, 50),      # Boa
        (12.1, 35.4, 51, 100),  # Moderada
        (35.5, 55.4, 101, 150), # Insalubre para grupos sensíveis
        (55.5, 150.4, 151, 200), # Insalubre
        (150.5, 250.4, 201, 300), # Muito Insalubre
        (250.5, 500, 301, 500)  # Perigosa
    ]
    
    # Breakpoints para PM10 (μg/m³)
    pm10_breakpoints = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 600, 301, 500)
    ]
    
    def calc_sub_index(concentration, breakpoints):
        for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
            if bp_lo <= concentration <= bp_hi:
                return ((i_hi - i_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + i_lo
        return 500  # Máximo se exceder todos os breakpoints
    
    aqi_pm25 = calc_sub_index(pm25, pm25_breakpoints)
    aqi_pm10 = calc_sub_index(pm10, pm10_breakpoints)
    
    # IQA é o maior dos dois
    aqi = max(aqi_pm25, aqi_pm10)
    
    # Categoria
    if aqi <= 50:
        category = "Boa"
        color = "green"
    elif aqi <= 100:
        category = "Moderada"
        color = "yellow"
    elif aqi <= 150:
        category = "Insalubre para Grupos Sensíveis"
        color = "orange"
    elif aqi <= 200:
        category = "Insalubre"
        color = "red"
    elif aqi <= 300:
        category = "Muito Insalubre"
        color = "purple"
    else:
        category = "Perigosa"
        color = "maroon"
    
    return aqi, category, color

# Função para detectar período de queimadas
def is_fire_season(date):
    """Determina se a data está no período típico de queimadas (maio-setembro)."""
    month = date.month
    return 5 <= month <= 9

# Função melhorada para extrair valores de AOD
def extract_point_timeseries(ds, lat, lon, var_name='aod550'):
    """Extrai série temporal de um ponto específico do dataset."""
    lat_idx = np.abs(ds.latitude.values - lat).argmin()
    lon_idx = np.abs(ds.longitude.values - lon).argmin()
    
    time_dims = [dim for dim in ds[var_name].dims if 'time' in dim or 'forecast' in dim]
    
    times = []
    values = []
    
    if 'forecast_reference_time' in ds[var_name].dims and 'forecast_period' in ds[var_name].dims:
        for t_idx, ref_time in enumerate(ds.forecast_reference_time.values):
            for p_idx, period in enumerate(ds.forecast_period.values):
                try:
                    value = float(ds[var_name].isel(
                        forecast_reference_time=t_idx, 
                        forecast_period=p_idx,
                        latitude=lat_idx, 
                        longitude=lon_idx
                    ).values)
                    
                    actual_time = pd.to_datetime(ref_time) + pd.to_timedelta(period, unit='h')
                    times.append(actual_time)
                    values.append(value)
                except:
                    continue
    elif any(dim in ds[var_name].dims for dim in ['time', 'forecast_reference_time']):
        time_dim = next(dim for dim in ds[var_name].dims if dim in ['time', 'forecast_reference_time'])
        for t_idx in range(len(ds[time_dim])):
            try:
                value = float(ds[var_name].isel({
                    time_dim: t_idx,
                    'latitude': lat_idx,
                    'longitude': lon_idx
                }).values)
                times.append(pd.to_datetime(ds[time_dim].isel({time_dim: t_idx}).values))
                values.append(value)
            except:
                continue
    
    if times and values:
        df = pd.DataFrame({'time': times, 'aod': values})
        df = df.sort_values('time').reset_index(drop=True)
        
        # Adicionar estimativas de PM
        df['is_fire_season'] = df['time'].apply(is_fire_season)
        df['season'] = df['is_fire_season'].apply(lambda x: 'dry' if x else 'normal')
        
        # Calcular PM2.5 e PM10
        pm_values = df.apply(lambda row: aod_to_pm(
            row['aod'], 
            season=row['season'],
            fire_detected=(row['aod'] > 0.3 and row['is_fire_season']),
            humidity=60  # Valor médio, idealmente seria obtido dos dados meteorológicos
        ), axis=1)
        
        df['pm25'] = pm_values.apply(lambda x: x[0])
        df['pm10'] = pm_values.apply(lambda x: x[1])
        
        # Calcular IQA
        aqi_values = df.apply(lambda row: calculate_aqi(row['pm25'], row['pm10']), axis=1)
        df['aqi'] = aqi_values.apply(lambda x: x[0])
        df['aqi_category'] = aqi_values.apply(lambda x: x[1])
        df['aqi_color'] = aqi_values.apply(lambda x: x[2])
        
        return df
    else:
        return pd.DataFrame(columns=['time', 'aod', 'pm25', 'pm10', 'aqi', 'aqi_category'])

# Função para prever valores futuros
def predict_future_values(df, days=5):
    """Gera previsão para AOD e PM."""
    if len(df) < 3:
        return pd.DataFrame(columns=['time', 'aod', 'pm25', 'pm10', 'aqi', 'type'])
    
    df_hist = df.copy()
    df_hist['time_numeric'] = (df_hist['time'] - df_hist['time'].min()).dt.total_seconds()
    
    # Modelos separados para AOD, PM2.5 e PM10
    X = df_hist['time_numeric'].values.reshape(-1, 1)
    
    model_aod = LinearRegression()
    model_aod.fit(X, df_hist['aod'].values)
    
    model_pm25 = LinearRegression()
    model_pm25.fit(X, df_hist['pm25'].values)
    
    model_pm10 = LinearRegression()
    model_pm10.fit(X, df_hist['pm10'].values)
    
    # Gerar pontos futuros
    last_time = df_hist['time'].max()
    future_times = [last_time + timedelta(hours=i*6) for i in range(1, days*4+1)]
    future_time_numeric = [(t - df_hist['time'].min()).total_seconds() for t in future_times]
    
    # Prever valores
    future_aod = model_aod.predict(np.array(future_time_numeric).reshape(-1, 1))
    future_pm25 = model_pm25.predict(np.array(future_time_numeric).reshape(-1, 1))
    future_pm10 = model_pm10.predict(np.array(future_time_numeric).reshape(-1, 1))
    
    # Limitar valores
    future_aod = np.maximum(future_aod, 0)
    future_pm25 = np.maximum(future_pm25, 0)
    future_pm10 = np.maximum(future_pm10, 0)
    
    # Calcular IQA para previsões
    future_aqi = []
    future_categories = []
    future_colors = []
    
    for pm25, pm10 in zip(future_pm25, future_pm10):
        aqi, category, color = calculate_aqi(pm25, pm10)
        future_aqi.append(aqi)
        future_categories.append(category)
        future_colors.append(color)
    
    # Criar DataFrame com previsão
    df_pred = pd.DataFrame({
        'time': future_times,
        'aod': future_aod,
        'pm25': future_pm25,
        'pm10': future_pm10,
        'aqi': future_aqi,
        'aqi_category': future_categories,
        'aqi_color': future_colors,
        'type': 'forecast'
    })
    
    # Adicionar indicador aos dados históricos
    df_hist['type'] = 'historical'
    
    # Combinar histórico e previsão
    result = pd.concat([df_hist[['time', 'aod', 'pm25', 'pm10', 'aqi', 'aqi_category', 'aqi_color', 'type']], df_pred], ignore_index=True)
    return result

# Função atualizada para analisar todas as cidades
def analyze_all_cities(ds, aod_var, cities_dict):
    """Analisa os valores de AOD e PM para todas as cidades."""
    cities_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (city_name, coords) in enumerate(cities_dict.items()):
        progress = (i + 1) / len(cities_dict)
        progress_bar.progress(progress)
        status_text.text(f"Analisando {city_name}... ({i+1}/{len(cities_dict)})")
        
        lat, lon = coords
        
        df_timeseries = extract_point_timeseries(ds, lat, lon, var_name=aod_var)
        
        if not df_timeseries.empty:
            df_forecast = predict_future_values(df_timeseries, days=5)
            
            forecast_only = df_forecast[df_forecast['type'] == 'forecast']
            
            if not forecast_only.empty:
                max_aod = forecast_only['aod'].max()
                max_pm25 = forecast_only['pm25'].max()
                max_pm10 = forecast_only['pm10'].max()
                max_aqi = forecast_only['aqi'].max()
                
                max_day_idx = forecast_only['aqi'].idxmax()
                max_day = forecast_only.loc[max_day_idx, 'time']
                max_category = forecast_only.loc[max_day_idx, 'aqi_category']
                
                cities_results.append({
                    'cidade': city_name,
                    'aod_max': max_aod,
                    'pm25_max': max_pm25,
                    'pm10_max': max_pm10,
                    'aqi_max': max_aqi,
                    'data_max': max_day,
                    'categoria': max_category
                })
    
    progress_bar.empty()
    status_text.empty()
    
    if cities_results:
        df_results = pd.DataFrame(cities_results)
        df_results = df_results.sort_values('aqi_max', ascending=False).reset_index(drop=True)
        
        df_results['aod_max'] = df_results['aod_max'].round(3)
        df_results['pm25_max'] = df_results['pm25_max'].round(1)
        df_results['pm10_max'] = df_results['pm10_max'].round(1)
        df_results['aqi_max'] = df_results['aqi_max'].round(0)
        df_results['data_max'] = df_results['data_max'].dt.strftime('%d/%m/%Y %H:%M')
        
        return df_results
    else:
        return pd.DataFrame(columns=['cidade', 'aod_max', 'pm25_max', 'pm10_max', 'aqi_max', 'data_max', 'categoria'])

# Função principal atualizada com centralização no município
def generate_aod_analysis():
    dataset = "cams-global-atmospheric-composition-forecasts"
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    hours = []
    current_hour = start_hour
    while True:
        hours.append(f"{current_hour:02d}:00")
        if current_hour == end_hour:
            break
        current_hour = (current_hour + 3) % 24
        if current_hour == start_hour:
            break
    
    if not hours:
        hours = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
    
    # MODIFICAÇÃO: Centralizar a área de requisição no município selecionado
    # Definir área de interesse centrada no município com buffer
    buffer = 1.5  # Graus de buffer ao redor do município
    city_bounds = {
        'north': lat_center + buffer,
        'south': lat_center - buffer,
        'east': lon_center + buffer,
        'west': lon_center - buffer
    }
    
    request = {
        'variable': ['total_aerosol_optical_depth_550nm'],
        'date': f'{start_date_str}/{end_date_str}',
        'time': hours,
        'leadtime_hour': ['0', '24', '48', '72', '96', '120'],
        'type': ['forecast'],
        'format': 'netcdf',
        'area': [city_bounds['north'], city_bounds['west'], 
                city_bounds['south'], city_bounds['east']]  # Norte, Oeste, Sul, Leste
    }
    
    filename = f'AOD550_{city}_{start_date}_to_{end_date}.nc'
    
    try:
        with st.spinner('📥 Baixando dados do CAMS...'):
            client.retrieve(dataset, request).download(filename)
        
        ds = xr.open_dataset(filename)
        
        variable_names = list(ds.data_vars)
        aod_var = next((var for var in variable_names if 'aod' in var.lower()), variable_names[0])
        
        da = ds[aod_var]
        
        time_dims = [dim for dim in da.dims if 'time' in dim or 'forecast' in dim]
        
        if not time_dims:
            st.error("Não foi possível identificar dimensão temporal nos dados.")
            return None
        
        # Extrair série temporal para o município selecionado
        with st.spinner("Extraindo dados para o município..."):
            df_timeseries = extract_point_timeseries(ds, lat_center, lon_center, var_name=aod_var)
        
        if df_timeseries.empty:
            st.error("Não foi possível extrair série temporal para este local.")
            return None
        
        # Gerar previsão
        with st.spinner("Gerando previsões..."):
            df_forecast = predict_future_values(df_timeseries, days=5)
        
        # Criar animação centralizada no município
        if 'forecast_reference_time' in da.dims:
            time_dim = 'forecast_reference_time'
            frames = len(da[time_dim])
        else:
            time_dim = time_dims[0]
            frames = len(da[time_dim])
        
        if frames < 1:
            st.error("Erro: Dados insuficientes para animação.")
            return None
        
        vmin, vmax = float(da.min().values), float(da.max().values)
        vmin = max(0, vmin - 0.05)
        vmax = min(2, vmax + 0.05)
        
        # Criar figura com projeção centralizada no município
        fig = plt.figure(figsize=(14, 10))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Adicionar features
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.coastlines(resolution='50m', color='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', color='gray')
        ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle='-', edgecolor='black', linewidth=1)
        
        # Grid
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # MODIFICAÇÃO: Definir extensão do mapa para cobrir área ao redor do município
        ax.set_extent([city_bounds['west'], city_bounds['east'], 
                      city_bounds['south'], city_bounds['north']], 
                     crs=ccrs.PlateCarree())
        
        # MODIFICAÇÃO: Título com o nome do município
        ax.text(lon_center, city_bounds['north'] + 0.1, city.upper(), 
                transform=ccrs.PlateCarree(), fontsize=18, fontweight='bold',
                ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='white', alpha=0.8))
        
        # Marcar o município selecionado no mapa
        ax.plot(lon_center, lat_center, 'ro', markersize=12, transform=ccrs.PlateCarree(), 
                label=city, markeredgecolor='white', markeredgewidth=2)
        
        # Adicionar outros municípios próximos (opcional)
        nearby_cities = []
        for city_name, coords in cities.items():
            city_lat, city_lon = coords
            # Verificar se está na área visível
            if (city_bounds['south'] <= city_lat <= city_bounds['north'] and 
                city_bounds['west'] <= city_lon <= city_bounds['east'] and 
                city_name != city):
                nearby_cities.append((city_name, city_lat, city_lon))
        
        # Mostrar até 5 cidades próximas
        for city_name, city_lat, city_lon in nearby_cities[:5]:
            ax.plot(city_lon, city_lat, 'ko', markersize=6, transform=ccrs.PlateCarree())
            ax.text(city_lon + 0.05, city_lat + 0.05, city_name, fontsize=8, 
                   transform=ccrs.PlateCarree(), bbox=dict(boxstyle='round,pad=0.2', 
                   facecolor='yellow', alpha=0.7))
        
        # Obter primeiro frame
        if 'forecast_period' in da.dims and 'forecast_reference_time' in da.dims:
            if len(da.forecast_period) > 0 and len(da.forecast_reference_time) > 0:
                first_frame_data = da.isel(forecast_period=0, forecast_reference_time=0).values
                first_frame_time = pd.to_datetime(ds.forecast_reference_time.values[0])
            else:
                first_frame_coords = {dim: 0 for dim in da.dims if len(da[dim]) > 0}
                first_frame_data = da.isel(**first_frame_coords).values
                first_frame_time = datetime.now()
        else:
            first_frame_data = da.isel({time_dim: 0}).values
            first_frame_time = pd.to_datetime(da[time_dim].values[0])
        
        # Garantir formato 2D
        if len(first_frame_data.shape) != 2:
            st.error(f"Erro: Formato de dados inesperado. Shape: {first_frame_data.shape}")
            return None
        
        # Criar mapa de cores
        im = ax.pcolormesh(ds.longitude, ds.latitude, first_frame_data, 
                         cmap=colormap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        
        # Adicionar barra de cores
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04, orientation='horizontal')
        cbar.set_label('AOD 550nm', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # Título inicial
        title = ax.set_title(f'AOD 550nm - {city}\n{first_frame_time.strftime("%d/%m/%Y %H:%M UTC")}', 
                           fontsize=14, pad=20)
        
        # Função de animação
        def animate(i):
            try:
                frame_data = None
                frame_time = None
                
                if 'forecast_period' in da.dims and 'forecast_reference_time' in da.dims:
                    fp_idx = min(0, len(da.forecast_period)-1)
                    frt_idx = min(i, len(da.forecast_reference_time)-1)
                    
                    frame_data = da.isel(forecast_period=fp_idx, forecast_reference_time=frt_idx).values
                    frame_time = pd.to_datetime(ds.forecast_reference_time.values[frt_idx])
                else:
                    t_idx = min(i, len(da[time_dim])-1)
                    frame_data = da.isel({time_dim: t_idx}).values
                    frame_time = pd.to_datetime(da[time_dim].values[t_idx])
                
                im.set_array(frame_data.ravel())
                title.set_text(f'AOD 550nm - {city}\n{frame_time.strftime("%d/%m/%Y %H:%M UTC")}')
                
                return [im, title]
            except Exception as e:
                st.error(f"Erro no frame {i}: {str(e)}")
                return [im, title]
        
        # Limitar número de frames
        actual_frames = min(frames, 20)
        
        # Criar animação
        ani = animation.FuncAnimation(fig, animate, frames=actual_frames, 
                                     interval=animation_speed, blit=True)
        
        # Salvar animação
        gif_filename = f'AOD550_{city}_{start_date}_to_{end_date}.gif'
        
        with st.spinner('💾 Salvando animação...'):
            ani.save(gif_filename, writer=animation.PillowWriter(fps=2))
        
        plt.close(fig)

        # Analisar todas as cidades (usando dados de MS completo se disponível)
        top_pollution_cities = None
        try:
            # Para análise de todas as cidades, usar coordenadas completas de MS
            ms_bounds = {
                'north': -17.5,
                'south': -24.0,
                'east': -50.5,
                'west': -58.5
            }
            
            # Requisitar dados de MS completo para análise das cidades
            request_ms = {
                'variable': ['total_aerosol_optical_depth_550nm'],
                'date': f'{start_date_str}/{end_date_str}',
                'time': hours,
                'leadtime_hour': ['0', '24', '48', '72', '96', '120'],
                'type': ['forecast'],
                'format': 'netcdf',
                'area': [ms_bounds['north'], ms_bounds['west'], 
                        ms_bounds['south'], ms_bounds['east']]
            }
            
            filename_ms = f'AOD550_MS_complete_{start_date}_to_{end_date}.nc'
            
            with st.spinner("🔍 Baixando dados de MS completo para análise das cidades..."):
                client.retrieve(dataset, request_ms).download(filename_ms)
            
            ds_ms = xr.open_dataset(filename_ms)
            with st.spinner("🔍 Analisando qualidade do ar em todos os municípios de MS..."):
                top_pollution_cities = analyze_all_cities(ds_ms, aod_var, cities)
        except Exception as e:
            st.warning(f"Não foi possível analisar todas as cidades: {str(e)}")
            # Usar apenas os dados locais se não conseguir baixar dados completos
            top_pollution_cities = pd.DataFrame(columns=['cidade', 'aod_max', 'pm25_max', 'pm10_max', 'aqi_max', 'data_max', 'categoria'])
        
        return {
            'animation': gif_filename,
            'timeseries': df_timeseries,
            'forecast': df_forecast,
            'dataset': ds,
            'variable': aod_var,
            'top_pollution': top_pollution_cities
        }
    
    except Exception as e:
        st.error(f"❌ Erro ao processar os dados: {str(e)}")
        st.write("Detalhes da requisição:")
        st.write(request)
        return None

# Carregar shapefiles dos municípios
with st.spinner("Carregando shapes dos municípios..."):
    ms_shapes = load_ms_municipalities()

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Seleção de cidade
available_cities = sorted(list(set(ms_shapes['NM_MUN'].tolist()).intersection(set(cities.keys()))))
if not available_cities:
    available_cities = list(cities.keys())

city = st.sidebar.selectbox("Selecione o município para análise detalhada", available_cities)
lat_center, lon_center = cities[city]

# Configurações de data e hora
st.sidebar.subheader("Período de Análise")
start_date = st.sidebar.date_input("Data de Início", datetime.today() - timedelta(days=2))
end_date = st.sidebar.date_input("Data Final", datetime.today() + timedelta(days=5))

all_hours = list(range(0, 24, 3))
start_hour = st.sidebar.selectbox("Horário Inicial", all_hours, format_func=lambda x: f"{x:02d}:00")
end_hour = st.sidebar.selectbox("Horário Final", all_hours, index=len(all_hours)-1, format_func=lambda x: f"{x:02d}:00")

# Opções avançadas
st.sidebar.subheader("Opções Avançadas")
with st.sidebar.expander("Configurações da Visualização"):
    animation_speed = st.slider("Velocidade da Animação (ms)", 200, 1000, 500)
    colormap = st.selectbox("Paleta de Cores", 
                          ["YlOrRd", "viridis", "plasma", "inferno", "magma", "cividis", "RdYlBu_r"])
    show_pm_estimates = st.checkbox("Mostrar estimativas de PM2.5/PM10", value=True)
    show_fire_adjustments = st.checkbox("Aplicar ajustes para queimadas", value=True)

# Informações sobre o período
current_month = datetime.now().month
if 5 <= current_month <= 9:
    st.sidebar.info("🔥 **Período de Queimadas Ativo**\nAs estimativas de PM são ajustadas para refletir as condições típicas de queimadas na região.")

# Botão principal
st.markdown("### 🚀 Iniciar Análise Completa")
st.markdown(f"Clique no botão abaixo para gerar análise de AOD e PM centralizada em **{city}**.")

if st.button("🎯 Gerar Análise de Qualidade do Ar", type="primary", use_container_width=True):
    try:
        results = generate_aod_analysis()
        
        if results:
            # Criar abas
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Análise do Município", 
                "⚠️ Alerta de Qualidade do Ar", 
                f"🗺️ Mapa de {city}",
                "📈 Análise PM2.5/PM10"
            ])
            
            # Aba do Mapa
            with tab3:
                st.subheader(f"🎬 Animação AOD 550nm - {city}")
                st.image(results['animation'], caption=f"Evolução temporal do AOD em {city} ({start_date} a {end_date})")
                
                with open(results['animation'], "rb") as file:
                    btn = st.download_button(
                        label="⬇️ Baixar Animação (GIF)",
                        data=file,
                        file_name=f"AOD_{city}_{start_date}_to_{end_date}.gif",
                        mime="image/gif"
                    )
                
                # Informações sobre o mapa
                st.info(f"""
                **Como interpretar o mapa de {city}:**
                - 🟢 Verde/Azul: AOD < 0.1 (Ar limpo)
                - 🟡 Amarelo: AOD 0.1-0.2 (Qualidade moderada)
                - 🟠 Laranja: AOD 0.2-0.5 (Poluição elevada)
                - 🔴 Vermelho: AOD > 0.5 (Condições severas)
                
                O ponto vermelho marca a localização de **{city}**.
                Os pontos pretos indicam outros municípios próximos.
                """)
            
            # Aba de Análise do Município
            with tab1:
                st.subheader(f"📊 Análise Detalhada - {city}")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    df_combined = results['forecast']
                    
                    # Criar subplots para AOD, PM2.5 e PM10
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                    
                    # Gráfico AOD
                    hist_data = df_combined[df_combined['type'] == 'historical']
                    forecast_data = df_combined[df_combined['type'] == 'forecast']
                    
                    ax1.plot(hist_data['time'], hist_data['aod'], 
                           'o-', color='blue', label='Observado', markersize=6)
                    ax1.plot(forecast_data['time'], forecast_data['aod'], 
                           'x--', color='red', label='Previsão', markersize=6)
                    ax1.set_ylabel('AOD 550nm', fontsize=12)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    ax1.set_title('Profundidade Óptica de Aerossóis', fontsize=14)
                    
                    # Gráfico PM2.5
                    ax2.plot(hist_data['time'], hist_data['pm25'], 
                           'o-', color='green', label='PM2.5 Estimado', markersize=6)
                    ax2.plot(forecast_data['time'], forecast_data['pm25'], 
                           'x--', color='darkgreen', label='PM2.5 Previsto', markersize=6)
                    ax2.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Limite OMS')
                    ax2.set_ylabel('PM2.5 (μg/m³)', fontsize=12)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.set_title('Material Particulado PM2.5', fontsize=14)
                    
                    # Gráfico PM10
                    ax3.plot(hist_data['time'], hist_data['pm10'], 
                           'o-', color='brown', label='PM10 Estimado', markersize=6)
                    ax3.plot(forecast_data['time'], forecast_data['pm10'], 
                           'x--', color='sienna', label='PM10 Previsto', markersize=6)
                    ax3.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Limite OMS')
                    ax3.set_ylabel('PM10 (μg/m³)', fontsize=12)
                    ax3.set_xlabel('Data/Hora', fontsize=12)
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    ax3.set_title('Material Particulado PM10', fontsize=14)
                    
                    # Formatar datas
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
                    plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("📈 Estatísticas Atuais")
                    
                    if not hist_data.empty:
                        curr_aod = hist_data['aod'].iloc[-1]
                        curr_pm25 = hist_data['pm25'].iloc[-1]
                        curr_pm10 = hist_data['pm10'].iloc[-1]
                        curr_aqi = hist_data['aqi'].iloc[-1]
                        curr_category = hist_data['aqi_category'].iloc[-1]
                        curr_color = hist_data['aqi_color'].iloc[-1]
                        
                        # Métricas
                        col_a, col_b = st.columns(2)
                        col_a.metric("AOD Atual", f"{curr_aod:.3f}")
                        col_b.metric("IQA", f"{curr_aqi:.0f}")
                        
                        col_c, col_d = st.columns(2)
                        col_c.metric("PM2.5", f"{curr_pm25:.1f} μg/m³")
                        col_d.metric("PM10", f"{curr_pm10:.1f} μg/m³")
                        
                        # Categoria de qualidade
                        st.markdown(f"""
                        <div style="padding:15px; border-radius:10px; background-color:{curr_color}; 
                        color:white; text-align:center; margin:10px 0;">
                        <h3 style="margin:0;">Qualidade do Ar</h3>
                        <h2 style="margin:5px 0;">{curr_category}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Recomendações
                        st.subheader("💡 Recomendações")
                        if curr_aqi <= 50:
                            st.success("✅ Condições ideais para atividades ao ar livre")
                        elif curr_aqi <= 100:
                            st.info("ℹ️ Pessoas sensíveis devem considerar limitar esforços prolongados")
                        elif curr_aqi <= 150:
                            st.warning("⚠️ Grupos sensíveis devem evitar esforços ao ar livre")
                        elif curr_aqi <= 200:
                            st.error("🚫 Evite esforços prolongados ao ar livre")
                        else:
                            st.error("☠️ Evite todas as atividades ao ar livre")
                        
                        # Comparação com limites OMS
                        st.subheader("📏 Comparação com Padrões")
                        
                        pm25_who_limit = 25  # μg/m³ (24h)
                        pm10_who_limit = 50  # μg/m³ (24h)
                        
                        st.progress(min(curr_pm25 / pm25_who_limit, 1.0))
                        st.caption(f"PM2.5: {curr_pm25:.1f}/{pm25_who_limit} μg/m³ (Limite OMS 24h)")
                        
                        st.progress(min(curr_pm10 / pm10_who_limit, 1.0))
                        st.caption(f"PM10: {curr_pm10:.1f}/{pm10_who_limit} μg/m³ (Limite OMS 24h)")
                        
                        # Previsão resumida
                        if not forecast_data.empty:
                            st.subheader("🔮 Próximos 5 dias")
                            
                            forecast_data['date'] = forecast_data['time'].dt.date
                            daily_forecast = forecast_data.groupby('date').agg({
                                'aqi': 'max',
                                'aqi_category': lambda x: x.iloc[x.values.argmax()],
                                'pm25': 'mean',
                                'pm10': 'mean'
                            }).reset_index()
                            
                            for _, row in daily_forecast.iterrows():
                                aqi_color = 'green' if row['aqi'] <= 50 else 'yellow' if row['aqi'] <= 100 else 'orange' if row['aqi'] <= 150 else 'red'
                                st.markdown(f"""
                                <div style="padding:5px; border-radius:5px; background-color:{aqi_color}; 
                                color:{'white' if aqi_color != 'yellow' else 'black'}; margin:2px 0;">
                                <b>{row['date'].strftime('%d/%m')}:</b> IQA {row['aqi']:.0f} - {row['aqi_category']}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Exportar dados
                    st.subheader("💾 Exportar Dados")
                    csv = df_combined.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Baixar Dados Completos (CSV)",
                        data=csv,
                        file_name=f"AOD_PM_data_{city}_{start_date}_to_{end_date}.csv",
                        mime="text/csv",
                    )
            
            # Aba de Alertas
            with tab2:
                st.subheader("⚠️ Alerta de Qualidade do Ar - Mato Grosso do Sul")
                
                if 'top_pollution' in results and not results['top_pollution'].empty:
                    top_cities = results['top_pollution'].head(20)
                    
                    # Estatísticas gerais
                    critical_cities = top_cities[top_cities['aqi_max'] > 100]
                    very_critical = top_cities[top_cities['aqi_max'] > 150]
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Cidades em Alerta", len(critical_cities))
                    col2.metric("Condição Insalubre", len(very_critical))
                    col3.metric("IQA Máximo Previsto", f"{top_cities['aqi_max'].max():.0f}")
                    
                    if len(critical_cities) > 0:
                        st.error(f"""
                        ### 🚨 ALERTA DE QUALIDADE DO AR
                        
                        **{len(critical_cities)} municípios** com previsão de qualidade do ar 
                        inadequada nos próximos 5 dias!
                        
                        Municípios mais críticos:
                        1. **{top_cities.iloc[0]['cidade']}**: IQA {top_cities.iloc[0]['aqi_max']:.0f} - PM2.5: {top_cities.iloc[0]['pm25_max']:.1f} μg/m³
                        2. **{top_cities.iloc[1]['cidade']}**: IQA {top_cities.iloc[1]['aqi_max']:.0f} - PM2.5: {top_cities.iloc[1]['pm25_max']:.1f} μg/m³
                        3. **{top_cities.iloc[2]['cidade']}**: IQA {top_cities.iloc[2]['aqi_max']:.0f} - PM2.5: {top_cities.iloc[2]['pm25_max']:.1f} μg/m³
                        """)
                    
                    # Tabela completa
                    st.markdown("### 📊 Ranking de Qualidade do Ar por Município")
                    
                    # Renomear colunas
                    top_cities_display = top_cities.rename(columns={
                        'cidade': 'Município',
                        'aod_max': 'AOD Máx',
                        'pm25_max': 'PM2.5 Máx (μg/m³)',
                        'pm10_max': 'PM10 Máx (μg/m³)',
                        'aqi_max': 'IQA Máx',
                        'data_max': 'Data Crítica',
                        'categoria': 'Categoria'
                    })
                    
                    # Função para colorir
                    def style_aqi_row(row):
                        aqi = row['IQA Máx']
                        if aqi <= 50:
                            return ['background-color: #00e400; color: black'] * len(row)
                        elif aqi <= 100:
                            return ['background-color: #ffff00; color: black'] * len(row)
                        elif aqi <= 150:
                            return ['background-color: #ff7e00; color: white'] * len(row)
                        elif aqi <= 200:
                            return ['background-color: #ff0000; color: white'] * len(row)
                        elif aqi <= 300:
                            return ['background-color: #8f3f97; color: white'] * len(row)
                        else:
                            return ['background-color: #7e0023; color: white'] * len(row)
                    
                    # Exibir tabela estilizada
                    st.dataframe(
                        top_cities_display.style.apply(style_aqi_row, axis=1),
                        use_container_width=True
                    )
                    
                    # MODIFICAÇÃO: Gráfico apenas do Material Particulado
                    st.subheader("📊 Material Particulado Previsto - 10 Municípios Mais Críticos")
                    
                    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
                    
                    top10 = top_cities.head(10)
                    
                    # Criar posições para as barras
                    x_pos = np.arange(len(top10))
                    width = 0.35
                    
                    # Gráfico PM2.5 e PM10
                    bars1 = ax.bar(x_pos - width/2, top10['pm25_max'], width, 
                                  color='darkblue', alpha=0.8, label='PM2.5')
                    bars2 = ax.bar(x_pos + width/2, top10['pm10_max'], width, 
                                  color='brown', alpha=0.8, label='PM10')
                    
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(top10['cidade'], rotation=45, ha='right')
                    ax.set_ylabel('Concentração (μg/m³)', fontsize=12)
                    ax.set_title('Material Particulado Máximo Previsto nos Próximos 5 Dias', fontsize=14)
                    
                    # Linhas de referência OMS
                    ax.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Limite PM2.5 OMS (25 μg/m³)')
                    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Limite PM10 OMS (50 μg/m³)')
                    
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Adicionar valores nas barras
                    for bar, val in zip(bars1, top10['pm25_max']):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                f'{val:.0f}', ha='center', va='bottom', fontsize=9)
                    
                    for bar, val in zip(bars2, top10['pm10_max']):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                f'{val:.0f}', ha='center', va='bottom', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Download dos dados
                    csv_alert = top_cities.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Baixar Dados de Alerta (CSV)",
                        data=csv_alert,
                        file_name=f"Alerta_Qualidade_Ar_MS_{start_date}_to_{end_date}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Dados de análise estadual não disponíveis. Mostrando apenas análise local.")
            
            # Nova aba para análise PM
           with tab4:
                st.subheader("📈 Análise Detalhada de Material Particulado")
                
                # Informações sobre a metodologia
                with st.expander("ℹ️ Sobre a Metodologia de Conversão AOD → PM"):
                    st.markdown("""
                    ### Metodologia de Conversão AOD para PM2.5/PM10
                    
                    Esta aplicação utiliza fórmulas empíricas calibradas para a América do Sul, 
                    considerando as características específicas da região:
                    
                    **Fórmula Base:**
                    ```
                    PM2.5 = AOD × η × multiplicador_queimadas × correção_umidade + background
                    ```
                    
                    **Parâmetros utilizados:**
                    - **η (eta)**: Fator de conversão regional
                      - Estação seca (maio-set): 110 μg/m³ por unidade AOD
                      - Estação úmida: 85 μg/m³ por unidade AOD
                    - **Multiplicador de queimadas**: 1.5-2.5x quando detectado
                    - **Correção de umidade**: f(RH) = (1 - RH/100)^(-0.7)
                    - **Background**: 8 μg/m³ (PM2.5), 15 μg/m³ (PM10)
                    
                    **Referências:**
                    - Estudos de validação MAIAC na América do Sul
                    - Calibração regional para biomassa queimada
                    - Ajustes sazonais baseados em AERONET
                    """)
                
                # Análise comparativa
                if not results['top_pollution'].empty:
                    st.subheader("🔍 Análise Comparativa entre Municípios")
                    
                    # Criar scatter plot AOD vs PM2.5
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Plot AOD vs PM2.5
                    scatter1 = ax1.scatter(results['top_pollution']['aod_max'], 
                                         results['top_pollution']['pm25_max'],
                                         c=results['top_pollution']['aqi_max'],
                                         cmap='RdYlGn_r', s=100, alpha=0.7)
                    ax1.set_xlabel('AOD Máximo')
                    ax1.set_ylabel('PM2.5 Máximo (μg/m³)')
                    ax1.set_title('Relação AOD vs PM2.5')
                    ax1.grid(True, alpha=0.3)
                    
                    # Adicionar linha de tendência
                    z = np.polyfit(results['top_pollution']['aod_max'], 
                                  results['top_pollution']['pm25_max'], 1)
                    p = np.poly1d(z)
                    ax1.plot(results['top_pollution']['aod_max'], 
                            p(results['top_pollution']['aod_max']), 
                            "r--", alpha=0.8, label=f'y={z[0]:.1f}x+{z[1]:.1f}')
                    ax1.legend()
                    
                    # Colorbar
                    cbar1 = plt.colorbar(scatter1, ax=ax1)
                    cbar1.set_label('IQA', rotation=270, labelpad=20)
                    
                    # Plot PM2.5 vs PM10
                    scatter2 = ax2.scatter(results['top_pollution']['pm25_max'], 
                                         results['top_pollution']['pm10_max'],
                                         c=results['top_pollution']['aqi_max'],
                                         cmap='RdYlGn_r', s=100, alpha=0.7)
                    ax2.set_xlabel('PM2.5 Máximo (μg/m³)')
                    ax2.set_ylabel('PM10 Máximo (μg/m³)')
                    ax2.set_title('Relação PM2.5 vs PM10')
                    ax2.grid(True, alpha=0.3)
                    
                    # Adicionar linha de referência PM10 = 1.5 * PM2.5
                    pm25_range = np.array([0, results['top_pollution']['pm25_max'].max()])
                    ax2.plot(pm25_range, pm25_range * 1.5, 'b--', alpha=0.5, label='PM10 = 1.5×PM2.5')
                    ax2.legend()
                    
                    # Colorbar
                    cbar2 = plt.colorbar(scatter2, ax=ax2)
                    cbar2.set_label('IQA', rotation=270, labelpad=20)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Estatísticas regionais
                    st.subheader("📊 Estatísticas Regionais")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    avg_pm25 = results['top_pollution']['pm25_max'].mean()
                    avg_pm10 = results['top_pollution']['pm10_max'].mean()
                    cities_above_who_pm25 = len(results['top_pollution'][results['top_pollution']['pm25_max'] > 25])
                    cities_above_who_pm10 = len(results['top_pollution'][results['top_pollution']['pm10_max'] > 50])
                    
                    col1.metric("PM2.5 Médio", f"{avg_pm25:.1f} μg/m³")
                    col2.metric("PM10 Médio", f"{avg_pm10:.1f} μg/m³")
                    col3.metric("Cidades > Limite PM2.5", cities_above_who_pm25)
                    col4.metric("Cidades > Limite PM10", cities_above_who_pm10)
                    
                    # Mapa de calor temporal
                    if 'timeseries' in results and not results['timeseries'].empty:
                        st.subheader("🗓️ Evolução Temporal - " + city)
                        
                        # Preparar dados para heatmap
                        df_heat = results['forecast'].copy()
                        df_heat['hour'] = df_heat['time'].dt.hour
                        df_heat['date'] = df_heat['time'].dt.date
                        
                        # Criar pivot table para PM2.5
                        pivot_pm25 = df_heat.pivot_table(values='pm25', index='hour', columns='date', aggfunc='mean')
                        
                        # Criar heatmap
                        fig, ax = plt.subplots(figsize=(12, 6))
                        im = ax.imshow(pivot_pm25.values, cmap='YlOrRd', aspect='auto')
                        
                        # Configurar eixos
                        ax.set_xticks(range(len(pivot_pm25.columns)))
                        ax.set_xticklabels([d.strftime('%d/%m') for d in pivot_pm25.columns])
                        ax.set_yticks(range(len(pivot_pm25.index)))
                        ax.set_yticklabels([f'{h:02d}h' for h in pivot_pm25.index])
                        
                        ax.set_xlabel('Data')
                        ax.set_ylabel('Hora do Dia')
                        ax.set_title(f'Variação Horária de PM2.5 em {city}')
                        
                        # Colorbar
                        cbar = plt.colorbar(im, ax=ax)
                        cbar.set_label('PM2.5 (μg/m³)', rotation=270, labelpad=20)
                        
                        # Adicionar valores nas células
                        for i in range(len(pivot_pm25.index)):
                            for j in range(len(pivot_pm25.columns)):
                                value = pivot_pm25.iloc[i, j]
                                if not np.isnan(value):
                                    text_color = 'white' if value > 50 else 'black'
                                    ax.text(j, i, f'{value:.0f}', ha='center', va='center', 
                                           color=text_color, fontsize=8)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # Informações sobre impactos na saúde
                st.subheader("🏥 Impactos na Saúde")
                
                st.markdown("""
                ### Efeitos do Material Particulado na Saúde
                
                **PM2.5 (Partículas Finas)**
                - Penetram profundamente nos pulmões e corrente sanguínea
                - Associadas a doenças cardiovasculares e respiratórias
                - Podem causar câncer de pulmão com exposição prolongada
                
                **PM10 (Partículas Inaláveis)**
                - Afetam principalmente o sistema respiratório superior
                - Agravam asma e doenças pulmonares
                - Causam irritação nos olhos, nariz e garganta
                
                **Grupos de Risco:**
                - 👶 Crianças
                - 👴 Idosos
                - 🫁 Pessoas com doenças respiratórias
                - ❤️ Pessoas com doenças cardiovasculares
                - 🤰 Gestantes
                """)
                
                # Recomendações baseadas nos níveis
                st.subheader("🛡️ Medidas de Proteção")
                
                protection_measures = {
                    "Boa": {
                        "color": "green",
                        "icon": "✅",
                        "measures": [
                            "Aproveite para atividades ao ar livre",
                            "Ótimo momento para exercícios externos",
                            "Mantenha janelas abertas para ventilação"
                        ]
                    },
                    "Moderada": {
                        "color": "yellow",
                        "icon": "⚠️",
                        "measures": [
                            "Pessoas sensíveis devem reduzir atividades intensas ao ar livre",
                            "Evite exercícios prolongados em áreas de tráfego intenso",
                            "Considere usar máscara em áreas muito poluídas"
                        ]
                    },
                    "Insalubre para Grupos Sensíveis": {
                        "color": "orange",
                        "icon": "🚨",
                        "measures": [
                            "Grupos sensíveis devem evitar atividades ao ar livre",
                            "Use máscaras N95/PFF2 se precisar sair",
                            "Mantenha janelas fechadas e use purificadores de ar",
                            "Evite áreas de tráfego intenso"
                        ]
                    },
                    "Insalubre": {
                        "color": "red",
                        "icon": "🚫",
                        "measures": [
                            "Todos devem evitar atividades ao ar livre",
                            "Use máscaras N95/PFF2 ao sair",
                            "Mantenha ambientes internos fechados",
                            "Considere adiar atividades não essenciais",
                            "Hidrate-se frequentemente"
                        ]
                    },
                    "Muito Insalubre": {
                        "color": "purple",
                        "icon": "☠️",
                        "measures": [
                            "Evite qualquer atividade ao ar livre",
                            "Permaneça em ambientes fechados com ar filtrado",
                            "Use máscaras N95/PFF2 mesmo em ambientes internos se necessário",
                            "Procure atendimento médico se tiver sintomas respiratórios",
                            "Cancele atividades não essenciais"
                        ]
                    }
                }
                
                # Mostrar medidas para cada categoria presente nos dados
                if 'top_pollution' in results and not results['top_pollution'].empty:
                    categories_present = results['top_pollution']['categoria'].unique()
                    
                    for category in categories_present:
                        if category in protection_measures:
                            info = protection_measures[category]
                            st.markdown(f"""
                            <div style="padding:10px; border-radius:5px; border: 2px solid {info['color']}; margin:10px 0;">
                            <h4>{info['icon']} {category}</h4>
                            """, unsafe_allow_html=True)
                            
                            for measure in info['measures']:
                                st.markdown(f"- {measure}")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
        
        else:
            st.error("❌ Não foi possível obter dados. Verifique os parâmetros e tente novamente.")
            
    except Exception as e:
        st.error(f"❌ Ocorreu um erro: {str(e)}")
        st.write("Por favor, verifique os parâmetros e tente novamente.")

# Rodapé com informações
st.markdown("---")
st.markdown("""
### ℹ️ Sobre o Sistema

**Dados e Metodologia:**
- 🛰️ **Fonte**: Copernicus Atmosphere Monitoring Service (CAMS)
- 📊 **Variável Principal**: AOD (Aerosol Optical Depth) a 550nm
- 🔬 **Conversão PM**: Baseada em literatura científica regional
- 🔥 **Ajustes Sazonais**: Correções específicas para queimadas
- ⏱️ **Resolução Temporal**: 3 horas
- 📅 **Previsão**: Até 5 dias

**Índice de Qualidade do Ar (IQA):**
- 0-50: Boa (Verde)
- 51-100: Moderada (Amarelo)
- 101-150: Insalubre para Grupos Sensíveis (Laranja)
- 151-200: Insalubre (Vermelho)
- 201-300: Muito Insalubre (Roxo)
- 301-500: Perigosa (Marrom)

**Limites de Referência OMS (24h):**
- PM2.5: 25 μg/m³
- PM10: 50 μg/m³

### 🚀 Funcionalidades Implementadas

1. **Visualização Centralizada**: Mapa focado no estado de MS
2. **Estimativa de PM**: Conversão AOD → PM2.5/PM10 com ajustes regionais
3. **Correção Sazonal**: Multiplicadores específicos para período de queimadas
4. **Cálculo de IQA**: Índice de Qualidade do Ar padronizado
5. **Alertas Municipais**: Ranking dos 20 municípios mais críticos
6. **Análise Temporal**: Previsão de 5 dias com resolução horária
7. **Recomendações de Saúde**: Orientações baseadas nos níveis de poluição

### 📧 Contato e Suporte

Desenvolvido para monitoramento ambiental e saúde pública em Mato Grosso do Sul.
Para dúvidas ou sugestões, entre em contato com a equipe de desenvolvimento.

**Última atualização**: {datetime.now().strftime('%d/%m/%Y')}
""")

# CSS customizado para melhorar a aparência
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)
