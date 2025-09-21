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
st.set_page_config(layout="wide", page_title="Monitor PM2.5/PM10 - MS", page_icon="🌍")

# ✅ Carregar autenticação a partir do secrets.toml
try:
    ads_url = st.secrets["ads"]["url"]
    ads_key = st.secrets["ads"]["key"]
    client = cdsapi.Client(url=ads_url, key=ads_key)
except Exception as e:
    st.error("❌ Erro ao carregar as credenciais do CDS API. Verifique seu secrets.toml.")
    st.stop()

# Função para baixar shapefile dos municípios de MS (modificada)
@st.cache_data
def load_ms_municipalities():
    try:
        url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/" + \
              "municipio_2022/UFs/MS/MS_Municipios_2022.zip"
        
        try:
            gdf = gpd.read_file(url)
            # Garantir que temos a coluna de nome do município
            if 'NM_MUN' not in gdf.columns and 'NM_MUNICIP' in gdf.columns:
                gdf['NM_MUN'] = gdf['NM_MUNICIP']
            elif 'NM_MUN' not in gdf.columns and 'NOME' in gdf.columns:
                gdf['NM_MUN'] = gdf['NOME']
            return gdf
        except Exception as e:
            st.warning(f"Erro ao carregar shapefile oficial do IBGE: {e}")
            # Fallback com dados mais realistas
            return create_fallback_shapefile()
    except Exception as e:
        st.warning(f"Não foi possível carregar os shapes dos municípios: {str(e)}")
        return create_fallback_shapefile()

# [AQUI VOCÊ COLA SUA LISTA DE CIDADES]
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

def create_fallback_shapefile():
    """Cria um shapefile simplificado caso o oficial falhe"""
    # Criar polígonos aproximados para alguns municípios principais
    from shapely.geometry import Polygon
    
    municipalities_data = []
    for city_name, (lat, lon) in cities.items():
        # Criar um polígono aproximado (quadrado) ao redor de cada cidade
        buffer_size = 0.15  # Aproximadamente 15km
        polygon = Polygon([
            (lon - buffer_size, lat - buffer_size),
            (lon + buffer_size, lat - buffer_size),
            (lon + buffer_size, lat + buffer_size),
            (lon - buffer_size, lat + buffer_size),
            (lon - buffer_size, lat - buffer_size)
        ])
        municipalities_data.append({
            'NM_MUN': city_name,
            'geometry': polygon
        })
    
    return gpd.GeoDataFrame(municipalities_data, crs="EPSG:4326")

# Títulos e introdução
st.title("🌍 Monitoramento PM2.5 e PM10 - Mato Grosso do Sul")
st.markdown("""
### Sistema Integrado de Monitoramento da Qualidade do Ar

Este aplicativo monitora diretamente as concentrações de Material Particulado (PM2.5 e PM10) 
para todos os municípios de Mato Grosso do Sul usando dados diretos do CAMS.

**Características desta versão:**
- 📊 Dados diretos de PM2.5 e PM10 do CAMS (sem conversão de AOD)
- 🎯 Visualização centralizada no município selecionado com contorno municipal
- 📈 Índice de Qualidade do Ar (IQA) calculado
- 🔮 Previsões para os próximos 5 dias
- 🎬 Animação temporal para PM2.5 e PM10
""")

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

# Função para extrair série temporal de PM2.5 e PM10
def extract_pm_timeseries(ds, lat, lon, pm25_var, pm10_var):
    """Extrai série temporal de PM2.5 e PM10 de um ponto específico do dataset."""
    lat_idx = np.abs(ds.latitude.values - lat).argmin()
    lon_idx = np.abs(ds.longitude.values - lon).argmin()
    
    times = []
    pm25_values = []
    pm10_values = []
    
    # Identificar dimensões temporais
    time_dims = [dim for dim in ds.dims if 'time' in dim or 'forecast' in dim]
    
    if 'forecast_reference_time' in ds.dims and 'forecast_period' in ds.dims:
        for t_idx, ref_time in enumerate(ds.forecast_reference_time.values):
            for p_idx, period in enumerate(ds.forecast_period.values):
                try:
                    pm25_val = float(ds[pm25_var].isel(
                        forecast_reference_time=t_idx, 
                        forecast_period=p_idx,
                        latitude=lat_idx, 
                        longitude=lon_idx
                    ).values)
                    
                    pm10_val = float(ds[pm10_var].isel(
                        forecast_reference_time=t_idx, 
                        forecast_period=p_idx,
                        latitude=lat_idx, 
                        longitude=lon_idx
                    ).values)
                    
                    # Converter unidades se necessário
                    if pm25_val < 1e-6:  # Se muito pequeno, provavelmente em kg/m³
                        pm25_val *= 1e9  # kg/m³ para μg/m³
                        pm10_val *= 1e9
                    elif pm25_val < 1e-3:  # Se pequeno, provavelmente em g/m³
                        pm25_val *= 1e6  # g/m³ para μg/m³
                        pm10_val *= 1e6
                    elif pm25_val > 1000:  # Se muito grande, dividir
                        pm25_val /= 1000
                        pm10_val /= 1000
                    
                    actual_time = pd.to_datetime(ref_time) + pd.to_timedelta(period, unit='h')
                    times.append(actual_time)
                    pm25_values.append(pm25_val)
                    pm10_values.append(pm10_val)
                except:
                    continue
    elif any(dim in ds.dims for dim in ['time', 'forecast_reference_time']):
        time_dim = next(dim for dim in ds.dims if dim in ['time', 'forecast_reference_time'])
        for t_idx in range(len(ds[time_dim])):
            try:
                pm25_val = float(ds[pm25_var].isel({
                    time_dim: t_idx,
                    'latitude': lat_idx,
                    'longitude': lon_idx
                }).values)
                
                pm10_val = float(ds[pm10_var].isel({
                    time_dim: t_idx,
                    'latitude': lat_idx,
                    'longitude': lon_idx
                }).values)
                
                # Converter unidades se necessário
                if pm25_val < 1e-6:
                    pm25_val *= 1e9
                    pm10_val *= 1e9
                elif pm25_val < 1e-3:
                    pm25_val *= 1e6
                    pm10_val *= 1e6
                elif pm25_val > 1000:
                    pm25_val /= 1000
                    pm10_val /= 1000
                
                times.append(pd.to_datetime(ds[time_dim].isel({time_dim: t_idx}).values))
                pm25_values.append(pm25_val)
                pm10_values.append(pm10_val)
            except:
                continue
    
    if times and pm25_values and pm10_values:
        df = pd.DataFrame({
            'time': times,
            'pm25': pm25_values,
            'pm10': pm10_values
        })
        df = df.sort_values('time').reset_index(drop=True)
        
        # Calcular IQA
        aqi_values = df.apply(lambda row: calculate_aqi(row['pm25'], row['pm10']), axis=1)
        df['aqi'] = aqi_values.apply(lambda x: x[0])
        df['aqi_category'] = aqi_values.apply(lambda x: x[1])
        df['aqi_color'] = aqi_values.apply(lambda x: x[2])
        
        return df
    else:
        return pd.DataFrame(columns=['time', 'pm25', 'pm10', 'aqi', 'aqi_category'])

# Função para prever valores futuros
def predict_future_values(df, days=5):
    """Gera previsão para PM2.5 e PM10."""
    if len(df) < 3:
        return pd.DataFrame(columns=['time', 'pm25', 'pm10', 'aqi', 'type'])
    
    df_hist = df.copy()
    df_hist['time_numeric'] = (df_hist['time'] - df_hist['time'].min()).dt.total_seconds()
    
    # Modelos separados para PM2.5 e PM10
    X = df_hist['time_numeric'].values.reshape(-1, 1)
    
    model_pm25 = LinearRegression()
    model_pm25.fit(X, df_hist['pm25'].values)
    
    model_pm10 = LinearRegression()
    model_pm10.fit(X, df_hist['pm10'].values)
    
    # Gerar pontos futuros
    last_time = df_hist['time'].max()
    future_times = [last_time + timedelta(hours=i*6) for i in range(1, days*4+1)]
    future_time_numeric = [(t - df_hist['time'].min()).total_seconds() for t in future_times]
    
    # Prever valores
    future_pm25 = model_pm25.predict(np.array(future_time_numeric).reshape(-1, 1))
    future_pm10 = model_pm10.predict(np.array(future_time_numeric).reshape(-1, 1))
    
    # Limitar valores
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
    result = pd.concat([df_hist[['time', 'pm25', 'pm10', 'aqi', 'aqi_category', 'aqi_color', 'type']], df_pred], ignore_index=True)
    return result

# Função para analisar todas as cidades
def analyze_all_cities(ds, pm25_var, pm10_var, cities_dict):
    """Analisa os valores de PM2.5 e PM10 para todas as cidades."""
    cities_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (city_name, coords) in enumerate(cities_dict.items()):
        progress = (i + 1) / len(cities_dict)
        progress_bar.progress(progress)
        status_text.text(f"Analisando {city_name}... ({i+1}/{len(cities_dict)})")
        
        lat, lon = coords
        
        df_timeseries = extract_pm_timeseries(ds, lat, lon, pm25_var, pm10_var)
        
        if not df_timeseries.empty:
            df_forecast = predict_future_values(df_timeseries, days=5)
            
            forecast_only = df_forecast[df_forecast['type'] == 'forecast']
            
            if not forecast_only.empty:
                max_pm25 = forecast_only['pm25'].max()
                max_pm10 = forecast_only['pm10'].max()
                max_aqi = forecast_only['aqi'].max()
                
                max_day_idx = forecast_only['aqi'].idxmax()
                max_day = forecast_only.loc[max_day_idx, 'time']
                max_category = forecast_only.loc[max_day_idx, 'aqi_category']
                
                cities_results.append({
                    'cidade': city_name,
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
        
        df_results['pm25_max'] = df_results['pm25_max'].round(1)
        df_results['pm10_max'] = df_results['pm10_max'].round(1)
        df_results['aqi_max'] = df_results['aqi_max'].round(0)
        df_results['data_max'] = df_results['data_max'].dt.strftime('%d/%m/%Y %H:%M')
        
        return df_results
    else:
        return pd.DataFrame(columns=['cidade', 'pm25_max', 'pm10_max', 'aqi_max', 'data_max', 'categoria'])

# Função para criar animação temporal melhorada com contorno municipal
def create_temporal_animation(ds, pm_var, pm_type, city, lat_center, lon_center, ms_shapes, animation_speed=500):
    """
    Cria animação temporal de PM2.5 ou PM10 com contorno do município destacado
    """
    # Converter unidades se necessário
    da_pm = ds[pm_var]
    if da_pm.max().values < 1e-6:
        da_pm = da_pm * 1e9
    elif da_pm.max().values < 1e-3:
        da_pm = da_pm * 1e6
    elif da_pm.max().values > 1000:
        da_pm = da_pm / 1000
    
    # Definir extensão para MS
    ms_extent = [-58.5, -50.5, -24.5, -17.0]
    
    # Identificar dimensões temporais
    time_dims = [dim for dim in da_pm.dims if 'time' in dim or 'forecast' in dim]
    
    if 'forecast_reference_time' in da_pm.dims:
        time_dim = 'forecast_reference_time'
        frames = len(da_pm[time_dim])
    else:
        time_dim = time_dims[0] if time_dims else None
        frames = len(da_pm[time_dim]) if time_dim else 1
    
    if frames < 1:
        st.error("Erro: Dados insuficientes para animação.")
        return None
    
    # Definir limites de cores
    vmin, vmax = float(da_pm.min().values), float(da_pm.max().values)
    if pm_type == "PM2.5":
        vmin = max(0, vmin - 5)
        vmax = min(150, vmax + 10)
        colormap = 'YlOrRd'
        limit_lines = [25, 35]  # OMS e EPA
        limit_labels = ['Limite OMS (25 μg/m³)', 'Limite EPA (35 μg/m³)']
        limit_colors = ['orange', 'red']
    else:  # PM10
        vmin = max(0, vmin - 10)
        vmax = min(300, vmax + 20)
        colormap = 'Oranges'
        limit_lines = [50, 150]  # OMS e EPA
        limit_labels = ['Limite OMS (50 μg/m³)', 'Limite EPA (150 μg/m³)']
        limit_colors = ['orange', 'red']
    
    # Criar figura
    fig = plt.figure(figsize=(16, 12))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Adicionar features do mapa
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, color='black')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1, edgecolor='darkgray')
    
    # Configurar grade
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Definir extensão
    ax.set_extent(ms_extent, crs=ccrs.PlateCarree())
    
    # Adicionar contorno dos municípios de MS
    if ms_shapes is not None and not ms_shapes.empty:
        try:
            ms_shapes.boundary.plot(ax=ax, color='black', linewidth=0.8, transform=ccrs.PlateCarree())
            
            # Destacar município selecionado
            selected_city = ms_shapes[ms_shapes['NM_MUN'].str.upper() == city.upper()]
            if not selected_city.empty:
                selected_city.plot(ax=ax, facecolor='none', edgecolor='red', 
                                 linewidth=3.0, transform=ccrs.PlateCarree())
        except Exception as e:
            print(f"Erro ao plotar shapefile: {e}")
    
    # Marcar município selecionado
    ax.plot(lon_center, lat_center, marker='o', markersize=15, 
             markerfacecolor='red', markeredgecolor='white', markeredgewidth=3,
             transform=ccrs.PlateCarree(), zorder=10, label=city)
    
    # Obter primeiro frame
    if 'forecast_period' in da_pm.dims and 'forecast_reference_time' in da_pm.dims:
        first_frame_data = da_pm.isel(forecast_period=0, forecast_reference_time=0).values
        first_frame_time = pd.to_datetime(ds.forecast_reference_time.values[0])
    else:
        first_frame_data = da_pm.isel({time_dim: 0}).values
        first_frame_time = pd.to_datetime(da_pm[time_dim].values[0])
    
    # Criar mapa de cores
    im = ax.pcolormesh(ds.longitude, ds.latitude, first_frame_data, 
                     cmap=colormap, vmin=vmin, vmax=vmax, 
                     transform=ccrs.PlateCarree(), alpha=0.8)
    
    # Barra de cores
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       fraction=0.046, pad=0.08, shrink=0.8)
    cbar.set_label(f'{pm_type} (μg/m³)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Título inicial
    title = ax.set_title(f'{pm_type} - {city.upper()}\n{first_frame_time.strftime("%d/%m/%Y %H:%M UTC")}', 
                        fontsize=18, fontweight='bold', pad=30)
    
    # Adicionar informações sobre limites
    info_text = ax.text(0.02, 0.98, f'Limites: {limit_labels[0].split("(")[1].split(")")[0]}, {limit_labels[1].split("(")[1].split(")")[0]}', 
                       transform=ax.transAxes, fontsize=12, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                       verticalalignment='top')
    
    # Legenda
    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=12)
    
    # Função de animação
    def animate(i):
        try:
            frame_data = None
            frame_time = None
            
            if 'forecast_period' in da_pm.dims and 'forecast_reference_time' in da_pm.dims:
                fp_idx = min(0, len(da_pm.forecast_period)-1)
                frt_idx = min(i, len(da_pm.forecast_reference_time)-1)
                
                frame_data = da_pm.isel(forecast_period=fp_idx, forecast_reference_time=frt_idx).values
                frame_time = pd.to_datetime(ds.forecast_reference_time.values[frt_idx])
            else:
                t_idx = min(i, len(da_pm[time_dim])-1)
                frame_data = da_pm.isel({time_dim: t_idx}).values
                frame_time = pd.to_datetime(da_pm[time_dim].values[t_idx])
            
            im.set_array(frame_data.ravel())
            title.set_text(f'{pm_type} - {city.upper()}\n{frame_time.strftime("%d/%m/%Y %H:%M UTC")}')
            
            return [im, title]
        except Exception as e:
            print(f"Erro no frame {i}: {str(e)}")
            return [im, title]
    
    # Limitar número de frames
    actual_frames = min(frames, 20)
    
    # Criar animação
    ani = animation.FuncAnimation(fig, animate, frames=actual_frames, 
                                 interval=animation_speed, blit=True)
    
    return fig, ani

# Função principal para análise de PM2.5 e PM10
def generate_pm_analysis():
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
    
    # Área de interesse - todo o estado do MS
    city_bounds = {
        'north': -17.0,
        'south': -24.5,
        'east': -50.5,
        'west': -58.5
    }
    
    # Requisição com PM2.5 e PM10 diretos do CAMS
    request = {
        'variable': [
            'particulate_matter_2.5um',
            'particulate_matter_10um'
        ],
        'date': f'{start_date_str}/{end_date_str}',
        'time': hours,
        'leadtime_hour': ['0', '24', '48', '72', '96', '120'],
        'type': ['forecast'],
        'format': 'netcdf',
        'area': [city_bounds['north'], city_bounds['west'], 
                city_bounds['south'], city_bounds['east']]
    }
    
    filename = f'PM25_PM10_{city}_{start_date}_to_{end_date}.nc'
    
    try:
        with st.spinner('📥 Baixando dados de PM2.5 e PM10 do CAMS...'):
            client.retrieve(dataset, request).download(filename)
        
        ds = xr.open_dataset(filename)
        
        # Identificar variáveis de PM2.5 e PM10
        variable_names = list(ds.data_vars)
        pm25_var = next((var for var in variable_names if 'pm2p5' in var.lower() or '2.5' in var), None)
        pm10_var = next((var for var in variable_names if 'pm10' in var.lower() or '10um' in var), None)
        
        if not pm25_var or not pm10_var:
            st.error("Variáveis de PM2.5 ou PM10 não encontradas nos dados.")
            st.write("Variáveis disponíveis:", variable_names)
            return None
        
        # Extrair série temporal para o município selecionado
        with st.spinner("Extraindo dados de PM para o município..."):
            df_timeseries = extract_pm_timeseries(ds, lat_center, lon_center, pm25_var, pm10_var)
        
        if df_timeseries.empty:
            st.error("Não foi possível extrair série temporal para este local.")
            return None
        
        # Gerar previsão
        with st.spinner("Gerando previsões..."):
            df_forecast = predict_future_values(df_timeseries, days=5)
        
        # Criar animações temporais
        with st.spinner('🎬 Criando animação temporal PM2.5...'):
            pm25_fig, pm25_ani = create_temporal_animation(
                ds, pm25_var, "PM2.5", city, lat_center, lon_center, ms_shapes, animation_speed
            )
            
            # Salvar animação PM2.5
            pm25_gif_filename = f'PM25_Temporal_{city}_{start_date}_to_{end_date}.gif'
            pm25_ani.save(pm25_gif_filename, writer=animation.PillowWriter(fps=2))
            plt.close(pm25_fig)
        
        # Criar animação PM10 se solicitada
        pm10_gif_filename = None
        if show_pm10_animation:
            with st.spinner('🎬 Criando animação temporal PM10...'):
                pm10_fig, pm10_ani = create_temporal_animation(
                    ds, pm10_var, "PM10", city, lat_center, lon_center, ms_shapes, animation_speed
                )
                
                # Salvar animação PM10
                pm10_gif_filename = f'PM10_Temporal_{city}_{start_date}_to_{end_date}.gif'
                pm10_ani.save(pm10_gif_filename, writer=animation.PillowWriter(fps=2))
                plt.close(pm10_fig)

        # Analisar todas as cidades
        top_pollution_cities = None
        try:
            # Converter unidades no dataset completo se necessário
            if ds[pm25_var].max().values < 1e-6:
                ds[pm25_var] = ds[pm25_var] * 1e9
                ds[pm10_var] = ds[pm10_var] * 1e9
            elif ds[pm25_var].max().values < 1e-3:
                ds[pm25_var] = ds[pm25_var] * 1e6
                ds[pm10_var] = ds[pm10_var] * 1e6
            elif ds[pm25_var].max().values > 1000:
                ds[pm25_var] = ds[pm25_var] / 1000
                ds[pm10_var] = ds[pm10_var] / 1000
            
            with st.spinner("🔍 Analisando qualidade do ar em todos os municípios de MS..."):
                top_pollution_cities = analyze_all_cities(ds, pm25_var, pm10_var, cities)
        except Exception as e:
            st.warning(f"Não foi possível analisar todas as cidades: {str(e)}")
            top_pollution_cities = pd.DataFrame(columns=['cidade', 'pm25_max', 'pm10_max', 'aqi_max', 'data_max', 'categoria'])
        
        return {
            'pm25_animation': pm25_gif_filename,
            'pm10_animation': pm10_gif_filename,
            'timeseries': df_timeseries,
            'forecast': df_forecast,
            'dataset': ds,
            'pm25_var': pm25_var,
            'pm10_var': pm10_var,
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
    show_pm10_animation = st.checkbox("Gerar animação também para PM10", value=False)

# Informações sobre dados diretos
st.sidebar.info("📊 **Dados Diretos CAMS**\nEste sistema utiliza concentrações de PM2.5 e PM10 medidas diretamente pelos sensores do CAMS, com contornos municipais destacados.")

# Botão principal
st.markdown("### 🚀 Iniciar Análise Completa")
st.markdown(f"Clique no botão abaixo para gerar análise de PM2.5 e PM10 centralizada em **{city}** com animações temporais.")

if st.button("🎯 Gerar Análise de Qualidade do Ar", type="primary", use_container_width=True):
    try:
        results = generate_pm_analysis()
        
        if results:
            # Criar abas
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Análise do Município", 
                "⚠️ Alerta de Qualidade do Ar", 
                f"🎬 Animações Temporais - {city}",
                "📈 Análise Detalhada PM"
            ])
            
            # Aba das Animações Temporais
            with tab3:
                st.subheader(f"🎬 Animações Temporais - {city}")
                
                # Animação PM2.5
                st.markdown("### PM2.5 - Evolução Temporal")
                if 'pm25_animation' in results and results['pm25_animation']:
                    st.image(results['pm25_animation'], 
                            caption=f"Evolução temporal do PM2.5 em {city} com contorno municipal destacado")
                    
                    # Download da animação PM2.5
                    try:
                        with open(results['pm25_animation'], "rb") as file:
                            st.download_button(
                                label="⬇️ Baixar Animação PM2.5 (GIF)",
                                data=file,
                                file_name=f"PM25_Temporal_{city}_{start_date}_to_{end_date}.gif",
                                mime="image/gif"
                            )
                    except:
                        pass
                
                # Animação PM10 (se solicitada)
                if show_pm10_animation and results.get('pm10_animation'):
                    st.markdown("### PM10 - Evolução Temporal")
                    st.image(results['pm10_animation'], 
                            caption=f"Evolução temporal do PM10 em {city} com contorno municipal destacado")
                    
                    # Download da animação PM10
                    try:
                        with open(results['pm10_animation'], "rb") as file:
                            st.download_button(
                                label="⬇️ Baixar Animação PM10 (GIF)",
                                data=file,
                                file_name=f"PM10_Temporal_{city}_{start_date}_to_{end_date}.gif",
                                mime="image/gif"
                            )
                    except:
                        pass
                
                # Informações sobre as animações
                st.info(f"""
                **Como interpretar as animações temporais de {city}:**
                
                🎯 **Localização**: O ponto vermelho marca a localização exata de {city}
                🔴 **Contorno Municipal**: A linha vermelha espessa destaca os limites do município
                ⏰ **Evolução Temporal**: Cada frame mostra a concentração em diferentes momentos
                
                **Escala de Cores:**
                - **PM2.5**: Verde (baixo) → Amarelo → Laranja → Vermelho (alto)
                - **PM10**: Branco/Amarelo claro (baixo) → Laranja → Vermelho escuro (alto)
                
                **Limites de Referência:**
                - **PM2.5**: OMS = 25 μg/m³ | EPA = 35 μg/m³
                - **PM10**: OMS = 50 μg/m³ | EPA = 150 μg/m³
                """)
            
            # Aba de Análise do Município
            with tab1:
                st.subheader(f"📊 Análise Detalhada - {city}")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    df_combined = results['forecast']
                    
                    # Criar subplots para PM2.5 e PM10
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                    
                    # Separar dados históricos e previsão
                    hist_data = df_combined[df_combined['type'] == 'historical']
                    forecast_data = df_combined[df_combined['type'] == 'forecast']
                    
                    # Gráfico PM2.5
                    ax1.plot(hist_data['time'], hist_data['pm25'], 
                           'o-', color='darkblue', label='PM2.5 Observado', markersize=6)
                    ax1.plot(forecast_data['time'], forecast_data['pm25'], 
                           'x--', color='red', label='PM2.5 Previsto', markersize=6)
                    ax1.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Limite OMS (25 μg/m³)')
                    ax1.axhline(y=35, color='red', linestyle='--', alpha=0.7, label='Limite EPA (35 μg/m³)')
                    ax1.set_ylabel('PM2.5 (μg/m³)', fontsize=12)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    ax1.set_title('Material Particulado PM2.5', fontsize=14)
                    
                    # Gráfico PM10
                    ax2.plot(hist_data['time'], hist_data['pm10'], 
                           'o-', color='brown', label='PM10 Observado', markersize=6)
                    ax2.plot(forecast_data['time'], forecast_data['pm10'], 
                           'x--', color='darkred', label='PM10 Previsto', markersize=6)
                    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Limite OMS (50 μg/m³)')
                    ax2.axhline(y=150, color='red', linestyle='--', alpha=0.7, label='Limite EPA (150 μg/m³)')
                    ax2.set_ylabel('PM10 (μg/m³)', fontsize=12)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.set_title('Material Particulado PM10', fontsize=14)
                    
                    # Gráfico IQA
                    ax3.plot(hist_data['time'], hist_data['aqi'], 
                           'o-', color='purple', label='IQA Observado', markersize=6)
                    ax3.plot(forecast_data['time'], forecast_data['aqi'], 
                           'x--', color='magenta', label='IQA Previsto', markersize=6)
                    
                    # Zonas de qualidade do ar
                    ax3.axhspan(0, 50, alpha=0.2, color='green', label='Boa')
                    ax3.axhspan(51, 100, alpha=0.2, color='yellow', label='Moderada')
                    ax3.axhspan(101, 150, alpha=0.2, color='orange', label='Insalubre p/ Sensíveis')
                    ax3.axhspan(151, 200, alpha=0.2, color='red', label='Insalubre')
                    
                    ax3.set_ylabel('IQA', fontsize=12)
                    ax3.set_xlabel('Data/Hora', fontsize=12)
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    ax3.set_title('Índice de Qualidade do Ar', fontsize=14)
                    
                    # Formatar datas
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
                    plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("📈 Estatísticas Atuais")
                    
                    if not hist_data.empty:
                        curr_pm25 = hist_data['pm25'].iloc[-1]
                        curr_pm10 = hist_data['pm10'].iloc[-1]
                        curr_aqi = hist_data['aqi'].iloc[-1]
                        curr_category = hist_data['aqi_category'].iloc[-1]
                        curr_color = hist_data['aqi_color'].iloc[-1]
                        
                        # Métricas
                        col_a, col_b = st.columns(2)
                        col_a.metric("PM2.5 Atual", f"{curr_pm25:.1f} μg/m³")
                        col_b.metric("PM10 Atual", f"{curr_pm10:.1f} μg/m³")
                        
                        st.metric("IQA Atual", f"{curr_aqi:.0f}")
                        
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
                        
                        # Comparação com limites
                        st.subheader("📏 Comparação com Padrões")
                        
                        pm25_who_limit = 25
                        pm10_who_limit = 50
                        
                        st.progress(min(curr_pm25 / pm25_who_limit, 1.0))
                        st.caption(f"PM2.5: {curr_pm25:.1f}/{pm25_who_limit} μg/m³ (Limite OMS 24h)")
                        
                        st.progress(min(curr_pm10 / pm10_who_limit, 1.0))
                        st.caption(f"PM10: {curr_pm10:.1f}/{pm10_who_limit} μg/m³ (Limite OMS 24h)")
                        
                        # Previsão resumida dos próximos 5 dias
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
                        file_name=f"PM_data_{city}_{start_date}_to_{end_date}.csv",
                        mime="text/csv",
                    )
            
            # Aba de Alertas (mesmo código anterior)
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
                        'pm25_max': 'PM2.5 Máx (μg/m³)',
                        'pm10_max': 'PM10 Máx (μg/m³)',
                        'aqi_max': 'IQA Máx',
                        'data_max': 'Data Crítica',
                        'categoria': 'Categoria'
                    })
                    
                    # Função para colorir linhas baseado no IQA
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
            
            # Aba de análise detalhada (código similar ao anterior, mantido para consistência)
            with tab4:
                st.subheader("📈 Análise Detalhada de Material Particulado")
                
                # Informações sobre dados diretos do CAMS
                with st.expander("ℹ️ Sobre os Dados Diretos do CAMS"):
                    st.markdown("""
                    ### Dados Diretos de PM2.5 e PM10 do CAMS
                    
                    Este aplicativo utiliza dados diretos de concentração de material particulado 
                    fornecidos pelo CAMS (Copernicus Atmosphere Monitoring Service):
                    
                    **Vantagens dos dados diretos:**
                    - ✅ Maior precisão: Sem necessidade de conversão de AOD
                    - ✅ Validação contínua: Dados calibrados com estações de monitoramento
                    - ✅ Resolução temporal: Dados a cada 3 horas
                    - ✅ Cobertura global: Disponível para todo o território brasileiro
                    - ✅ Visualização municipal: Contornos dos municípios destacados
                    
                    **Características técnicas:**
                    - **Resolução espacial**: ~0.4° x 0.4° (≈ 44 km)
                    - **Resolução temporal**: 3 horas
                    - **Unidades**: μg/m³ (microgramas por metro cúbico)
                    - **Fonte**: Sensores e modelos atmosféricos integrados
                    - **Validação**: Comparação contínua com estações de superfície
                    
                    **Variáveis utilizadas:**
                    - `particulate_matter_2.5um`: PM2.5 (partículas < 2.5 μm)
                    - `particulate_matter_10um`: PM10 (partículas < 10 μm)
                    """)
                
                df_combined = results['forecast']
                
                if not df_combined.empty:
                    # Separar dados históricos e previsões
                    hist_data = df_combined[df_combined['type'] == 'historical']
                    forecast_data = df_combined[df_combined['type'] == 'forecast']
                    
                    # Análise estatística
                    if not hist_data.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Estatísticas Históricas")
                            
                            # Métricas estatísticas
                            stats_data = {
                                'Métrica': ['Média', 'Mediana', 'Máximo', 'Mínimo', 'Desvio Padrão'],
                                'PM2.5 (μg/m³)': [
                                    f"{hist_data['pm25'].mean():.1f}",
                                    f"{hist_data['pm25'].median():.1f}",
                                    f"{hist_data['pm25'].max():.1f}",
                                    f"{hist_data['pm25'].min():.1f}",
                                    f"{hist_data['pm25'].std():.1f}"
                                ],
                                'PM10 (μg/m³)': [
                                    f"{hist_data['pm10'].mean():.1f}",
                                    f"{hist_data['pm10'].median():.1f}",
                                    f"{hist_data['pm10'].max():.1f}",
                                    f"{hist_data['pm10'].min():.1f}",
                                    f"{hist_data['pm10'].std():.1f}"
                                ],
                                'IQA': [
                                    f"{hist_data['aqi'].mean():.0f}",
                                    f"{hist_data['aqi'].median():.0f}",
                                    f"{hist_data['aqi'].max():.0f}",
                                    f"{hist_data['aqi'].min():.0f}",
                                    f"{hist_data['aqi'].std():.0f}"
                                ]
                            }
                            
                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df, use_container_width=True)
                        
                        with col2:
                            # Comparação com padrões internacionais
                            st.subheader("🌍 Comparação com Padrões")
                            
                            standards = {
                                'Padrão': ['OMS (24h)', 'EPA (24h)', 'CONAMA (24h)', 'UE (24h)'],
                                'PM2.5 (μg/m³)': [25, 35, 60, 25],
                                'PM10 (μg/m³)': [50, 150, 150, 50]
                            }
                            
                            standards_df = pd.DataFrame(standards)
                            
                            if not df_combined.empty:
                                current_pm25 = df_combined['pm25'].iloc[-1]
                                current_pm10 = df_combined['pm10'].iloc[-1]
                                
                                standards_df['Status PM2.5'] = standards_df['PM2.5 (μg/m³)'].apply(
                                    lambda x: '✅ OK' if current_pm25 <= x else '❌ Excede'
                                )
                                standards_df['Status PM10'] = standards_df['PM10 (μg/m³)'].apply(
                                    lambda x: '✅ OK' if current_pm10 <= x else '❌ Excede'
                                )
                            
                            st.dataframe(standards_df, use_container_width=True)
                
                else:
                    st.warning("Dados insuficientes para análise detalhada.")
    
    except Exception as e:
        st.error(f"❌ Erro durante a análise: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Rodapé informativo
st.markdown("---")
st.markdown("""
### ℹ️ Informações Importantes

**Sobre as Animações Temporais:**
- As animações mostram a evolução temporal de PM2.5 e PM10 com contornos municipais
- Cada frame representa um momento específico da previsão CAMS
- O município selecionado é destacado com contorno vermelho e marcador

**Dados Fornecidos por:**
- CAMS (Copernicus Atmosphere Monitoring Service) - União Europeia
- Shapefiles: IBGE (Instituto Brasileiro de Geografia e Estatística)
- Processamento: Sistema desenvolvido para monitoramento ambiental de MS

**Desenvolvido para:** Monitoramento da Qualidade do Ar em Mato Grosso do Sul com Visualização Municipal
""")

# Informações de contato/suporte
with st.expander("📞 Suporte e Informações Técnicas"):
    st.markdown("""
    ### Suporte Técnico
    
    **Parâmetros do Sistema:**
    - Resolução espacial: ~0.4° x 0.4° (≈ 44 km)
    - Resolução temporal: 3 horas
    - Previsão: Até 5 dias
    - Variáveis principais: PM2.5 e PM10 diretos
    - Visualização: Contornos municipais destacados
    
    **Vantagens das Animações Temporais:**
    - Eliminação de incertezas de conversão AOD→PM
    - Visualização clara da evolução temporal
    - Destaque do município de interesse
    - Calibração contínua com estações de superfície
    - Maior precisão para tomada de decisões
    
    **Para Melhor Precisão:**
    - Use dados de múltiplos pontos temporais
    - Considere condições meteorológicas locais
    - Valide com medições locais quando disponível
    - Monitore tendências de longo prazo
    - Observe a evolução temporal nas animações
    
    **Sobre as Animações:**
    - Velocidade ajustável (200-1000ms por frame)
    - PM2.5 sempre disponível, PM10 opcional
    - Contornos municipais do IBGE
    - Escalas de cores padronizadas (OMS/EPA)
    - Formatos GIF para fácil compartilhamento
    """)
