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
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Configuração inicial da página
st.set_page_config(layout="wide", page_title="Monitoramento Avançado da Qualidade do Ar - MS")

# ✅ Carregar autenticação a partir do secrets.toml
try:
    ads_url = st.secrets["ads"]["url"]
    ads_key = st.secrets["ads"]["key"]
    client = cdsapi.Client(url=ads_url, key=ads_key)
except Exception as e:
    st.error("❌ Erro ao carregar as credenciais do CDS API. Verifique seu secrets.toml.")
    st.stop()

# Função para carregar shapefile dos municípios de MS
@st.cache_data
def load_ms_municipalities():
    try:
        # URL para o shapefile de municípios do MS
        url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/" + \
              "municipio_2022/UFs/MS/MS_Municipios_2022.zip"
        
        # Tentativa de carregar os dados
        try:
            gdf = gpd.read_file(url)
            return gdf
        except:
            # Fallback: criar geodataframe com municípios do MS
            data = {
                'NM_MUN': [
                    'Campo Grande', 'Dourados', 'Três Lagoas', 'Corumbá', 'Ponta Porã',
                    'Naviraí', 'Nova Andradina', 'Aquidauana', 'Maracaju', 'Sidrolândia',
                    'Paranaíba', 'Rio Brilhante', 'Coxim', 'Amambai', 'Chapadão do Sul',
                    'Miranda', 'Jardim', 'Aparecida do Taboado', 'Costa Rica', 'Cassilândia',
                    'Bonito', 'Ivinhema', 'Anastácio', 'Ribas do Rio Pardo', 'Bataguassu',
                    'Nova Alvorada do Sul', 'Itaquiraí', 'Mundo Novo', 'Ladário', 'Bela Vista',
                    'Água Clara', 'São Gabriel do Oeste', 'Caarapó', 'Porto Murtinho', 'Camapuã',
                    'Batayporã', 'Brasilândia', 'Terenos', 'Eldorado', 'Guia Lopes da Laguna'
                ],
                'geometry': [gpd.points_from_xy([lon], [lat])[0].buffer(0.2) for lat, lon in [
                    [-20.4697, -54.6201], [-22.2231, -54.812], [-20.7849, -51.7005], [-19.0082, -57.651], [-22.5334, -55.7271],
                    [-23.0618, -54.1990], [-22.2306, -53.3438], [-20.4697, -55.7880], [-21.6100, -55.1678], [-20.9300, -54.9567],
                    [-19.6746, -51.1909], [-21.8021, -54.5452], [-18.5013, -54.7647], [-23.1058, -55.2253], [-18.7913, -52.6269],
                    [-20.2472, -56.3797], [-21.4831, -56.1380], [-20.0873, -51.0961], [-18.5432, -53.1386], [-19.1179, -51.7368],
                    [-21.1261, -56.4834], [-22.3048, -53.8185], [-20.4839, -55.8413], [-20.4444, -53.7590], [-22.2962, -52.4231],
                    [-21.4516, -54.3825], [-23.4785, -54.1878], [-23.9412, -54.2806], [-19.0103, -57.5962], [-22.1064, -56.5265],
                    [-20.4431, -52.8973], [-19.3956, -54.5507], [-22.6365, -54.8210], [-22.0500, -57.8833], [-19.5306, -54.0439],
                    [-22.3085, -53.3199], [-21.2564, -52.1950], [-20.4403, -54.8603], [-23.7868, -54.2839], [-21.4551, -56.1122]
                ]]
            }
            gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
            return gdf
    except Exception as e:
        st.warning(f"Não foi possível carregar os shapes dos municípios: {str(e)}")
        # Retornar DataFrame vazio com estrutura esperada
        return gpd.GeoDataFrame(columns=['NM_MUN', 'geometry'], crs="EPSG:4326")

# Dicionário com todas as cidades de MS e suas coordenadas
ms_cities = {
    "Campo Grande": [-20.4697, -54.6201], "Dourados": [-22.2231, -54.812], "Três Lagoas": [-20.7849, -51.7005],
    "Corumbá": [-19.0082, -57.651], "Ponta Porã": [-22.5334, -55.7271], "Naviraí": [-23.0618, -54.1990],
    "Nova Andradina": [-22.2306, -53.3438], "Aquidauana": [-20.4697, -55.7880], "Maracaju": [-21.6100, -55.1678],
    "Sidrolândia": [-20.9300, -54.9567], "Paranaíba": [-19.6746, -51.1909], "Rio Brilhante": [-21.8021, -54.5452],
    "Coxim": [-18.5013, -54.7647], "Amambai": [-23.1058, -55.2253], "Chapadão do Sul": [-18.7913, -52.6269],
    "Miranda": [-20.2472, -56.3797], "Jardim": [-21.4831, -56.1380], "Aparecida do Taboado": [-20.0873, -51.0961],
    "Costa Rica": [-18.5432, -53.1386], "Cassilândia": [-19.1179, -51.7368], "Bonito": [-21.1261, -56.4834],
    "Ivinhema": [-22.3048, -53.8185], "Anastácio": [-20.4839, -55.8413], "Ribas do Rio Pardo": [-20.4444, -53.7590],
    "Bataguassu": [-22.2962, -52.4231], "Nova Alvorada do Sul": [-21.4516, -54.3825], "Itaquiraí": [-23.4785, -54.1878],
    "Mundo Novo": [-23.9412, -54.2806], "Ladário": [-19.0103, -57.5962], "Bela Vista": [-22.1064, -56.5265],
    "Água Clara": [-20.4431, -52.8973], "São Gabriel do Oeste": [-19.3956, -54.5507], "Caarapó": [-22.6365, -54.8210],
    "Porto Murtinho": [-22.0500, -57.8833], "Camapuã": [-19.5306, -54.0439], "Batayporã": [-22.3085, -53.3199],
    "Brasilândia": [-21.2564, -52.1950], "Terenos": [-20.4403, -54.8603], "Eldorado": [-23.7868, -54.2839],
    "Guia Lopes da Laguna": [-21.4551, -56.1122], "Fátima do Sul": [-22.3789, -54.5131], "Bandeirantes": [-19.9279, -54.3584],
    "Sonora": [-17.5727, -54.7551], "Antônio João": [-22.1927, -55.9511], "Nioaque": [-21.1416, -55.8297],
    "Dois Irmãos do Buriti": [-20.6847, -55.2956], "Pedro Gomes": [-18.0996, -54.5507], "Deodápolis": [-22.2785, -54.1682],
    "Angélica": [-22.1527, -53.7708], "Vicentina": [-22.4095, -54.4415], "Itaporã": [-22.0789, -54.7902]
}

# Definir limites de qualidade do ar para PM2.5, PM10 e AOD
air_quality_limits = {
    'pm2p5': [12, 35.4, 55.4, 150.4, 250.4],  # μg/m³
    'pm10': [54, 154, 254, 354, 424],         # μg/m³
    'aod550': [0.1, 0.2, 0.5, 1.0, 2.0]       # adimensional
}

# Cores para os níveis de qualidade do ar
aqi_colors = {
    'Boa': '#00e400',
    'Moderada': '#ffff00',
    'Insalubre para grupos sensíveis': '#ff7e00',
    'Insalubre': '#ff0000',
    'Muito Insalubre': '#99004c',
    'Perigosa': '#7e0023'
}

# Função para classificar a qualidade do ar com base nos limites
def classify_air_quality(value, pollutant):
    limits = air_quality_limits[pollutant]
    categories = ['Boa', 'Moderada', 'Insalubre para grupos sensíveis', 
                 'Insalubre', 'Muito Insalubre', 'Perigosa']
    
    for i, limit in enumerate(limits):
        if value < limit:
            return categories[i], aqi_colors[categories[i]]
    return categories[-1], aqi_colors[categories[-1]]

# Função para extrair série temporal de um ponto para qualquer variável
def extract_point_timeseries(ds, lat, lon, var_name):
    """Extrai série temporal de um ponto específico do dataset."""
    lat_idx = np.abs(ds.latitude.values - lat).argmin()
    lon_idx = np.abs(ds.longitude.values - lon).argmin()
    
    # Identificar as dimensões temporais
    time_dims = [dim for dim in ds[var_name].dims if 'time' in dim or 'forecast' in dim]
    
    # Criar dataframe para armazenar valores por tempo
    times = []
    values = []
    
    # Se tivermos forecast_reference_time e forecast_period
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
                    
                    # Calcular o tempo real somando o período à referência
                    actual_time = pd.to_datetime(ref_time) + pd.to_timedelta(period, unit='h')
                    times.append(actual_time)
                    values.append(value)
                except:
                    continue
    # Caso tenha apenas uma dimensão de tempo
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
    
    # Criar DataFrame
    if times and values:
        df = pd.DataFrame({'time': times, 'value': values})
        df = df.sort_values('time').reset_index(drop=True)
        return df
    else:
        return pd.DataFrame(columns=['time', 'value'])

# Função para prever valores futuros usando regressão linear
def predict_future_values(df, days=5):
    """Gera uma previsão simples baseada nos dados históricos."""
    if len(df) < 3:  # Precisa de pelo menos 3 pontos para uma previsão mínima
        return pd.DataFrame(columns=['time', 'value', 'type'])
    
    # Preparar dados para regressão
    df_hist = df.copy()
    df_hist['time_numeric'] = (df_hist['time'] - df_hist['time'].min()).dt.total_seconds()
    
    # Modelo de regressão linear simples
    X = df_hist['time_numeric'].values.reshape(-1, 1)
    y = df_hist['value'].values
    model = LinearRegression()
    model.fit(X, y)
    
    # Gerar pontos futuros
    last_time = df_hist['time'].max()
    future_times = [last_time + timedelta(hours=i*6) for i in range(1, days*4+1)]  # 4 pontos por dia (6h)
    future_time_numeric = [(t - df_hist['time'].min()).total_seconds() for t in future_times]
    
    # Prever valores
    future_values = model.predict(np.array(future_time_numeric).reshape(-1, 1))
    
    # Limitar valores previstos (não podem ser negativos)
    future_values = np.maximum(future_values, 0)
    
    # Criar DataFrame com previsão
    df_pred = pd.DataFrame({
        'time': future_times,
        'value': future_values,
        'type': 'forecast'
    })
    
    # Adicionar indicador aos dados históricos
    df_hist['type'] = 'historical'
    
    # Combinar histórico e previsão
    result = pd.concat([df_hist[['time', 'value', 'type']], df_pred], ignore_index=True)
    return result

# Função para baixar dados de AOD, PM2.5 e PM10 do CAMS
def download_pollutant_data(start_date, end_date, start_hour, end_hour, lat_center, lon_center, map_width):
    dataset = "cams-global-atmospheric-composition-forecasts"
    
    # Format dates and times correctly for ADS API
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Create list of hours in the correct format
    hours = []
    current_hour = start_hour
    while True:
        hours.append(f"{current_hour:02d}:00")
        if current_hour == end_hour:
            break
        current_hour = (current_hour + 3) % 24
        if current_hour == start_hour:  # Evitar loop infinito
            break
    
    # Se não tivermos horas definidas, usar padrão
    if not hours:
        hours = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
    
    # Preparar request para API
    request = {
        'variable': [
            'total_aerosol_optical_depth_550nm',     # AOD
            'particulate_matter_10um_aerosol_concentration',  # PM10 
            'particulate_matter_2.5um_aerosol_concentration'  # PM2.5
        ],
        'date': f'{start_date_str}/{end_date_str}',
        'time': hours,
        'leadtime_hour': ['0', '24', '48', '72', '96', '120'],  # Previsões de até 5 dias
        'type': ['forecast'],
        'format': 'netcdf',
        'area': [lat_center + map_width/2, lon_center - map_width/2, 
                lat_center - map_width/2, lon_center + map_width/2]
    }
    
    filename = f'AirQuality_MS_{start_date}_to_{end_date}.nc'
    
    with st.spinner('📥 Baixando dados do CAMS...'):
        client.retrieve(dataset, request).download(filename)
    
    return filename, xr.open_dataset(filename)

# Função para plotar mapa de poluentes
def plot_pollutant_map(ds, pollutant_var, lat_center, lon_center, map_width, city, municipality_shape, colormap):
    """Cria um mapa estático de um poluente específico para o último timestamp"""
    # Encontrar a última data/hora disponível
    time_dims = [dim for dim in ds[pollutant_var].dims if 'time' in dim or 'forecast' in dim]
    
    # Preparar dados para o mapa
    if 'forecast_reference_time' in ds[pollutant_var].dims and 'forecast_period' in ds[pollutant_var].dims:
        # Pegar o último timestamp disponível
        last_frt_idx = len(ds.forecast_reference_time) - 1
        last_fp_idx = 0  # Primeiro período de previsão (mais atual)
        data = ds[pollutant_var].isel(forecast_reference_time=last_frt_idx, forecast_period=last_fp_idx)
        timestamp = pd.to_datetime(ds.forecast_reference_time.values[last_frt_idx])
    else:
        # Caso tenha apenas uma dimensão de tempo
        time_dim = time_dims[0]
        last_t_idx = len(ds[time_dim]) - 1
        data = ds[pollutant_var].isel({time_dim: last_t_idx})
        timestamp = pd.to_datetime(ds[time_dim].values[last_t_idx])
    
    # Determinar range de cores
    vmin, vmax = float(ds[pollutant_var].min().values), float(ds[pollutant_var].max().values)
    
    # Limites específicos para cada poluente
    if 'pm2p5' in pollutant_var.lower():
        vmin, vmax = 0, 100  # μg/m³
        pollutant_name = "PM2.5"
        unit = "μg/m³"
    elif 'pm10' in pollutant_var.lower():
        vmin, vmax = 0, 200  # μg/m³
        pollutant_name = "PM10"
        unit = "μg/m³"
    else:  # AOD
        vmin, vmax = 0, 1.5
        pollutant_name = "AOD 550nm"
        unit = ""
    
    # Criar figura
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Adicionar features do mapa
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
    
    # Adicionar grid
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Definir extensão do mapa
    ax.set_extent([lon_center - map_width/2, lon_center + map_width/2, 
                  lat_center - map_width/2, lat_center + map_width/2], 
                 crs=ccrs.PlateCarree())
    
    # Criar mapa de cores
    im = ax.pcolormesh(ds.longitude, ds.latitude, data, 
                     cmap=colormap, vmin=vmin, vmax=vmax)
    
    # Adicionar barra de cores
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(f'{pollutant_name} ({unit})' if unit else pollutant_name)
    
    # Adicionar título
    ax.set_title(f'{pollutant_name} em {city} - {timestamp}', fontsize=14)
    
    # Adicionar shape do município selecionado se disponível
    if municipality_shape:
        try:
            if hasattr(municipality_shape, '__geo_interface__'):
                ax.add_geometries([municipality_shape], crs=ccrs.PlateCarree(), 
                                facecolor='none', edgecolor='red', linewidth=2, zorder=3)
                
            # Adicionar rótulo do município
            ax.text(lon_center, lat_center, city, fontsize=12, fontweight='bold', 
                   ha='center', va='center', color='red',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                   transform=ccrs.PlateCarree(), zorder=4)
        except Exception as e:
            st.warning(f"Não foi possível desenhar o shape do município: {str(e)}")
    
    return fig

# Função para criar a animação de um poluente
def create_pollutant_animation(ds, pollutant_var, lat_center, lon_center, map_width, city, 
                              municipality_shape, colormap, animation_speed):
    """Cria uma animação para um poluente específico"""
    # Verificar dimensões
    da = ds[pollutant_var]
    
    # Identificar dimensões temporais
    time_dims = [dim for dim in da.dims if 'time' in dim or 'forecast' in dim]
    
    if not time_dims:
        st.error(f"Não foi possível identificar dimensão temporal nos dados para {pollutant_var}.")
        return None
    
    # Determinar nome e unidade do poluente
    if 'pm2p5' in pollutant_var.lower():
        vmin, vmax = 0, 100  # μg/m³
        pollutant_name = "PM2.5"
        unit = "μg/m³"
    elif 'pm10' in pollutant_var.lower():
        vmin, vmax = 0, 200  # μg/m³
        pollutant_name = "PM10"
        unit = "μg/m³"
    else:  # AOD
        vmin, vmax = 0, 1.5
        pollutant_name = "AOD 550nm"
        unit = ""
    
    # Criar figura
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Adicionar features do mapa
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
    
    # Adicionar grid
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Definir extensão do mapa
    ax.set_extent([lon_center - map_width/2, lon_center + map_width/2, 
                  lat_center - map_width/2, lat_center + map_width/2], 
                 crs=ccrs.PlateCarree())
    
    # Obter primeiro frame para inicializar
    first_frame_data = None
    first_frame_time = None
    
    if 'forecast_period' in da.dims and 'forecast_reference_time' in da.dims:
        if len(da.forecast_period) > 0 and len(da.forecast_reference_time) > 0:
            first_frame_data = da.isel(forecast_period=0, forecast_reference_time=0).values
            first_frame_time = pd.to_datetime(ds.forecast_reference_time.values[0])
        else:
            first_frame_coords = {dim: 0 for dim in da.dims if len(da[dim]) > 0}
            first_frame_data = da.isel(**first_frame_coords).values
            first_frame_time = datetime.now()
    else:
        first_frame_data = da.isel({time_dims[0]: 0}).values
        first_frame_time = pd.to_datetime(da[time_dims[0]].values[0])
    
    # Garantir formato 2D
    if len(first_frame_data.shape) != 2:
        st.error(f"Erro: Formato de dados inesperado para {pollutant_var}. Shape: {first_frame_data.shape}")
        return None
    
    # Criar mapa de cores
    im = ax.pcolormesh(ds.longitude, ds.latitude, first_frame_data, 
                     cmap=colormap, vmin=vmin, vmax=vmax)
    
    # Adicionar barra de cores
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(f'{pollutant_name} ({unit})' if unit else pollutant_name)
    
    # Adicionar título inicial
    title = ax.set_title(f'{pollutant_name} em {city} - {first_frame_time}', fontsize=14)
    
    # Adicionar shape do município selecionado se disponível
    if municipality_shape:
        try:
            if hasattr(municipality_shape, '__geo_interface__'):
                ax.add_geometries([municipality_shape], crs=ccrs.PlateCarree(), 
                                facecolor='none', edgecolor='red', linewidth=2, zorder=3)
                
            # Adicionar rótulo do município
            ax.text(lon_center, lat_center, city, fontsize=12, fontweight='bold', 
                   ha='center', va='center', color='red',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                   transform=ccrs.PlateCarree(), zorder=4)
        except Exception as e:
            pass
    
    # Função de animação
    def animate(i):
        try:
            # Selecionar frame de acordo com a estrutura dos dados
            frame_data = None
            frame_time = None
            
            if 'forecast_period' in da.dims and 'forecast_reference_time' in da.dims:
                # Determinar índices válidos
                fp_idx = min(0, len(da.forecast_period)-1)
                frt_idx = min(i, len(da.forecast_reference_time)-1)
                
                frame_data = da.isel(forecast_period=fp_idx, forecast_reference_time=frt_idx).values
                frame_time = pd.to_datetime(ds.forecast_reference_time.values[frt_idx])
            else:
                # Selecionar pelo índice na dimensão de tempo
                t_idx = min(i, len(da[time_dims[0]])-1)
                frame_data = da.isel({time_dims[0]: t_idx}).values
                frame_time = pd.to_datetime(da[time_dims[0]].values[t_idx])
            
            # Atualizar dados
            im.set_array(frame_data.ravel())
            
            # Atualizar título com timestamp
            title.set_text(f'{pollutant_name} em {city} - {frame_time}')
            
            return [im, title]
        except Exception as e:
            return [im, title]
    
    # Identificar frames disponíveis
    if 'forecast_reference_time' in da.dims:
        frames = len(da.forecast_reference_time)
    else:
        frames = len(da[time_dims[0]])
    
    # Limitar número de frames para evitar problemas
    actual_frames = min(frames, 20)  # Máximo de 20 frames
    
    # Criar animação
    ani = animation.FuncAnimation(fig, animate, frames=actual_frames, 
                                 interval=animation_speed, blit=True)
    
    # Salvar animação
    gif_filename = f'{pollutant_name}_{city}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.gif'
    
    with st.spinner(f'💾 Salvando animação de {pollutant_name}...'):
        ani.save(gif_filename, writer=animation.PillowWriter(fps=2))
    
    plt.close(fig)
    
    return gif_filename

# Função para analisar todas as cidades de MS
def analyze_all_cities(ds, pollutant_vars, start_date, end_date):
    """Analisa dados de todos os poluentes para todas as cidades de MS e gera ranking"""
    all_cities_data = []
    
    with st.spinner("Analisando dados para todas as cidades de MS..."):
        # Criar barra de progresso
        progress_bar = st.progress(0)
        total_cities = len(ms_cities)
        
        # Processar cada cidade
        for i, (city_name, coordinates) in enumerate(ms_cities.items()):
            lat, lon = coordinates
            
            city_data = {'cidade': city_name, 'latitude': lat, 'longitude': lon}
            
            # Analisar cada poluente
            for var_name in pollutant_vars:
                pollutant_key = 'aod550' if 'aod' in var_name.lower() else \
                               'pm10' if 'pm10' in var_name.lower() else 'pm2p5'
                
                # Extrair série temporal
                df = extract_point_timeseries(ds, lat, lon, var_name)
                
                if not df.empty:
                    # Estatísticas básicas
                    curr_value = df['value'].iloc[-1] if len(df) > 0 else np.nan
                    max_value = df['value'].max() if len(df) > 0 else np.nan
                    mean_value = df['value'].mean() if len(df) > 0 else np.nan
                    
                    # Gerar previsão para os próximos 5 dias
                    forecast_df = predict_future_values(df, days=5)
                    
                    if not forecast_df.empty:
                        forecast_data = forecast_df[forecast_df['type'] == 'forecast']
                        
                        if not forecast_data.empty:
                            # Encontrar o máximo previsto e a data correspondente
                            max_forecast = forecast_data['value'].max()
                            max_forecast_date = forecast_data.loc[forecast_data['value'].idxmax(), 'time']
                            
                            # Verificar se há algum dia com valor acima do limiar de alerta
                            forecast_data['date'] = forecast_data['time'].dt.date
                            daily_max = forecast_data.groupby('date')['value'].max().reset_index()
                            
                            # Determinar limiar de alerta com base no poluente
                            alert_threshold = 0
                            if pollutant_key == 'aod550':
                                alert_threshold = 0.5  # Valor insalubre para AOD
                            elif pollutant_key == 'pm2p5':
                                alert_threshold = 55.4  # Valor insalubre para PM2.5
                            elif pollutant_key == 'pm10':
                                alert_threshold = 154  # Valor insalubre para PM10
                            
                            alert_days = daily_max[daily_max['value'] > alert_threshold]
                            has_alert = not alert_days.empty
                            
                            # Armazenar dados
                            city_data[f'{pollutant_key}_atual'] = curr_value
                            city_data[f'{pollutant_key}_max'] = max_value
                            city_data[f'{pollutant_key}_medio'] = mean_value
                            city_data[f'{pollutant_key}_max_previsto'] = max_forecast
                            city_data[f'{pollutant_key}_data_max_previsto'] = max_forecast_date
                            city_data[f'{pollutant_key}_alerta'] = has_alert
                            
                            # Se há alerta, armazenar detalhes
                            if has_alert:
                                alert_dates = alert_days['date'].dt.strftime('%d/%m/%Y').tolist()
                                city_data[f'{pollutant_key}_dias_alerta'] = ', '.join(alert_dates)
            
            # Atualizar barra de progresso
            progress_bar.progress((i + 1) / total_cities)
            
            # Adicionar dados da cidade à lista
            all_cities_data.append(city_data)
    
    # Criar DataFrame com todos os dados
    all_cities_df = pd.DataFrame(all_cities_data)
    
    return all_cities_df

# Função para criar ranking de cidades mais afetadas
def create_city_rankings(all_cities_df, pollutant):
    """Cria um ranking das cidades mais afetadas para um determinado poluente"""
    # Verificar se o DataFrame contém dados para o poluente
    if f'{pollutant}_max_previsto' not in all_cities_df.columns:
        return pd.DataFrame()
    
    # Ordenar pelo valor máximo previsto (decrescente)
    ranking = all_cities_df[['cidade', f'{pollutant}_max_previsto', f'{pollutant}_data_max_previsto']]
    ranking = ranking.sort_values(by=f'{pollutant}_max_previsto', ascending=False)
    ranking = ranking.reset_index(drop=True)
    
    # Renomear colunas para melhor visualização
    ranking.columns = ['Cidade', 'Valor Máximo Previsto', 'Data da Previsão']
    
    # Limitar para as 20 cidades mais afetadas
    return ranking.head(20)

# Função para gerar gráfico de previsão de poluente
def plot_pollutant_forecast(df_forecast, pollutant_name, city):
    """Cria um gráfico de linha com dados históricos e previsões para um poluente"""
    # Verificar se há dados suficientes
    if df_forecast.empty or len(df_forecast) < 3:
        return None
    
    # Determinar unidade e título com base no tipo de poluente
    if pollutant_name == 'PM2.5':
        unit = 'μg/m³'
        y_max = 100
    elif pollutant_name == 'PM10':
        unit = 'μg/m³'
        y_max = 200
    else:  # AOD
        unit = ''
        y_max = 1.5
        pollutant_name = 'AOD 550nm'
    
    # Criar gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dados históricos
    hist_data = df_forecast[df_forecast['type'] == 'historical']
    ax.plot(hist_data['time'], hist_data['value'], 
           marker='o', linestyle='-', color='blue', label='Observado')
    
    # Dados de previsão
    forecast_data = df_forecast[df_forecast['type'] == 'forecast']
    ax.plot(forecast_data['time'], forecast_data['value'], 
           marker='x', linestyle='--', color='red', label='Previsão')
    
    # Formatar eixos
    ax.set_title(f'{pollutant_name} em {city}: Valores Observados e Previstos', fontsize=14)
    ax.set_xlabel('Data/Hora', fontsize=12)
    ax.set_ylabel(f'{pollutant_name} {unit}' if unit else pollutant_name, fontsize=12)
    
    # Limitar eixo y
    ax.set_ylim(0, min(y_max, max(df_forecast['value'].max() * 1.2, 1)))
    
    # Formatar datas no eixo x
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
    plt.xticks(rotation=45)
    
    # Adicionar legenda e grade
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Adicionar faixas de qualidade do ar
    if pollutant_name == 'PM2.5':
        ax.axhspan(0, 12, alpha=0.2, color=aqi_colors['Boa'])
        ax.axhspan(12, 35.4, alpha=0.2, color=aqi_colors['Moderada'])
        ax.axhspan(35.4, 55.4, alpha=0.2, color=aqi_colors['Insalubre para grupos sensíveis'])
        ax.axhspan(55.4, 150.4, alpha=0.2, color=aqi_colors['Insalubre'])
        ax.axhspan(150.4, 250.4, alpha=0.2, color=aqi_colors['Muito Insalubre'])
        ax.axhspan(250.4, y_max, alpha=0.2, color=aqi_colors['Perigosa'])
    elif pollutant_name == 'PM10':
        ax.axhspan(0, 54, alpha=0.2, color=aqi_colors['Boa'])
        ax.axhspan(54, 154, alpha=0.2, color=aqi_colors['Moderada'])
        ax.axhspan(154, 254, alpha=0.2, color=aqi_colors['Insalubre para grupos sensíveis'])
        ax.axhspan(254, 354, alpha=0.2, color=aqi_colors['Insalubre'])
        ax.axhspan(354, 424, alpha=0.2, color=aqi_colors['Muito Insalubre'])
        ax.axhspan(424, y_max, alpha=0.2, color=aqi_colors['Perigosa'])
    else:  # AOD
        ax.axhspan(0, 0.1, alpha=0.2, color=aqi_colors['Boa'])
        ax.axhspan(0.1, 0.2, alpha=0.2, color=aqi_colors['Moderada'])
        ax.axhspan(0.2, 0.5, alpha=0.2, color=aqi_colors['Insalubre para grupos sensíveis'])
        ax.axhspan(0.5, 1.0, alpha=0.2, color=aqi_colors['Insalubre'])
        ax.axhspan(1.0, 2.0, alpha=0.2, color=aqi_colors['Muito Insalubre'])
        ax.axhspan(2.0, y_max, alpha=0.2, color=aqi_colors['Perigosa'])
    
    plt.tight_layout()
    return fig

# Função para criar um mapa de calor das concentrações de poluentes em MS
def create_heatmap(all_cities_df, pollutant, ms_shapes):
    """Cria um mapa de calor das concentrações de poluentes no estado de MS"""
    # Verificar se o DataFrame tem dados para o poluente
    if f'{pollutant}_max_previsto' not in all_cities_df.columns:
        return None
    
    # Criar figura
    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Adicionar features do mapa
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
    
    # Adicionar shapes dos municípios de MS
    if not ms_shapes.empty:
        ms_shapes.boundary.plot(ax=ax, linewidth=0.5, color='gray')
    
    # Configurar título e labels
    pollutant_label = 'AOD 550nm' if pollutant == 'aod550' else \
                     'PM10 (μg/m³)' if pollutant == 'pm10' else 'PM2.5 (μg/m³)'
    ax.set_title(f'Mapa de {pollutant_label} previsto para MS', fontsize=16)
    
    # Extrair coordenadas e valores
    lats = all_cities_df['latitude'].values
    lons = all_cities_df['longitude'].values
    values = all_cities_df[f'{pollutant}_max_previsto'].values
    
    # Determinar escala de cores com base no poluente
    if pollutant == 'aod550':
        vmin, vmax = 0, 1.5
        cmap = plt.cm.get_cmap('YlOrRd')
    elif pollutant == 'pm10':
        vmin, vmax = 0, 200
        cmap = plt.cm.get_cmap('YlOrRd')
    else:  # pm2p5
        vmin, vmax = 0, 100
        cmap = plt.cm.get_cmap('YlOrRd')
    
    # Scatter plot com tamanho proporcional ao valor
    norm = plt.Normalize(vmin, vmax)
    sizes = 20 + (values - vmin) / (vmax - vmin) * 300  # Tamanho entre 20 e 300 pontos
    scatter = ax.scatter(lons, lats, c=values, cmap=cmap, 
                        norm=norm, s=sizes, alpha=0.7, edgecolor='k', linewidth=0.5)
    
    # Adicionar nomes das cidades principais (top 5 valores)
    top_indices = np.argsort(values)[-5:]  # 5 maiores valores
    for i in top_indices:
        city_name = all_cities_df.iloc[i]['cidade']
        ax.text(lons[i], lats[i], city_name, fontsize=9, ha='center', va='bottom',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Adicionar colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label(pollutant_label)
    
    # Definir limites do mapa (MS)
    ms_bounds = [-58.0, -51.0, -24.0, -17.0]  # [lon_min, lon_max, lat_min, lat_max]
    ax.set_extent(ms_bounds, crs=ccrs.PlateCarree())
    
    # Adicionar grid
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    return fig

# Função para criar alertas de poluição para os próximos 5 dias
def generate_air_quality_alerts(all_cities_df):
    """Gera alertas de qualidade do ar para os próximos 5 dias"""
    # Verificar quais poluentes temos dados
    pollutants = []
    if 'aod550_alerta' in all_cities_df.columns:
        pollutants.append('aod550')
    if 'pm10_alerta' in all_cities_df.columns:
        pollutants.append('pm10')
    if 'pm2p5_alerta' in all_cities_df.columns:
        pollutants.append('pm2p5')
    
    if not pollutants:
        return pd.DataFrame()
    
    # Criar lista para armazenar alertas
    alerts = []
    
    # Processar cada poluente
    for pollutant in pollutants:
        # Filtrar cidades com alerta para este poluente
        cities_with_alert = all_cities_df[all_cities_df[f'{pollutant}_alerta'] == True]
        
        # Nome amigável do poluente
        pollutant_name = 'AOD 550nm' if pollutant == 'aod550' else \
                        'PM10' if pollutant == 'pm10' else 'PM2.5'
        
        # Adicionar cada cidade com alerta à lista
        for _, city in cities_with_alert.iterrows():
            alerts.append({
                'Cidade': city['cidade'],
                'Poluente': pollutant_name,
                'Valor Máximo Previsto': city[f'{pollutant}_max_previsto'],
                'Dias de Alerta': city.get(f'{pollutant}_dias_alerta', ''),
                'Categoria': classify_air_quality(city[f'{pollutant}_max_previsto'], pollutant)[0]
            })
    
    # Converter para DataFrame e ordenar
    if alerts:
        alerts_df = pd.DataFrame(alerts)
        alerts_df = alerts_df.sort_values(['Categoria', 'Valor Máximo Previsto'], ascending=[True, False])
        return alerts_df
    else:
        return pd.DataFrame()
    
# Interface principal da aplicação
def main_app():
    # Títulos e introdução
    st.title("🌀 Monitoramento Avançado da Qualidade do Ar - Mato Grosso do Sul")
    st.markdown("""
    Este aplicativo permite visualizar e analisar dados de qualidade do ar para municípios de Mato Grosso do Sul:
    
    - **AOD (550nm)**: Profundidade Óptica de Aerossóis - indicador da quantidade de partículas na atmosfera
    - **PM2.5**: Material Particulado com diâmetro menor que 2.5 micrômetros
    - **PM10**: Material Particulado com diâmetro menor que 10 micrômetros
    
    Os dados são obtidos em tempo real do CAMS (Copernicus Atmosphere Monitoring Service) com previsões para os próximos 5 dias.
    """)
    
    # Carregar shapefiles dos municípios do MS
    with st.spinner("Carregando shapes dos municípios..."):
        ms_shapes = load_ms_municipalities()
    
    # Sidebar para configurações
    st.sidebar.header("⚙️ Configurações")
    
    # Seleção de cidade com os shapes disponíveis
    available_cities = sorted(list(set(ms_shapes['NM_MUN'].tolist()).intersection(set(ms_cities.keys()))))
    if not available_cities:
        available_cities = list(ms_cities.keys())  # Fallback para a lista original
    
    city = st.sidebar.selectbox("Selecione o município", available_cities)
    lat_center, lon_center = ms_cities[city]
    
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
        map_width = st.slider("Largura do Mapa (graus)", 5, 20, 10)
        animation_speed = st.slider("Velocidade da Animação (ms)", 200, 1000, 500)
        colormap = st.selectbox("Paleta de Cores", 
                              ["YlOrRd", "viridis", "plasma", "inferno", "magma", "cividis"])
    
    # Botão para iniciar análise completa
    analyze_btn = st.button("🔍 Gerar Análise Completa", type="primary")
    
    if analyze_btn:
        # Baixar dados de poluentes
        filename, ds = download_pollutant_data(
            start_date, end_date, start_hour, end_hour, lat_center, lon_center, map_width
        )
        
        # Verificar variáveis disponíveis
        variable_names = list(ds.data_vars)
        
        # Identificar variáveis de poluentes
        aod_var = next((var for var in variable_names if 'aod' in var.lower() or 'aerosol_optical_depth' in var.lower()), None)
        pm10_var = next((var for var in variable_names if 'pm10' in var.lower() or 'particulate_matter_10' in var.lower()), None)
        pm25_var = next((var for var in variable_names if 'pm2p5' in var.lower() or 'particulate_matter_2.5' in var.lower()), None)
        
        pollutant_vars = [var for var in [aod_var, pm10_var, pm25_var] if var is not None]
        
        if not pollutant_vars:
            st.error("Não foi possível identificar variáveis de poluentes nos dados.")
            st.stop()
        
        # Extrair séries temporais para o município selecionado
        timeseries_data = {}
        forecast_data = {}
        
        with st.spinner("Extraindo dados de poluentes..."):
            for var_name in pollutant_vars:
                # Extrair série temporal
                ts = extract_point_timeseries(ds, lat_center, lon_center, var_name)
                
                if not ts.empty:
                    # Gerar previsão
                    fc = predict_future_values(ts, days=5)
                    
                    # Armazenar dados
                    pollutant_key = 'aod550' if 'aod' in var_name.lower() else \
                                  'pm10' if 'pm10' in var_name.lower() else 'pm2p5'
                    
                    timeseries_data[pollutant_key] = ts
                    forecast_data[pollutant_key] = fc
        
        # Encontrar o município no geodataframe
        municipality_shape = None
        if not ms_shapes.empty:
            city_shape = ms_shapes[ms_shapes['NM_MUN'] == city]
            if not city_shape.empty:
                municipality_shape = city_shape.iloc[0].geometry
        
        # Gerar animações para cada poluente
        animation_files = {}
        with st.spinner("Gerando animações de poluentes..."):
            for var_name in pollutant_vars:
                pollutant_key = 'aod550' if 'aod' in var_name.lower() else \
                              'pm10' if 'pm10' in var_name.lower() else 'pm2p5'
                
                animation_file = create_pollutant_animation(
                    ds, var_name, lat_center, lon_center, map_width,
                    city, municipality_shape, colormap, animation_speed
                )
                
                if animation_file:
                    animation_files[pollutant_key] = animation_file
        
        # Analisar dados para todas as cidades de MS
        all_cities_data = analyze_all_cities(ds, pollutant_vars, start_date, end_date)
        
        # Criar ranking das 20 cidades mais afetadas
        rankings = {}
        for pollutant in ['aod550', 'pm10', 'pm2p5']:
            if f'{pollutant}_max_previsto' in all_cities_data.columns:
                rankings[pollutant] = create_city_rankings(all_cities_data, pollutant)
        
        # Gerar alertas de qualidade do ar
        alerts = generate_air_quality_alerts(all_cities_data)
        
        # Exibir resultados - Interface com abas
        tabs = st.tabs([
            "📊 Visão Geral", 
            "🔍 Dados Detalhados", 
            "🏙️ Ranking de Cidades", 
            "⚠️ Alertas", 
            "🗺️ Mapas de Calor"
        ])
        
        # Tab 1: Visão Geral
        with tabs[0]:
            st.header(f"Visão Geral da Qualidade do Ar em {city}")
            
            # Mostrar animação principal
            if 'aod550' in animation_files:
                st.subheader("🎬 Animação de AOD 550nm")
                st.image(animation_files['aod550'], caption=f"AOD 550nm em {city} ({start_date} a {end_date})")
            
            # Mostrar estatísticas resumidas
            st.subheader("📈 Sumário da Qualidade do Ar")
            
            # Criar colunas para métricas de poluentes
            metric_cols = st.columns(3)
            
            # Adicionar métricas para cada poluente
            pollutant_names = {
                'aod550': 'AOD 550nm', 
                'pm10': 'PM10 (μg/m³)', 
                'pm2p5': 'PM2.5 (μg/m³)'
            }
            
            for i, (pollutant_key, pollutant_name) in enumerate(pollutant_names.items()):
                if pollutant_key in forecast_data and not forecast_data[pollutant_key].empty:
                    df = forecast_data[pollutant_key]
                    
                    # Dados históricos
                    hist_data = df[df['type'] == 'historical']
                    
                    # Dados de previsão
                    forecast_df = df[df['type'] == 'forecast']
                    
                    # Valores atuais e previstos
                    current_val = hist_data['value'].iloc[-1] if not hist_data.empty else 0
                    max_forecast = forecast_df['value'].max() if not forecast_df.empty else 0
                    
                    # Classificação da qualidade do ar
                    current_category, current_color = classify_air_quality(current_val, pollutant_key)
                    forecast_category, forecast_color = classify_air_quality(max_forecast, pollutant_key)
                    
                    # Exibir métricas
                    with metric_cols[i]:
                        st.metric(
                            label=f"{pollutant_name}",
                            value=f"{current_val:.2f}",
                            delta=f"Máx previsto: {max_forecast:.2f}"
                        )
                        
                        # Mostrar classificação com cor
                        st.markdown(f"""
                        <div style="padding:5px; border-radius:5px; background-color:{current_color}; 
                                  color:white; text-align:center; margin:5px 0;">
                        Atual: {current_category}
                        </div>
                        <div style="padding:5px; border-radius:5px; background-color:{forecast_color}; 
                                  color:white; text-align:center; margin:5px 0;">
                        Previsão: {forecast_category}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Mostrar alertas específicos para a cidade selecionada
            if not alerts.empty:
                city_alerts = alerts[alerts['Cidade'] == city]
                
                if not city_alerts.empty:
                    st.subheader("⚠️ Alertas para Este Município")
                    for _, alert in city_alerts.iterrows():
                        st.warning(f"""
                        **Alerta de {alert['Poluente']}**: Níveis {alert['Categoria']} previstos
                        para os dias {alert['Dias de Alerta']}.
                        Valor máximo previsto: {alert['Valor Máximo Previsto']:.2f}
                        """)
        
        # Tab 2: Dados Detalhados
        with tabs[1]:
            st.header(f"Análise Detalhada da Qualidade do Ar em {city}")
            
            # Criar subtabs para cada poluente
            pollutant_tabs = st.tabs(["AOD 550nm", "PM10", "PM2.5"])
            
            # AOD 550nm
            with pollutant_tabs[0]:
                if 'aod550' in forecast_data and not forecast_data['aod550'].empty:
                    # Mostrar gráfico de previsão
                    st.subheader("📈 Série Temporal e Previsão")
                    forecast_fig = plot_pollutant_forecast(forecast_data['aod550'], "AOD", city)
                    if forecast_fig:
                        st.pyplot(forecast_fig)
                    
                    # Mostrar estatísticas
                    df = forecast_data['pm10']
                    hist_data = df[df['type'] == 'historical']
                    
                    if not hist_data.empty:
                        st.subheader("📊 Estatísticas")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("PM10 Atual", f"{hist_data['value'].iloc[-1]:.1f} μg/m³")
                        col1.metric("PM10 Médio", f"{hist_data['value'].mean():.1f} μg/m³")
                        col2.metric("PM10 Máximo", f"{hist_data['value'].max():.1f} μg/m³")
                        col2.metric("PM10 Mínimo", f"{hist_data['value'].min():.1f} μg/m³")
                        col3.metric("Desvio Padrão", f"{hist_data['value'].std():.1f} μg/m³")
                        col3.metric("Tendência", f"{'+' if hist_data['value'].iloc[-1] > hist_data['value'].iloc[0] else '-'}{abs(hist_data['value'].iloc[-1] - hist_data['value'].iloc[0]):.1f} μg/m³")
                else:
                    st.info("Dados de PM10 não disponíveis para este município.")
                    
            # PM2.5
            with pollutant_tabs[2]:
                if 'pm2p5' in forecast_data and not forecast_data['pm2p5'].empty:
                    # Mostrar gráfico de previsão
                    st.subheader("📈 Série Temporal e Previsão")
                    forecast_fig = plot_pollutant_forecast(forecast_data['pm2p5'], "PM2.5", city)
                    if forecast_fig:
                        st.pyplot(forecast_fig)
                    
                    # Mostrar estatísticas
                    df = forecast_data['pm2p5']
                    hist_data = df[df['type'] == 'historical']
                    
                    if not hist_data.empty:
                        st.subheader("📊 Estatísticas")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("PM2.5 Atual", f"{hist_data['value'].iloc[-1]:.1f} μg/m³")
                        col1.metric("PM2.5 Médio", f"{hist_data['value'].mean():.1f} μg/m³")
                        col2.metric("PM2.5 Máximo", f"{hist_data['value'].max():.1f} μg/m³")
                        col2.metric("PM2.5 Mínimo", f"{hist_data['value'].min():.1f} μg/m³")
                        col3.metric("Desvio Padrão", f"{hist_data['value'].std():.1f} μg/m³")
                        col3.metric("Tendência", f"{'+' if hist_data['value'].iloc[-1] > hist_data['value'].iloc[0] else '-'}{abs(hist_data['value'].iloc[-1] - hist_data['value'].iloc[0]):.1f} μg/m³")
                else:
                    st.info("Dados de PM2.5 não disponíveis para este município.")
            
            # Animações de todos os poluentes
            st.subheader("🎬 Animações de Poluentes")
            animation_col1, animation_col2, animation_col3 = st.columns(3)
            
            with animation_col1:
                if 'aod550' in animation_files:
                    st.image(animation_files['aod550'], caption="AOD 550nm", use_column_width=True)
                    with open(animation_files['aod550'], "rb") as file:
                        btn = st.download_button(
                            label="⬇️ Baixar Animação AOD",
                            data=file,
                            file_name=f"AOD_{city}_{start_date}_to_{end_date}.gif",
                            mime="image/gif"
                        )
            
            with animation_col2:
                if 'pm10' in animation_files:
                    st.image(animation_files['pm10'], caption="PM10", use_column_width=True)
                    with open(animation_files['pm10'], "rb") as file:
                        btn = st.download_button(
                            label="⬇️ Baixar Animação PM10",
                            data=file,
                            file_name=f"PM10_{city}_{start_date}_to_{end_date}.gif",
                            mime="image/gif"
                        )
            
            with animation_col3:
                if 'pm2p5' in animation_files:
                    st.image(animation_files['pm2p5'], caption="PM2.5", use_column_width=True)
                    with open(animation_files['pm2p5'], "rb") as file:
                        btn = st.download_button(
                            label="⬇️ Baixar Animação PM2.5",
                            data=file,
                            file_name=f"PM25_{city}_{start_date}_to_{end_date}.gif",
                            mime="image/gif"
                        )
        
        # Tab 3: Ranking de Cidades
        with tabs[2]:
            st.header("🏙️ Ranking das 20 Cidades Mais Afetadas")
            
            # Criar subtabs para cada poluente
            ranking_tabs = st.tabs(["AOD 550nm", "PM10", "PM2.5"])
            
            with ranking_tabs[0]:
                if 'aod550' in rankings and not rankings['aod550'].empty:
                    st.subheader("Top 20 Cidades - AOD 550nm")
                    
                    # Formatar o DataFrame para exibição
                    display_df = rankings['aod550'].copy()
                    display_df['Valor Máximo Previsto'] = display_df['Valor Máximo Previsto'].apply(lambda x: f"{x:.3f}")
                    display_df['Data da Previsão'] = display_df['Data da Previsão'].dt.strftime('%d/%m/%Y %H:%M')
                    
                    # Adicionar colunas de qualidade do ar e cor
                    categories = []
                    colors = []
                    for _, row in rankings['aod550'].iterrows():
                        cat, color = classify_air_quality(row['Valor Máximo Previsto'], 'aod550')
                        categories.append(cat)
                        colors.append(color)
                    
                    display_df['Categoria'] = categories
                    
                    # Exibir tabela
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Criar gráfico de barras
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars = ax.barh(display_df['Cidade'], rankings['aod550']['Valor Máximo Previsto'], color=colors)
                    
                    # Adicionar rótulos
                    for bar, category, value in zip(bars, categories, rankings['aod550']['Valor Máximo Previsto']):
                        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                               f"{value:.3f} ({category})", va='center')
                    
                    ax.set_xlabel('AOD 550nm Máximo Previsto')
                    ax.set_title('Top 20 Cidades com Maiores Valores de AOD Previstos')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Dados de ranking de AOD 550nm não disponíveis.")
            
            with ranking_tabs[1]:
                if 'pm10' in rankings and not rankings['pm10'].empty:
                    st.subheader("Top 20 Cidades - PM10")
                    
                    # Formatar o DataFrame para exibição
                    display_df = rankings['pm10'].copy()
                    display_df['Valor Máximo Previsto'] = display_df['Valor Máximo Previsto'].apply(lambda x: f"{x:.1f} μg/m³")
                    display_df['Data da Previsão'] = display_df['Data da Previsão'].dt.strftime('%d/%m/%Y %H:%M')
                    
                    # Adicionar colunas de qualidade do ar e cor
                    categories = []
                    colors = []
                    for _, row in rankings['pm10'].iterrows():
                        cat, color = classify_air_quality(row['Valor Máximo Previsto'], 'pm10')
                        categories.append(cat)
                        colors.append(color)
                    
                    display_df['Categoria'] = categories
                    
                    # Exibir tabela
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Criar gráfico de barras
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars = ax.barh(display_df['Cidade'], rankings['pm10']['Valor Máximo Previsto'], color=colors)
                    
                    # Adicionar rótulos
                    for bar, category, value in zip(bars, categories, rankings['pm10']['Valor Máximo Previsto']):
                        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                               f"{value:.1f} μg/m³ ({category})", va='center')
                    
                    ax.set_xlabel('PM10 Máximo Previsto (μg/m³)')
                    ax.set_title('Top 20 Cidades com Maiores Valores de PM10 Previstos')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Dados de ranking de PM10 não disponíveis.")
            
            with ranking_tabs[2]:
                if 'pm2p5' in rankings and not rankings['pm2p5'].empty:
                    st.subheader("Top 20 Cidades - PM2.5")
                    
                    # Formatar o DataFrame para exibição
                    display_df = rankings['pm2p5'].copy()
                    display_df['Valor Máximo Previsto'] = display_df['Valor Máximo Previsto'].apply(lambda x: f"{x:.1f} μg/m³")
                    display_df['Data da Previsão'] = display_df['Data da Previsão'].dt.strftime('%d/%m/%Y %H:%M')
                    
                    # Adicionar colunas de qualidade do ar e cor
                    categories = []
                    colors = []
                    for _, row in rankings['pm2p5'].iterrows():
                        cat, color = classify_air_quality(row['Valor Máximo Previsto'], 'pm2p5')
                        categories.append(cat)
                        colors.append(color)
                    
                    display_df['Categoria'] = categories
                    
                    # Exibir tabela
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Criar gráfico de barras
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars = ax.barh(display_df['Cidade'], rankings['pm2p5']['Valor Máximo Previsto'], color=colors)
                    
                    # Adicionar rótulos
                    for bar, category, value in zip(bars, categories, rankings['pm2p5']['Valor Máximo Previsto']):
                        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                               f"{value:.1f} μg/m³ ({category})", va='center')
                    
                    ax.set_xlabel('PM2.5 Máximo Previsto (μg/m³)')
                    ax.set_title('Top 20 Cidades com Maiores Valores de PM2.5 Previstos')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Dados de ranking de PM2.5 não disponíveis.")
            
            # Exportar ranking completo
            if any(not rankings[p].empty for p in rankings):
                st.subheader("💾 Exportar Dados")
                
                # Preparar dados para exportação
                export_data = {}
                for pollutant, ranking_df in rankings.items():
                    if not ranking_df.empty:
                        export_data[pollutant] = ranking_df
                
                if export_data:
                    # Criar um Excel na memória
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        for pollutant, df in export_data.items():
                            pollutant_name = 'AOD' if pollutant == 'aod550' else pollutant.upper()
                            df.to_excel(writer, sheet_name=f'Ranking_{pollutant_name}', index=False)
                    
                    # Botão de download
                    st.download_button(
                        label="⬇️ Baixar Rankings (Excel)",
                        data=output.getvalue(),
                        file_name=f"Rankings_Qualidade_Ar_MS_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        # Tab 4: Alertas
        with tabs[3]:
            st.header("⚠️ Alertas de Qualidade do Ar")
            
            if not alerts.empty:
                # Mostrar sumário de alertas
                st.subheader("Sumário de Alertas para os Próximos 5 Dias")
                
                # Contagem de alertas por categoria
                alert_counts = alerts['Categoria'].value_counts().reset_index()
                alert_counts.columns = ['Categoria', 'Contagem']
                
                # Contagem de alertas por poluente
                pollutant_counts = alerts['Poluente'].value_counts().reset_index()
                pollutant_counts.columns = ['Poluente', 'Contagem']
                
                # Exibir em colunas
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Alertas por Categoria:")
                    for _, row in alert_counts.iterrows():
                        cat = row['Categoria']
                        count = row['Contagem']
                        color = next((c for c, n in aqi_colors.items() if n == cat), '#ff0000')
                        st.markdown(f"""
                        <div style="padding:5px; border-radius:5px; background-color:{color}; 
                                 color:white; text-align:center; margin:5px 0;">
                        {cat}: {count} município(s)
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.write("Alertas por Poluente:")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(pollutant_counts['Contagem'], labels=pollutant_counts['Poluente'], 
                          autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
                    ax.axis('equal')
                    st.pyplot(fig)
                
                # Exibir tabela completa de alertas
                st.subheader("Todos os Alertas")
                
                # Adicionar cores às células conforme categoria
                def highlight_category(val):
                    color = next((v for k, v in aqi_colors.items() if k == val), '')
                    return f'background-color: {color}; color: white'
                
                styled_alerts = alerts.style.applymap(highlight_category, subset=['Categoria'])
                st.dataframe(styled_alerts, use_container_width=True)
                
                # Mapa de alertas
                st.subheader("Mapa de Municípios com Alertas")
                
                # Criar mapa
                alert_cities = list(alerts['Cidade'].unique())
                
                # Filtrar dados para o mapa
                map_data = all_cities_data[all_cities_data['cidade'].isin(alert_cities)]
                
                if not map_data.empty and not ms_shapes.empty:
                    # Criar figura
                    fig = plt.figure(figsize=(12, 10))
                    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
                    
                    # Adicionar features do mapa
                    ax.coastlines(resolution='10m')
                    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
                    ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
                    
                    # Adicionar shapes dos municípios de MS
                    ms_shapes.boundary.plot(ax=ax, linewidth=0.5, color='gray')
                    
                    # Extrair coordenadas
                    lats = map_data['latitude'].values
                    lons = map_data['longitude'].values
                    
                    # Criar lista de cores para cada cidade
                    city_colors = []
                    for city in map_data['cidade']:
                        city_alerts = alerts[alerts['Cidade'] == city]
                        if not city_alerts.empty:
                            # Pegar a categoria mais severa
                            categories = ['Perigosa', 'Muito Insalubre', 'Insalubre', 
                                         'Insalubre para grupos sensíveis', 'Moderada', 'Boa']
                            for cat in categories:
                                if cat in city_alerts['Categoria'].values:
                                    city_colors.append(aqi_colors[cat])
                                    break
                            else:
                                city_colors.append('#ff0000')  # Vermelho por padrão
                        else:
                            city_colors.append('#ff0000')
                    
                    # Scatter plot
                    scatter = ax.scatter(lons, lats, c=city_colors, s=100, 
                                        edgecolor='k', linewidth=0.5, alpha=0.8)
                    
                    # Adicionar nomes das cidades
                    for i, city in enumerate(map_data['cidade']):
                        ax.text(lons[i], lats[i], city, fontsize=8, ha='center', va='bottom',
                               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
                    
                    # Definir limites do mapa (MS)
                    ms_bounds = [-58.0, -51.0, -24.0, -17.0]  # [lon_min, lon_max, lat_min, lat_max]
                    ax.set_extent(ms_bounds, crs=ccrs.PlateCarree())
                    
                    # Adicionar título
                    ax.set_title('Municípios com Alertas de Qualidade do Ar', fontsize=14)
                    
                    # Adicionar grid
                    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
                    gl.top_labels = False
                    gl.right_labels = False
                    
                    # Exibir mapa
                    st.pyplot(fig)
                else:
                    st.info("Dados insuficientes para gerar o mapa de alertas.")
                
                # Exportar alertas
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    alerts.to_excel(writer, sheet_name='Alertas', index=False)
                
                st.download_button(
                    label="⬇️ Baixar Alertas (Excel)",
                    data=output.getvalue(),
                    file_name=f"Alertas_Qualidade_Ar_MS_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.success("😊 Boas notícias! Não há alertas de qualidade do ar para os próximos 5 dias.")
                st.write("""
                Todos os municípios de Mato Grosso do Sul apresentam níveis seguros de poluentes 
                atmosféricos para o período analisado.
                """)
        
        # Tab 5: Mapas de Calor
        with tabs[4]:
            st.header("🗺️ Mapas de Calor de Poluentes - MS")
            
            # Criar subtabs para cada poluente
            heatmap_tabs = st.tabs(["AOD 550nm", "PM10", "PM2.5"])
            
            with heatmap_tabs[0]:
                if 'aod550_max_previsto' in all_cities_data.columns:
                    st.subheader("Mapa de Calor - AOD 550nm")
                    heatmap_fig = create_heatmap(all_cities_data, 'aod550', ms_shapes)
                    if heatmap_fig:
                        st.pyplot(heatmap_fig)
                    else:
                        st.info("Não foi possível gerar o mapa de calor para AOD 550nm.")
                else:
                    st.info("Dados de AOD 550nm não disponíveis para o mapa de calor.")
            
            with heatmap_tabs[1]:
                if 'pm10_max_previsto' in all_cities_data.columns:
                    st.subheader("Mapa de Calor - PM10")
                    heatmap_fig = create_heatmap(all_cities_data, 'pm10', ms_shapes)
                    if heatmap_fig:
                        st.pyplot(heatmap_fig)
                    else:
                        st.info("Não foi possível gerar o mapa de calor para PM10.")
                else:
                    st.info("Dados de PM10 não disponíveis para o mapa de calor.")
            
            with heatmap_tabs[2]:
                if 'pm2p5_max_previsto' in all_cities_data.columns:
                    st.subheader("Mapa de Calor - PM2.5")
                    heatmap_fig = create_heatmap(all_cities_data, 'pm2p5', ms_shapes)
                    if heatmap_fig:
                        st.pyplot(heatmap_fig)
                    else:
                        st.info("Não foi possível gerar o mapa de calor para PM2.5.")
                else:
                    st.info("Dados de PM2.5 não disponíveis para o mapa de calor.")
            
            # Exportar dados de todas as cidades
            if not all_cities_data.empty:
                st.subheader("💾 Exportar Dados")
                
                # Preparar para exportação
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    all_cities_data.to_excel(writer, sheet_name='Dados_Completos', index=False)
                
                st.download_button(
                    label="⬇️ Baixar Dados Completos (Excel)",
                    data=output.getvalue(),
                    file_name=f"Dados_Qualidade_Ar_MS_Completos_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# Executar a aplicação principal
if __name__ == "__main__":
    main_app()
    
    # Adicionar informações sobre interpretação dos dados na parte inferior
    st.markdown("---")
    st.markdown("""
    ### ℹ️ Sobre os dados
    - **Fonte**: Copernicus Atmosphere Monitoring Service (CAMS)
    - **Variáveis**:
        - **AOD 550nm**: Profundidade Óptica de Aerossóis (550nm) - medida adimensional da extinção de luz devido aos aerossóis
        - **PM10**: Material Particulado com diâmetro menor que 10 micrômetros (μg/m³)
        - **PM2.5**: Material Particulado com diâmetro menor que 2.5 micrômetros (μg/m³)
    - **Resolução temporal**: 3 horas
    - **Atualização**: Diária
    - **Previsão**: Até 5 dias

    ### 📖 Como interpretar a qualidade do ar:
    """)
    
    # Criar tabela de interpretação
    interpretations = {
        'Categoria': ['Boa', 'Moderada', 'Insalubre para grupos sensíveis', 'Insalubre', 'Muito Insalubre', 'Perigosa'],
        'PM2.5 (μg/m³)': ['0-12', '12.1-35.4', '35.5-55.4', '55.5-150.4', '150.5-250.4', '>250.5'],
        'PM10 (μg/m³)': ['0-54', '54.1-154', '154.1-254', '254.1-354', '354.1-424', '>424.1'],
        'AOD 550nm': ['0-0.1', '0.1-0.2', '0.2-0.5', '0.5-1.0', '1.0-2.0', '>2.0'],
        'Impactos à Saúde': [
            'Qualidade do ar satisfatória, com pouco ou nenhum risco à saúde.',
            'Qualidade do ar aceitável, mas pode causar problemas a pessoas muito sensíveis.',
            'Pode causar efeitos em pessoas de grupos sensíveis (crianças, idosos, pessoas com doenças respiratórias).',
            'Todos podem começar a sentir efeitos na saúde; pessoas sensíveis podem ter efeitos mais graves.',
            'Alerta de saúde: todos podem sofrer efeitos mais graves.',
            'Alerta de emergência: toda a população está sob risco de sofrer efeitos graves.'
        ]
    }
    
    # Criar DataFrame com as interpretações
    interp_df = pd.DataFrame(interpretations)
    
    # Função para colorir as células de acordo com a categoria
    def color_categories(val):
        if val in aqi_colors:
            return f'background-color: {aqi_colors[val]}; color: white'
        return ''
    
    # Aplicar estilo
    styled_df = interp_df.style.applymap(color_categories, subset=['Categoria'])
    
    # Exibir tabela de interpretação
    st.dataframe(styled_df, use_container_width=True)
    
    # Informações adicionais
    st.markdown("""
    ### 📋 Ações recomendadas durante períodos de má qualidade do ar:
    
    - **Insalubre para grupos sensíveis** (laranja):
        - Pessoas sensíveis devem limitar atividades ao ar livre
        - Mantenha janelas fechadas se possível
        - Use purificadores de ar em ambientes internos
        
    - **Insalubre** (vermelho):
        - Todos devem reduzir atividades físicas prolongadas ao ar livre
        - Grupos sensíveis devem permanecer em ambientes internos
        - Use máscaras N95 se precisar sair
        
    - **Muito Insalubre ou Perigosa** (roxo/marrom):
        - Evite todas as atividades ao ar livre
        - Permaneça em ambientes internos com purificação de ar
        - Procure atendimento médico imediato em caso de sintomas respiratórios
    
    Desenvolvido para monitoramento e previsão de qualidade do ar no estado de Mato Grosso do Sul - Brasil.
    
    **Última atualização**: {data_atual}
    """.format(data_atual=datetime.now().strftime('%d/%m/%Y %H:%M')))

    df = forecast_data['aod550']
    hist_data = df[df['type'] == 'historical']
                    
                    if not hist_data.empty:
                        st.subheader("📊 Estatísticas")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("AOD Atual", f"{hist_data['value'].iloc[-1]:.3f}")
                        col1.metric("AOD Médio", f"{hist_data['value'].mean():.3f}")
                        col2.metric("AOD Máximo", f"{hist_data['value'].max():.3f}")
                        col2.metric("AOD Mínimo", f"{hist_data['value'].min():.3f}")
                        col3.metric("Desvio Padrão", f"{hist_data['value'].std():.3f}")
                        col3.metric("Tendência", f"{'+' if hist_data['value'].iloc[-1] > hist_data['value'].iloc[0] else '-'}{abs(hist_data['value'].iloc[-1] - hist_data['value'].iloc[0]):.3f}")
                else:
                    st.info("Dados de AOD 550nm não disponíveis para este município.")
            
            # PM10
            with pollutant_tabs[1]:
                if 'pm10' in forecast_data and not forecast_data['pm10'].empty:
                    # Mostrar gráfico de previsão
                    st.subheader("📈 Série Temporal e Previsão")
                    forecast_fig = plot_pollutant_forecast(forecast_data['pm10'], "PM10", city)
                    if forecast_fig:
                        st.pyplot(forecast_fig)
                    
                    # Mostrar estatísticas
