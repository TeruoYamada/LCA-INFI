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
import matplotlib.patches as patches

# Configuração inicial da página
st.set_page_config(layout="wide", page_title="Monitor PM2.5/PM10 - MS", page_icon="🌍")

# Carregar autenticação do CDS API
try:
    ads_url = st.secrets["ads"]["url"]
    ads_key = st.secrets["ads"]["key"]
    client = cdsapi.Client(url=ads_url, key=ads_key)
except Exception as e:
    st.error("❌ Erro ao carregar as credenciais do CDS API. Verifique seu secrets.toml.")
    st.stop()

# Lista dos municípios de MS com coordenadas
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

# Coordenadas do centro geográfico de MS
MS_CENTER_LAT = -20.5147
MS_CENTER_LON = -54.5416

# Função para carregar shapefile dos municípios de MS
@st.cache_data
def load_ms_municipalities():
    try:
        url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/" + \
              "municipio_2022/UFs/MS/MS_Municipios_2022.zip"
        
        try:
            gdf = gpd.read_file(url)
            return gdf
        except:
            # Fallback: criar shapes simplificados
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

def calculate_aqi(pm25, pm10):
    """Calcula o Índice de Qualidade do Ar baseado em PM2.5 e PM10."""
    # Breakpoints para PM2.5 (μg/m³)
    pm25_breakpoints = [
        (0, 12, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500, 301, 500)
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
        return 500
    
    aqi_pm25 = calc_sub_index(pm25, pm25_breakpoints)
    aqi_pm10 = calc_sub_index(pm10, pm10_breakpoints)
    
    aqi = max(aqi_pm25, aqi_pm10)
    
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

def extract_point_timeseries(ds, lat, lon, pm25_var='particulate_matter_2.5um', pm10_var='particulate_matter_10um'):
    """Extrai série temporal de PM2.5 e PM10 para um ponto específico."""
    lat_idx = np.abs(ds.latitude.values - lat).argmin()
    lon_idx = np.abs(ds.longitude.values - lon).argmin()
    
    times = []
    pm25_values = []
    pm10_values = []
    
    # Determinar dimensões temporais
    if 'forecast_reference_time' in ds[pm25_var].dims and 'forecast_period' in ds[pm25_var].dims:
        for t_idx, ref_time in enumerate(ds.forecast_reference_time.values):
            for p_idx, period in enumerate(ds.forecast_period.values):
                try:
                    pm25_val = float(ds[pm25_var].isel(
                        forecast_reference_time=t_idx, 
                        forecast_period=p_idx,
                        latitude=lat_idx, 
                        longitude=lon_idx
                    ).values) * 1e9  # Converter para μg/m³
                    
                    pm10_val = float(ds[pm10_var].isel(
                        forecast_reference_time=t_idx, 
                        forecast_period=p_idx,
                        latitude=lat_idx, 
                        longitude=lon_idx
                    ).values) * 1e9  # Converter para μg/m³
                    
                    actual_time = pd.to_datetime(ref_time) + pd.to_timedelta(period, unit='h')
                    times.append(actual_time)
                    pm25_values.append(pm25_val)
                    pm10_values.append(pm10_val)
                except:
                    continue
    elif 'time' in ds[pm25_var].dims:
        for t_idx in range(len(ds.time)):
            try:
                pm25_val = float(ds[pm25_var].isel(
                    time=t_idx,
                    latitude=lat_idx,
                    longitude=lon_idx
                ).values) * 1e9
                
                pm10_val = float(ds[pm10_var].isel(
                    time=t_idx,
                    latitude=lat_idx,
                    longitude=lon_idx
                ).values) * 1e9
                
                times.append(pd.to_datetime(ds.time.isel(time=t_idx).values))
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

def analyze_all_cities(ds, pm25_var, pm10_var, cities_dict):
    """Analisa PM2.5 e PM10 para todas as cidades."""
    cities_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (city_name, coords) in enumerate(cities_dict.items()):
        progress = (i + 1) / len(cities_dict)
        progress_bar.progress(progress)
        status_text.text(f"Analisando {city_name}... ({i+1}/{len(cities_dict)})")
        
        lat, lon = coords
        
        df_timeseries = extract_point_timeseries(ds, lat, lon, pm25_var, pm10_var)
        
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

def generate_pm_analysis():
    """Função principal para análise de PM2.5/PM10."""
    dataset = "cams-global-atmospheric-composition-forecasts"
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Gerar lista de horários
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
    
    # Área centrada no município selecionado
    buffer = 1.5
    city_bounds = {
        'north': lat_center + buffer,
        'south': lat_center - buffer,
        'east': lon_center + buffer,
        'west': lon_center - buffer
    }
    
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
    
    filename = f'PM_data_{city}_{start_date}_to_{end_date}.nc'
    
    try:
        with st.spinner('📥 Baixando dados do CAMS...'):
            client.retrieve(dataset, request).download(filename)
        
        ds = xr.open_dataset(filename)
        
        # Identificar variáveis PM2.5 e PM10
        variable_names = list(ds.data_vars)
        pm25_var = next((var for var in variable_names if '2.5' in var), None)
        pm10_var = next((var for var in variable_names if '10' in var and '2.5' not in var), None)
        
        if not pm25_var or not pm10_var:
            st.error("Variáveis PM2.5/PM10 não encontradas nos dados.")
            return None
        
        # Extrair série temporal para o município
        with st.spinner("Extraindo dados para o município..."):
            df_timeseries = extract_point_timeseries(ds, lat_center, lon_center, pm25_var, pm10_var)
        
        if df_timeseries.empty:
            st.error("Não foi possível extrair série temporal para este local.")
            return None
        
        # Gerar previsão
        with st.spinner("Gerando previsões..."):
            df_forecast = predict_future_values(df_timeseries, days=5)
        
        # Criar animação
        da_pm25 = ds[pm25_var] * 1e9  # Converter para μg/m³
        
        # Identificar dimensões temporais
        time_dims = [dim for dim in da_pm25.dims if 'time' in dim or 'forecast' in dim]
        
        if 'forecast_reference_time' in da_pm25.dims:
            time_dim = 'forecast_reference_time'
            frames = len(da_pm25[time_dim])
        else:
            time_dim = time_dims[0] if time_dims else 'time'
            frames = len(da_pm25[time_dim]) if time_dim in da_pm25.dims else 1
        
        if frames < 1:
            st.error("Dados insuficientes para animação.")
            return None
        
        # Calcular limites da escala
        vmin, vmax = 0, float(da_pm25.max().values)
        vmax = min(vmax, 150)  # Limitar a escala máxima
        
        # Criar figura com projeção centrada no município
        fig = plt.figure(figsize=(14, 10))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Adicionar features geográficas
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.coastlines(resolution='50m', color='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', color='gray')
        ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle='-', edgecolor='black', linewidth=1)
        
        # Grid
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Definir extensão do mapa
        ax.set_extent([city_bounds['west'], city_bounds['east'], 
                      city_bounds['south'], city_bounds['north']], 
                     crs=ccrs.PlateCarree())
        
        # Adicionar shape do município selecionado
        try:
            city_shape = ms_shapes[ms_shapes['NM_MUN'] == city]
            if not city_shape.empty:
                city_shape.plot(ax=ax, transform=ccrs.PlateCarree(), 
                               facecolor='none', edgecolor='red', linewidth=3, alpha=0.8)
        except:
            # Fallback: círculo ao redor do município
            circle = patches.Circle((lon_center, lat_center), 0.1, 
                                  transform=ccrs.PlateCarree(), 
                                  fill=False, edgecolor='red', linewidth=3)
            ax.add_patch(circle)
        
        # Título com nome do município
        ax.text(lon_center, city_bounds['north'] + 0.1, city.upper(), 
                transform=ccrs.PlateCarree(), fontsize=18, fontweight='bold',
                ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='white', alpha=0.8))
        
        # Marcar o município selecionado
        ax.plot(lon_center, lat_center, 'ro', markersize=12, transform=ccrs.PlateCarree(), 
                label=city, markeredgecolor='white', markeredgewidth=2)
        
        # Obter primeiro frame
        if 'forecast_period' in da_pm25.dims and 'forecast_reference_time' in da_pm25.dims:
            if len(da_pm25.forecast_period) > 0 and len(da_pm25.forecast_reference_time) > 0:
                first_frame_data = da_pm25.isel(forecast_period=0, forecast_reference_time=0).values
                first_frame_time = pd.to_datetime(ds.forecast_reference_time.values[0])
            else:
                first_frame_coords = {dim: 0 for dim in da_pm25.dims if len(da_pm25[dim]) > 0}
                first_frame_data = da_pm25.isel(**first_frame_coords).values
                first_frame_time = datetime.now()
        else:
            first_frame_data = da_pm25.isel({time_dim: 0}).values
            first_frame_time = pd.to_datetime(da_pm25[time_dim].values[0]) if time_dim in da_pm25.dims else datetime.now()
        
        # Criar mapa de cores
        im = ax.pcolormesh(ds.longitude, ds.latitude, first_frame_data, 
                         cmap=colormap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        
        # Barra de cores
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04, orientation='horizontal')
        cbar.set_label('PM2.5 (μg/m³)', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # Título inicial
        title = ax.set_title(f'PM2.5 - {city}\n{first_frame_time.strftime("%d/%m/%Y %H:%M UTC")}', 
                           fontsize=14, pad=20)
        
        # Função de animação
        def animate(i):
            try:
                frame_data = None
                frame_time = None
                
                if 'forecast_period' in da_pm25.dims and 'forecast_reference_time' in da_pm25.dims:
                    fp_idx = min(0, len(da_pm25.forecast_period)-1)
                    frt_idx = min(i, len(da_pm25.forecast_reference_time)-1)
                    
                    frame_data = da_pm25.isel(forecast_period=fp_idx, forecast_reference_time=frt_idx).values
                    frame_time = pd.to_datetime(ds.forecast_reference_time.values[frt_idx])
                else:
                    t_idx = min(i, len(da_pm25[time_dim])-1)
                    frame_data = da_pm25.isel({time_dim: t_idx}).values
                    frame_time = pd.to_datetime(da_pm25[time_dim].values[t_idx])
                
                im.set_array(frame_data.ravel())
                title.set_text(f'PM2.5 - {city}\n{frame_time.strftime("%d/%m/%Y %H:%M UTC")}')
                
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
        gif_filename = f'PM25_{city}_{start_date}_to_{end_date}.gif'
        
        with st.spinner('💾 Salvando animação...'):
            ani.save(gif_filename, writer=animation.PillowWriter(fps=2))
        
        plt.close(fig)

        # Analisar todas as cidades de MS
        top_pollution_cities = None
        try:
            # Requisitar dados de MS completo
            ms_bounds = {
                'north': -17.5,
                'south': -24.0,
                'east': -50.5,
                'west': -58.5
            }
            
            request_ms = {
                'variable': [
                    'particulate_matter_2.5um',
                    'particulate_matter_10um'
                ],
                'date': f'{start_date_str}/{end_date_str}',
                'time': hours,
                'leadtime_hour': ['0', '24', '48', '72', '96', '120'],
                'type': ['forecast'],
                'format': 'netcdf',
                'area': [ms_bounds['north'], ms_bounds['west'], 
                        ms_bounds['south'], ms_bounds['east']]
            }
            
            filename_ms = f'PM_MS_complete_{start_date}_to_{end_date}.nc'
            
            with st.spinner("🔍 Baixando dados de MS completo..."):
                client.retrieve(dataset, request_ms).download(filename_ms)
            
            ds_ms = xr.open_dataset(filename_ms)
            pm25_var_ms = 'particulate_matter_2.5um'
            pm10_var_ms = 'particulate_matter_10um'
            
            # Verificar se as variáveis existem no dataset de MS
            if pm25_var_ms in ds_ms.data_vars and pm10_var_ms in ds_ms.data_vars:
                with st.spinner("🔍 Analisando qualidade do ar em todos os municípios..."):
                    top_pollution_cities = analyze_all_cities(ds_ms, pm25_var_ms, pm10_var_ms, cities)
            else:
                st.warning(f"Variáveis PM não encontradas no dataset de MS. Disponíveis: {list(ds_ms.data_vars)}")
                top_pollution_cities = pd.DataFrame(columns=['cidade', 'pm25_max', 'pm10_max', 'aqi_max', 'data_max', 'categoria'])
        except Exception as e:
            st.warning(f"Não foi possível analisar todas as cidades: {str(e)}")
            top_pollution_cities = pd.DataFrame(columns=['cidade', 'pm25_max', 'pm10_max', 'aqi_max', 'data_max', 'categoria'])
        
        return {
            'animation': gif_filename,
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

# Títulos e introdução
st.title("🌍 Monitoramento PM2.5/PM10 - Mato Grosso do Sul")
st.markdown("""
### Sistema Integrado de Monitoramento da Qualidade do Ar

Este aplicativo monitora as concentrações de Material Particulado (PM2.5 e PM10) 
diretamente dos dados do CAMS para todos os municípios de Mato Grosso do Sul.

**Características desta versão:**
- 📊 Dados diretos de PM2.5 e PM10 do CAMS-ECMWF
- 🎯 Visualização centralizada no município selecionado
- 🗺️ Shapes dos municípios sobrepostos no mapa
- 📈 Índice de Qualidade do Ar (IQA) calculado
- 🔮 Previsão para os próximos 5 dias
""")

# Carregar shapefiles dos municípios
with st.spinner("Carregando shapes dos municípios..."):
    ms_shapes = load_ms_municipalities()

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Seleção de cidade
available_cities = sorted(list(set(ms_shapes['NM_MUN'].tolist()).intersection(set(cities.keys()))))
if not available_cities:
    available_cities = list(cities.keys())

city = st.sidebar.selectbox("Selecione o município", available_cities)
lat_center, lon_center = cities[city]

# Configurações de data e hora
st.sidebar.subheader("Período de Análise")
start_date = st.sidebar.date_input("Data de Início", datetime.today() - timedelta(days=2))
end_date = st.sidebar.date_input("Data Final", datetime.today() + timedelta(days=5))

all_hours = list(range(0, 24, 3))
start_hour = st.sidebar.selectbox("Horário Inicial", all_hours, format_func=lambda x: f"{x:02d}:00")
end_hour = st.sidebar.selectbox("Horário Final", all_hours, index=len(all_hours)-1, format_func=lambda x: f"{x:02d}:00")

# Opções de visualização
st.sidebar.subheader("Opções de Visualização")
with st.sidebar.expander("Configurações da Animação"):
    animation_speed = st.slider("Velocidade da Animação (ms)", 200, 1000, 500)
    colormap = st.selectbox("Paleta de Cores", 
                          ["YlOrRd", "viridis", "plasma", "inferno", "magma", "cividis", "RdYlBu_r"])

# Botão principal
st.markdown("### 🚀 Iniciar Análise")
st.markdown(f"Análise de PM2.5/PM10 centralizada em **{city}**")

if st.button("🎯 Gerar Análise de Qualidade do Ar", type="primary", use_container_width=True):
    try:
        results = generate_pm_analysis()
        
        if results:
            # Criar abas
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Análise do Município", 
                "⚠️ Alerta Estadual", 
                f"🗺️ Mapa de {city}",
                "📈 Análise Detalhada"
            ])
            
            # Aba do Mapa
            with tab3:
                st.subheader(f"🎬 Animação PM2.5 - {city}")
                st.image(results['animation'], caption=f"Evolução temporal do PM2.5 em {city}")
                
                with open(results['animation'], "rb") as file:
                    btn = st.download_button(
                        label="⬇️ Baixar Animação (GIF)",
                        data=file,
                        file_name=f"PM25_{city}_{start_date}_to_{end_date}.gif",
                        mime="image/gif"
                    )
                
                st.info(f"""
                **Interpretação do mapa de {city}:**
                - 🟢 Verde: PM2.5 < 15 μg/m³ (Boa qualidade)
                - 🟡 Amarelo: PM2.5 15-35 μg/m³ (Moderada)
                - 🟠 Laranja: PM2.5 35-55 μg/m³ (Insalubre para sensíveis)
                - 🔴 Vermelho: PM2.5 > 55 μg/m³ (Insalubre)
                
                O contorno vermelho marca o limite do município de **{city}**.
                """)
            
            # Aba de Análise do Município
            with tab1:
                st.subheader(f"📊 Análise Detalhada - {city}")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    df_combined = results['forecast']
                    
                    # Gráfico de PM2.5 e PM10
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                    
                    hist_data = df_combined[df_combined['type'] == 'historical']
                    forecast_data = df_combined[df_combined['type'] == 'forecast']
                    
                    # PM2.5
                    ax1.plot(hist_data['time'], hist_data['pm25'], 
                           'o-', color='blue', label='Observado', markersize=6)
                    ax1.plot(forecast_data['time'], forecast_data['pm25'], 
                           'x--', color='red', label='Previsão', markersize=6)
                    ax1.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Limite OMS')
                    ax1.set_ylabel('PM2.5 (μg/m³)', fontsize=12)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    ax1.set_title('Material Particulado PM2.5', fontsize=14)
                    
                    # PM10
                    ax2.plot(hist_data['time'], hist_data['pm10'], 
                           'o-', color='brown', label='Observado', markersize=6)
                    ax2.plot(forecast_data['time'], forecast_data['pm10'], 
                           's--', color='darkred', label='Previsão', markersize=6)
                    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Limite OMS')
                    ax2.set_ylabel('PM10 (μg/m³)', fontsize=12)
                    ax2.set_xlabel('Data/Hora', fontsize=12)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.set_title('Material Particulado PM10', fontsize=14)
                    
                    # Formatar datas
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
                    plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("📈 Status Atual")
                    
                    if not hist_data.empty:
                        curr_pm25 = hist_data['pm25'].iloc[-1]
                        curr_pm10 = hist_data['pm10'].iloc[-1]
                        curr_aqi = hist_data['aqi'].iloc[-1]
                        curr_category = hist_data['aqi_category'].iloc[-1]
                        curr_color = hist_data['aqi_color'].iloc[-1]
                        
                        # Métricas
                        col_a, col_b = st.columns(2)
                        col_a.metric("PM2.5", f"{curr_pm25:.1f} μg/m³")
                        col_b.metric("PM10", f"{curr_pm10:.1f} μg/m³")
                        
                        st.metric("IQA", f"{curr_aqi:.0f}", help="Índice de Qualidade do Ar")
                        
                        # Status da qualidade do ar
                        st.markdown(f"""
                        <div style="padding:15px; border-radius:10px; background-color:{curr_color}; 
                        color:white; text-align:center; margin:10px 0;">
                        <h3 style="margin:0;">Qualidade do Ar</h3>
                        <h2 style="margin:5px 0;">{curr_category}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Comparação com padrões OMS
                        st.subheader("📏 Comparação com OMS")
                        
                        pm25_who_limit = 25
                        pm10_who_limit = 50
                        
                        pm25_progress = min(curr_pm25 / pm25_who_limit, 1.0)
                        pm10_progress = min(curr_pm10 / pm10_who_limit, 1.0)
                        
                        st.progress(pm25_progress)
                        st.caption(f"PM2.5: {curr_pm25:.1f}/{pm25_who_limit} μg/m³")
                        
                        st.progress(pm10_progress)
                        st.caption(f"PM10: {curr_pm10:.1f}/{pm10_who_limit} μg/m³")
                        
                        # Recomendações
                        st.subheader("💡 Recomendações")
                        if curr_aqi <= 50:
                            st.success("✅ Condições ideais para atividades ao ar livre")
                        elif curr_aqi <= 100:
                            st.info("ℹ️ Pessoas sensíveis devem considerar limitar esforços prolongados")
                        elif curr_aqi <= 150:
                            st.warning("⚠️ Grupos sensíveis devem evitar esforços ao ar livre")
                        else:
                            st.error("🚫 Evite atividades prolongadas ao ar livre")
                    
                    # Exportar dados
                    st.subheader("💾 Exportar")
                    csv = df_combined.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Baixar Dados (CSV)",
                        data=csv,
                        file_name=f"PM_data_{city}_{start_date}_to_{end_date}.csv",
                        mime="text/csv",
                    )
            
            # Aba de Alertas Estaduais
            with tab2:
                st.subheader("⚠️ Situação da Qualidade do Ar - Mato Grosso do Sul")
                
                if 'top_pollution' in results and not results['top_pollution'].empty:
                    top_cities = results['top_pollution'].head(20)
                    
                    # Estatísticas gerais
                    critical_cities = top_cities[top_cities['aqi_max'] > 100]
                    very_critical = top_cities[top_cities['aqi_max'] > 150]
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Cidades em Alerta", len(critical_cities))
                    col2.metric("Condição Insalubre", len(very_critical))
                    col3.metric("IQA Máximo", f"{top_cities['aqi_max'].max():.0f}")
                    
                    if len(critical_cities) > 0:
                        st.error(f"""
                        ### 🚨 ALERTA DE QUALIDADE DO AR
                        
                        **{len(critical_cities)} municípios** com previsão de qualidade inadequada!
                        
                        Mais críticos:
                        1. **{top_cities.iloc[0]['cidade']}**: IQA {top_cities.iloc[0]['aqi_max']:.0f}
                        2. **{top_cities.iloc[1]['cidade']}**: IQA {top_cities.iloc[1]['aqi_max']:.0f}
                        3. **{top_cities.iloc[2]['cidade']}**: IQA {top_cities.iloc[2]['aqi_max']:.0f}
                        """)
                    
                    # Tabela dos municípios
                    st.markdown("### 📊 Ranking por Município")
                    
                    top_cities_display = top_cities.rename(columns={
                        'cidade': 'Município',
                        'pm25_max': 'PM2.5 Máx (μg/m³)',
                        'pm10_max': 'PM10 Máx (μg/m³)',
                        'aqi_max': 'IQA Máx',
                        'data_max': 'Data Crítica',
                        'categoria': 'Categoria'
                    })
                    
                    st.dataframe(top_cities_display, use_container_width=True)
                    
                    # Gráfico das 10 piores cidades
                    st.subheader("📊 Top 10 - Concentrações Máximas Previstas")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                    
                    top10 = top_cities.head(10)
                    
                    # PM2.5
                    bars1 = ax1.barh(range(len(top10)), top10['pm25_max'], color='darkblue', alpha=0.8)
                    ax1.set_yticks(range(len(top10)))
                    ax1.set_yticklabels(top10['cidade'])
                    ax1.set_xlabel('PM2.5 (μg/m³)')
                    ax1.set_title('PM2.5 Máximo Previsto')
                    ax1.axvline(x=25, color='orange', linestyle='--', label='Limite OMS')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # PM10
                    bars2 = ax2.barh(range(len(top10)), top10['pm10_max'], color='brown', alpha=0.8)
                    ax2.set_yticks(range(len(top10)))
                    ax2.set_yticklabels(top10['cidade'])
                    ax2.set_xlabel('PM10 (μg/m³)')
                    ax2.set_title('PM10 Máximo Previsto')
                    ax2.axvline(x=50, color='orange', linestyle='--', label='Limite OMS')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Download
                    csv_alert = top_cities.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Baixar Dados Estaduais (CSV)",
                        data=csv_alert,
                        file_name=f"Alerta_PM_MS_{start_date}_to_{end_date}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Dados estaduais não disponíveis. Mostrando apenas análise local.")
            
            # Aba de Análise Detalhada
            with tab4:
                st.subheader("📈 Análise Estatística e Tendências")
                
                df_combined = results['forecast']
                
                if not df_combined.empty:
                    hist_data = df_combined[df_combined['type'] == 'historical']
                    forecast_data = df_combined[df_combined['type'] == 'forecast']
                    
                    # Estatísticas descritivas
                    if not hist_data.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Estatísticas Históricas")
                            
                            stats_data = {
                                'Métrica': ['Média', 'Mediana', 'Máximo', 'Mínimo', 'Desvio'],
                                'PM2.5': [
                                    f"{hist_data['pm25'].mean():.1f}",
                                    f"{hist_data['pm25'].median():.1f}",
                                    f"{hist_data['pm25'].max():.1f}",
                                    f"{hist_data['pm25'].min():.1f}",
                                    f"{hist_data['pm25'].std():.1f}"
                                ],
                                'PM10': [
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
                            st.subheader("📈 Distribuição dos Valores")
                            
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
                            
                            # Histograma PM2.5
                            ax1.hist(hist_data['pm25'], bins=10, alpha=0.7, color='blue', edgecolor='black')
                            ax1.axvline(hist_data['pm25'].mean(), color='red', linestyle='--', 
                                       label=f'Média: {hist_data["pm25"].mean():.1f}')
                            ax1.axvline(25, color='orange', linestyle=':', label='OMS: 25 μg/m³')
                            ax1.set_xlabel('PM2.5 (μg/m³)')
                            ax1.set_ylabel('Frequência')
                            ax1.set_title('Distribuição PM2.5')
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                            
                            # Histograma PM10
                            ax2.hist(hist_data['pm10'], bins=10, alpha=0.7, color='brown', edgecolor='black')
                            ax2.axvline(hist_data['pm10'].mean(), color='red', linestyle='--', 
                                       label=f'Média: {hist_data["pm10"].mean():.1f}')
                            ax2.axvline(50, color='orange', linestyle=':', label='OMS: 50 μg/m³')
                            ax2.set_xlabel('PM10 (μg/m³)')
                            ax2.set_ylabel('Frequência')
                            ax2.set_title('Distribuição PM10')
                            ax2.legend()
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Análise de correlação
                    if not hist_data.empty and len(hist_data) > 5:
                        st.subheader("🔍 Correlação PM2.5 vs PM10")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        ax.scatter(hist_data['pm25'], hist_data['pm10'], alpha=0.6, s=50)
                        
                        # Linha de tendência
                        z = np.polyfit(hist_data['pm25'], hist_data['pm10'], 1)
                        p = np.poly1d(z)
                        ax.plot(hist_data['pm25'], p(hist_data['pm25']), "r--", alpha=0.8)
                        
                        correlation = hist_data['pm25'].corr(hist_data['pm10'])
                        
                        ax.set_xlabel('PM2.5 (μg/m³)')
                        ax.set_ylabel('PM10 (μg/m³)')
                        ax.set_title(f'Correlação PM2.5 vs PM10 (R = {correlation:.3f})')
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                    
                    # Previsão para os próximos dias
                    if not forecast_data.empty:
                        st.subheader("🔮 Previsão - Próximos 5 Dias")
                        
                        forecast_data['date'] = forecast_data['time'].dt.date
                        daily_forecast = forecast_data.groupby('date').agg({
                            'pm25': ['mean', 'max'],
                            'pm10': ['mean', 'max'],
                            'aqi': 'max'
                        }).round(1)
                        
                        daily_forecast.columns = ['PM2.5 Média', 'PM2.5 Max', 'PM10 Média', 'PM10 Max', 'IQA Max']
                        daily_forecast = daily_forecast.reset_index()
                        daily_forecast['date'] = daily_forecast['date'].apply(lambda x: x.strftime('%d/%m/%Y'))
                        
                        st.dataframe(daily_forecast.rename(columns={'date': 'Data'}), use_container_width=True)
                    
                    # Padrões internacionais
                    st.subheader("🌍 Comparação com Padrões")
                    
                    standards = pd.DataFrame({
                        'Organização': ['OMS', 'EPA (EUA)', 'CONAMA (Brasil)', 'União Europeia'],
                        'PM2.5 (μg/m³)': [25, 35, 60, 25],
                        'PM10 (μg/m³)': [50, 150, 150, 50]
                    })
                    
                    st.dataframe(standards, use_container_width=True)
                    
                    st.markdown("""
                    **Notas importantes:**
                    - Valores baseados em médias de 24h
                    - OMS: Organização Mundial da Saúde (padrão mais restritivo)
                    - EPA: Agência de Proteção Ambiental dos EUA
                    - CONAMA: Conselho Nacional do Meio Ambiente (Brasil)
                    """)
    
    except Exception as e:
        st.error(f"❌ Erro durante a análise: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Rodapé
st.markdown("---")
st.markdown("""
### ℹ️ Informações do Sistema

**Fonte dos Dados:**
- CAMS (Copernicus Atmosphere Monitoring Service) - ECMWF
- Dados diretos de PM2.5 e PM10 (sem conversões)
- Resolução: ~0.4° x 0.4° (≈ 44 km)
- Atualização: A cada 12 horas

**Limitações:**
- Resolução espacial limitada para análises muito locais
- Dados dependem da disponibilidade do CAMS
- Para decisões críticas, recomenda-se validação com medições locais

**Desenvolvido para:** Secretaria de Estado de Meio Ambiente, Desenvolvimento, Ciência, Tecnologia e Inovação (SEMADESC) - MS
""")

with st.expander("📋 Informações Técnicas"):
    st.markdown("""
    ### Especificações Técnicas
    
    **Variáveis Monitoradas:**
    - PM2.5: Material particulado < 2.5 μm
    - PM10: Material particulado < 10 μm
    - IQA: Índice de Qualidade do Ar (baseado em padrões EPA/OMS)
    
    **Conversões:**
    - Dados CAMS em kg/m³ convertidos para μg/m³
    - IQA calculado usando breakpoints EPA adaptados
    
    **Cobertura:**
    - 79 municípios de Mato Grosso do Sul
    - Previsão: até 5 dias
    - Frequência temporal: 3 horas
    
    **Metodologia IQA:**
    ```
    0-50:   Boa
    51-100: Moderada
    101-150: Insalubre para Grupos Sensíveis
    151-200: Insalubre
    201-300: Muito Insalubre
    301-500: Perigosa
    ```
    
    **Validação:**
    - Dados validados com estações de monitoramento quando disponíveis
    - Comparação com padrões internacionais (OMS, EPA, CONAMA)
    - Incertezas típicas: ±20-30% para PM2.5, ±15-25% para PM10
    """)

# Informações de contato
st.markdown("""
### 📞 Suporte

Para dúvidas técnicas ou reportar problemas:
- **E-mail:** [suporte.ambiente@ms.gov.br](mailto:suporte.ambiente@ms.gov.br)
- **Telefone:** (67) 3318-6000
- **Horário:** Segunda a Sexta, 8h às 18h

### 🔗 Links Úteis

- [Portal SEMADESC](https://www.semadesc.ms.gov.br/)
- [CAMS - Copernicus](https://atmosphere.copernicus.eu/)
- [Padrões de Qualidade do Ar - OMS](https://www.who.int/news-room/feature-stories/detail/what-are-the-who-air-quality-guidelines)
- [CONAMA - Padrões Nacionais](http://conama.mma.gov.br/)

---
**Sistema desenvolvido em:** {datetime.now().strftime('%B %Y')} | **Versão:** 2.0 | **Última atualização:** {datetime.now().strftime('%d/%m/%Y')}
""")

# Footer com informações legais
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px; margin-top: 20px; padding: 10px; border-top: 1px solid #ddd;'>
    <p>© 2024 Secretaria de Estado de Meio Ambiente, Desenvolvimento, Ciência, Tecnologia e Inovação - SEMADESC/MS</p>
    <p>Dados fornecidos pelo CAMS (Copernicus Atmosphere Monitoring Service) sob licença Copernicus</p>
    <p>Este sistema é fornecido apenas para fins informativos. Para decisões críticas de saúde pública, consulte sempre fontes oficiais.</p>
</div>
""", unsafe_allow_html=True)
