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

# Função melhorada para criar mapas diretos do CAMS
def create_enhanced_pm_maps(ds, pm25_var, pm10_var, city, lat_center, lon_center, ms_shapes, frame_idx=0):
    """
    Cria dois mapas (PM2.5 e PM10) usando diretamente os dados CAMS,
    centrados em Mato Grosso do Sul com o município destacado.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    import pandas as pd
    
    # Configurar figura com dois subplots verticais
    fig = plt.figure(figsize=(16, 12))
    
    # Definir extensão para MS baseada nos dados disponíveis
    lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
    lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
    
    print(f"Extensão dos dados CAMS: Lon {lon_min:.2f} a {lon_max:.2f}, Lat {lat_min:.2f} a {lat_max:.2f}")
    
    # Subplot 1: PM2.5
    ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.8, color='black')
    ax1.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1, edgecolor='darkgray')
    
    # Configurar grade
    gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl1.top_labels = False
    gl1.right_labels = False
    
    # Usar a extensão dos dados CAMS
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Plotar dados de PM2.5 diretamente do CAMS
    try:
        da_pm25 = ds[pm25_var]
        
        # Usar os dados originais do CAMS sem conversão inicial
        print(f"Valores originais PM2.5: min={da_pm25.min().values:.2e}, max={da_pm25.max().values:.2e}")
        
        # Obter frame específico
        if 'forecast_reference_time' in da_pm25.dims and 'forecast_period' in da_pm25.dims:
            frame_data = da_pm25.isel(forecast_reference_time=0, forecast_period=min(frame_idx, len(da_pm25.forecast_period)-1)).values
            frame_time = pd.to_datetime(ds.forecast_reference_time.values[0])
        else:
            time_dims = [d for d in da_pm25.dims if 'time' in d or 'forecast' in d]
            if time_dims:
                time_dim = time_dims[0]
                frame_data = da_pm25.isel({time_dim: min(frame_idx, len(da_pm25[time_dim])-1)}).values
                frame_time = pd.to_datetime(da_pm25[time_dim].values[min(frame_idx, len(da_pm25[time_dim])-1)])
            else:
                frame_data = da_pm25.values
                frame_time = pd.to_datetime('now')
        
        # Definir limites de cores baseados nos dados reais
        vmin_pm25 = max(0, np.nanpercentile(frame_data, 5))
        vmax_pm25 = np.nanpercentile(frame_data, 95)
        
        print(f"Limites PM2.5: {vmin_pm25:.2e} a {vmax_pm25:.2e}")
        
        # Plotar PM2.5
        im1 = ax1.pcolormesh(ds.longitude, ds.latitude, frame_data, 
                            cmap='YlOrRd', vmin=vmin_pm25, vmax=vmax_pm25, 
                            transform=ccrs.PlateCarree(), alpha=0.8)
        
        # Barra de cores para PM2.5
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', 
                            fraction=0.046, pad=0.08, shrink=0.8)
        
        # Determinar unidade baseada na magnitude dos dados
        if vmax_pm25 < 1e-6:
            unit_label = 'PM2.5 (kg/m³)'
        elif vmax_pm25 < 1e-3:
            unit_label = 'PM2.5 (g/m³)'
        else:
            unit_label = 'PM2.5 (μg/m³)'
            
        cbar1.set_label(unit_label, fontsize=12)
        
    except Exception as e:
        print(f"Erro ao plotar PM2.5: {e}")
        im1 = None
    
    # Adicionar shapefile de MS se disponível
    if ms_shapes is not None and not ms_shapes.empty:
        try:
            ms_shapes.boundary.plot(ax=ax1, color='black', linewidth=0.8, transform=ccrs.PlateCarree())
            
            # Destacar município selecionado
            selected_city = ms_shapes[ms_shapes['NM_MUN'].str.upper() == city.upper()]
            if not selected_city.empty:
                selected_city.plot(ax=ax1, facecolor='none', edgecolor='red', 
                                 linewidth=3.0, transform=ccrs.PlateCarree())
        except Exception as e:
            print(f"Erro ao plotar shapefile no PM2.5: {e}")
    
    # Marcar cidade
    ax1.plot(lon_center, lat_center, marker='o', markersize=12, 
             markerfacecolor='red', markeredgecolor='white', markeredgewidth=2,
             transform=ccrs.PlateCarree(), zorder=10)
    
    # Título do primeiro mapa
    ax1.set_title(f'PM2.5 - {city.upper()}\n{frame_time.strftime("%d/%m/%Y %H:%M UTC")}', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Subplot 2: PM10
    ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.8, color='black')
    ax2.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1, edgecolor='darkgray')
    
    # Configurar grade
    gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
    gl2.top_labels = False
    gl2.right_labels = False
    
    # Definir extensão
    ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Plotar dados de PM10
    try:
        da_pm10 = ds[pm10_var]
        
        # Usar os dados originais do CAMS
        print(f"Valores originais PM10: min={da_pm10.min().values:.2e}, max={da_pm10.max().values:.2e}")
        
        # Obter frame específico
        if 'forecast_reference_time' in da_pm10.dims and 'forecast_period' in da_pm10.dims:
            frame_data_pm10 = da_pm10.isel(forecast_reference_time=0, forecast_period=min(frame_idx, len(da_pm10.forecast_period)-1)).values
        else:
            time_dims = [d for d in da_pm10.dims if 'time' in d or 'forecast' in d]
            if time_dims:
                time_dim = time_dims[0]
                frame_data_pm10 = da_pm10.isel({time_dim: min(frame_idx, len(da_pm10[time_dim])-1)}).values
            else:
                frame_data_pm10 = da_pm10.values
        
        # Definir limites de cores para PM10
        vmin_pm10 = max(0, np.nanpercentile(frame_data_pm10, 5))
        vmax_pm10 = np.nanpercentile(frame_data_pm10, 95)
        
        print(f"Limites PM10: {vmin_pm10:.2e} a {vmax_pm10:.2e}")
        
        # Plotar PM10
        im2 = ax2.pcolormesh(ds.longitude, ds.latitude, frame_data_pm10, 
                            cmap='Oranges', vmin=vmin_pm10, vmax=vmax_pm10, 
                            transform=ccrs.PlateCarree(), alpha=0.8)
        
        # Barra de cores para PM10
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', 
                            fraction=0.046, pad=0.08, shrink=0.8)
        
        # Determinar unidade baseada na magnitude dos dados
        if vmax_pm10 < 1e-6:
            unit_label = 'PM10 (kg/m³)'
        elif vmax_pm10 < 1e-3:
            unit_label = 'PM10 (g/m³)'
        else:
            unit_label = 'PM10 (μg/m³)'
            
        cbar2.set_label(unit_label, fontsize=12)
        
    except Exception as e:
        print(f"Erro ao plotar PM10: {e}")
        im2 = None
    
    # Adicionar shapefile de MS
    if ms_shapes is not None and not ms_shapes.empty:
        try:
            ms_shapes.boundary.plot(ax=ax2, color='black', linewidth=0.8, transform=ccrs.PlateCarree())
            
            # Destacar município selecionado
            selected_city = ms_shapes[ms_shapes['NM_MUN'].str.upper() == city.upper()]
            if not selected_city.empty:
                selected_city.plot(ax=ax2, facecolor='none', edgecolor='red', 
                                 linewidth=3.0, transform=ccrs.PlateCarree())
        except Exception as e:
            print(f"Erro ao plotar shapefile no PM10: {e}")
    
    # Marcar cidade
    ax2.plot(lon_center, lat_center, marker='o', markersize=12, 
             markerfacecolor='red', markeredgecolor='white', markeredgewidth=2,
             transform=ccrs.PlateCarree(), zorder=10)
    
    # Título do segundo mapa
    ax2.set_title(f'PM10 - {city.upper()}\n{frame_time.strftime("%d/%m/%Y %H:%M UTC")}', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Adicionar informações sobre unidades
    try:
        if im1 is not None:
            ax1.text(0.02, 0.98, f'Dados diretos CAMS - Valores originais', 
                    transform=ax1.transAxes, fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        if im2 is not None:
            ax2.text(0.02, 0.98, f'Dados diretos CAMS - Valores originais', 
                    transform=ax2.transAxes, fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top')
    except Exception as e:
        print(f"Erro ao adicionar texto de referência: {e}")
    
    return fig

# Função para integrar na análise principal
def create_enhanced_static_maps(results, city, lat_center, lon_center, ms_shapes, start_date):
    """
    Cria mapas estáticos melhorados e salva como arquivo.
    """
    try:
        ds = results['dataset']
        pm25_var = results['pm25_var']
        pm10_var = results['pm10_var']
        
        # Criar mapas melhorados
        enhanced_fig = create_enhanced_pm_maps(
            ds, pm25_var, pm10_var, city, lat_center, lon_center, ms_shapes, frame_idx=0
        )
        
        # Salvar mapa
        enhanced_map_filename = f'Enhanced_PM_Maps_{city}_{start_date}.png'
        enhanced_fig.savefig(enhanced_map_filename, dpi=300, bbox_inches='tight')
        plt.close(enhanced_fig)
        
        return enhanced_map_filename
        
    except Exception as e:
        print(f"Erro ao criar mapas melhorados: {e}")
        return None

# Configuração inicial da página
st.set_page_config(layout="wide", page_title="Monitor PM2.5/PM10 - MS", page_icon="🌍")

# Carregar autenticação a partir do secrets.toml
try:
    ads_url = st.secrets["ads"]["url"]
    ads_key = st.secrets["ads"]["key"]
    client = cdsapi.Client(url=ads_url, key=ads_key)
except Exception as e:
    st.error("Erro ao carregar as credenciais do CDS API. Verifique seu secrets.toml.")
    st.stop()

# Função para baixar shapefile dos municípios de MS
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
            return create_fallback_shapefile()
    except Exception as e:
        st.warning(f"Não foi possível carregar os shapes dos municípios: {str(e)}")
        return create_fallback_shapefile()

# Lista de cidades de MS com coordenadas
cities = {
    "Campo Grande": [-20.4697, -54.6201],
    "Dourados": [-22.2231, -54.812],
    "Três Lagoas": [-20.7849, -51.7005],
    "Corumbá": [-19.0082, -57.651],
    "Aparecida do Taboado": [-20.0873, -51.0961]
    # Adicione sua lista completa aqui
}

def create_fallback_shapefile():
    """Cria um shapefile simplificado caso o oficial falhe"""
    from shapely.geometry import Polygon
    
    municipalities_data = []
    for city_name, (lat, lon) in cities.items():
        buffer_size = 0.15
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
st.title("Monitoramento PM2.5 e PM10 - Mato Grosso do Sul")
st.markdown("""
### Sistema Integrado de Monitoramento da Qualidade do Ar

Este aplicativo monitora diretamente as concentrações de Material Particulado (PM2.5 e PM10) 
para todos os municípios de Mato Grosso do Sul usando dados diretos do CAMS.

**Características desta versão:**
- Dados diretos de PM2.5 e PM10 do CAMS (valores originais)
- Visualização centralizada no município selecionado
- Mapas baseados na extensão real dos dados CAMS
- Análise com valores nativos do sistema CAMS
""")

# Função para calcular IQA baseado nos valores originais CAMS
def calculate_aqi_cams(pm25, pm10):
    """
    Calcula IQA usando os valores diretos do CAMS,
    aplicando conversão apenas quando necessário para o cálculo do IQA.
    """
    # Detectar e converter apenas para cálculo do IQA
    pm25_calc = pm25
    pm10_calc = pm10
    
    # Se valores muito pequenos, assumir kg/m³ e converter para μg/m³
    if pm25 < 1e-6:
        pm25_calc = pm25 * 1e9
        pm10_calc = pm10 * 1e9
    elif pm25 < 1e-3:
        pm25_calc = pm25 * 1e6
        pm10_calc = pm10 * 1e6
    
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
    
    aqi_pm25 = calc_sub_index(pm25_calc, pm25_breakpoints)
    aqi_pm10 = calc_sub_index(pm10_calc, pm10_breakpoints)
    
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

# Função para extrair série temporal mantendo valores originais do CAMS
def extract_pm_timeseries(ds, lat, lon, pm25_var, pm10_var):
    """Extrai série temporal mantendo os valores originais do CAMS."""
    lat_idx = np.abs(ds.latitude.values - lat).argmin()
    lon_idx = np.abs(ds.longitude.values - lon).argmin()
    
    times = []
    pm25_values = []
    pm10_values = []
    
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
                    
                    # Manter valores originais CAMS - sem conversão
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
                
                # Manter valores originais CAMS
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
        
        # Calcular IQA (função faz conversão apenas para cálculo)
        aqi_values = df.apply(lambda row: calculate_aqi_cams(row['pm25'], row['pm10']), axis=1)
        df['aqi'] = aqi_values.apply(lambda x: x[0])
        df['aqi_category'] = aqi_values.apply(lambda x: x[1])
        df['aqi_color'] = aqi_values.apply(lambda x: x[2])
        
        return df
    else:
        return pd.DataFrame(columns=['time', 'pm25', 'pm10', 'aqi', 'aqi_category'])

# Função para prever valores futuros
def predict_future_values(df, days=5):
    """Gera previsão mantendo a escala original dos dados."""
    if len(df) < 3:
        return pd.DataFrame(columns=['time', 'pm25', 'pm10', 'aqi', 'type'])
    
    df_hist = df.copy()
    df_hist['time_numeric'] = (df_hist['time'] - df_hist['time'].min()).dt.total_seconds()
    
    X = df_hist['time_numeric'].values.reshape(-1, 1)
    
    model_pm25 = LinearRegression()
    model_pm25.fit(X, df_hist['pm25'].values)
    
    model_pm10 = LinearRegression()
    model_pm10.fit(X, df_hist['pm10'].values)
    
    last_time = df_hist['time'].max()
    future_times = [last_time + timedelta(hours=i*6) for i in range(1, days*4+1)]
    future_time_numeric = [(t - df_hist['time'].min()).total_seconds() for t in future_times]
    
    future_pm25 = model_pm25.predict(np.array(future_time_numeric).reshape(-1, 1))
    future_pm10 = model_pm10.predict(np.array(future_time_numeric).reshape(-1, 1))
    
    future_pm25 = np.maximum(future_pm25, 0)
    future_pm10 = np.maximum(future_pm10, 0)
    
    future_aqi = []
    future_categories = []
    future_colors = []
    
    for pm25, pm10 in zip(future_pm25, future_pm10):
        aqi, category, color = calculate_aqi_cams(pm25, pm10)
        future_aqi.append(aqi)
        future_categories.append(category)
        future_colors.append(color)
    
    df_pred = pd.DataFrame({
        'time': future_times,
        'pm25': future_pm25,
        'pm10': future_pm10,
        'aqi': future_aqi,
        'aqi_category': future_categories,
        'aqi_color': future_colors,
        'type': 'forecast'
    })
    
    df_hist['type'] = 'historical'
    
    result = pd.concat([df_hist[['time', 'pm25', 'pm10', 'aqi', 'aqi_category', 'aqi_color', 'type']], df_pred], ignore_index=True)
    return result

# Função para analisar todas as cidades mantendo valores CAMS
def analyze_all_cities(ds, pm25_var, pm10_var, cities_dict):
    """Analisa valores mantendo os dados originais do CAMS."""
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
                # Manter valores originais do CAMS
                max_pm25 = forecast_only['pm25'].max()
                max_pm10 = forecast_only['pm10'].max()
                max_aqi = forecast_only['aqi'].max()
                
                max_day_idx = forecast_only['aqi'].idxmax()
                max_day = forecast_only.loc[max_day_idx, 'time']
                max_category = forecast_only.loc[max_day_idx, 'aqi_category']
                
                cities_results.append({
                    'cidade': city_name,
                    'pm25_max_cams': max_pm25,  # Valores originais CAMS
                    'pm10_max_cams': max_pm10,  # Valores originais CAMS
                    'aqi_max': max_aqi,
                    'data_max': max_day,
                    'categoria': max_category
                })
    
    progress_bar.empty()
    status_text.empty()
    
    if cities_results:
        df_results = pd.DataFrame(cities_results)
        df_results = df_results.sort_values('aqi_max', ascending=False).reset_index(drop=True)
        
        # Manter precisão científica dos dados CAMS
        df_results['data_max'] = df_results['data_max'].dt.strftime('%d/%m/%Y %H:%M')
        
        return df_results
    else:
        return pd.DataFrame(columns=['cidade', 'pm25_max_cams', 'pm10_max_cams', 'aqi_max', 'data_max', 'categoria'])

# Função principal para análise
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
    
    if not hours:
        hours = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
    
    # Área de interesse: todo o estado do MS
    city_bounds = {
        'north': -17.0,
        'south': -24.5,
        'east': -50.5,
        'west': -58.5
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
    
    filename = f'PM25_PM10_{city}_{start_date}_to_{end_date}.nc'
    
    try:
        with st.spinner('Baixando dados de PM2.5 e PM10 do CAMS...'):
            client.retrieve(dataset, request).download(filename)
        
        ds = xr.open_dataset(filename)
        
        variable_names = list(ds.data_vars)
        pm25_var = next((var for var in variable_names if 'pm2p5' in var.lower() or '2.5' in var), None)
        pm10_var = next((var for var in variable_names if 'pm10' in var.lower() or '10um' in var), None)
        
        if not pm25_var or not pm10_var:
            st.error("Variáveis de PM2.5 ou PM10 não encontradas nos dados.")
            st.write("Variáveis disponíveis:", variable_names)
            return None
        
        # Extrair série temporal mantendo valores originais
        with st.spinner("Extraindo dados de PM para o município..."):
            df_timeseries = extract_pm_timeseries(ds, lat_center, lon_center, pm25_var, pm10_var)
        
        if df_timeseries.empty:
            st.error("Não foi possível extrair série temporal para este local.")
            return None
        
        with st.spinner("Gerando previsões..."):
            df_forecast = predict_future_values(df_timeseries, days=5)
        
        # Criar animação mantendo dados originais
        da_pm25 = ds[pm25_var]
        
        time_dims = [dim for dim in da_pm25.dims if 'time' in dim or 'forecast' in dim]
        
        if 'forecast_reference_time' in da_pm25.dims:
            time_dim = 'forecast_reference_time'
            frames = len(da_pm25[time_dim])
        else:
            time_dim = time_dims[0]
            frames = len(da_pm25[time_dim])
        
        if frames < 1:
            st.error("Erro: Dados insuficientes para animação.")
            return None
        
        # Usar percentis para definir limites da animação
        vmin, vmax = float(np.nanpercentile(da_pm25.values, 5)), float(np.nanpercentile(da_pm25.values, 95))
        
        # Criar figura para animação PM2.5
        fig = plt.figure(figsize=(14, 10))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.coastlines(resolution='50m', color='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', color='gray')
        ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle='-', edgecolor='black', linewidth=1)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Usar extensão dos dados CAMS
        lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
        lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        ax.text(lon_center, lat_max + 0.1, city.upper(), 
                transform=ccrs.PlateCarree(), fontsize=18, fontweight='bold',
                ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='white', alpha=0.8))
        
        ax.plot(lon_center, lat_center, 'ro', markersize=12, transform=ccrs.PlateCarree(), 
                label=city, markeredgecolor='white', markeredgewidth=2)
        
        # Obter primeiro frame
        if 'forecast_period' in da_pm25.dims and 'forecast_reference_time' in da_pm25.dims:
            first_frame_data = da_pm25.isel(forecast_period=0, forecast_reference_time=0).values
            first_frame_time = pd.to_datetime(ds.forecast_reference_time.values[0])
        else:
            first_frame_data = da_pm25.isel({time_dim: 0}).values
            first_frame_time = pd.to_datetime(da_pm25[time_dim].values[0])
        
        im = ax.pcolormesh(ds.longitude, ds.latitude, first_frame_data, 
                         cmap='YlOrRd', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04, orientation='horizontal')
        
        # Determinar unidade da barra de cores
        if vmax < 1e-6:
            unit_label = 'PM2.5 (kg/m³)'
        elif vmax < 1e-3:
            unit_label = 'PM2.5 (g/m³)'  
        else:
            unit_label = 'PM2.5 (μg/m³)'
            
        cbar.set_label(unit_label, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        title = ax.set_title(f'PM2.5 - {city}\n{first_frame_time.strftime("%d/%m/%Y %H:%M UTC")}', 
                           fontsize=14, pad=20)
        
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
        
        actual_frames = min(frames, 20)
        
        ani = animation.FuncAnimation(fig, animate, frames=actual_frames, 
                                     interval=animation_speed, blit=True)
        
        gif_filename = f'PM25_{city}_{start_date}_to_{end_date}.gif'
        
        with st.spinner('Salvando animação...'):
            ani.save(gif_filename, writer=animation.PillowWriter(fps=2))
        
        plt.close(fig)

        # Criar mapas estáticos melhorados
        with st.spinner('Criando mapas melhorados de PM2.5 e PM10...'):
            enhanced_map_filename = create_enhanced_static_maps(
                {'dataset': ds, 'pm25_var': pm25_var, 'pm10_var': pm10_var}, 
                city, lat_center, lon_center, ms_shapes, start_date
            )

        # Analisar todas as cidades mantendo dados originais CAMS
        top_pollution_cities = None
        try:
            with st.spinner("Analisando qualidade do ar em todos os municípios de MS..."):
                top_pollution_cities = analyze_all_cities(ds, pm25_var, pm10_var, cities)
        except Exception as e:
            st.warning(f"Não foi possível analisar todas as cidades: {str(e)}")
            top_pollution_cities = pd.DataFrame(columns=['cidade', 'pm25_max_cams', 'pm10_max_cams', 'aqi_max', 'data_max', 'categoria'])
        
        return {
            'animation': gif_filename,
            'enhanced_maps': enhanced_map_filename,
            'timeseries': df_timeseries,
            'forecast': df_forecast,
            'dataset': ds,
            'pm25_var': pm25_var,
            'pm10_var': pm10_var,
            'top_pollution': top_pollution_cities
        }
    
    except Exception as e:
        st.error(f"Erro ao processar os dados: {str(e)}")
        st.write("Detalhes da requisição:")
        st.write(request)
        return None

# Carregar shapefiles dos municípios
with st.spinner("Carregando shapes dos municípios..."):
    ms_shapes = load_ms_municipalities()

# Sidebar para configurações
st.sidebar.header("Configurações")

available_cities = sorted(list(set(ms_shapes['NM_MUN'].tolist()).intersection(set(cities.keys()))))
if not available_cities:
    available_cities = list(cities.keys())

city = st.sidebar.selectbox("Selecione o município para análise detalhada", available_cities)
lat_center, lon_center = cities[city]

st.sidebar.subheader("Período de Análise")
start_date = st.sidebar.date_input("Data de Início", datetime.today() - timedelta(days=2))
end_date = st.sidebar.date_input("Data Final", datetime.today() + timedelta(days=5))

all_hours = list(range(0, 24, 3))
start_hour = st.sidebar.selectbox("Horário Inicial", all_hours, format_func=lambda x: f"{x:02d}:00")
end_hour = st.sidebar.selectbox("Horário Final", all_hours, index=len(all_hours)-1, format_func=lambda x: f"{x:02d}:00")

st.sidebar.subheader("Opções Avançadas")
with st.sidebar.expander("Configurações da Visualização"):
    animation_speed = st.slider("Velocidade da Animação (ms)", 200, 1000, 500)
    show_pm10_animation = st.checkbox("Gerar animação também para PM10", value=False)

st.sidebar.info("Dados Diretos CAMS\nEste sistema utiliza os valores originais do CAMS sem conversões de unidade.")

# Botão principal
st.markdown("### Iniciar Análise Completa")
st.markdown(f"Clique no botão abaixo para gerar análise de PM2.5 e PM10 centralizada em **{city}**.")

if st.button("Gerar Análise de Qualidade do Ar", type="primary", use_container_width=True):
    try:
        results = generate_pm_analysis()
        
        if results:
            tab1, tab2, tab3, tab4 = st.tabs([
                "Análise do Município", 
                "Alerta de Qualidade do Ar", 
                f"Mapa de {city}",
                "Análise Detalhada PM"
            ])
            
            # Aba do Mapa
            with tab3:
                st.subheader(f"Mapas de PM2.5 e PM10 - {city}")
                
                if 'enhanced_maps' in results and results['enhanced_maps']:
                    st.image(results['enhanced_maps'], 
                            caption=f"Concentrações de PM2.5 e PM10 (valores originais CAMS) - Foco em {city}")
                    
                    try:
                        with open(results['enhanced_maps'], "rb") as file:
                            st.download_button(
                                label="Baixar Mapas (PNG)",
                                data=file,
                                file_name=f"PM25_PM10_Maps_{city}_{start_date}.png",
                                mime="image/png"
                            )
                    except:
                        pass
                
                with st.expander("Ver Animação Temporal PM2.5"):
                    st.image(results['animation'], 
                            caption=f"Evolução temporal do PM2.5 em {city} ({start_date} a {end_date})")
                    
                    with open(results['animation'], "rb") as file:
                        st.download_button(
                            label="Baixar Animação (GIF)",
                            data=file,
                            file_name=f"PM25_{city}_{start_date}_to_{end_date}.gif",
                            mime="image/gif"
                        )
                
                st.info(f"""
                **Mapas baseados nos dados diretos do CAMS:**
                
                - Extensão: Baseada na cobertura real dos dados CAMS
                - Valores: Mantidos na escala original do sistema CAMS
                - Unidades: Detectadas automaticamente (kg/m³, g/m³ ou μg/m³)
                - Município: {city} destacado em vermelho
                - Resolução: ~0.4° x 0.4° (≈ 44 km)
                """)
            
            # Aba de Alertas com valores CAMS originais
            with tab2:
                st.subheader("Alerta de Qualidade do Ar - Mato Grosso do Sul")
                
                if 'top_pollution' in results and not results['top_pollution'].empty:
                    top_cities = results['top_pollution'].head(20)
                    
                    critical_cities = top_cities[top_cities['aqi_max'] > 100]
                    very_critical = top_cities[top_cities['aqi_max'] > 150]
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Cidades em Alerta", len(critical_cities))
                    col2.metric("Condição Insalubre", len(very_critical))
                    col3.metric("IQA Máximo Previsto", f"{top_cities['aqi_max'].max():.0f}")
                    
                    if len(critical_cities) > 0:
                        st.error(f"""
                        ### ALERTA DE QUALIDADE DO AR
                        
                        **{len(critical_cities)} municípios** com previsão de qualidade do ar 
                        inadequada nos próximos 5 dias!
                        
                        Municípios mais críticos:
                        1. **{top_cities.iloc[0]['cidade']}**: IQA {top_cities.iloc[0]['aqi_max']:.0f}
                        2. **{top_cities.iloc[1]['cidade']}**: IQA {top_cities.iloc[1]['aqi_max']:.0f}
                        3. **{top_cities.iloc[2]['cidade']}**: IQA {top_cities.iloc[2]['aqi_max']:.0f}
                        """)
                    
                    st.markdown("### Ranking de Qualidade do Ar por Município (Dados CAMS Originais)")
                    
                    # Renomear colunas mantendo valores CAMS originais
                    top_cities_display = top_cities.rename(columns={
                        'cidade': 'Município',
                        'pm25_max_cams': 'PM2.5 Máx (CAMS)',
                        'pm10_max_cams': 'PM10 Máx (CAMS)',
                        'aqi_max': 'IQA Máx',
                        'data_max': 'Data Crítica',
                        'categoria': 'Categoria'
                    })
                    
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
                    
                    st.dataframe(
                        top_cities_display.style.apply(style_aqi_row, axis=1),
                        use_container_width=True
                    )
                    
                    csv_alert = top_cities.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Baixar Dados de Alerta (CSV)",
                        data=csv_alert,
                        file_name=f"Alerta_Qualidade_Ar_MS_CAMS_{start_date}_to_{end_date}.csv",
                        mime="text/csv",
                    )
                    
                    st.info("""
                    **Nota sobre os dados:**
                    - Valores PM2.5/PM10 são mantidos conforme exportados pelo CAMS
                    - Unidades variam conforme a escala original (kg/m³, g/m³ ou μg/m³)
                    - IQA calculado após conversão apropriada apenas para índice
                    - Dados representam máximos previstos nos próximos 5 dias
                    """)
                else:
                    st.info("Dados de análise estadual não disponíveis.")
            
            # Restante das abas (tab1 e tab4) mantêm estrutura similar
            # mas com foco nos valores originais CAMS
            
            with tab1:
                st.subheader(f"Análise Detalhada - {city}")
                st.info("Análise baseada nos valores originais do sistema CAMS")
                
                df_combined = results['forecast']
                
                if not df_combined.empty:
                    hist_data = df_combined[df_combined['type'] == 'historical']
                    forecast_data = df_combined[df_combined['type'] == 'forecast']
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                        
                        # Gráfico PM2.5 (valores originais)
                        ax1.plot(hist_data['time'], hist_data['pm25'], 
                               'o-', color='darkblue', label='PM2.5 Observado (CAMS)', markersize=6)
                        ax1.plot(forecast_data['time'], forecast_data['pm25'], 
                               'x--', color='red', label='PM2.5 Previsto', markersize=6)
                        
                        # Detectar unidade para título
                        max_pm25 = hist_data['pm25'].max() if not hist_data.empty else 0
                        if max_pm25 < 1e-6:
                            pm25_unit = 'kg/m³'
                        elif max_pm25 < 1e-3:
                            pm25_unit = 'g/m³'
                        else:
                            pm25_unit = 'μg/m³'
                            
                        ax1.set_ylabel(f'PM2.5 ({pm25_unit})', fontsize=12)
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        ax1.set_title(f'Material Particulado PM2.5 (Valores CAMS Originais)', fontsize=14)
                        
                        # Gráfico PM10
                        ax2.plot(hist_data['time'], hist_data['pm10'], 
                               'o-', color='brown', label='PM10 Observado (CAMS)', markersize=6)
                        ax2.plot(forecast_data['time'], forecast_data['pm10'], 
                               'x--', color='darkred', label='PM10 Previsto', markersize=6)
                        
                        max_pm10 = hist_data['pm10'].max() if not hist_data.empty else 0
                        if max_pm10 < 1e-6:
                            pm10_unit = 'kg/m³'
                        elif max_pm10 < 1e-3:
                            pm10_unit = 'g/m³'
                        else:
                            pm10_unit = 'μg/m³'
                            
                        ax2.set_ylabel(f'PM10 ({pm10_unit})', fontsize=12)
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        ax2.set_title(f'Material Particulado PM10 (Valores CAMS Originais)', fontsize=14)
                        
                        # Gráfico IQA
                        ax3.plot(hist_data['time'], hist_data['aqi'], 
                               'o-', color='purple', label='IQA Observado', markersize=6)
                        ax3.plot(forecast_data['time'], forecast_data['aqi'], 
                               'x--', color='magenta', label='IQA Previsto', markersize=6)
                        
                        ax3.axhspan(0, 50, alpha=0.2, color='green', label='Boa')
                        ax3.axhspan(51, 100, alpha=0.2, color='yellow', label='Moderada')
                        ax3.axhspan(101, 150, alpha=0.2, color='orange', label='Insalubre p/ Sensíveis')
                        ax3.axhspan(151, 200, alpha=0.2, color='red', label='Insalubre')
                        
                        ax3.set_ylabel('IQA', fontsize=12)
                        ax3.set_xlabel('Data/Hora', fontsize=12)
                        ax3.legend()
                        ax3.grid(True, alpha=0.3)
                        ax3.set_title('Índice de Qualidade do Ar', fontsize=14)
                        
                        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
                        plt.xticks(rotation=45)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("Estatísticas Atuais")
                        
                        if not hist_data.empty:
                            curr_pm25 = hist_data['pm25'].iloc[-1]
                            curr_pm10 = hist_data['pm10'].iloc[-1]
                            curr_aqi = hist_data['aqi'].iloc[-1]
                            curr_category = hist_data['aqi_category'].iloc[-1]
                            curr_color = hist_data['aqi_color'].iloc[-1]
                            
                            col_a, col_b = st.columns(2)
                            col_a.metric("PM2.5 Atual", f"{curr_pm25:.2e}")
                            col_b.metric("PM10 Atual", f"{curr_pm10:.2e}")
                            
                            st.metric("IQA Atual", f"{curr_aqi:.0f}")
                            
                            st.markdown(f"""
                            <div style="padding:15px; border-radius:10px; background-color:{curr_color}; 
                            color:white; text-align:center; margin:10px 0;">
                            <h3 style="margin:0;">Qualidade do Ar</h3>
                            <h2 style="margin:5px 0;">{curr_category}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.info(f"""
                            **Valores originais CAMS:**
                            - PM2.5: {curr_pm25:.2e} (unidade original)
                            - PM10: {curr_pm10:.2e} (unidade original)
                            - IQA calculado após conversão apropriada
                            """)
                        
                        csv = df_combined.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Baixar Dados Completos (CSV)",
                            data=csv,
                            file_name=f"PM_data_CAMS_{city}_{start_date}_to_{end_date}.csv",
                            mime="text/csv",
                        )
            
            with tab4:
                st.subheader("Análise Detalhada - Dados CAMS Originais")
                
                with st.expander("Sobre os Dados Diretos do CAMS"):
                    st.markdown("""
                    ### Dados Diretos de PM2.5 e PM10 do CAMS
                    
                    **Principais características:**
                    - Valores mantidos conforme exportados pelo CAMS
                    - Sem conversões de unidade aplicadas
                    - Unidades variam: kg/m³, g/m³ ou μg/m³
                    - Conversão apenas para cálculo do IQA quando necessário
                    - Mapas baseados na extensão real dos dados
                    - Resolução espacial: ~0.4° x 0.4° (≈ 44 km)
                    """)
                
                st.info("Esta versão preserva os valores exatos conforme fornecidos pelo sistema CAMS, permitindo análise com dados não processados.")
    
    except Exception as e:
        st.error(f"Erro durante a análise: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

st.markdown("---")
st.markdown("""
### Informações Importantes

**Sobre os Dados Diretos CAMS:**
- Valores mantidos conforme exportação original do sistema
- Unidades preservadas (kg/m³, g/m³ ou μg/m³)
- Mapas baseados na extensão real da cobertura de dados
- Maior fidelidade aos dados de origem

**Desenvolvido para:** Monitoramento da Qualidade do Ar em Mato Grosso do Sul
""")
