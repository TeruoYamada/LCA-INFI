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
import seaborn as sns

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
        
        # Converter para μg/m³ se necessário
        pm25_values = da_pm25.values.copy()
        if pm25_values.max() < 1e-6:  # kg/m³
            pm25_values = pm25_values * 1e9
        elif pm25_values.max() < 1e-3:  # g/m³
            pm25_values = pm25_values * 1e6
        
        print(f"Valores convertidos PM2.5: min={pm25_values.min():.2f}, max={pm25_values.max():.2f} μg/m³")
        
        # Obter frame específico
        if 'forecast_reference_time' in da_pm25.dims and 'forecast_period' in da_pm25.dims:
            frame_data = pm25_values[0, min(frame_idx, pm25_values.shape[1]-1), :, :]
            frame_time = pd.to_datetime(ds.forecast_reference_time.values[0])
        else:
            time_dims = [d for d in da_pm25.dims if 'time' in d or 'forecast' in d]
            if time_dims:
                time_dim = time_dims[0]
                frame_data = pm25_values[min(frame_idx, pm25_values.shape[0]-1), :, :]
                frame_time = pd.to_datetime(da_pm25[time_dim].values[min(frame_idx, len(da_pm25[time_dim])-1)])
            else:
                frame_data = pm25_values
                frame_time = pd.to_datetime('now')
        
        # Definir limites de cores baseados nos dados convertidos
        vmin_pm25 = max(0, np.nanpercentile(frame_data, 5))
        vmax_pm25 = np.nanpercentile(frame_data, 95)
        
        print(f"Limites PM2.5: {vmin_pm25:.2f} a {vmax_pm25:.2f} μg/m³")
        
        # Plotar PM2.5
        im1 = ax1.pcolormesh(ds.longitude, ds.latitude, frame_data, 
                            cmap='YlOrRd', vmin=vmin_pm25, vmax=vmax_pm25, 
                            transform=ccrs.PlateCarree(), alpha=0.8)
        
        # Barra de cores para PM2.5
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', 
                            fraction=0.046, pad=0.08, shrink=0.8)
        cbar1.set_label('PM2.5 (μg/m³)', fontsize=12)
        
    except Exception as e:
        print(f"Erro ao plotar PM2.5: {e}")
        im1 = None
    
    # Adicionar shapefile de MS se disponível
    if ms_shapes is not None and not ms_shapes.empty:
        try:
            ms_shapes.boundary.plot(ax=ax1, color='black', linewidth=1.5, transform=ccrs.PlateCarree())
            
            # Destacar município selecionado
            selected_city = ms_shapes[ms_shapes['NM_MUN'].str.upper() == city.upper()]
            if not selected_city.empty:
                selected_city.boundary.plot(ax=ax1, color='red', 
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
        
        # Converter para μg/m³ se necessário
        pm10_values = da_pm10.values.copy()
        if pm10_values.max() < 1e-6:  # kg/m³
            pm10_values = pm10_values * 1e9
        elif pm10_values.max() < 1e-3:  # g/m³
            pm10_values = pm10_values * 1e6
        
        print(f"Valores convertidos PM10: min={pm10_values.min():.2f}, max={pm10_values.max():.2f} μg/m³")
        
        # Obter frame específico
        if 'forecast_reference_time' in da_pm10.dims and 'forecast_period' in da_pm10.dims:
            frame_data_pm10 = pm10_values[0, min(frame_idx, pm10_values.shape[1]-1), :, :]
        else:
            time_dims = [d for d in da_pm10.dims if 'time' in d or 'forecast' in d]
            if time_dims:
                time_dim = time_dims[0]
                frame_data_pm10 = pm10_values[min(frame_idx, pm10_values.shape[0]-1), :, :]
            else:
                frame_data_pm10 = pm10_values
        
        # Definir limites de cores para PM10
        vmin_pm10 = max(0, np.nanpercentile(frame_data_pm10, 5))
        vmax_pm10 = np.nanpercentile(frame_data_pm10, 95)
        
        print(f"Limites PM10: {vmin_pm10:.2f} a {vmax_pm10:.2f} μg/m³")
        
        # Plotar PM10
        im2 = ax2.pcolormesh(ds.longitude, ds.latitude, frame_data_pm10, 
                            cmap='Oranges', vmin=vmin_pm10, vmax=vmax_pm10, 
                            transform=ccrs.PlateCarree(), alpha=0.8)
        
        # Barra de cores para PM10
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', 
                            fraction=0.046, pad=0.08, shrink=0.8)
        cbar2.set_label('PM10 (μg/m³)', fontsize=12)
        
    except Exception as e:
        print(f"Erro ao plotar PM10: {e}")
        im2 = None
    
    # Adicionar shapefile de MS
    if ms_shapes is not None and not ms_shapes.empty:
        try:
            ms_shapes.boundary.plot(ax=ax2, color='black', linewidth=1.5, transform=ccrs.PlateCarree())
            
            # Destacar município selecionado
            selected_city = ms_shapes[ms_shapes['NM_MUN'].str.upper() == city.upper()]
            if not selected_city.empty:
                selected_city.boundary.plot(ax=ax2, color='red', 
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
    
    return fig

# Função para criar animação melhorada
def create_enhanced_animation(ds, pm_var, city, lat_center, lon_center, ms_shapes, start_date, end_date):
    """
    Cria animação com contorno do MS e município destacado.
    """
    da_pm = ds[pm_var]
    
    # Converter para μg/m³
    pm_values = da_pm.values.copy()
    if pm_values.max() < 1e-6:  # kg/m³
        pm_values = pm_values * 1e9
    elif pm_values.max() < 1e-3:  # g/m³
        pm_values = pm_values * 1e6
    
    # Configuração da figura
    fig = plt.figure(figsize=(14, 10))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, color='black')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=1, edgecolor='darkgray')
    
    # Adicionar contorno do MS
    if ms_shapes is not None and not ms_shapes.empty:
        ms_shapes.boundary.plot(ax=ax, color='black', linewidth=2, transform=ccrs.PlateCarree())
        
        # Destacar município selecionado
        selected_city = ms_shapes[ms_shapes['NM_MUN'].str.upper() == city.upper()]
        if not selected_city.empty:
            selected_city.boundary.plot(ax=ax, color='red', linewidth=3.0, transform=ccrs.PlateCarree())
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Usar extensão dos dados CAMS
    lon_min, lon_max = float(ds.longitude.min()), float(ds.longitude.max())
    lat_min, lat_max = float(ds.latitude.min()), float(ds.latitude.max())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    ax.plot(lon_center, lat_center, 'ro', markersize=12, transform=ccrs.PlateCarree(), 
            markeredgecolor='white', markeredgewidth=2)
    
    # Definir limites de cores
    vmin = max(0, np.nanpercentile(pm_values, 5))
    vmax = np.nanpercentile(pm_values, 95)
    
    # Primeiro frame
    if 'forecast_period' in da_pm.dims and 'forecast_reference_time' in da_pm.dims:
        first_frame_data = pm_values[0, 0, :, :]
        first_frame_time = pd.to_datetime(ds.forecast_reference_time.values[0])
        frames = min(pm_values.shape[1], 20)
    else:
        time_dims = [d for d in da_pm.dims if 'time' in d or 'forecast' in d]
        time_dim = time_dims[0] if time_dims else None
        first_frame_data = pm_values[0, :, :] if time_dim else pm_values
        first_frame_time = pd.to_datetime(da_pm[time_dim].values[0]) if time_dim else pd.to_datetime('now')
        frames = min(pm_values.shape[0] if time_dim else 1, 20)
    
    im = ax.pcolormesh(ds.longitude, ds.latitude, first_frame_data, 
                      cmap='YlOrRd', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, orientation='horizontal')
    pm_type = 'PM2.5' if 'pm2p5' in pm_var.lower() or '2.5' in pm_var else 'PM10'
    cbar.set_label(f'{pm_type} (μg/m³)', fontsize=12)
    
    title = ax.set_title(f'{pm_type} - {city.upper()}\n{first_frame_time.strftime("%d/%m/%Y %H:%M UTC")}', 
                        fontsize=14, pad=20)
    
    def animate(i):
        try:
            if 'forecast_period' in da_pm.dims and 'forecast_reference_time' in da_pm.dims:
                frame_data = pm_values[0, min(i, pm_values.shape[1]-1), :, :]
                frame_time = pd.to_datetime(ds.forecast_reference_time.values[0]) + pd.Timedelta(hours=i*3)
            else:
                frame_data = pm_values[min(i, pm_values.shape[0]-1), :, :]
                frame_time = pd.to_datetime(da_pm[time_dim].values[min(i, len(da_pm[time_dim])-1)])
            
            im.set_array(frame_data.ravel())
            title.set_text(f'{pm_type} - {city.upper()}\n{frame_time.strftime("%d/%m/%Y %H:%M UTC")}')
            
            return [im, title]
        except Exception as e:
            print(f"Erro no frame {i}: {str(e)}")
            return [im, title]
    
    ani = animation.FuncAnimation(fig, animate, frames=frames, 
                                 interval=500, blit=True)
    
    return ani, fig

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
para todos os municípios de Mato Grosso do Sul usando dados do CAMS.

**Características desta versão:**
- Dados em μg/m³ (microgramas por metro cúbico)
- Mapas com contornos do MS e município selecionado
- Animação temporal com frames de material particulado
- Top 10 cidades com maiores concentrações
- Análise baseada na extensão real dos dados CAMS
""")

# Função para calcular IQA mantendo valores em μg/m³
def calculate_aqi_ugm3(pm25, pm10):
    """
    Calcula IQA usando valores já em μg/m³.
    """
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

# Função para extrair série temporal convertendo para μg/m³
def extract_pm_timeseries(ds, lat, lon, pm25_var, pm10_var):
    """Extrai série temporal convertendo valores para μg/m³."""
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
                    
                    # Converter para μg/m³
                    if pm25_val < 1e-6:  # kg/m³
                        pm25_val *= 1e9
                        pm10_val *= 1e9
                    elif pm25_val < 1e-3:  # g/m³
                        pm25_val *= 1e6
                        pm10_val *= 1e6
                    
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
                
                # Converter para μg/m³
                if pm25_val < 1e-6:  # kg/m³
                    pm25_val *= 1e9
                    pm10_val *= 1e9
                elif pm25_val < 1e-3:  # g/m³
                    pm25_val *= 1e6
                    pm10_val *= 1e6
                
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
        aqi_values = df.apply(lambda row: calculate_aqi_ugm3(row['pm25'], row['pm10']), axis=1)
        df['aqi'] = aqi_values.apply(lambda x: x[0])
        df['aqi_category'] = aqi_values.apply(lambda x: x[1])
        df['aqi_color'] = aqi_values.apply(lambda x: x[2])
        
        return df
    else:
        return pd.DataFrame(columns=['time', 'pm25', 'pm10', 'aqi', 'aqi_category'])

# Função para prever valores futuros
def predict_future_values(df, days=5):
    """Gera previsão mantendo valores em μg/m³."""
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
        aqi, category, color = calculate_aqi_ugm3(pm25, pm10)
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

# Função para analisar todas as cidades convertendo para μg/m³
def analyze_all_cities(ds, pm25_var, pm10_var, cities_dict):
    """Analisa valores convertendo para μg/m³."""
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
                # Valores já em μg/m³
                max_pm25 = forecast_only['pm25'].max()
                max_pm10 = forecast_only['pm10'].max()
                max_aqi = forecast_only['aqi'].max()
                
                max_day_idx = forecast_only['aqi'].idxmax()
                max_day = forecast_only.loc[max_day_idx, 'time']
                max_category = forecast_only.loc[max_day_idx, 'aqi_category']
                
                cities_results.append({
                    'cidade': city_name,
                    'pm25_max': max_pm25,  # Valores em μg/m³
                    'pm10_max': max_pm10,  # Valores em μg/m³
                    'aqi_max': max_aqi,
                    'data_max': max_day,
                    'categoria': max_category
                })
    
    progress_bar.empty()
    status_text.empty()
    
    if cities_results:
        df_results = pd.DataFrame(cities_results)
        df_results = df_results.sort_values('aqi_max', ascending=False).reset_index(drop=True)
        
        df_results['data_max'] = df_results['data_max'].dt.strftime('%d/%m/%Y %H:%M')
        
        return df_results
    else:
        return pd.DataFrame(columns=['cidade', 'pm25_max', 'pm10_max', 'aqi_max', 'data_max', 'categoria'])

# Função para criar gráficos do Top 10
def create_top10_charts(df_results):
    """Cria gráficos de barras para o Top 10 de PM2.5 e PM10."""
    
    # Top 10 PM2.5
    top10_pm25 = df_results.nlargest(10, 'pm25_max')
    
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    bars1 = ax1.bar(range(len(top10_pm25)), top10_pm25['pm25_max'], 
                   color='red', alpha=0.7)
    ax1.set_xlabel('Municípios', fontsize=12)
    ax1.set_ylabel('PM2.5 (μg/m³)', fontsize=12)
    ax1.set_title('Top 10 Municípios - Maiores Concentrações de PM2.5', 
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(top10_pm25)))
    ax1.set_xticklabels(top10_pm25['cidade'], rotation=45, ha='right')
    
    # Adicionar valores nas barras
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Top 10 PM10
    top10_pm10 = df_results.nlargest(10, 'pm10_max')
    
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    bars2 = ax2.bar(range(len(top10_pm10)), top10_pm10['pm10_max'], 
                   color='orange', alpha=0.7)
    ax2.set_xlabel('Municípios', fontsize=12)
    ax2.set_ylabel('PM10 (μg/m³)', fontsize=12)
    ax2.set_title('Top 10 Municípios - Maiores Concentrações de PM10', 
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(top10_pm10)))
    ax2.set_xticklabels(top10_pm10['cidade'], rotation=45, ha='right')
    
    # Adicionar valores nas barras
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    return fig1, fig2, top10_pm25, top10_pm10
