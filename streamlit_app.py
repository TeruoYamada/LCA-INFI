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

# Função para criar animação de PM2.5 ou PM10 com contorno do município
def create_pm_animation(ds, pm_var, city, lat_center, lon_center, ms_shapes, start_date, pm_type="PM2.5"):
    """
    Cria animação temporal de PM2.5 ou PM10 com destaque para o município selecionado.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    import pandas as pd
    
    # Obter dados do poluente (SEM CONVERSÃO DE UNIDADES)
    da_pm = ds[pm_var]
    
    # Identificar dimensões temporais
    time_dims = [dim for dim in da_pm.dims if 'time' in dim or 'forecast' in dim]
    
    if 'forecast_reference_time' in da_pm.dims:
        time_dim = 'forecast_reference_time'
        frames = len(da_pm[time_dim])
    else:
        time_dim = time_dims[0]
        frames = len(da_pm[time_dim])
    
    if frames < 1:
        st.error("Erro: Dados insuficientes para animação.")
        return None
    
    # Definir limites de cores (SEM ASSUMIR UNIDADES ESPECÍFICAS)
    vmin, vmax = float(da_pm.min().values), float(da_pm.max().values)
    # Usar 5% de margem para melhor visualização
    range_margin = (vmax - vmin) * 0.05
    vmin = max(0, vmin - range_margin)
    vmax = vmax + range_margin
    
    if pm_type == "PM2.5":
        colormap = 'YlOrRd'
    else:  # PM10
        colormap = 'Oranges'
    
    # Definir extensão para MS
    ms_extent = [-58.5, -50.5, -24.5, -17.0]  # [west, east, south, north]
    
    # Criar figura para animação
    fig = plt.figure(figsize=(14, 10))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Adicionar features do mapa
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', color='gray')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle='-', edgecolor='black', linewidth=1)
    
    # Grid
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Definir extensão do mapa
    ax.set_extent(ms_extent, crs=ccrs.PlateCarree())
    
    # Adicionar contornos dos municípios de MS
    if ms_shapes is not None and not ms_shapes.empty:
        try:
            # Plotar todos os municípios
            ms_shapes.boundary.plot(ax=ax, color='black', linewidth=0.6, 
                                   transform=ccrs.PlateCarree(), alpha=0.7)
            
            # Destacar município selecionado
            selected_city = ms_shapes[ms_shapes['NM_MUN'].str.upper() == city.upper()]
            if not selected_city.empty:
                selected_city.boundary.plot(ax=ax, color='red', linewidth=3.0, 
                                           transform=ccrs.PlateCarree())
                selected_city.plot(ax=ax, facecolor='none', edgecolor='red', 
                                  linewidth=3.0, transform=ccrs.PlateCarree(), alpha=0.8)
        except Exception as e:
            print(f"Erro ao plotar shapefile: {e}")
    
    # Marcar o município selecionado
    ax.plot(lon_center, lat_center, 'ro', markersize=12, transform=ccrs.PlateCarree(), 
            markeredgecolor='white', markeredgewidth=2, zorder=10)
    
    # Obter primeiro frame
    if 'forecast_period' in da_pm.dims and 'forecast_reference_time' in da_pm.dims:
        first_frame_data = da_pm.isel(forecast_period=0, forecast_reference_time=0).values
        first_frame_time = pd.to_datetime(ds.forecast_reference_time.values[0])
    else:
        first_frame_data = da_pm.isel({time_dim: 0}).values
        first_frame_time = pd.to_datetime(da_pm[time_dim].values[0])
    
    # Criar mapa de cores inicial
    im = ax.pcolormesh(ds.longitude, ds.latitude, first_frame_data, 
                     cmap=colormap, vmin=vmin, vmax=vmax, 
                     transform=ccrs.PlateCarree(), alpha=0.8)
    
    # Barra de cores (sem assumir unidades específicas)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, orientation='horizontal')
    units_label = da_pm.attrs.get('units', 'unidade original')
    cbar.set_label(f'{pm_type} ({units_label})', fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Título inicial
    title = ax.set_title(f'{pm_type} - {city}\n{first_frame_time.strftime("%d/%m/%Y %H:%M UTC")}', 
                       fontsize=16, pad=20, weight='bold')
    
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
            title.set_text(f'{pm_type} - {city}\n{frame_time.strftime("%d/%m/%Y %H:%M UTC")}')
            
            return [im, title]
        except Exception as e:
            print(f"Erro no frame {i}: {str(e)}")
            return [im, title]
    
    # Limitar número de frames
    actual_frames = min(frames, 20)
    
    return fig, animate, actual_frames

# Função para extrair série temporal de PM2.5 e PM10 (SEM CONVERSÃO)
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
                    
                    # SEM CONVERSÃO DE UNIDADES - valores originais
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
                
                # SEM CONVERSÃO DE UNIDADES - valores originais
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
        
        # Calcular IQA com valores originais (pode não funcionar se não estiver em μg/m³)
        try:
            aqi_values = df.apply(lambda row: calculate_aqi_original_units(row['pm25'], row['pm10']), axis=1)
            df['aqi'] = aqi_values.apply(lambda x: x[0])
            df['aqi_category'] = aqi_values.apply(lambda x: x[1])
            df['aqi_color'] = aqi_values.apply(lambda x: x[2])
        except:
            # Se falhar, usar valores relativos
            df['aqi'] = 0
            df['aqi_category'] = 'Não calculado'
            df['aqi_color'] = 'gray'
        
        return df
    else:
        return pd.DataFrame(columns=['time', 'pm25', 'pm10', 'aqi', 'aqi_category'])

# Função para calcular IQA com unidades originais (pode não funcionar corretamente)
def calculate_aqi_original_units(pm25, pm10):
    """
    Tenta calcular IQA com unidades originais.
    AVISO: Pode não funcionar se os dados não estiverem em μg/m³
    """
    # Se os valores são muito pequenos, provavelmente não estão em μg/m³
    if pm25 < 1e-3 or pm10 < 1e-3:
        return 0, "Unidade incompatível", "gray"
    
    # Se os valores são muito grandes, podem estar em outra unidade
    if pm25 > 1000 or pm10 > 1000:
        return 0, "Unidade incompatível", "gray"
    
    # Assumir que estão em μg/m³ e calcular normalmente
    return calculate_aqi_standard(pm25, pm10)

# Função padrão para calcular IQA (assumindo μg/m³)
def calculate_aqi_standard(pm25, pm10):
    """
    Calcula o Índice de Qualidade do Ar baseado em PM2.5 e PM10.
    Usa os padrões da EPA adaptados para o Brasil.
    ASSUME que os valores estão em μg/m³.
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

# Função para prever valores futuros (sem conversão)
def predict_future_values(df, days=5):
    """Gera previsão para PM2.5 e PM10 com valores originais."""
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
    
    # Limitar valores (manter não-negativos)
    future_pm25 = np.maximum(future_pm25, 0)
    future_pm10 = np.maximum(future_pm10, 0)
    
    # Calcular IQA para previsões (pode falhar se unidades não forem μg/m³)
    future_aqi = []
    future_categories = []
    future_colors = []
    
    for pm25, pm10 in zip(future_pm25, future_pm10):
        try:
            aqi, category, color = calculate_aqi_original_units(pm25, pm10)
            future_aqi.append(aqi)
            future_categories.append(category)
            future_colors.append(color)
        except:
            future_aqi.append(0)
            future_categories.append('Não calculado')
            future_colors.append('gray')
    
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

# Função para analisar todas as cidades (sem conversão)
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
        
        # Não assumir unidades específicas para formatação
        df_results['pm25_max'] = df_results['pm25_max'].round(6)
        df_results['pm10_max'] = df_results['pm10_max'].round(6)
        df_results['aqi_max'] = df_results['aqi_max'].round(0)
        df_results['data_max'] = df_results['data_max'].dt.strftime('%d/%m/%Y %H:%M')
        
        return df_results
    else:
        return pd.DataFrame(columns=['cidade', 'pm25_max', 'pm10_max', 'aqi_max', 'data_max', 'categoria'])

# Configuração inicial da página
st.set_page_config(layout="wide", page_title="Monitor PM2.5/PM10 - MS", page_icon="🌍")

# Carregar autenticação a partir do secrets.toml
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

# Lista de cidades de MS
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
