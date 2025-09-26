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
    
    # Obter dados do poluente
    da_pm = ds[pm_var]
    
    # Converter unidades se necessário
    if da_pm.max().values < 1e-6:  # Se muito pequeno
        da_pm = da_pm * 1e9
    elif da_pm.max().values < 1e-3:  # Se pequeno
        da_pm = da_pm * 1e6
    elif da_pm.max().values > 1000:  # Se muito grande
        da_pm = da_pm / 1000
    
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
    
    # Definir limites de cores
    vmin, vmax = float(da_pm.min().values), float(da_pm.max().values)
    if pm_type == "PM2.5":
        vmin = max(0, vmin - 5)
        vmax = min(200, vmax + 10)
        colormap = 'YlOrRd'
    else:  # PM10
        vmin = max(0, vmin - 5)
        vmax = min(300, vmax + 15)
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
    
    # Barra de cores
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, orientation='horizontal')
    cbar.set_label(f'{pm_type} (μg/m³)', fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Título inicial
    title = ax.set_title(f'{pm_type} - {city}\n{first_frame_time.strftime("%d/%m/%Y %H:%M UTC")}', 
                       fontsize=16, pad=20, weight='bold')
    
    # Adicionar limites de referência na barra de cores
    if pm_type == "PM2.5":
        # Adicionar linhas de referência OMS e EPA
        ax.text(0.02, 0.02, 'Limites: OMS=25 μg/m³, EPA=35 μg/m³', 
                transform=ax.transAxes, fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='orange'),
                verticalalignment='bottom')
    else:  # PM10
        ax.text(0.02, 0.02, 'Limites: OMS=50 μg/m³, EPA=150 μg/m³', 
                transform=ax.transAxes, fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='orange'),
                verticalalignment='bottom')
    
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

# Função para validar e corrigir dados de PM
def validate_and_convert_pm_data(pm25, pm10):
    """
    Valida e converte dados de PM para as unidades corretas (μg/m³)
    
    Args:
        pm25, pm10: Valores de concentração em unidades desconhecidas
    
    Returns:
        tuple: (pm25_corrected, pm10_corrected, conversion_applied) em μg/m³
    """
    
    # Converter para float
    pm25 = float(pm25)
    pm10 = float(pm10)
    
    # Detectar unidades baseado na magnitude
    if pm25 < 1e-9 or pm10 < 1e-9:
        # Muito pequeno - provavelmente em kg/m³
        pm25 *= 1e9  # kg/m³ para μg/m³
        pm10 *= 1e9
        conversion_applied = "kg/m³ -> μg/m³"
    elif pm25 < 1e-6 or pm10 < 1e-6:
        # Pequeno - provavelmente em g/m³  
        pm25 *= 1e6  # g/m³ para μg/m³
        pm10 *= 1e6
        conversion_applied = "g/m³ -> μg/m³"
    elif pm25 < 1e-3 or pm10 < 1e-3:
        # Médio pequeno - provavelmente em mg/m³
        pm25 *= 1e3  # mg/m³ para μg/m³
        pm10 *= 1e3
        conversion_applied = "mg/m³ -> μg/m³"
    elif pm25 > 10000 or pm10 > 10000:
        # Muito grande - provavelmente em ng/m³
        pm25 /= 1000  # ng/m³ para μg/m³
        pm10 /= 1000
        conversion_applied = "ng/m³ -> μg/m³"
    else:
        # Assumir que já está em μg/m³
        conversion_applied = "Assumido μg/m³"
    
    # Validar PM2.5 <= PM10 (fisicamente correto)
    if pm25 > pm10:
        print(f"Aviso: PM2.5 ({pm25:.1f}) > PM10 ({pm10:.1f}) - fisicamente improvável")
    
    return round(pm25, 1), round(pm10, 1), conversion_applied

# Função para extrair série temporal de PM2.5 e PM10 (melhorada)
def extract_pm_timeseries(ds, lat, lon, pm25_var, pm10_var):
    """Extrai série temporal de PM2.5 e PM10 de um ponto específico do dataset."""
    lat_idx = np.abs(ds.latitude.values - lat).argmin()
    lon_idx = np.abs(ds.longitude.values - lon).argmin()
    
    times = []
    pm25_values = []
    pm10_values = []
    conversions_log = []
    
    # Identificar dimensões temporais
    time_dims = [dim for dim in ds.dims if 'time' in dim or 'forecast' in dim]
    
    if 'forecast_reference_time' in ds.dims and 'forecast_period' in ds.dims:
        for t_idx, ref_time in enumerate(ds.forecast_reference_time.values):
            for p_idx, period in enumerate(ds.forecast_period.values):
                try:
                    pm25_raw = float(ds[pm25_var].isel(
                        forecast_reference_time=t_idx, 
                        forecast_period=p_idx,
                        latitude=lat_idx, 
                        longitude=lon_idx
                    ).values)
                    
                    pm10_raw = float(ds[pm10_var].isel(
                        forecast_reference_time=t_idx, 
                        forecast_period=p_idx,
                        latitude=lat_idx, 
                        longitude=lon_idx
                    ).values)
                    
                    # Validar e converter unidades
                    pm25_val, pm10_val, conversion = validate_and_convert_pm_data(pm25_raw, pm10_raw)
                    
                    actual_time = pd.to_datetime(ref_time) + pd.to_timedelta(period, unit='h')
                    times.append(actual_time)
                    pm25_values.append(pm25_val)
                    pm10_values.append(pm10_val)
                    conversions_log.append(conversion)
                except Exception as e:
                    print(f"Erro ao processar ponto temporal {t_idx},{p_idx}: {e}")
                    continue
    elif any(dim in ds.dims for dim in ['time', 'forecast_reference_time']):
        time_dim = next(dim for dim in ds.dims if dim in ['time', 'forecast_reference_time'])
        for t_idx in range(len(ds[time_dim])):
            try:
                pm25_raw = float(ds[pm25_var].isel({
                    time_dim: t_idx,
                    'latitude': lat_idx,
                    'longitude': lon_idx
                }).values)
                
                pm10_raw = float(ds[pm10_var].isel({
                    time_dim: t_idx,
                    'latitude': lat_idx,
                    'longitude': lon_idx
                }).values)
                
                # Validar e converter unidades
                pm25_val, pm10_val, conversion = validate_and_convert_pm_data(pm25_raw, pm10_raw)
                
                times.append(pd.to_datetime(ds[time_dim].isel({time_dim: t_idx}).values))
                pm25_values.append(pm25_val)
                pm10_values.append(pm10_val)
                conversions_log.append(conversion)
            except Exception as e:
                print(f"Erro ao processar ponto temporal {t_idx}: {e}")
                continue
    
    if times and pm25_values and pm10_values:
        # Log das conversões aplicadas
        unique_conversions = list(set(conversions_log))
        print(f"Conversões aplicadas: {unique_conversions}")
        
        df = pd.DataFrame({
            'time': times,
            'pm25': pm25_values,
            'pm10': pm10_values
        })
        df = df.sort_values('time').reset_index(drop=True)
        
        # Calcular IQA com função corrigida
        aqi_results = []
        for _, row in df.iterrows():
            aqi, category, color = calculate_aqi(row['pm25'], row['pm10'])
            aqi_results.append((aqi, category, color))
        
        df['aqi'] = [result[0] for result in aqi_results]
        df['aqi_category'] = [result[1] for result in aqi_results]
        df['aqi_color'] = [result[2] for result in aqi_results]
        
        # Estatísticas de validação
        print(f"Dados extraídos: {len(df)} pontos temporais")
        print(f"PM2.5 range: {df['pm25'].min():.1f} - {df['pm25'].max():.1f} μg/m³")
        print(f"PM10 range: {df['pm10'].min():.1f} - {df['pm10'].max():.1f} μg/m³")
        print(f"IQA range: {df['aqi'].min():.0f} - {df['aqi'].max():.0f}")
        
        return df
    else:
        print("Erro: Não foi possível extrair dados válidos")
        return pd.DataFrame(columns=['time', 'pm25', 'pm10', 'aqi', 'aqi_category', 'aqi_color'])

# Função corrigida para calcular IQA (Índice de Qualidade do Ar)
def calculate_aqi(pm25, pm10):
    """
    Calcula o Índice de Qualidade do Ar baseado em PM2.5 e PM10.
    Usa os padrões da EPA dos EUA, adaptados para uso no Brasil.
    
    Args:
        pm25 (float): Concentração de PM2.5 em μg/m³
        pm10 (float): Concentração de PM10 em μg/m³
    
    Returns:
        tuple: (aqi_value, category, color)
    """
    
    # Validar inputs
    if pm25 < 0 or pm10 < 0:
        return 0, "Dados Inválidos", "#gray"
    
    # Breakpoints para PM2.5 (μg/m³) - EPA padrão
    pm25_breakpoints = [
        (0.0, 12.0, 0, 50),      # Boa
        (12.1, 35.4, 51, 100),   # Moderada  
        (35.5, 55.4, 101, 150),  # Insalubre para Grupos Sensíveis
        (55.5, 150.4, 151, 200), # Insalubre
        (150.5, 250.4, 201, 300), # Muito Insalubre
        (250.5, 500.4, 301, 500)  # Perigosa
    ]
    
    # Breakpoints para PM10 (μg/m³) - EPA padrão
    pm10_breakpoints = [
        (0, 54, 0, 50),          # Boa
        (55, 154, 51, 100),      # Moderada
        (155, 254, 101, 150),    # Insalubre para Grupos Sensíveis
        (255, 354, 151, 200),    # Insalubre
        (355, 424, 201, 300),    # Muito Insalubre
        (425, 604, 301, 500)     # Perigosa
    ]
    
    def calc_sub_index(concentration, breakpoints):
        """
        Calcula o sub-índice para um poluente específico
        """
        # Arredondar concentração para 1 casa decimal
        concentration = round(concentration, 1)
        
        for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
            if bp_lo <= concentration <= bp_hi:
                # Fórmula EPA: I = ((I_hi - I_lo)/(BP_hi - BP_lo)) * (C - BP_lo) + I_lo
                sub_index = ((i_hi - i_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + i_lo
                return round(sub_index, 0)
        
        # Se exceder todos os breakpoints, retorna 500 (máximo)
        return 500
    
    # Calcular sub-índices
    aqi_pm25 = calc_sub_index(pm25, pm25_breakpoints)
    aqi_pm10 = calc_sub_index(pm10, pm10_breakpoints)
    
    # O IQA final é o MAIOR dos dois sub-índices
    final_aqi = max(aqi_pm25, aqi_pm10)
    
    # Determinar categoria e cor
    if final_aqi <= 50:
        category = "Boa"
        color = "#00e400"  # Verde
    elif final_aqi <= 100:
        category = "Moderada"
        color = "#ffff00"  # Amarelo
    elif final_aqi <= 150:
        category = "Insalubre para Grupos Sensíveis"
        color = "#ff7e00"  # Laranja
    elif final_aqi <= 200:
        category = "Insalubre"
        color = "#ff0000"  # Vermelho
    elif final_aqi <= 300:
        category = "Muito Insalubre"
        color = "#8f3f97"  # Roxo
    else:
        category = "Perigosa"
        color = "#7e0023"  # Vinho
    
    return int(final_aqi), category, color

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

# Função para analisar todas as cidades (melhorada)
def analyze_all_cities(ds, pm25_var, pm10_var, cities_dict):
    """Analisa os valores de PM2.5 e PM10 para todas as cidades com validação melhorada."""
    cities_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Aplicar conversão de unidades no dataset completo primeiro
    print("Aplicando conversões de unidade no dataset...")
    
    # Detectar e converter unidades do dataset
    pm25_sample = float(ds[pm25_var].isel(latitude=0, longitude=0, **{list(ds[pm25_var].dims)[0]: 0}).values)
    pm10_sample = float(ds[pm10_var].isel(latitude=0, longitude=0, **{list(ds[pm10_var].dims)[0]: 0}).values)
    
    _, _, conversion_needed = validate_and_convert_pm_data(pm25_sample, pm10_sample)
    print(f"Conversão detectada para dataset: {conversion_needed}")
    
    # Aplicar conversão se necessário
    if "kg/m³" in conversion_needed:
        ds[pm25_var] = ds[pm25_var] * 1e9
        ds[pm10_var] = ds[pm10_var] * 1e9
        print("Aplicada conversão kg/m³ -> μg/m³")
    elif "g/m³" in conversion_needed:
        ds[pm25_var] = ds[pm25_var] * 1e6
        ds[pm10_var] = ds[pm10_var] * 1e6
        print("Aplicada conversão g/m³ -> μg/m³")
    elif "mg/m³" in conversion_needed:
        ds[pm25_var] = ds[pm25_var] * 1e3
        ds[pm10_var] = ds[pm10_var] * 1e3
        print("Aplicada conversão mg/m³ -> μg/m³")
    elif "ng/m³" in conversion_needed:
        ds[pm25_var] = ds[pm25_var] / 1000
        ds[pm10_var] = ds[pm10_var] / 1000
        print("Aplicada conversão ng/m³ -> μg/m³")
    
    for i, (city_name, coords) in enumerate(cities_dict.items()):
        progress = (i + 1) / len(cities_dict)
        progress_bar.progress(progress)
        status_text.text(f"Analisando {city_name}... ({i+1}/{len(cities_dict)})")
        
        lat, lon = coords
        
        try:
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
                    
                    # Log para debug das primeiras cidades
                    if i < 3:
                        print(f"{city_name}: PM2.5={max_pm25:.1f}, PM10={max_pm10:.1f}, IQA={max_aqi:.0f}")
        except Exception as e:
            print(f"Erro ao analisar {city_name}: {e}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if cities_results:
        df_results = pd.DataFrame(cities_results)
        df_results = df_results.sort_values('aqi_max', ascending=False).reset_index(drop=True)
        
        df_results['pm25_max'] = df_results['pm25_max'].round(1)
        df_results['pm10_max'] = df_results['pm10_max'].round(1)
        df_results['aqi_max'] = df_results['aqi_max'].round(0)
        df_results['data_max'] = df_results['data_max'].dt.strftime('%d/%m/%Y %H:%M')
        
        # Log estatísticas gerais
        print(f"Análise completa: {len(df_results)} cidades")
        print(f"IQA range: {df_results['aqi_max'].min():.0f} - {df_results['aqi_max'].max():.0f}")
        print(f"Cidades com IQA > 100: {len(df_results[df_results['aqi_max'] > 100])}")
        
        return df_results
    else:
        print("Nenhuma cidade analisada com sucesso")
        return pd.DataFrame(columns=['cidade', 'pm25_max', 'pm10_max', 'aqi_max', 'data_max', 'categoria'])
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

# Função corrigida para estilizar linhas baseado no IQA
def get_aqi_color(aqi_value):
    """Retorna cor CSS baseada no valor do IQA"""
    if aqi_value <= 50:
        return '#00e400'  # Verde (Boa)
    elif aqi_value <= 100:
        return '#ffff00'  # Amarelo (Moderada)
    elif aqi_value <= 150:
        return '#ff7e00'  # Laranja (Insalubre para Sensíveis)
    elif aqi_value <= 200:
        return '#ff0000'  # Vermelho (Insalubre)
    elif aqi_value <= 300:
        return '#8f3f97'  # Roxo (Muito Insalubre)
    else:
        return '#7e0023'  # Vinho (Perigosa)

def style_aqi_table(df):
    """Aplica estilos condicionais à tabela baseado no IQA"""
    def apply_styles(row):
        aqi = row['IQA Máx']
        bg_color = get_aqi_color(aqi)
        text_color = 'white' if aqi > 100 else 'black'
        return [f'background-color: {bg_color}; color: {text_color}'] * len(row)
    
    return df.style.apply(apply_styles, axis=1)

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
    
    return gpd.GeoDataFrame(municipalities_data, crs="EPSG:4326")

# Títulos e introdução
st.title("🌍 Monitoramento PM2.5 e PM10 - Mato Grosso do Sul")
st.markdown("""
### Sistema Integrado de Monitoramento da Qualidade do Ar

Este aplicativo monitora diretamente as concentrações de Material Particulado (PM2.5 e PM10) 
para todos os municípios de Mato Grosso do Sul usando dados diretos do CAMS.

**Características desta versão:**
- 📊 Dados diretos de PM2.5 e PM10 do CAMS (sem conversão de AOD)
- 🎯 Visualização centralizada no município selecionado com contornos municipais
- 🎬 Animações temporais para PM2.5 e PM10
- 📈 Índice de Qualidade do Ar (IQA) calculado
- 🔮 Previsões para os próximos 5 dias
""")

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
        
        # Criar animação para PM2.5
        with st.spinner('🎬 Criando animação de PM2.5...'):
            animation_result = create_pm_animation(
                ds, pm25_var, city, lat_center, lon_center, ms_shapes, start_date, "PM2.5"
            )
            
            if animation_result is None:
                return None
                
            fig_pm25, animate_pm25, frames_pm25 = animation_result
            
            # Criar animação PM2.5
            ani_pm25 = animation.FuncAnimation(fig_pm25, animate_pm25, frames=frames_pm25, 
                                             interval=animation_speed, blit=True)
            
            # Salvar animação PM2.5
            gif_filename_pm25 = f'PM25_{city}_{start_date}_to_{end_date}.gif'
            ani_pm25.save(gif_filename_pm25, writer=animation.PillowWriter(fps=2))
            plt.close(fig_pm25)
        
        # Criar animação para PM10
        gif_filename_pm10 = None
        if show_pm10_animation:
            with st.spinner('🎬 Criando animação de PM10...'):
                animation_result_pm10 = create_pm_animation(
                    ds, pm10_var, city, lat_center, lon_center, ms_shapes, start_date, "PM10"
                )
                
                if animation_result_pm10 is not None:
                    fig_pm10, animate_pm10, frames_pm10 = animation_result_pm10
                    
                    # Criar animação PM10
                    ani_pm10 = animation.FuncAnimation(fig_pm10, animate_pm10, frames=frames_pm10, 
                                                     interval=animation_speed, blit=True)
                    
                    # Salvar animação PM10
                    gif_filename_pm10 = f'PM10_{city}_{start_date}_to_{end_date}.gif'
                    ani_pm10.save(gif_filename_pm10, writer=animation.PillowWriter(fps=2))
                    plt.close(fig_pm10)

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
            'animation_pm25': gif_filename_pm25,
            'animation_pm10': gif_filename_pm10,
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
    finally:
        # Limpar arquivo temporário
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                pass

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
    show_pm10_animation = st.checkbox("Gerar animação também para PM10", value=True)

# Informações sobre dados diretos
st.sidebar.info("📊 **Dados Diretos CAMS**\nEste sistema utiliza concentrações de PM2.5 e PM10 medidas diretamente pelos sensores do CAMS, com contornos municipais destacados.")

# Botão principal
st.markdown("### 🚀 Iniciar Análise Completa")
st.markdown(f"Clique no botão abaixo para gerar análise de PM2.5 e PM10 centralizada em **{city}** com contornos municipais.")

if st.button("🎯 Gerar Análise de Qualidade do Ar", type="primary", use_container_width=True):
    try:
        results = generate_pm_analysis()
        
        if results:
            # Criar abas
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Análise do Município", 
                "⚠️ Alerta de Qualidade do Ar", 
                f"🎬 Animações - {city}",
                "📈 Análise Detalhada PM"
            ])
            
            # Aba das Animações
            with tab3:
                st.subheader(f"🎬 Animações Temporais - {city}")
                
                # Animação PM2.5
                st.markdown("### 📊 Evolução Temporal - PM2.5")
                if os.path.exists(results['animation_pm25']):
                    st.image(results['animation_pm25'], 
                            caption=f"Evolução temporal do PM2.5 em {city} com contornos municipais destacados ({start_date} a {end_date})")
                    
                    # Botão de download PM2.5
                    with open(results['animation_pm25'], "rb") as file:
                        st.download_button(
                            label="⬇️ Baixar Animação PM2.5 (GIF)",
                            data=file,
                            file_name=f"PM25_{city}_{start_date}_to_{end_date}.gif",
                            mime="image/gif"
                        )
                
                # Animação PM10 (se solicitada)
                if results['animation_pm10'] and os.path.exists(results['animation_pm10']):
                    st.markdown("### 📊 Evolução Temporal - PM10")
                    st.image(results['animation_pm10'], 
                            caption=f"Evolução temporal do PM10 em {city} com contornos municipais destacados ({start_date} a {end_date})")
                    
                    # Botão de download PM10
                    with open(results['animation_pm10'], "rb") as file:
                        st.download_button(
                            label="⬇️ Baixar Animação PM10 (GIF)",
                            data=file,
                            file_name=f"PM10_{city}_{start_date}_to_{end_date}.gif",
                            mime="image/gif"
                        )
                
                # Informações interpretativas
                st.info(f"""
                **Como interpretar as animações de {city}:**
                
                **Elementos do Mapa:**
                - 🔴 **Ponto vermelho**: Localização exata de {city}
                - 🔲 **Contorno vermelho espesso**: Limites do município de {city}
                - ⚫ **Linhas pretas finas**: Contornos de todos os municípios de MS
                - 🌡️ **Cores**: Intensidade das concentrações de material particulado
                
                **Escala de Cores PM2.5 (Vermelho-Amarelo):**
                - 🟢 Verde/Azul: < 12 μg/m³ (Boa qualidade)
                - 🟡 Amarelo: 12-25 μg/m³ (Moderada)
                - 🟠 Laranja: 25-35 μg/m³ (Limite OMS excedido)
                - 🔴 Vermelho: > 35 μg/m³ (Insalubre)
                
                **Escala de Cores PM10 (Tons de Laranja):**
                - 🟢 Verde claro: < 25 μg/m³ (Boa qualidade)
                - 🟡 Amarelo claro: 25-50 μg/m³ (Moderada)
                - 🟠 Laranja: 50-150 μg/m³ (Limite OMS excedido)
                - 🔴 Vermelho escuro: > 150 μg/m³ (Insalubre)
                
                📍 O município selecionado está destacado para facilitar o acompanhamento da evolução temporal.
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
                    if not hist_data.empty:
                        ax1.plot(hist_data['time'], hist_data['pm25'], 
                               'o-', color='darkblue', label='PM2.5 Observado', markersize=6)
                    if not forecast_data.empty:
                        ax1.plot(forecast_data['time'], forecast_data['pm25'], 
                               'x--', color='red', label='PM2.5 Previsto', markersize=6)
                    ax1.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Limite OMS (25 μg/m³)')
                    ax1.axhline(y=35, color='red', linestyle='--', alpha=0.7, label='Limite EPA (35 μg/m³)')
                    ax1.set_ylabel('PM2.5 (μg/m³)', fontsize=12)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    ax1.set_title('Material Particulado PM2.5', fontsize=14)
                    
                    # Gráfico PM10
                    if not hist_data.empty:
                        ax2.plot(hist_data['time'], hist_data['pm10'], 
                               'o-', color='brown', label='PM10 Observado', markersize=6)
                    if not forecast_data.empty:
                        ax2.plot(forecast_data['time'], forecast_data['pm10'], 
                               'x--', color='darkred', label='PM10 Previsto', markersize=6)
                    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Limite OMS (50 μg/m³)')
                    ax2.axhline(y=150, color='red', linestyle='--', alpha=0.7, label='Limite EPA (150 μg/m³)')
                    ax2.set_ylabel('PM10 (μg/m³)', fontsize=12)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.set_title('Material Particulado PM10', fontsize=14)
                    
                    # Gráfico IQA
                    if not hist_data.empty:
                        ax3.plot(hist_data['time'], hist_data['aqi'], 
                               'o-', color='purple', label='IQA Observado', markersize=6)
                    if not forecast_data.empty:
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
                                aqi_color = get_aqi_color(row['aqi'])
                                text_color = 'white' if row['aqi'] > 100 else 'black'
                                st.markdown(f"""
                                <div style="padding:5px; border-radius:5px; background-color:{aqi_color}; 
                                color:{text_color}; margin:2px 0;">
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
                    
                    # Renomear colunas para exibição
                    top_cities_display = top_cities.rename(columns={
                        'cidade': 'Município',
                        'pm25_max': 'PM2.5 Máx (μg/m³)',
                        'pm10_max': 'PM10 Máx (μg/m³)',
                        'aqi_max': 'IQA Máx',
                        'data_max': 'Data Crítica',
                        'categoria': 'Categoria'
                    })
                    
                    # Exibir tabela com estilos corrigidos
                    st.dataframe(
                        style_aqi_table(top_cities_display),
                        use_container_width=True
                    )
                    
                    # Gráfico dos 10 municípios mais críticos
                    st.subheader("📊 Material Particulado - 10 Municípios Mais Críticos")
                    
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
            
            # Aba de análise detalhada
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
                            
                            # Análise de correlação PM2.5 vs PM10
                            if len(hist_data) > 2:
                                corr_pm = hist_data['pm25'].corr(hist_data['pm10'])
                                
                                st.subheader("🔍 Correlações")
                                st.metric("PM2.5 vs PM10", f"{corr_pm:.3f}")
                                
                                # Tendência temporal
                                hist_data_copy = hist_data.copy()
                                hist_data_copy['time_numeric'] = (hist_data_copy['time'] - hist_data_copy['time'].min()).dt.total_seconds()
                                
                                slope_pm25, _, r_pm25, _, _ = stats.linregress(hist_data_copy['time_numeric'], hist_data_copy['pm25'])
                                slope_pm10, _, r_pm10, _, _ = stats.linregress(hist_data_copy['time_numeric'], hist_data_copy['pm10'])
                                
                                st.subheader("📈 Tendências Temporais")
                                trend_pm25 = "↗️ Crescente" if slope_pm25 > 0 else "↘️ Decrescente" if slope_pm25 < 0 else "➡️ Estável"
                                trend_pm10 = "↗️ Crescente" if slope_pm10 > 0 else "↘️ Decrescente" if slope_pm10 < 0 else "➡️ Estável"
                                
                                st.write(f"**PM2.5**: {trend_pm25} (R² = {r_pm25**2:.3f})")
                                st.write(f"**PM10**: {trend_pm10} (R² = {r_pm10**2:.3f})")
                        
                        with col2:
                            st.subheader("🎯 Distribuição dos Valores")
                            
                            # Histogramas
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
                            
                            # Histograma PM2.5
                            ax1.hist(hist_data['pm25'], bins=15, alpha=0.7, color='darkblue', edgecolor='black')
                            ax1.axvline(hist_data['pm25'].mean(), color='red', linestyle='--', label=f'Média: {hist_data["pm25"].mean():.1f}')
                            ax1.axvline(25, color='orange', linestyle=':', label='Limite OMS: 25 μg/m³')
                            ax1.set_xlabel('PM2.5 (μg/m³)')
                            ax1.set_ylabel('Frequência')
                            ax1.set_title('Distribuição PM2.5')
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                            
                            # Histograma PM10
                            ax2.hist(hist_data['pm10'], bins=15, alpha=0.7, color='brown', edgecolor='black')
                            ax2.axvline(hist_data['pm10'].mean(), color='red', linestyle='--', label=f'Média: {hist_data["pm10"].mean():.1f}')
                            ax2.axvline(50, color='orange', linestyle=':', label='Limite OMS: 50 μg/m³')
                            ax2.set_xlabel('PM10 (μg/m³)')
                            ax2.set_ylabel('Frequência')
                            ax2.set_title('Distribuição PM10')
                            ax2.legend()
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Gráfico de correlação PM2.5 vs PM10
                    if not hist_data.empty and len(hist_data) > 2:
                        st.subheader("📈 Correlação PM2.5 vs PM10")
                        
                        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                        
                        # Scatter plot
                        ax.scatter(hist_data['pm25'], hist_data['pm10'], alpha=0.6, color='purple')
                        
                        # Linha de tendência
                        z = np.polyfit(hist_data['pm25'], hist_data['pm10'], 1)
                        p = np.poly1d(z)
                        ax.plot(hist_data['pm25'], p(hist_data['pm25']), "r--", alpha=0.8)
                        
                        ax.set_xlabel('PM2.5 (μg/m³)')
                        ax.set_ylabel('PM10 (μg/m³)')
                        ax.set_title(f'Correlação PM2.5 vs PM10\nR² = {hist_data["pm25"].corr(hist_data["pm10"])**2:.3f}')
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
                else:
                    st.warning("Dados insuficientes para análise detalhada de PM.")
    
    except Exception as e:
        st.error(f"❌ Erro durante a análise: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Rodapé informativo
st.markdown("---")
st.markdown("""
### ℹ️ Informações Importantes

**Sobre os Dados Diretos:**
- As concentrações de PM2.5/PM10 são obtidas diretamente do CAMS, sem conversões
- Dados calibrados e validados continuamente com estações de monitoramento
- Precisão superior aos métodos de conversão de AOD

**Novidades desta versão:**
- 🗺️ Contornos municipais destacados nas animações
- 🎯 Município selecionado evidenciado em vermelho
- 🎬 Animações separadas para PM2.5 e PM10
- 📊 Visualização focada no estado de Mato Grosso do Sul

**Dados Fornecidos por:**
- CAMS (Copernicus Atmosphere Monitoring Service) - União Europeia
- Processamento: Sistema desenvolvido para monitoramento ambiental de MS

**Desenvolvido para:** Monitoramento da Qualidade do Ar em Mato Grosso do Sul
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
    - Contornos: Municípios de MS com destaque do selecionado
    
    **Vantagens dos Dados Diretos:**
    - Eliminação de incertezas de conversão AOD→PM
    - Calibração contínua com estações de superfície
    - Maior precisão para tomada de decisões
    - Validação internacional
    
    **Melhorias na Visualização:**
    - Contornos municipais para referência geográfica
    - Destaque do município selecionado
    - Animações separadas para cada poluente
    - Escalas de cores otimizadas
    
    **Para Melhor Precisão:**
    - Use dados de múltiplos pontos temporais
    - Considere condições meteorológicas locais
    - Valide com medições locais quando disponível
    - Monitore tendências de longo prazo
    """)
