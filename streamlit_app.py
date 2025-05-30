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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Configuração inicial da página
st.set_page_config(layout="wide", page_title="Visualizador de AOD - MS")

# ✅ Carregar autenticação a partir do secrets.toml
try:
    ads_url = st.secrets["ads"]["url"]
    ads_key = st.secrets["ads"]["key"]
    client = cdsapi.Client(url=ads_url, key=ads_key)
except Exception as e:
    st.error("❌ Erro ao carregar as credenciais do CDS API. Verifique seu secrets.toml.")
    st.stop()

# Função para baixar shapefile dos municípios de MS (simplificado)
@st.cache_data
def load_ms_municipalities():
    try:
        # URL para um shapefile de municípios do MS (substitua pelo URL correto)
        # Este é um exemplo - você precisará de um URL real para os dados
        url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/" + \
              "municipio_2022/UFs/MS/MS_Municipios_2022.zip"
        
        # Tentativa de carregar os dados
        try:
            gdf = gpd.read_file(url)
            return gdf
        except:
            # Fallback: criar geodataframe simplificado com alguns municípios
            # Isso é apenas para demonstração se não conseguir carregar o shapefile real
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
        # Retornar DataFrame vazio com estrutura esperada
        return gpd.GeoDataFrame(columns=['NM_MUN', 'geometry'], crs="EPSG:4326")

# 🎯 Lista completa dos municípios de MS com coordenadas
# Expandida para incluir todos os 79 municípios do MS
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

# Títulos e introdução
st.title("🌀 Monitoramento e Previsão de AOD (550nm) - Mato Grosso do Sul")
st.markdown("""
Este aplicativo permite visualizar e analisar dados de Profundidade Óptica de Aerossóis (AOD) a 550nm 
para municípios de Mato Grosso do Sul. Os dados são obtidos em tempo real do CAMS (Copernicus Atmosphere 
Monitoring Service).
""")

# Função para extrair valores de AOD para um ponto específico
def extract_point_timeseries(ds, lat, lon, var_name='aod550'):
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
def moving_average_forecast(df, window=3, days=5):
    """Previsão usando média móvel simples."""
    if len(df) < window:
        return pd.DataFrame(columns=['time', 'aod', 'type', 'method'])
    
    # Calcular média móvel dos últimos valores
    last_values = df['aod'].tail(window).mean()
    
    # Gerar pontos futuros
    last_time = df['time'].max()
    future_times = [last_time + timedelta(hours=i*6) for i in range(1, days*4+1)]
    
    # Usar média móvel como previsão (assumindo tendência estável)
    future_aod = [last_values] * len(future_times)
    
    return pd.DataFrame({
        'time': future_times,
        'aod': future_aod,
        'type': 'forecast',
        'method': 'moving_average'
    })
 def exponential_smoothing_forecast(df, alpha=0.3, days=5):
    """Previsão usando suavização exponencial."""
    if len(df) < 2:
        return pd.DataFrame(columns=['time', 'aod', 'type', 'method'])
    
    # Aplicar suavização exponencial
    values = df['aod'].values
    smoothed = [values[0]]
    
    for i in range(1, len(values)):
        smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])
    
    # Usar último valor suavizado como previsão
    last_smoothed = smoothed[-1]
    
    # Gerar pontos futuros
    last_time = df['time'].max()
    future_times = [last_time + timedelta(hours=i*6) for i in range(1, days*4+1)]
    future_aod = [last_smoothed] * len(future_times)
    
    return pd.DataFrame({
        'time': future_times,
        'aod': future_aod,
        'type': 'forecast',
        'method': 'exponential_smoothing'
    })
def polynomial_forecast(df, degree=2, days=5):
    """Previsão usando ajuste polinomial."""
    if len(df) < degree + 1:
        return pd.DataFrame(columns=['time', 'aod', 'type', 'method'])
    
    # Preparar dados
    df_work = df.copy()
    df_work['time_numeric'] = (df_work['time'] - df_work['time'].min()).dt.total_seconds()
    
    # Ajustar polinômio
    coeffs = np.polyfit(df_work['time_numeric'], df_work['aod'], degree)
    poly_func = np.poly1d(coeffs)
    
    # Gerar pontos futuros
    last_time = df['time'].max()
    future_times = [last_time + timedelta(hours=i*6) for i in range(1, days*4+1)]
    
    # Calcular valores futuros
    future_time_numeric = [(t - df['time'].min()).total_seconds() for t in future_times]
    future_aod = poly_func(future_time_numeric)
    
    # Limitar valores negativos
    future_aod = np.maximum(future_aod, 0)
    
    return pd.DataFrame({
        'time': future_times,
        'aod': future_aod,
        'type': 'forecast',
        'method': f'polynomial_deg{degree}'
    })
def seasonal_decomposition_forecast(df, days=5):
    """Previsão baseada em decomposição sazonal simplificada."""
    if len(df) < 8:  # Precisa de pelo menos 8 pontos para detectar padrões
        return pd.DataFrame(columns=['time', 'aod', 'type', 'method'])
    
    # Calcular tendência simples (regressão linear)
    df_work = df.copy()
    df_work['time_numeric'] = (df_work['time'] - df_work['time'].min()).dt.total_seconds()
    
    # Tendência linear
    slope, intercept = np.polyfit(df_work['time_numeric'], df_work['aod'], 1)
    trend = slope * df_work['time_numeric'] + intercept
    
    # Componente sazonal (ciclo diário simplificado)
    df_work['hour'] = df_work['time'].dt.hour
    hourly_means = df_work.groupby('hour')['aod'].mean()
    
    # Gerar pontos futuros
    last_time = df['time'].max()
    future_times = [last_time + timedelta(hours=i*6) for i in range(1, days*4+1)]
    
    future_aod = []
    for future_time in future_times:
        # Calcular tendência futura
        future_time_numeric = (future_time - df['time'].min()).total_seconds()
        future_trend = slope * future_time_numeric + intercept
        
        # Adicionar componente sazonal
        hour = future_time.hour
        if hour in hourly_means.index:
            seasonal_component = hourly_means[hour] - df['aod'].mean()
        else:
            seasonal_component = 0
        
        predicted_value = future_trend + seasonal_component
        future_aod.append(max(0, predicted_value))
    
    return pd.DataFrame({
        'time': future_times,
        'aod': future_aod,
        'type': 'forecast',
        'method': 'seasonal_decomposition'
    })
def random_forest_forecast(df, days=5):
    """Previsão usando Random Forest com features temporais."""
    if len(df) < 5:
        return pd.DataFrame(columns=['time', 'aod', 'type', 'method'])
    
    # Criar features temporais
    df_work = df.copy()
    df_work['hour'] = df_work['time'].dt.hour
    df_work['day_of_year'] = df_work['time'].dt.dayofyear
    df_work['time_numeric'] = (df_work['time'] - df_work['time'].min()).dt.total_seconds()
    
    # Criar features de lag (valores anteriores)
    for lag in [1, 2, 3]:
        df_work[f'aod_lag_{lag}'] = df_work['aod'].shift(lag)
    
    # Remover linhas com NaN
    df_clean = df_work.dropna()
    
    if len(df_clean) < 3:
        return pd.DataFrame(columns=['time', 'aod', 'type', 'method'])
    
    # Preparar dados para treinamento
    feature_cols = ['hour', 'day_of_year', 'time_numeric'] + [f'aod_lag_{i}' for i in [1, 2, 3]]
    X = df_clean[feature_cols].values
    y = df_clean['aod'].values
    
    # Treinar modelo
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)
    
    # Gerar previsões futuras
    last_time = df['time'].max()
    future_times = [last_time + timedelta(hours=i*6) for i in range(1, days*4+1)]
    
    future_aod = []
    last_values = df['aod'].tail(3).values  # Últimos 3 valores para lags
    
    for i, future_time in enumerate(future_times):
        # Criar features para o ponto futuro
        hour = future_time.hour
        day_of_year = future_time.timetuple().tm_yday
        time_numeric = (future_time - df['time'].min()).total_seconds()
        
        # Usar valores anteriores como lags
        if i == 0:
            lags = last_values
        elif i == 1:
            lags = np.append(last_values[1:], [future_aod[0]])
        elif i == 2:
            lags = np.append(last_values[2:], future_aod[:2])
        else:
            lags = future_aod[i-3:i]
        
        # Garantir que temos 3 lags
        if len(lags) < 3:
            lags = np.pad(lags, (3-len(lags), 0), mode='edge')
        
        features = np.array([[hour, day_of_year, time_numeric] + lags[-3:].tolist()])
        prediction = rf.predict(features)[0]
        future_aod.append(max(0, prediction))
    
    return pd.DataFrame({
        'time': future_times,
        'aod': future_aod,
        'type': 'forecast',
        'method': 'random_forest'
    })
def ensemble_forecast(df, days=5):
    """Previsão ensemble combinando múltiplos métodos."""
    methods = [
        moving_average_forecast,
        exponential_smoothing_forecast,
        lambda x, d: polynomial_forecast(x, degree=2, days=d),
        seasonal_decomposition_forecast,
        random_forest_forecast
    ]
    
    forecasts = []
    weights = [0.15, 0.2, 0.2, 0.25, 0.2]  # Pesos para cada método
    
    # Gerar previsões de todos os métodos
    for method in methods:
        try:
            forecast = method(df, days)
            if not forecast.empty:
                forecasts.append(forecast['aod'].values)
        except:
            continue
    
    if not forecasts:
        return pd.DataFrame(columns=['time', 'aod', 'type', 'method'])
    
    # Calcular média ponderada
    forecasts_array = np.array(forecasts)
    if len(forecasts) != len(weights):
        # Usar pesos iguais se número de métodos for diferente
        weights = [1/len(forecasts)] * len(forecasts)
    
    ensemble_aod = np.average(forecasts_array, axis=0, weights=weights[:len(forecasts)])
    
    # Gerar pontos futuros
    last_time = df['time'].max()
    future_times = [last_time + timedelta(hours=i*6) for i in range(1, days*4+1)]
    
    return pd.DataFrame({
        'time': future_times,
        'aod': ensemble_aod,
        'type': 'forecast',
        'method': 'ensemble'
    })
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
        df = pd.DataFrame({'time': times, 'aod': values})
        df = df.sort_values('time').reset_index(drop=True)
        return df
    else:
        return pd.DataFrame(columns=['time', 'aod'])

# Função para prever valores futuros de AOD
def predict_future_aod_advanced(df, method='ensemble', days=5):
    """Gera previsão usando o método especificado."""
    if len(df) < 2:
        return pd.DataFrame(columns=['time', 'aod', 'type', 'method'])
    
    # Adicionar dados históricos
    df_hist = df.copy()
    df_hist['type'] = 'historical'
    df_hist['method'] = 'observed'
    
    # Escolher método de previsão
    if method == 'linear_regression':
        # Regressão linear (método original)
        df_hist['time_numeric'] = (df_hist['time'] - df_hist['time'].min()).dt.total_seconds()
        X = df_hist['time_numeric'].values.reshape(-1, 1)
        y = df_hist['aod'].values
        model = LinearRegression()
        model.fit(X, y)
        
        last_time = df_hist['time'].max()
        future_times = [last_time + timedelta(hours=i*6) for i in range(1, days*4+1)]
        future_time_numeric = [(t - df_hist['time'].min()).total_seconds() for t in future_times]
        future_aod = model.predict(np.array(future_time_numeric).reshape(-1, 1))
        future_aod = np.maximum(future_aod, 0)
        
        df_pred = pd.DataFrame({
            'time': future_times,
            'aod': future_aod,
            'type': 'forecast',
            'method': 'linear_regression'
        })
    
    elif method == 'moving_average':
        df_pred = moving_average_forecast(df, days=days)
    elif method == 'exponential_smoothing':
        df_pred = exponential_smoothing_forecast(df, days=days)
    elif method == 'polynomial':
        df_pred = polynomial_forecast(df, degree=2, days=days)
    elif method == 'seasonal':
        df_pred = seasonal_decomposition_forecast(df, days=days)
    elif method == 'random_forest':
        df_pred = random_forest_forecast(df, days=days)
    elif method == 'ensemble':
        df_pred = ensemble_forecast(df, days=days)
    else:
        # Default para ensemble
        df_pred = ensemble_forecast(df, days=days)
    
    # Combinar histórico e previsão
    result = pd.concat([df_hist[['time', 'aod', 'type', 'method']], df_pred], ignore_index=True)
    return result
def compare_forecast_methods(df, days=5):
    """Compara diferentes métodos de previsão."""
    methods = {
        'Regressão Linear': 'linear_regression',
        'Média Móvel': 'moving_average',
        'Suavização Exponencial': 'exponential_smoothing',
        'Polinomial': 'polynomial',
        'Decomposição Sazonal': 'seasonal',
        'Random Forest': 'random_forest',
        'Ensemble': 'ensemble'
    }
    
    forecasts = {}
    
    for name, method in methods.items():
        try:
            forecast = predict_future_aod_advanced(df, method=method, days=days)
            forecasts[name] = forecast
        except Exception as e:
            st.warning(f"Erro no método {name}: {str(e)}")
            continue
    
    return forecasts

# NOVA FUNÇÃO: Analisar AOD para todas as cidades e gerar tabela de alertas
def analyze_all_cities(ds, aod_var, cities_dict):
    """Analisa os valores de AOD para todas as cidades e retorna as 20 mais críticas."""
    cities_results = []
    
    # Para cada cidade, extrair série temporal e determinar valor máximo previsto
    with st.spinner(f"Analisando AOD para todos os municípios de MS... (0/{len(cities_dict)})"):
        for i, (city_name, coords) in enumerate(cities_dict.items()):
            # Atualize o spinner a cada 10 cidades para não sobrecarregar a interface
            if i % 10 == 0:
                st.spinner(f"Analisando AOD para todos os municípios de MS... ({i}/{len(cities_dict)})")
            
            lat, lon = coords
            
            # Extrair série temporal para a cidade
            df_timeseries = extract_point_timeseries(ds, lat, lon, var_name=aod_var)
            
            if not df_timeseries.empty:
                # Gerar previsão
                df_forecast = predict_future_aod(df_timeseries, days=5)
                
                # Filtrar apenas dados de previsão
                forecast_only = df_forecast[df_forecast['type'] == 'forecast']
                
                if not forecast_only.empty:
                    # Obter valor máximo previsto e quando ocorrerá
                    max_aod = forecast_only['aod'].max()
                    max_day = forecast_only.loc[forecast_only['aod'].idxmax(), 'time']
                    
                    # Categorizar nível de poluição
                    pollution_level = "Baixo"
                    if max_aod >= 0.5:
                        pollution_level = "Muito Alto"
                    elif max_aod >= 0.2:
                        pollution_level = "Alto"
                    elif max_aod >= 0.1:
                        pollution_level = "Moderado"
                    
                    # Adicionar resultado à lista
                    cities_results.append({
                        'cidade': city_name,
                        'aod_max': max_aod,
                        'data_max': max_day,
                        'nivel': pollution_level
                    })
    
    # Criar DataFrame com os resultados
    if cities_results:
        df_results = pd.DataFrame(cities_results)
        
        # Ordenar por AOD máximo (decrescente)
        df_results = df_results.sort_values('aod_max', ascending=False).reset_index(drop=True)
        
        # Formatar o DataFrame para exibição
        df_results['aod_max'] = df_results['aod_max'].round(3)
        df_results['data_max'] = df_results['data_max'].dt.strftime('%d/%m/%Y %H:%M')
        
        return df_results
    else:
        return pd.DataFrame(columns=['cidade', 'aod_max', 'data_max', 'nivel'])

# Função principal para gerar análise de AOD
def generate_aod_analysis():
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
        'variable': ['total_aerosol_optical_depth_550nm'],
        'date': f'{start_date_str}/{end_date_str}',
        'time': hours,
        'leadtime_hour': ['0', '24', '48', '72', '96', '120'],  # Incluir previsões de até 5 dias
        'type': ['forecast'],
        'format': 'netcdf',
        'area': [lat_center + map_width/2, lon_center - map_width/2, 
                lat_center - map_width/2, lon_center + map_width/2]
    }
    
    filename = f'AOD550_{city}_{start_date}_to_{end_date}.nc'
    
    try:
        with st.spinner('📥 Baixando dados do CAMS...'):
            client.retrieve(dataset, request).download(filename)
        
        ds = xr.open_dataset(filename)
        
        # Verificar variáveis disponíveis
        variable_names = list(ds.data_vars)
        st.write(f"Variáveis disponíveis: {variable_names}")
        
        # Usar a variável 'aod550' encontrada nos dados
        aod_var = next((var for var in variable_names if 'aod' in var.lower()), variable_names[0])
        
        st.write(f"Usando variável: {aod_var}")
        da = ds[aod_var]
        
        # Verificar dimensões
        st.write(f"Dimensões: {da.dims}")
        
        # Identificar dimensões temporais
        time_dims = [dim for dim in da.dims if 'time' in dim or 'forecast' in dim]
        
        if not time_dims:
            st.error("Não foi possível identificar dimensão temporal nos dados.")
            return None
        
        # Extrair série temporal para o ponto central (cidade selecionada)
        with st.spinner("Extraindo série temporal para o município..."):
            df_timeseries = extract_point_timeseries(ds, lat_center, lon_center, var_name=aod_var)
        
        if df_timeseries.empty:
            st.error("Não foi possível extrair série temporal para este local.")
            return None
        
        # Gerar previsão para os próximos dias
        with st.spinner("Gerando previsão de AOD..."):
            df_forecast = predict_future_aod(df_timeseries, days=5)  # Aumentado para 5 dias
        
        # Encontrar o município no geodataframe
        municipality_shape = None
        if not ms_shapes.empty:
            city_shape = ms_shapes[ms_shapes['NM_MUN'] == city]
            if not city_shape.empty:
                municipality_shape = city_shape.iloc[0].geometry
        
        # --- Criação da animação ---
        # Identificar frames disponíveis
        if 'forecast_reference_time' in da.dims:
            time_dim = 'forecast_reference_time'
            frames = len(da[time_dim])
        else:
            time_dim = time_dims[0]
            frames = len(da[time_dim])
        
        st.write(f"✅ Total de frames disponíveis: {frames}")
        
        if frames < 1:
            st.error("Erro: Dados insuficientes para animação.")
            return None
        
        # Determinar range de cores
        vmin, vmax = float(da.min().values), float(da.max().values)
        vmin = max(0, vmin - 0.05)
        vmax = min(2, vmax + 0.05)  # AOD geralmente não ultrapassa 2
        
        # Criar figura
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Adicionar features básicas
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
            first_frame_data = da.isel({time_dim: 0}).values
            first_frame_time = pd.to_datetime(da[time_dim].values[0])
        
        # Garantir formato 2D
        if len(first_frame_data.shape) != 2:
            st.error(f"Erro: Formato de dados inesperado. Shape: {first_frame_data.shape}")
            return None
        
        # Criar mapa de cores
        im = ax.pcolormesh(ds.longitude, ds.latitude, first_frame_data, 
                         cmap=colormap, vmin=vmin, vmax=vmax)
        
        # Adicionar barra de cores
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('AOD 550nm')
        
        # Adicionar título inicial
        title = ax.set_title(f'AOD 550nm em {city} - {first_frame_time}', fontsize=14)
        
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
                    t_idx = min(i, len(da[time_dim])-1)
                    frame_data = da.isel({time_dim: t_idx}).values
                    frame_time = pd.to_datetime(da[time_dim].values[t_idx])
                
                # Atualizar dados
                im.set_array(frame_data.ravel())
                
                # Atualizar título com timestamp
                title.set_text(f'AOD 550nm em {city} - {frame_time}')
                
                return [im, title]
            except Exception as e:
                st.error(f"Erro no frame {i}: {str(e)}")
                return [im, title]
        
        # Limitar número de frames para evitar problemas
        actual_frames = min(frames, 20)  # Máximo de 20 frames
        
        # Criar animação
        ani = animation.FuncAnimation(fig, animate, frames=actual_frames, 
                                     interval=animation_speed, blit=True)
        
        # Salvar animação
        gif_filename = f'AOD550_{city}_{start_date}_to_{end_date}.gif'
        
        with st.spinner('💾 Salvando animação...'):
            ani.save(gif_filename, writer=animation.PillowWriter(fps=2))
        
        plt.close(fig)

        # NOVO: Analisar dados para todas as cidades do MS
        top_pollution_cities = None
        with st.spinner("🔍 Analisando todas as cidades do MS para alerta de poluição..."):
            top_pollution_cities = analyze_all_cities(ds, aod_var, cities)
        
        return {
            'animation': gif_filename,
            'timeseries': df_timeseries,
            'forecast': df_forecast,
            'dataset': ds,
            'variable': aod_var,
            'top_pollution': top_pollution_cities  # Novo item no dicionário de resultados
        }
    
    except Exception as e:
        st.error(f"❌ Erro ao processar os dados: {str(e)}")
        st.write("Detalhes da requisição:")
        st.write(request)
        return None

# Carregar shapefiles dos municípios do MS
with st.spinner("Carregando shapes dos municípios..."):
    ms_shapes = load_ms_municipalities()

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Seleção de cidade com os shapes disponíveis
available_cities = sorted(list(set(ms_shapes['NM_MUN'].tolist()).intersection(set(cities.keys()))))
if not available_cities:
    available_cities = list(cities.keys())  # Fallback para a lista original

city = st.sidebar.selectbox("Selecione o município", available_cities)
lat_center, lon_center = cities[city]

# Adicionar seleção do método de previsão na sidebar
st.sidebar.header("🔮 Método de Previsão")
forecast_method = st.sidebar.selectbox(
    "Escolha o método:",
    ['ensemble', 'linear_regression', 'moving_average', 'exponential_smoothing', 
     'polynomial', 'seasonal', 'random_forest'],
    format_func=lambda x: {
        'ensemble': '🎯 Ensemble (Recomendado)',
        'linear_regression': '📈 Regressão Linear',
        'moving_average': '📊 Média Móvel',
        'exponential_smoothing': '🌊 Suavização Exponencial',
        'polynomial': '📐 Ajuste Polinomial',
        'seasonal': '🔄 Decomposição Sazonal',
        'random_forest': '🌳 Random Forest'
    }[x]
)

# Adicionar opção para comparar métodos
compare_methods = st.sidebar.checkbox("🔬 Comparar todos os métodos", value=False)

# Configurações de data e hora
st.sidebar.subheader("Período de Análise")
start_date = st.sidebar.date_input("Data de Início", datetime.today() - timedelta(days=2))
end_date = st.sidebar.date_input("Data Final", datetime.today() + timedelta(days=5))  # Estendido para 5 dias

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

# Agora, vamos adicionar o botão logo após o texto introdutório
st.markdown("### 🚀 Iniciar Análise de AOD")
st.markdown("Clique no botão abaixo para gerar análise completa de AOD para todos os municípios de MS.")

# Botão para iniciar análise
if st.button("🎞️ Gerar Análise Completa", type="primary", use_container_width=True):
    try:
        # Executar análise e obter resultados
        results = generate_aod_analysis()
        
        if results:
            # Layout com abas para diferentes visualizações
            tab1, tab2, tab3 = st.tabs(["📊 Análise do Município", "⚠️ Alerta de Poluição para MS", "🗺️ Mapa e Animação"])
            
            with tab3:
                st.subheader("🎬 Animação de AOD 550nm")
                st.image(results['animation'], caption=f"AOD 550nm em {city} ({start_date} a {end_date})")
                
                # Adicionar opções para baixar
                with open(results['animation'], "rb") as file:
                    btn = st.download_button(
                        label="⬇️ Baixar Animação (GIF)",
                        data=file,
                        file_name=f"AOD_{city}_{start_date}_to_{end_date}.gif",
                        mime="image/gif"
                    )
            
            with tab1:
                st.subheader("📊 Série Temporal e Previsão")
                
                # Layout de duas colunas
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Preparar dados para gráfico
                    df_combined = results['forecast']
                    
                    # Criar gráfico
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Dados históricos
                    hist_data = df_combined[df_combined['type'] == 'historical']
                    ax.plot(hist_data['time'], hist_data['aod'], 
                           marker='o', linestyle='-', color='blue', label='Observado')
                    
                    # Dados de previsão
                    forecast_data = df_combined[df_combined['type'] == 'forecast']
                    ax.plot(forecast_data['time'], forecast_data['aod'], 
                           marker='x', linestyle='--', color='red', label='Previsão')
                    
                    # Formatar eixos
                    ax.set_title(f'AOD 550nm em {city}: Valores Observados e Previstos', fontsize=14)
                    ax.set_xlabel('Data/Hora', fontsize=12)
                    ax.set_ylabel('AOD 550nm', fontsize=12)
                    
                    # Formatar datas no eixo x
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
                    plt.xticks(rotation=45)
                    
                    # Adicionar legenda e grade
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Adicionar faixa de qualidade do ar
                    ax.axhspan(0, 0.1, alpha=0.2, color='green', label='Boa')
                    ax.axhspan(0.1, 0.2, alpha=0.2, color='yellow', label='Moderada')
                    ax.axhspan(0.2, 0.5, alpha=0.2, color='orange', label='Insalubre')
                    ax.axhspan(0.5, 2.0, alpha=0.2, color='red', label='Perigosa')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    # Estatísticas
                    st.subheader("📈 Estatísticas de AOD")
                    
                    # Calcular estatísticas
                    if not hist_data.empty:
                        curr_aod = hist_data['aod'].iloc[-1]
                        max_aod = hist_data['aod'].max()
                        mean_aod = hist_data['aod'].mean()
                        
                        # Categorizar qualidade do ar baseado no AOD
                        def aod_category(value):
                            if value < 0.1:
                                return "Boa", "green"
                            elif value < 0.2:
                                return "Moderada", "orange"
                            elif value < 0.5:
                                return "Insalubre para grupos sensíveis", "red"
                            else:
                                return "Perigosa", "darkred"
                        
                        current_cat, current_color = aod_category(curr_aod)
                        
                        # Mostrar métricas
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("AOD Atual", f"{curr_aod:.3f}")
                        col_b.metric("AOD Máximo", f"{max_aod:.3f}")
                        col_c.metric("AOD Médio", f"{mean_aod:.3f}")
                        
                        # Mostrar categoria da qualidade do ar
                        st.markdown(f"""
                        <div style="padding:10px; border-radius:5px; background-color:{current_color}; color:white; text-align:center; margin:10px 0;">
                        <h3 style="margin:0;">Qualidade do Ar: {current_cat}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Previsão para os próximos dias
                        if not forecast_data.empty:
                            st.subheader("🔮 Previsão para os próximos dias")
                            
                            # Agrupar por dia
                            forecast_data['date'] = forecast_data['time'].dt.date
                            daily_forecast = forecast_data.groupby('date')['aod'].mean().reset_index()
                            
                            for i, row in daily_forecast.iterrows():
                                day_cat, day_color = aod_category(row['aod'])
                                st.markdown(f"""
                                <div style="padding:5px; border-radius:3px; background-color:{day_color}; color:white; margin:2px 0;">
                                <b>{row['date'].strftime('%d/%m/%Y')}:</b> AOD médio previsto: {row['aod']:.3f} - {day_cat}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Exportar dados
                    st.subheader("💾 Exportar Dados")
                    csv = df_combined.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Baixar Dados (CSV)",
                        data=csv,
                        file_name=f"AOD_data_{city}_{start_date}_to_{end_date}.csv",
                        mime="text/csv",
                    )
            
            # NOVA ABA: Alerta de Poluição para MS
            with tab2:
                st.subheader("⚠️ Alerta de Poluição para Municípios de MS")
                
                # Verificar se temos os dados de todas as cidades
                if 'top_pollution' in results and not results['top_pollution'].empty:
                    top_cities = results['top_pollution'].head(20)  # Pegar as 20 primeiras
                    
                    # Criar uma tabela formatada e colorida com as cidades mais críticas
                    st.markdown("### 🔴 Top 20 Municípios com Maior Previsão de AOD")
                    
                    # Adicionar legenda de cores
                    st.markdown("""
                    <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                        <div style="display: flex; align-items: center;">
                            <div style="width: 20px; height: 20px; background-color: darkred; margin-right: 5px;"></div>
                            <span>AOD ≥ 0.5 (Muito Alto)</span>
                        </div>
                        <div style="display: flex; align-items: center;">
                            <div style="width: 20px; height: 20px; background-color: red; margin-right: 5px;"></div>
                            <span>AOD ≥ 0.2 (Alto)</span>
                        </div>
                        <div style="display: flex; align-items: center;">
                            <div style="width: 20px; height: 20px; background-color: orange; margin-right: 5px;"></div>
                            <span>AOD ≥ 0.1 (Moderado)</span>
                        </div>
                        <div style="display: flex; align-items: center;">
                            <div style="width: 20px; height: 20px; background-color: green; margin-right: 5px;"></div>
                            <span>AOD < 0.1 (Baixo)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Renomear colunas para exibição
                    top_cities_display = top_cities.rename(columns={
                        'cidade': 'Município', 
                        'aod_max': 'AOD Máximo', 
                        'data_max': 'Data do Pico',
                        'nivel': 'Nível de Alerta'
                    })
                    
                    # Função para colorir as linhas baseado no valor de AOD
                    def highlight_aod(val):
                        try:
                            aod = float(val['AOD Máximo'])
                            if aod >= 0.5:
                                return ['background-color: darkred; color: white'] * len(val)
                            elif aod >= 0.2:
                                return ['background-color: red; color: white'] * len(val)
                            elif aod >= 0.1:
                                return ['background-color: orange; color: black'] * len(val)
                            else:
                                return ['background-color: green; color: white'] * len(val)
                        except:
                            return [''] * len(val)
                    
                    # Exibir tabela formatada
                    st.dataframe(
                        top_cities_display.style.apply(highlight_aod, axis=1),
                        use_container_width=True
                    )
                    
                    # Adicionar um aviso se houver cidades com nível alto ou muito alto
                    high_risk_cities = top_cities[top_cities['aod_max'] >= 0.2]
                    
                    if not high_risk_cities.empty:
                        st.warning(f"""
                        ### ⚠️ ALERTA DE POLUIÇÃO ATMOSFÉRICA
                        
                        Detectamos previsão de níveis elevados de AOD (≥ 0.2) para {len(high_risk_cities)} municípios nos próximos 5 dias!
                        
                        Os municípios mais críticos são:
                        - **{high_risk_cities.iloc[0]['cidade']}**: AOD {high_risk_cities.iloc[0]['aod_max']:.3f} em {high_risk_cities.iloc[0]['data_max']}
                        - **{high_risk_cities.iloc[1]['cidade'] if len(high_risk_cities) > 1 else ''}**: AOD {high_risk_cities.iloc[1]['aod_max']:.3f if len(high_risk_cities) > 1 else 0} em {high_risk_cities.iloc[1]['data_max'] if len(high_risk_cities) > 1 else ''}
                        - **{high_risk_cities.iloc[2]['cidade'] if len(high_risk_cities) > 2 else ''}**: AOD {high_risk_cities.iloc[2]['aod_max']:.3f if len(high_risk_cities) > 2 else 0} em {high_risk_cities.iloc[2]['data_max'] if len(high_risk_cities) > 2 else ''}
                        
                        Recomenda-se atenção especial a pessoas com problemas respiratórios nestas localidades.
                        """)
                    
                    # Exportar dados da tabela
                    csv_top_cities = top_cities.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Baixar Tabela de Alerta (CSV)",
                        data=csv_top_cities,
                        file_name=f"AOD_alerta_MS_{start_date}_to_{end_date}.csv",
                        mime="text/csv",
                    )
                    
                    # Criar gráfico de barras com as 10 cidades mais críticas
                    st.subheader("📊 Previsão de AOD Máximo - Top 10 Municípios")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Selecionar top 10
                    top10 = top_cities.head(10)
                    
                    # Criar barras com cores baseadas no nível de AOD
                    colors = []
                    for aod in top10['aod_max']:
                        if aod >= 0.5:
                            colors.append('darkred')
                        elif aod >= 0.2:
                            colors.append('red')
                        elif aod >= 0.1:
                            colors.append('orange')
                        else:
                            colors.append('green')
                    
                    # Plotar gráfico
                    bars = ax.bar(top10['cidade'], top10['aod_max'], color=colors)
                    
                    # Adicionar rótulos
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.3f}', ha='center', va='bottom', 
                                fontsize=10, rotation=0)
                    
                    # Formatação do gráfico
                    ax.set_title('Top 10 Municípios com Maior Previsão de AOD', fontsize=14)
                    ax.set_xlabel('Município', fontsize=12)
                    ax.set_ylabel('AOD Máximo Previsto', fontsize=12)
                    ax.set_ylim(0, max(top10['aod_max']) * 1.2)  # Ajustar limite do eixo Y
                    ax.axhline(y=0.5, linestyle='--', color='darkred', alpha=0.7)
                    ax.axhline(y=0.2, linestyle='--', color='red', alpha=0.7)
                    ax.axhline(y=0.1, linestyle='--', color='orange', alpha=0.7)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                else:
                    st.error("❌ Não foi possível obter dados de previsão para os municípios de MS.")
                    st.info("Tente novamente com um período diferente ou verifique a conexão com a API do CAMS.")
    except Exception as e:
        st.error(f"❌ Ocorreu um erro ao gerar a análise: {str(e)}")
        st.write("Por favor, verifique os parâmetros e tente novamente.")

# Adicionar informações na parte inferior
st.markdown("---")
st.markdown("""
### ℹ️ Sobre os dados
- **Fonte**: Copernicus Atmosphere Monitoring Service (CAMS)
- **Variável**: Profundidade Óptica de Aerossóis (AOD) a 550nm
- **Resolução temporal**: 3 horas
- **Atualização**: Diária
- **Previsão**: Até 5 dias à frente

### 📖 Como interpretar:
- **AOD < 0.1**: Qualidade do ar boa
- **AOD 0.1-0.2**: Qualidade do ar moderada
- **AOD 0.2-0.5**: Insalubre para grupos sensíveis
- **AOD > 0.5**: Qualidade do ar perigosa

### 🔍 Novas funcionalidades:
- **Alerta de Poluição**: Monitoramento automático dos 79 municípios de MS
- **Previsão de 5 dias**: Análise de tendências e picos de AOD
- **Top 20 Municípios**: Identificação das áreas mais críticas

Desenvolvido para monitoramento de aerossóis no estado de Mato Grosso do Sul - Brasil.
""")
