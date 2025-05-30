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
def predict_future_aod_advanced(df, days=5, method='ensemble'):
    """
    Gera previsões de AOD usando múltiplos métodos estatísticos.
    
    Parâmetros:
    - df: DataFrame com colunas 'time' e 'aod'
    - days: número de dias para previsão
    - method: 'linear', 'polynomial', 'arima', 'exponential', 'random_forest', 'ensemble'
    
    Retorna:
    - DataFrame com previsões e métricas de qualidade
    """
    
    if len(df) < 5:
        return pd.DataFrame(columns=['time', 'aod', 'type', 'method', 'confidence_lower', 'confidence_upper'])
    
    # Preparar dados
    df_work = df.copy().sort_values('time').reset_index(drop=True)
    df_work['time_numeric'] = (df_work['time'] - df_work['time'].min()).dt.total_seconds() / 3600  # horas
    
    # Definir pontos futuros
    last_time = df_work['time'].max()
    future_times = [last_time + timedelta(hours=i*6) for i in range(1, days*4+1)]
    future_hours = [(t - df_work['time'].min()).total_seconds() / 3600 for t in future_times]
    
    predictions = {}
    metrics = {}
    
    # 1. REGRESSÃO LINEAR SIMPLES
    def linear_regression():
        X = df_work['time_numeric'].values.reshape(-1, 1)
        y = df_work['aod'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Previsão
        future_X = np.array(future_hours).reshape(-1, 1)
        pred = model.predict(future_X)
        pred = np.maximum(pred, 0)  # AOD não pode ser negativo
        
        # Métricas (usando validação nos últimos pontos)
        if len(y) > 3:
            train_pred = model.predict(X)
            mae = mean_absolute_error(y, train_pred)
            mse = mean_squared_error(y, train_pred)
            r2 = r2_score(y, train_pred)
        else:
            mae = mse = r2 = np.nan
            
        return pred, {'mae': mae, 'mse': mse, 'r2': r2}
    
    # 2. REGRESSÃO POLINOMIAL
    def polynomial_regression(degree=2):
        X = df_work['time_numeric'].values.reshape(-1, 1)
        y = df_work['aod'].values
        
        # Transformação polinomial
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        
        model = Ridge(alpha=0.1)  # Ridge para evitar overfitting
        model.fit(X_poly, y)
        
        # Previsão
        future_X = np.array(future_hours).reshape(-1, 1)
        future_X_poly = poly_features.transform(future_X)
        pred = model.predict(future_X_poly)
        pred = np.maximum(pred, 0)
        
        # Métricas
        if len(y) > 3:
            train_pred = model.predict(X_poly)
            mae = mean_absolute_error(y, train_pred)
            mse = mean_squared_error(y, train_pred)
            r2 = r2_score(y, train_pred)
        else:
            mae = mse = r2 = np.nan
            
        return pred, {'mae': mae, 'mse': mse, 'r2': r2}
    
    # 3. MODELO ARIMA
    def arima_forecast():
        try:
            y = df_work['aod'].values
            
            # Auto-ARIMA simplificado
            model = ARIMA(y, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Previsão
            forecast = fitted_model.forecast(steps=len(future_times))
            pred = np.maximum(forecast, 0)
            
            # Métricas
            residuals = fitted_model.resid
            mae = np.mean(np.abs(residuals))
            mse = np.mean(residuals**2)
            r2 = 1 - (np.var(residuals) / np.var(y))
            
            return pred, {'mae': mae, 'mse': mse, 'r2': r2}
            
        except Exception as e:
            # Fallback para média móvel se ARIMA falhar
            window = min(3, len(df_work))
            mean_value = df_work['aod'].rolling(window=window).mean().iloc[-1]
            pred = np.full(len(future_times), mean_value)
            return pred, {'mae': np.nan, 'mse': np.nan, 'r2': np.nan}
    
    # 4. SUAVIZAÇÃO EXPONENCIAL
    def exponential_smoothing():
        try:
            y = df_work['aod'].values
            
            # Modelo de Holt-Winters simples (sem sazonalidade)
            model = ExponentialSmoothing(y, trend='add', seasonal=None)
            fitted_model = model.fit()
            
            # Previsão
            forecast = fitted_model.forecast(steps=len(future_times))
            pred = np.maximum(forecast, 0)
            
            # Métricas
            fitted_values = fitted_model.fittedvalues
            residuals = y - fitted_values
            mae = np.mean(np.abs(residuals))
            mse = np.mean(residuals**2)
            r2 = 1 - (np.var(residuals) / np.var(y))
            
            return pred, {'mae': mae, 'mse': mse, 'r2': r2}
            
        except Exception as e:
            # Fallback
            trend = np.polyfit(range(len(df_work)), df_work['aod'], 1)[0]
            last_value = df_work['aod'].iloc[-1]
            pred = [last_value + trend * i for i in range(1, len(future_times) + 1)]
            pred = np.maximum(pred, 0)
            return pred, {'mae': np.nan, 'mse': np.nan, 'r2': np.nan}
    
    # 5. RANDOM FOREST
    def random_forest():
        if len(df_work) < 10:  # RF precisa de mais dados
            return linear_regression()
            
        # Features: hora do dia, dia da semana, tendência temporal
        features = []
        for i, row in df_work.iterrows():
            features.append([
                row['time_numeric'],
                row['time'].hour,
                row['time'].weekday(),
                i  # índice sequencial
            ])
        
        X = np.array(features)
        y = df_work['aod'].values
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Previsão
        future_features = []
        for i, t in enumerate(future_times):
            future_features.append([
                future_hours[i],
                t.hour,
                t.weekday(),
                len(df_work) + i
            ])
        
        future_X = np.array(future_features)
        pred = model.predict(future_X)
        pred = np.maximum(pred, 0)
        
        # Métricas
        train_pred = model.predict(X)
        mae = mean_absolute_error(y, train_pred)
        mse = mean_squared_error(y, train_pred)
        r2 = r2_score(y, train_pred)
        
        return pred, {'mae': mae, 'mse': mse, 'r2': r2}
    
    # Executar métodos conforme solicitado
    if method == 'linear' or method == 'ensemble':
        predictions['linear'], metrics['linear'] = linear_regression()
    
    if method == 'polynomial' or method == 'ensemble':
        predictions['polynomial'], metrics['polynomial'] = polynomial_regression()
    
    if method == 'arima' or method == 'ensemble':
        predictions['arima'], metrics['arima'] = arima_forecast()
    
    if method == 'exponential' or method == 'ensemble':
        predictions['exponential'], metrics['exponential'] = exponential_smoothing()
    
    if method == 'random_forest' or method == 'ensemble':
        predictions['random_forest'], metrics['random_forest'] = random_forest()
    
    # Se método específico foi escolhido, retornar apenas ele
    if method != 'ensemble':
        pred_values = predictions[method]
        method_metrics = metrics[method]
        
        # Criar intervalos de confiança simples (± 20% da previsão)
        confidence_range = pred_values * 0.2
        
        df_result = pd.DataFrame({
            'time': future_times,
            'aod': pred_values,
            'type': 'forecast',
            'method': method,
            'confidence_lower': pred_values - confidence_range,
            'confidence_upper': pred_values + confidence_range
        })
        
        # Adicionar dados históricos
        df_hist = df_work[['time', 'aod']].copy()
        df_hist['type'] = 'historical'
        df_hist['method'] = method
        df_hist['confidence_lower'] = df_hist['aod']
        df_hist['confidence_upper'] = df_hist['aod']
        
        return pd.concat([df_hist, df_result], ignore_index=True)
    
    # ENSEMBLE: Combinar previsões com pesos baseados na qualidade
    else:
        # Calcular pesos baseados no R²
        weights = {}
        total_r2 = 0
        
        for pred_method, metric in metrics.items():
            r2_value = metric.get('r2', 0)
            if np.isnan(r2_value) or r2_value < 0:
                r2_value = 0.1  # peso mínimo
            weights[pred_method] = max(r2_value, 0.1)
            total_r2 += weights[pred_method]
        
        # Normalizar pesos
        for pred_method in weights:
            weights[pred_method] /= total_r2
        
        # Combinar previsões
        ensemble_pred = np.zeros(len(future_times))
        for pred_method, pred_values in predictions.items():
            ensemble_pred += pred_values * weights[pred_method]
        
        # Calcular intervalo de confiança baseado na variabilidade entre métodos
        all_preds = np.array(list(predictions.values()))
        pred_std = np.std(all_preds, axis=0)
        
        df_result = pd.DataFrame({
            'time': future_times,
            'aod': ensemble_pred,
            'type': 'forecast',
            'method': 'ensemble',
            'confidence_lower': ensemble_pred - 1.96 * pred_std,
            'confidence_upper': ensemble_pred + 1.96 * pred_std
        })
        
        # Adicionar dados históricos
        df_hist = df_work[['time', 'aod']].copy()
        df_hist['type'] = 'historical'
        df_hist['method'] = 'ensemble'
        df_hist['confidence_lower'] = df_hist['aod']
        df_hist['confidence_upper'] = df_hist['aod']
        
        # Adicionar informações sobre os métodos individuais
        individual_results = []
        for pred_method, pred_values in predictions.items():
            df_individual = pd.DataFrame({
                'time': future_times,
                'aod': pred_values,
                'type': 'forecast_individual',
                'method': pred_method,
                'confidence_lower': pred_values,
                'confidence_upper': pred_values
            })
            individual_results.append(df_individual)
        
        # Combinar todos os resultados
        all_results = [df_hist, df_result] + individual_results
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Adicionar informações sobre a qualidade dos métodos
        final_df.attrs['metrics'] = metrics
        final_df.attrs['weights'] = weights
        
        return final_df

# Função para avaliar e comparar métodos
def evaluate_prediction_methods(df, test_size=0.3):
    """
    Avalia diferentes métodos de previsão usando validação temporal.
    """
    if len(df) < 10:
        return None
    
    # Dividir dados em treino e teste (temporal)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    methods = ['linear', 'polynomial', 'arima', 'exponential', 'random_forest']
    results = {}
    
    for method in methods:
        try:
            # Fazer previsão para o período de teste
            forecast_df = predict_future_aod_advanced(
                train_df, 
                days=len(test_df)//4 + 1, 
                method=method
            )
            
            # Extrair apenas previsões
            predictions = forecast_df[forecast_df['type'] == 'forecast']['aod'].values
            
            # Ajustar tamanhos se necessário
            min_len = min(len(predictions), len(test_df))
            if min_len > 0:
                pred_subset = predictions[:min_len]
                actual_subset = test_df['aod'].values[:min_len]
                
                # Calcular métricas
                mae = mean_absolute_error(actual_subset, pred_subset)
                mse = mean_squared_error(actual_subset, pred_subset)
                rmse = np.sqrt(mse)
                
                results[method] = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAPE': np.mean(np.abs((actual_subset - pred_subset) / actual_subset)) * 100
                }
        except Exception as e:
            results[method] = {
                'MAE': np.inf,
                'MSE': np.inf,
                'RMSE': np.inf,
                'MAPE': np.inf,
                'Error': str(e)
            }
    
    return results

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

st.sidebar.subheader("Método de Previsão")
prediction_method = st.sidebar.selectbox(
    "Selecione o método de previsão",
    ['ensemble', 'linear', 'polynomial', 'arima', 'exponential', 'random_forest'],)
    index=0  # ensemble como padrão

# Seleção de cidade com os shapes disponíveis
available_cities = sorted(list(set(ms_shapes['NM_MUN'].tolist()).intersection(set(cities.keys()))))
if not available_cities:
    available_cities = list(cities.keys())  # Fallback para a lista original

city = st.sidebar.selectbox("Selecione o município", available_cities)
lat_center, lon_center = cities[city]

# No processamento:
df_forecast = predict_future_aod_advanced(df_timeseries, days=5, method=prediction_method)

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

# No processamento:
df_forecast = predict_future_aod_advanced(df_timeseries, days=5, method=prediction_method)

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
