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
from matplotlib.gridspec import GridSpec

# Configuração inicial da página
st.set_page_config(layout="wide", page_title="Monitoramento de Qualidade do Ar - MS")

# ✅ Carregar autenticação a partir do secrets.toml
try:
    ads_url = st.secrets["ads"]["url"]
    ads_key = st.secrets["ads"]["key"]
    client = cdsapi.Client(url=ads_url, key=ads_key)
except Exception as e:
    st.error("❌ Erro ao carregar as credenciais do CDS API. Verifique seu secrets.toml.")
    st.stop()

# 🌎 Lista expandida de cidades do MS
cities = {
    "Campo Grande": [-20.4697, -54.6201],
    "Dourados": [-22.2231, -54.812],
    "Três Lagoas": [-20.7849, -51.7005],
    "Corumbá": [-19.0082, -57.651],
    "Ponta Porã": [-22.5334, -55.7271],
    "Bonito": [-21.1261, -56.4836],
    "Nova Andradina": [-22.2332, -53.3437],
    "Aquidauana": [-20.4697, -55.7868],
    "Naviraí": [-23.0616, -54.1990],
    "Coxim": [-18.5065, -54.7610],
    "Paranaíba": [-19.6746, -51.1911],
    "Maracaju": [-21.6136, -55.1678],
    "Miranda": [-20.2410, -56.3752],
    "Chapadão do Sul": [-18.7884, -52.6264],
    "Costa Rica": [-18.5451, -53.1298],
    "Jardim": [-21.4799, -56.1489],
    "Rio Brilhante": [-21.8033, -54.5425],
    "Sidrolândia": [-20.9275, -54.9592],
    "Amambai": [-23.1058, -55.2253],
    "Porto Murtinho": [-21.6989, -57.8828]
}

# Função para baixar shapefile dos municípios de MS (simplificado)
@st.cache_data
def load_ms_municipalities():
    try:
        # URL para um shapefile de municípios do MS
        url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/" + \
              "municipio_2022/UFs/MS/MS_Municipios_2022.zip"
        
        # Tentativa de carregar os dados
        try:
            gdf = gpd.read_file(url)
            return gdf
        except:
            # Fallback: criar geodataframe simplificado com cidades
            data = {
                'NM_MUN': list(cities.keys()),
                'geometry': [
                    gpd.points_from_xy([lon], [lat])[0].buffer(0.2) 
                    for city, (lat, lon) in cities.items()
                ]
            }
            gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
            return gdf
    except Exception as e:
        st.warning(f"Não foi possível carregar os shapes dos municípios: {str(e)}")
        # Retornar DataFrame vazio com estrutura esperada
        return gpd.GeoDataFrame(columns=['NM_MUN', 'geometry'], crs="EPSG:4326")

# Dicionário de referência para parâmetros de qualidade do ar
air_quality_params = {
    "total_aerosol_optical_depth_550nm": {
        "nome": "AOD 550nm",
        "unidade": "sem unidade",
        "limites": {
            "bom": 0.1,
            "moderado": 0.2,
            "insalubre": 0.5,
            "perigoso": 1.0
        },
        "cores": ["green", "yellow", "orange", "red", "darkred"]
    },
    "nitrogen_dioxide_column_number_density": {
        "nome": "Dióxido de Nitrogênio (NO₂)",
        "unidade": "mol/m²",
        "limites": {
            "bom": 1e-5,
            "moderado": 5e-5,
            "insalubre": 1e-4,
            "perigoso": 2e-4
        },
        "cores": ["green", "yellow", "orange", "red", "darkred"]
    },
    "sulphur_dioxide_column_number_density": {
        "nome": "Dióxido de Enxofre (SO₂)",
        "unidade": "mol/m²",
        "limites": {
            "bom": 5e-6,
            "moderado": 2e-5,
            "insalubre": 5e-5,
            "perigoso": 1e-4
        },
        "cores": ["green", "yellow", "orange", "red", "darkred"]
    },
    "carbon_monoxide_column_number_density": {
        "nome": "Monóxido de Carbono (CO)",
        "unidade": "mol/m²",
        "limites": {
            "bom": 0.02,
            "moderado": 0.035,
            "insalubre": 0.05,
            "perigoso": 0.1
        },
        "cores": ["green", "yellow", "orange", "red", "darkred"]
    },
    "ozone_column_number_density": {
        "nome": "Ozônio (O₃)",
        "unidade": "mol/m²",
        "limites": {
            "bom": 0.006,
            "moderado": 0.01,
            "insalubre": 0.015,
            "perigoso": 0.02
        },
        "cores": ["green", "yellow", "orange", "red", "darkred"]
    },
    "nitric_oxide_column_number_density": {
        "nome": "Óxido Nítrico (NO)",
        "unidade": "mol/m²",
        "limites": {
            "bom": 1e-6,
            "moderado": 5e-6,
            "insalubre": 1e-5,
            "perigoso": 2e-5
        },
        "cores": ["green", "yellow", "orange", "red", "darkred"]
    }
}

# Títulos e introdução
st.title("🌀 Monitoramento e Previsão de Qualidade do Ar - Mato Grosso do Sul")
st.markdown("""
Este aplicativo permite visualizar e analisar dados de qualidade do ar para municípios 
de Mato Grosso do Sul. Os dados são obtidos em tempo real do CAMS (Copernicus Atmosphere 
Monitoring Service).

Indicadores disponíveis:
- **AOD 550nm**: Profundidade Óptica de Aerossóis
- **NO₂**: Dióxido de Nitrogênio
- **SO₂**: Dióxido de Enxofre
- **CO**: Monóxido de Carbono
- **O₃**: Ozônio
- **NO**: Óxido Nítrico
""")

# Carregar shapefiles dos municípios do MS
with st.spinner("Carregando shapes dos municípios..."):
    ms_shapes = load_ms_municipalities()

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Seleção de cidade
available_cities = sorted(list(cities.keys()))
city = st.sidebar.selectbox("Selecione o município", available_cities)
lat_center, lon_center = cities[city]

# Seleção de parâmetros
param_options = list(air_quality_params.keys())
param_names = [air_quality_params[p]["nome"] for p in param_options]
param_dict = dict(zip(param_names, param_options))

selected_params = st.sidebar.multiselect(
    "Selecione os parâmetros",
    param_names,
    default=[param_names[0]]
)

# Converte nomes de parâmetros em chaves para API
selected_param_keys = [param_dict[param] for param in selected_params]

# Configurações de data e hora
st.sidebar.subheader("Período de Análise")
start_date = st.sidebar.date_input("Data de Início", datetime.today() - timedelta(days=2))
end_date = st.sidebar.date_input("Data Final", datetime.today() + timedelta(days=3))

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

# Função para extrair valores para um ponto específico
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

# Função para prever valores futuros
def predict_future_values(df, days=3):
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
    
    # Limitar valores previstos (não pode ser negativo)
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

# Função para calcular categoria de qualidade do ar
def get_air_quality_category(value, param_key):
    """Retorna categoria e cor para o valor de um parâmetro."""
    if param_key not in air_quality_params:
        return "Desconhecido", "gray"
    
    limits = air_quality_params[param_key]["limites"]
    colors = air_quality_params[param_key]["cores"]
    
    if value < limits["bom"]:
        return "Bom", colors[0]
    elif value < limits["moderado"]:
        return "Moderado", colors[1]
    elif value < limits["insalubre"]:
        return "Insalubre para grupos sensíveis", colors[2]
    elif value < limits["perigoso"]:
        return "Insalubre", colors[3]
    else:
        return "Perigoso", colors[4]

# Função para gerar relatório de qualidade do ar
def generate_air_quality_report(dataframes, param_keys):
    """Gera relatório de qualidade do ar com base em séries temporais."""
    if not dataframes or not param_keys:
        return None
    
    # Criar DataFrame para relatório
    report = pd.DataFrame()
    
    # Para cada parâmetro, calcular estatísticas
    for param_key, df in zip(param_keys, dataframes):
        if df.empty:
            continue
            
        # Dados históricos apenas
        hist_data = df[df['type'] == 'historical']
        if hist_data.empty:
            continue
            
        # Estatísticas básicas
        current = hist_data['value'].iloc[-1]
        max_val = hist_data['value'].max()
        min_val = hist_data['value'].min()
        mean_val = hist_data['value'].mean()
        
        # Tendência (últimas 24h)
        recent_cutoff = hist_data['time'].max() - pd.Timedelta(hours=24)
        recent_data = hist_data[hist_data['time'] >= recent_cutoff]
        
        if len(recent_data) >= 2:
            first_val = recent_data['value'].iloc[0]
            last_val = recent_data['value'].iloc[-1]
            trend_pct = ((last_val - first_val) / first_val * 100) if first_val > 0 else 0
        else:
            trend_pct = 0
            
        # Categoria atual
        category, color = get_air_quality_category(current, param_key)
        
        # Adicionar ao DataFrame
        param_info = air_quality_params[param_key]
        new_row = pd.DataFrame({
            'param_key': [param_key],
            'param_name': [param_info['nome']],
            'current': [current],
            'min': [min_val],
            'max': [max_val],
            'mean': [mean_val],
            'trend_pct': [trend_pct],
            'category': [category],
            'color': [color],
            'unit': [param_info['unidade']]
        })
        
        report = pd.concat([report, new_row], ignore_index=True)
    
    return report

# Função principal para gerar análise de qualidade do ar
def generate_air_quality_analysis():
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
    
    # Verificar se temos parâmetros selecionados
    if not selected_param_keys:
        st.error("Selecione pelo menos um parâmetro para análise.")
        return None
    
    # Preparar request para API
    request = {
        'variable': selected_param_keys,
        'date': f'{start_date_str}/{end_date_str}',
        'time': hours,
        'leadtime_hour': ['0', '24', '48', '72'],  # Incluir previsões de até 3 dias
        'type': ['forecast'],
        'format': 'netcdf',
        'area': [lat_center + map_width/2, lon_center - map_width/2, 
                lat_center - map_width/2, lon_center + map_width/2]
    }
    
    filename = f'AirQuality_{city}_{start_date}_to_{end_date}.nc'
    
    try:
        with st.spinner('📥 Baixando dados do CAMS...'):
            client.retrieve(dataset, request).download(filename)
        
        ds = xr.open_dataset(filename)
        
        # Verificar variáveis disponíveis
        variable_names = list(ds.data_vars)
        st.write(f"Variáveis disponíveis: {variable_names}")
        
        # Filtra apenas as variáveis que estão nos parâmetros selecionados
        available_params = [var for var in selected_param_keys if var in variable_names]
        
        if not available_params:
            st.error("Nenhum dos parâmetros selecionados está disponível nos dados.")
            return None
        
        # Para cada parâmetro, extrair série temporal
        timeseries_data = {}
        forecast_data = {}
        
        for param in available_params:
            # Extrair série temporal
            with st.spinner(f"Extraindo série temporal para {air_quality_params[param]['nome']}..."):
                ts_df = extract_point_timeseries(ds, lat_center, lon_center, var_name=param)
                
                if not ts_df.empty:
                    timeseries_data[param] = ts_df
                    
                    # Gerar previsão
                    with st.spinner(f"Gerando previsão para {air_quality_params[param]['nome']}..."):
                        forecast_df = predict_future_values(ts_df, days=3)
                        forecast_data[param] = forecast_df
        
        # Encontrar o município no geodataframe
        municipality_shape = None
        if not ms_shapes.empty:
            city_shape = ms_shapes[ms_shapes['NM_MUN'] == city]
            if not city_shape.empty:
                municipality_shape = city_shape.iloc[0].geometry
        
        # Para o primeiro parâmetro, criar a animação
        main_param = available_params[0]
        da = ds[main_param]
        
        # Verificar dimensões
        st.write(f"Dimensões do parâmetro principal: {da.dims}")
        
        # Identificar dimensões temporais
        time_dims = [dim for dim in da.dims if 'time' in dim or 'forecast' in dim]
        
        if not time_dims:
            st.error("Não foi possível identificar dimensão temporal nos dados.")
            return None
        
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
        vmin = max(0, vmin - 0.05 * vmin)
        vmax = min(vmax * 1.05, vmax * 2)  # Evitar outliers
        
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
        cbar.set_label(f"{air_quality_params[main_param]['nome']} ({air_quality_params[main_param]['unidade']})")
        
        # Adicionar título inicial
        title = ax.set_title(f"{air_quality_params[main_param]['nome']} em {city} - {first_frame_time}", fontsize=14)
        
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
                title.set_text(f"{air_quality_params[main_param]['nome']} em {city} - {frame_time}")
                
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
        gif_filename = f'AirQuality_{city}_{start_date}_to_{end_date}.gif'
        
        with st.spinner('💾 Salvando animação...'):
            ani.save(gif_filename, writer=animation.PillowWriter(fps=2))
        
        plt.close(fig)
        
        # Gerar relatório
        report_data = []
        for param in available_params:
            if param in forecast_data:
                report_data.append(forecast_data[param])
                
        with st.spinner('📊 Gerando relatório de qualidade do ar...'):
            report = generate_air_quality_report(report_data, available_params)
        
        return {
            'animation': gif_filename,
            'timeseries': timeseries_data,
            'forecast': forecast_data,
            'dataset': ds,
            'parameters': available_params,
            'report': report
        }
    
    except Exception as e:
        st.error(f"❌ Erro ao processar os dados: {str(e)}")
        st.write("Detalhes da requisição:")
        st.write(request)
        return None

# Função para desenhar gráfico com múltiplos parâmetros
# Função para desenhar gráfico com múltiplos parâmetros
def plot_multi_parameter_chart(forecasts, params):
    """Gera um gráfico com múltiplos parâmetros em subplots."""
    if not forecasts or not params:
        return None
        
    # Determinar número de subplots necessários
    n_plots = len(params)
    if n_plots == 0:
        return None
    
    # Criar figura com subplots
    fig = plt.figure(figsize=(12, n_plots * 3))
    gs = GridSpec(n_plots, 1, figure=fig)
    
    # Para cada parâmetro, criar um subplot
    for i, param in enumerate(params):
        if param not in forecasts:
            continue
            
        df = forecasts[param]
        if df.empty:
            continue
            
        ax = fig.add_subplot(gs[i, 0])
        
        # Separar dados históricos e previsão
        hist_data = df[df['type'] == 'historical']
        forecast_data = df[df['type'] == 'forecast']
        
        # Plotar dados históricos
        ax.plot(hist_data['time'], hist_data['value'], 
               marker='o', linestyle='-', color='blue', label='Dados Observados')
        
        # Plotar dados de previsão
        if not forecast_data.empty:
            ax.plot(forecast_data['time'], forecast_data['value'], 
                   marker='x', linestyle='--', color='red', label='Previsão')
        
        # Configurar eixos
        ax.set_title(f"{air_quality_params[param]['nome']}", fontsize=12)
        ax.set_ylabel(f"{air_quality_params[param]['unidade']}", fontsize=10)
        
        if i == n_plots - 1:  # Último subplot
            ax.set_xlabel('Data/Hora', fontsize=10)
            
        # Formatar datas no eixo x
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
        plt.xticks(rotation=45)
        
        # Adicionar legenda e grade
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Adicionar faixas de qualidade do ar
        limits = air_quality_params[param]['limites']
        colors = air_quality_params[param]['cores']
        
        if limits:
            y_min, y_max = ax.get_ylim()
            y_max = max(y_max, limits['perigoso'] * 1.2)
            
            # Desenhar faixas coloridas
            ax.axhspan(0, limits['bom'], alpha=0.1, color=colors[0], label='Bom')
            ax.axhspan(limits['bom'], limits['moderado'], alpha=0.1, color=colors[1])
            ax.axhspan(limits['moderado'], limits['insalubre'], alpha=0.1, color=colors[2])
            ax.axhspan(limits['insalubre'], limits['perigoso'], alpha=0.1, color=colors[3])
            ax.axhspan(limits['perigoso'], y_max, alpha=0.1, color=colors[4])
            
            # Ajustar limites do eixo y
            ax.set_ylim(0, y_max)
    
    plt.tight_layout()
    return fig

# Função para gerar relatório detalhado de qualidade do ar
def generate_detailed_report(report_df, forecasts, params):
    """Gera relatório detalhado com tendências e previsões."""
    if report_df is None or report_df.empty:
        return None
        
    # Criar DataFrame para o relatório
    report_html = "<div style='padding:10px; background-color:#f8f9fa; border-radius:5px;'>"
    report_html += "<h3>📋 Relatório de Qualidade do Ar</h3>"
    report_html += f"<p>Município: <b>{city}</b> | Período: {start_date} a {end_date}</p>"
    
    # Tabela com situação atual
    report_html += "<h4>Situação Atual dos Parâmetros</h4>"
    
    # Cabeçalho da tabela
    report_html += """
    <table style='width:100%; border-collapse:collapse;'>
    <tr style='background-color:#e9ecef;'>
        <th style='padding:8px; text-align:left; border:1px solid #dee2e6;'>Parâmetro</th>
        <th style='padding:8px; text-align:center; border:1px solid #dee2e6;'>Valor Atual</th>
        <th style='padding:8px; text-align:center; border:1px solid #dee2e6;'>Categoria</th>
        <th style='padding:8px; text-align:center; border:1px solid #dee2e6;'>Tendência 24h</th>
    </tr>
    """
    
    # Linhas da tabela
    for _, row in report_df.iterrows():
        # Formatação do valor atual
        value_str = f"{row['current']:.4f}"
        if row['current'] < 0.01:
            value_str = f"{row['current']:.2e}"
            
        # Formatação da tendência
        if row['trend_pct'] > 5:
            trend_icon = "⬆️"
            trend_desc = f"+{row['trend_pct']:.1f}%"
            trend_color = "#dc3545"  # vermelho
        elif row['trend_pct'] < -5:
            trend_icon = "⬇️"
            trend_desc = f"{row['trend_pct']:.1f}%"
            trend_color = "#28a745"  # verde
        else:
            trend_icon = "↔️"
            trend_desc = "Estável"
            trend_color = "#6c757d"  # cinza
            
        # Cor da categoria
        category_color = row['color']
            
        report_html += f"""
        <tr>
            <td style='padding:8px; border:1px solid #dee2e6;'>{row['param_name']}</td>
            <td style='padding:8px; text-align:center; border:1px solid #dee2e6;'>{value_str} {row['unit']}</td>
            <td style='padding:8px; text-align:center; border:1px solid #dee2e6; background-color:{category_color}; color:white;'>{row['category']}</td>
            <td style='padding:8px; text-align:center; border:1px solid #dee2e6; color:{trend_color};'>{trend_icon} {trend_desc}</td>
        </tr>
        """
    
    report_html += "</table>"
    
    # Previsão para próximos dias
    report_html += "<h4>Previsão para os Próximos Dias</h4>"
    
    # Para cada parâmetro, analisar previsão
    for param in params:
        if param not in forecasts or param not in air_quality_params:
            continue
            
        df = forecasts[param]
        if df.empty:
            continue
            
        forecast_data = df[df['type'] == 'forecast']
        if forecast_data.empty:
            continue
            
        # Agrupar por dia
        forecast_data['date'] = forecast_data['time'].dt.date
        daily_forecast = forecast_data.groupby('date')['value'].agg(['mean', 'max']).reset_index()
        
        param_name = air_quality_params[param]['nome']
        report_html += f"<p><b>{param_name}</b></p>"
        
        # Mini tabela de previsão
        report_html += """
        <table style='width:100%; border-collapse:collapse; margin-bottom:15px;'>
        <tr style='background-color:#e9ecef;'>
            <th style='padding:6px; text-align:left; border:1px solid #dee2e6;'>Data</th>
            <th style='padding:6px; text-align:center; border:1px solid #dee2e6;'>Média</th>
            <th style='padding:6px; text-align:center; border:1px solid #dee2e6;'>Máximo</th>
            <th style='padding:6px; text-align:center; border:1px solid #dee2e6;'>Condição</th>
        </tr>
        """
        
        for _, day in daily_forecast.iterrows():
            # Calcular categoria com base no valor médio
            category, color = get_air_quality_category(day['mean'], param)
            
            # Formatação dos valores
            mean_str = f"{day['mean']:.4f}"
            max_str = f"{day['max']:.4f}"
            
            if day['mean'] < 0.01:
                mean_str = f"{day['mean']:.2e}"
            if day['max'] < 0.01:
                max_str = f"{day['max']:.2e}"
                
            report_html += f"""
            <tr>
                <td style='padding:6px; border:1px solid #dee2e6;'>{day['date'].strftime('%d/%m/%Y')}</td>
                <td style='padding:6px; text-align:center; border:1px solid #dee2e6;'>{mean_str}</td>
                <td style='padding:6px; text-align:center; border:1px solid #dee2e6;'>{max_str}</td>
                <td style='padding:6px; text-align:center; border:1px solid #dee2e6; background-color:{color}; color:white;'>{category}</td>
            </tr>
            """
            
        report_html += "</table>"
    
    # Análise final
    worst_param = None
    worst_category = "Bom"
    
    # Encontrar o pior parâmetro
    for _, row in report_df.iterrows():
        categories = ["Bom", "Moderado", "Insalubre para grupos sensíveis", "Insalubre", "Perigoso"]
        category_index = categories.index(row['category']) if row['category'] in categories else -1
        worst_index = categories.index(worst_category) if worst_category in categories else -1
        
        if category_index > worst_index:
            worst_category = row['category']
            worst_param = row['param_name']
    
    if worst_param:
        report_html += f"""
        <div style='padding:10px; background-color:#f1f8ff; border-left:4px solid #0366d6; margin-top:15px;'>
            <h4>Análise Geral da Qualidade do Ar</h4>
            <p>A qualidade do ar em {city} está classificada como <b style='color:{report_df[report_df['param_name'] == worst_param]['color'].iloc[0]};'>{worst_category}</b>, 
            principalmente devido aos níveis de <b>{worst_param}</b>.</p>
        """
        
        # Adicionar recomendações com base na categoria
        if worst_category == "Bom":
            report_html += "<p>✅ Condições favoráveis para atividades ao ar livre.</p>"
        elif worst_category == "Moderado":
            report_html += "<p>⚠️ Pessoas muito sensíveis devem considerar limitar atividades prolongadas ao ar livre.</p>"
        elif worst_category == "Insalubre para grupos sensíveis":
            report_html += "<p>⚠️ Grupos sensíveis (crianças, idosos e pessoas com doenças respiratórias) devem evitar atividades ao ar livre.</p>"
        elif worst_category == "Insalubre":
            report_html += "<p>🚫 Reduzir atividades ao ar livre. Grupos sensíveis devem permanecer em ambientes internos.</p>"
        else:  # Perigoso
            report_html += "<p>⛔ Evitar qualquer atividade ao ar livre. Manter janelas fechadas.</p>"
            
        report_html += "</div>"
    
    report_html += "</div>"
    
    return report_html

# Botão para iniciar análise
if st.button("🎞️ Gerar Análise Completa", type="primary"):
    # Verificar se há parâmetros selecionados
    if not selected_params:
        st.error("Selecione pelo menos um parâmetro para análise.")
    else:
        # Executar análise e obter resultados
        results = generate_air_quality_analysis()
        
        if results:
            # Layout de duas colunas para conteúdo principal
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("🎬 Animação de Qualidade do Ar")
                main_param = results['parameters'][0]
                param_name = air_quality_params[main_param]['nome']
                st.image(results['animation'], caption=f"{param_name} em {city} ({start_date} a {end_date})")
                
                # Adicionar opções para baixar
                with open(results['animation'], "rb") as file:
                    btn = st.download_button(
                        label="⬇️ Baixar Animação (GIF)",
                        data=file,
                        file_name=f"AirQuality_{city}_{start_date}_to_{end_date}.gif",
                        mime="image/gif"
                    )
            
            with col2:
                # Se tivermos apenas um parâmetro
                if len(results['parameters']) == 1:
                    param = results['parameters'][0]
                    st.subheader(f"📊 Série Temporal: {air_quality_params[param]['nome']}")
                    
                    # Criar gráfico
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    df_combined = results['forecast'][param]
                    
                    # Dados históricos
                    hist_data = df_combined[df_combined['type'] == 'historical']
                    ax.plot(hist_data['time'], hist_data['value'], 
                           marker='o', linestyle='-', color='blue', label='Observado')
                    
                    # Dados de previsão
                    forecast_data = df_combined[df_combined['type'] == 'forecast']
                    ax.plot(forecast_data['time'], forecast_data['value'], 
                           marker='x', linestyle='--', color='red', label='Previsão')
                    
                    # Formatar eixos
                    ax.set_title(f"{air_quality_params[param]['nome']} em {city}", fontsize=14)
                    ax.set_xlabel('Data/Hora', fontsize=12)
                    ax.set_ylabel(f"{air_quality_params[param]['unidade']}", fontsize=12)
                    
                    # Formatar datas no eixo x
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
                    plt.xticks(rotation=45)
                    
                    # Adicionar legenda e grade
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Adicionar faixas de qualidade do ar
                    limits = air_quality_params[param]['limites']
                    colors = air_quality_params[param]['cores']
                    
                    y_min, y_max = ax.get_ylim()
                    y_max = max(y_max, limits['perigoso'] * 1.2)
                    
                    ax.axhspan(0, limits['bom'], alpha=0.2, color=colors[0], label='Boa')
                    ax.axhspan(limits['bom'], limits['moderado'], alpha=0.2, color=colors[1], label='Moderada')
                    ax.axhspan(limits['moderado'], limits['insalubre'], alpha=0.2, color=colors[2], label='Insalubre para grupos sensíveis')
                    ax.axhspan(limits['insalubre'], limits['perigoso'], alpha=0.2, color=colors[3], label='Insalubre')
                    ax.axhspan(limits['perigoso'], y_max, alpha=0.2, color=colors[4], label='Perigosa')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                # Se tivermos múltiplos parâmetros
                else:
                    st.subheader("📊 Comparação de Parâmetros")
                    
                    # Criar gráfico multi-parâmetros
                    fig_multi = plot_multi_parameter_chart(results['forecast'], results['parameters'])
                    if fig_multi:
                        st.pyplot(fig_multi)
                    else:
                        st.warning("Não foi possível gerar o gráfico comparativo.")
            
            # Relatório completo abaixo 
            st.subheader("📑 Relatório Detalhado de Qualidade do Ar")
            
            if results['report'] is not None and not results['report'].empty:
                # Gerar relatório HTML detalhado
                report_html = generate_detailed_report(
                    results['report'], 
                    results['forecast'], 
                    results['parameters']
                )
                
                if report_html:
                    st.markdown(report_html, unsafe_allow_html=True)
                    
                    # Estatísticas dos parâmetros
                    with st.expander("📊 Estatísticas Detalhadas"):
                        for param in results['parameters']:
                            if param in results['forecast']:
                                df = results['forecast'][param]
                                hist_data = df[df['type'] == 'historical']
                                
                                if not hist_data.empty:
                                    st.write(f"### {air_quality_params[param]['nome']}")
                                    
                                    # Estatísticas básicas
                                    stats_df = pd.DataFrame({
                                        'Estatística': ['Mínimo', 'Média', 'Máximo', 'Desvio Padrão'],
                                        'Valor': [
                                            f"{hist_data['value'].min():.6f}",
                                            f"{hist_data['value'].mean():.6f}",
                                            f"{hist_data['value'].max():.6f}",
                                            f"{hist_data['value'].std():.6f}"
                                        ]
                                    })
                                    
                                    st.dataframe(stats_df, hide_index=True)
                                    
                                    # Histograma da distribuição
                                    fig, ax = plt.subplots(figsize=(8, 4))
                                    sns.histplot(hist_data['value'], kde=True, ax=ax)
                                    ax.set_title(f"Distribuição de {air_quality_params[param]['nome']}")
                                    ax.set_xlabel(f"{air_quality_params[param]['unidade']}")
                                    st.pyplot(fig)
                                    
                                    # Linha horizontal
                                    st.markdown("---")
                
                # Exportar dados
                with st.expander("💾 Exportar Dados"):
                    # Preparar dados para exportação
                    export_dfs = {}
                    
                    for param in results['parameters']:
                        if param in results['forecast']:
                            param_name = air_quality_params[param]['nome']
                            df = results['forecast'][param].copy()
                            # Renomear colunas para melhor compreensão
                            df = df.rename(columns={
                                'value': param_name,
                                'time': 'Data/Hora',
                                'type': 'Tipo'
                            })
                            export_dfs[param_name] = df
                    
                    # Opções de parâmetros para exportar
                    export_param = st.selectbox(
                        "Selecione o parâmetro para exportar", 
                        list(export_dfs.keys())
                    )
                    
                    if export_param in export_dfs:
                        # Formato de exportação
                        export_format = st.radio(
                            "Formato de exportação",
                            options=["CSV", "Excel", "JSON"],
                            horizontal=True
                        )
                        
                        df_to_export = export_dfs[export_param]
                        
                        if export_format == "CSV":
                            csv = df_to_export.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Baixar dados (CSV)",
                                data=csv,
                                file_name=f"{export_param}_{city}_{start_date}_to_{end_date}.csv",
                                mime="text/csv",
                            )
                        elif export_format == "Excel":
                            # Buffer para Excel
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                df_to_export.to_excel(writer, sheet_name=export_param, index=False)
                            
                            buffer.seek(0)
                            st.download_button(
                                label="⬇️ Baixar dados (Excel)",
                                data=buffer,
                                file_name=f"{export_param}_{city}_{start_date}_to_{end_date}.xlsx",
                                mime="application/vnd.ms-excel",
                            )
                        else:  # JSON
                            json_str = df_to_export.to_json(orient='records', date_format='iso')
                            st.download_button(
                                label="⬇️ Baixar dados (JSON)",
                                data=json_str,
                                file_name=f"{export_param}_{city}_{start_date}_to_{end_date}.json",
                                mime="application/json",
                            )
                            
                # Opção de relatório em PDF (simulado)
                with st.expander("📄 Gerar Relatório em PDF"):
                    st.info("Esta funcionalidade simularia a geração de um relatório em PDF completo com os dados analisados.")
                    
                    report_type = st.radio(
                        "Tipo de relatório",
                        ["Resumido", "Completo", "Técnico"],
                        horizontal=True
                    )
                    
                    if st.button("Simular geração de PDF"):
                        with st.spinner("Preparando relatório..."):
                            st.success(f"Relatório {report_type} gerado com sucesso! Esta é uma simulação da funcionalidade.")
                            
                            # Mostrar preview do que seria o PDF
                            st.markdown(f"""
                            **Preview do relatório {report_type}**
                            
                            # Monitoramento de Qualidade do Ar em {city}
                            Período: {start_date} a {end_date}
                            
                            ## Parâmetros Monitorados
                            - {', '.join([air_quality_params[p]['nome'] for p in results['parameters']])}
                            
                            ## Conclusão
                            Este relatório apresenta a análise completa dos parâmetros de qualidade do ar monitorados.
                            """)

# Adicionar informações na parte inferior
st.markdown("---")
st.markdown("""
### ℹ️ Sobre os dados
- **Fonte**: Copernicus Atmosphere Monitoring Service (CAMS)
- **Parâmetros**:
  - **AOD 550nm**: Profundidade Óptica de Aerossóis (indicador de partículas suspensas)
  - **NO₂**: Dióxido de Nitrogênio (poluente de tráfego e indústrias)
  - **SO₂**: Dióxido de Enxofre (poluente industrial)
  - **CO**: Monóxido de Carbono (poluente de combustão)
  - **O₃**: Ozônio (poluente fotoquímico)
  - **NO**: Óxido Nítrico (precursor de NO₂)
- **Resolução temporal**: 3 horas
- **Atualização**: Diária

### 📖 Guia de interpretação da qualidade do ar:
| Categoria | Recomendação |
|-----------|--------------|
| **Boa** | Ideal para atividades ao ar livre |
| **Moderada** | Pessoas excepcionalmente sensíveis devem considerar limitação de esforço prolongado |
| **Insalubre para grupos sensíveis** | Idosos, crianças e pessoas com problemas respiratórios devem limitar esforço prolongado |
| **Insalubre** | Todos devem reduzir o esforço prolongado; grupos sensíveis devem evitar atividades ao ar livre |
| **Perigosa** | Todos devem evitar qualquer atividade ao ar livre; grupos sensíveis devem permanecer em ambiente interno |

Desenvolvido para monitoramento da qualidade do ar no estado de Mato Grosso do Sul - Brasil.
""")
        
