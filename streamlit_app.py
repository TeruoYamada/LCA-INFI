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

# Lista expandida de cidades do Mato Grosso do Sul
cities = {
    "Campo Grande": [-20.4697, -54.6201],
    "Dourados": [-22.2231, -54.812],
    "Três Lagoas": [-20.7849, -51.7005],
    "Corumbá": [-19.0082, -57.651],
    "Ponta Porã": [-22.5334, -55.7271],
    "Naviraí": [-23.0618, -54.1990],
    "Nova Andradina": [-22.2330, -53.3430],
    "Aquidauana": [-20.4666, -55.7868],
    "Bonito": [-21.1256, -56.4836],
    "Coxim": [-18.5013, -54.7603],
    "Jardim": [-21.4799, -56.1489],
    "Miranda": [-20.2422, -56.3780],
    "Chapadão do Sul": [-18.7908, -52.6263],
    "Paranaíba": [-19.6746, -51.1909],
    "Maracaju": [-21.6114, -55.1697]
}

# Parâmetros de qualidade do ar para monitoramento
air_quality_parameters = {
    "Carbon monoxide": {
        "variable": "carbon_monoxide", 
        "threshold_good": 4.4,   # ppm
        "threshold_moderate": 9.4,
        "threshold_unhealthy": 12.4,
        "unit": "ppm",
        "color_map": "YlOrRd"
    },
    "Nitrogen dioxide": {
        "variable": "nitrogen_dioxide", 
        "threshold_good": 53,    # ppb
        "threshold_moderate": 100,
        "threshold_unhealthy": 360,
        "unit": "ppb",
        "color_map": "RdPu"
    },
    "Ozone": {
        "variable": "ozone", 
        "threshold_good": 54,    # ppb
        "threshold_moderate": 70,
        "threshold_unhealthy": 85,
        "unit": "ppb",
        "color_map": "PuBu"
    },
    "Sulphur dioxide": {
        "variable": "sulphur_dioxide", 
        "threshold_good": 35,    # ppb
        "threshold_moderate": 75,
        "threshold_unhealthy": 185,
        "unit": "ppb",
        "color_map": "BuPu"
    },
    "Aerosol Optical Depth": {
        "variable": "total_aerosol_optical_depth_550nm", 
        "threshold_good": 0.1,
        "threshold_moderate": 0.2,
        "threshold_unhealthy": 0.5,
        "unit": "",
        "color_map": "YlOrRd"
    }
}

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
                'NM_MUN': list(cities.keys()),
                'geometry': [
                    gpd.points_from_xy([coords[1]], [coords[0]])[0].buffer(0.2)
                    for city, coords in cities.items()
                ]
            }
            gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
            return gdf
    except Exception as e:
        st.warning(f"Não foi possível carregar os shapes dos municípios: {str(e)}")
        # Retornar DataFrame vazio com estrutura esperada
        return gpd.GeoDataFrame(columns=['NM_MUN', 'geometry'], crs="EPSG:4326")

# Títulos e introdução
st.title("🌀 Monitoramento de Qualidade do Ar - Mato Grosso do Sul")
st.markdown("""
Este aplicativo permite visualizar e analisar dados de qualidade do ar para municípios de 
Mato Grosso do Sul. Os dados são obtidos em tempo real do CAMS (Copernicus Atmosphere 
Monitoring Service).

**Parâmetros monitorados:**
- Monóxido de Carbono (CO)
- Dióxido de Nitrogênio (NO₂)
- Ozônio (O₃)
- Dióxido de Enxofre (SO₂)
- Profundidade Óptica de Aerossóis (AOD) a 550nm
""")

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

# Seleção de parâmetro de qualidade do ar
parameter = st.sidebar.selectbox("Parâmetro de qualidade do ar", 
                             list(air_quality_parameters.keys()))

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

# Função para extrair valores de um parâmetro para um ponto específico
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

# Função para determinar a categoria de qualidade do ar
def air_quality_category(value, parameter_info):
    """Determina a categoria de qualidade do ar com base no valor e parâmetro."""
    if value < parameter_info["threshold_good"]:
        return "Boa", "green"
    elif value < parameter_info["threshold_moderate"]:
        return "Moderada", "orange"
    elif value < parameter_info["threshold_unhealthy"]:
        return "Insalubre para grupos sensíveis", "red"
    else:
        return "Perigosa", "darkred"

# Função principal para gerar análise
def generate_air_quality_analysis():
    dataset = "cams-global-atmospheric-composition-forecasts"
    parameter_info = air_quality_parameters[parameter]
    var_name = parameter_info["variable"]
    
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
        'variable': [var_name],
        'date': f'{start_date_str}/{end_date_str}',
        'time': hours,
        'leadtime_hour': ['0', '24', '48', '72'],  # Incluir previsões de até 3 dias
        'type': ['forecast'],
        'format': 'netcdf',
        'area': [lat_center + map_width/2, lon_center - map_width/2, 
                lat_center - map_width/2, lon_center + map_width/2]
    }
    
    filename = f'{var_name}_{city}_{start_date}_to_{end_date}.nc'
    
    try:
        with st.spinner(f'📥 Baixando dados de {parameter} do CAMS...'):
            client.retrieve(dataset, request).download(filename)
        
        ds = xr.open_dataset(filename)
        
        # Verificar variáveis disponíveis
        variable_names = list(ds.data_vars)
        st.write(f"Variáveis disponíveis: {variable_names}")
        
        # Usar a variável encontrada nos dados
        if var_name not in variable_names:
            # Tentar encontrar uma variável similar
            var_found = next((var for var in variable_names if var_name.lower() in var.lower()), variable_names[0])
            st.warning(f"Variável {var_name} não encontrada. Usando {var_found} em seu lugar.")
            var_name = var_found
        
        st.write(f"Usando variável: {var_name}")
        da = ds[var_name]
        
        # Verificar dimensões
        st.write(f"Dimensões: {da.dims}")
        
        # Identificar dimensões temporais
        time_dims = [dim for dim in da.dims if 'time' in dim or 'forecast' in dim]
        
        if not time_dims:
            st.error("Não foi possível identificar dimensão temporal nos dados.")
            return None
        
        # Extrair série temporal para o ponto central (cidade selecionada)
        with st.spinner(f"Extraindo série temporal de {parameter} para {city}..."):
            df_timeseries = extract_point_timeseries(ds, lat_center, lon_center, var_name=var_name)
        
        if df_timeseries.empty:
            st.error("Não foi possível extrair série temporal para este local.")
            return None
        
        # Gerar previsão para os próximos dias
        with st.spinner(f"Gerando previsão de {parameter}..."):
            df_forecast = predict_future_values(df_timeseries, days=3)
        
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
        
        # Determinar range de cores com base nos limiares do parâmetro
        vmin = 0
        vmax = parameter_info["threshold_unhealthy"] * 1.5  # 150% do limiar "insalubre"
        
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
        
        # Criar mapa de cores usando colormap do parâmetro ou o selecionado pelo usuário
        cmap = colormap if colormap != "YlOrRd" else parameter_info["color_map"]
        im = ax.pcolormesh(ds.longitude, ds.latitude, first_frame_data, 
                         cmap=cmap, vmin=vmin, vmax=vmax)
        
        # Adicionar barra de cores
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label(f'{parameter} ({parameter_info["unit"]})')
        
        # Adicionar título inicial
        title = ax.set_title(f'{parameter} em {city} - {first_frame_time}', fontsize=14)
        
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
                title.set_text(f'{parameter} em {city} - {frame_time}')
                
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
        gif_filename = f'{var_name}_{city}_{start_date}_to_{end_date}.gif'
        
        with st.spinner('💾 Salvando animação...'):
            ani.save(gif_filename, writer=animation.PillowWriter(fps=2))
        
        plt.close(fig)
        
        return {
            'animation': gif_filename,
            'timeseries': df_timeseries,
            'forecast': df_forecast,
            'dataset': ds,
            'variable': var_name,
            'parameter_info': parameter_info
        }
    
    except Exception as e:
        st.error(f"❌ Erro ao processar os dados: {str(e)}")
        st.write("Detalhes da requisição:")
        st.write(request)
        return None

# Função para gerar relatório completo de todos os parâmetros
def generate_complete_report():
    st.subheader("📊 Relatório Completo de Qualidade do Ar")
    
    # Criar abas para cada parâmetro
    tabs = st.tabs([param for param in air_quality_parameters.keys()])
    
    all_data = {}
    
    # Para cada parâmetro, gerar análise
    for i, (param_name, param_info) in enumerate(air_quality_parameters.items()):
        with tabs[i]:
            st.write(f"### {param_name}")
            
            # Tentar obter dados para este parâmetro
            try:
                with st.spinner(f"Analisando {param_name}..."):
                    # Configurar requisição
                    start_date_str = start_date.strftime('%Y-%m-%d')
                    end_date_str = end_date.strftime('%Y-%m-%d')
                    
                    # Criar lista de horas no formato correto
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
                        hours = ['00:00', '12:00']
                    
                    # Preparar request para API
                    request = {
                        'variable': [param_info["variable"]],
                        'date': f'{start_date_str}/{end_date_str}',
                        'time': hours,
                        'leadtime_hour': ['0', '24'],  # Versão simplificada para o relatório
                        'type': ['forecast'],
                        'format': 'netcdf',
                        'area': [lat_center + map_width/2, lon_center - map_width/2, 
                                lat_center - map_width/2, lon_center + map_width/2]
                    }
                    
                    filename = f'{param_info["variable"]}_{city}_report.nc'
                    
                    # Baixar dados
                    client.retrieve("cams-global-atmospheric-composition-forecasts", request).download(filename)
                    ds = xr.open_dataset(filename)
                    
                    # Extrair série temporal
                    df = extract_point_timeseries(ds, lat_center, lon_center, var_name=param_info["variable"])
                    
                    if not df.empty:
                        # Salvar dados
                        all_data[param_name] = df
                        
                        # Mostrar gráfico
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(df['time'], df['value'], marker='o', linestyle='-')
                        ax.set_title(f'{param_name} em {city}')
                        ax.set_ylabel(f'{param_name} ({param_info["unit"]})')
                        ax.grid(True, alpha=0.3)
                        
                        # Adicionar limites de qualidade do ar
                        ax.axhspan(0, param_info["threshold_good"], alpha=0.2, color='green', label='Boa')
                        ax.axhspan(param_info["threshold_good"], param_info["threshold_moderate"], alpha=0.2, color='yellow', label='Moderada')
                        ax.axhspan(param_info["threshold_moderate"], param_info["threshold_unhealthy"], alpha=0.2, color='orange', label='Insalubre')
                        ax.axhspan(param_info["threshold_unhealthy"], param_info["threshold_unhealthy"]*2, alpha=0.2, color='red', label='Perigosa')
                        
                        ax.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Estatísticas
                        current_val = df['value'].iloc[-1] if len(df) > 0 else 0
                        avg_val = df['value'].mean() if len(df) > 0 else 0
                        max_val = df['value'].max() if len(df) > 0 else 0
                        
                        # Determinar categoria
                        category, color = air_quality_category(current_val, param_info)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric(f"Atual", f"{current_val:.2f} {param_info['unit']}")
                        col2.metric(f"Média", f"{avg_val:.2f} {param_info['unit']}")
                        col3.metric(f"Máximo", f"{max_val:.2f} {param_info['unit']}")
                        
                        # Mostrar categoria
                        st.markdown(f"""
                        <div style="padding:10px; border-radius:5px; background-color:{color}; color:white; text-align:center; margin:10px 0;">
                        <h3 style="margin:0;">Status: {category}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning(f"Não foi possível obter dados para {param_name}")
            
            except Exception as e:
                st.error(f"Erro ao analisar {param_name}: {str(e)}")
    
    # Após obter dados de todos os parâmetros, gerar relatório consolidado
    if all_data:
        st.subheader("📑 Relatório Consolidado")
        
        # Criar um dataframe consolidado
        consolidated_df = None
        
        for param_name, df in all_data.items():
            if consolidated_df is None:
                consolidated_df = df.rename(columns={'value': param_name})
            else:
                # Juntar em um único dataframe
                param_df = df.rename(columns={'value': param_name})
                consolidated_df = pd.merge_asof(consolidated_df, param_df, on='time', direction='nearest')
        
        if consolidated_df is not None:
            # Mostrar dados consolidados
            st.write("### Dados Consolidados")
            st.dataframe(consolidated_df.set_index('time'))
            
            # Gráfico de correlação
            if len(all_data) > 1:
                st.write("### Matriz de Correlação")
                
                # Calcular correlação
                corr_matrix = consolidated_df.drop(columns=['time']).corr()
                
                # Plotar matriz de correlação
                fig, ax = plt.subplots(figsize=(10, 8))
                cax = ax.matshow(corr_matrix, cmap='coolwarm')
                fig.colorbar(cax)
                
                # Configurar rótulos
                tick_labels = list(corr_matrix.columns)
                ax.set_xticks(range(len(tick_labels)))
                ax.set_yticks(range(len(tick_labels)))
                ax.set_xticklabels(tick_labels, rotation=45)
                ax.set_yticklabels(tick_labels)
                
                # Adicionar valores na matriz
                for i in range(len(tick_labels)):
                    for j in range(len(tick_labels)):
                        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                                ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Gráfico de linhas comparativo
                st.write("### Comparação Temporal")
                
                # Normalizar dados para comparação visual
                normalized_df = consolidated_df.copy()
                for col in normalized_df.columns:
                    if col != 'time':
                        # Normalizar entre 0 e 1
                        min_val = normalized_df[col].min()
                        max_val = normalized_df[col].max()
                        if max_val > min_val:
                            normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                
                # Plotar gráfico de linhas
                fig, ax = plt.subplots(figsize=(12, 6))
                
                for col in normalized_df.columns:
                    if col != 'time':
                        ax.plot(normalized_df['time'], normalized_df[col], marker='o', linestyle='-', label=col)
                
                ax.set_title(f'Comparação Normalizada dos Parâmetros em {city}')
                ax.set_ylabel('Valor Normalizado (0-1)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Exportar dados
            csv = consolidated_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Baixar Relatório Consolidado (CSV)",
                data=csv,
                file_name=f"relatorio_qualidade_ar_{city}_{start_date}_to_{end_date}.csv",
                mime="text/csv",
            )
        else:
            st.warning("Não foi possível criar um relatório consolidado com os dados disponíveis.")

# Botões de ação na página principal
col1, col2 = st.columns(2)

with col1:
    if st.button("🎞️ Analisar Parâmetro Específico", type="primary"):
        # Executar análise do parâmetro selecionado
        results = generate_air_quality_analysis()
        
        if results:
            # Layout de duas colunas
            col_a, col_b = st.columns([3, 2])
            
            with col_a:
                st.subheader(f"🎬 Animação de {parameter}")
                st.image(results['animation'], caption=f"{parameter} em {city} ({start_date} a {end_date})")
                
                # Adicionar opções para baixar
                with open(results['animation'], "rb") as file:
                    btn = st.download_button(
                        label=f"⬇️ Baixar Animação (GIF)",
                        data=file,
                        file_name=f"{parameter}_{city}_{start_date}_to_{end_date}.gif",
                        mime="image/gif"
                    )
            
            with col_b:
                st.subheader(f"📊 Série Temporal e Previsão de {parameter}")
                
                # Preparar dados para gráfico
                df_combined = results['forecast']
                parameter_info = results['parameter_info']
                
                # Criar gráfico
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Dados históricos
                hist_data = df_combined[df_combined['type'] == 'historical']
                ax.plot(hist_data['time'], hist_data['value'], 
                       marker='o', linestyle='-', color='blue', label='Observado')
                
                # Dados de previsão
                forecast_data = df_combined[df_combined['type'] == 'forecast']
                ax.plot(forecast_data['time'], forecast_data['value'], 
                       marker='x', linestyle='--', color='red', label='Previsão')
                
                # Formatar eixos
                ax.set_title(f'{parameter} em {city}: Valores Observados e Previstos', fontsize=14)
                ax.set_xlabel('Data/Hora', fontsize=12)
                ax.set_ylabel(f'{parameter} ({parameter_info["unit"]})', fontsize=12)
                
                # Formatar datas no eixo x
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
                plt.xticks(rotation=45)
                
                # Adicionar legenda e grade
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Adicionar faixa de qualidade do ar
                ax.axhspan(0, parameter_info["threshold_good"], alpha=0.2, color='green', label='Boa')
                ax.axhspan(parameter_info["threshold_good"], parameter_info["threshold_moderate"], alpha=0.2, color='yellow', label='Moderada')
                ax.axhspan(parameter_info["threshold_moderate"], parameter_info["threshold_unhealthy"], alpha=0.2, color='orange', label='Insalubre')
                ax.axhspan(parameter_info["threshold_unhealthy"], parameter_info["threshold_unhealthy"]*2, alpha=0.2, color='red', label='Perigosa')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Estatísticas
                st.subheader(f"📈 Estatísticas de {parameter}")
                
                # Calcular estatísticas
                if not hist_data.empty:
                    curr_val = hist_data['value'].iloc[-1]
                    max_val = hist_data['value'].max()
                    mean_val = hist_data['value'].mean()
                    
                    # Categorizar qualidade do ar baseado no parâmetro
                    current_cat, current_color = air_quality_category(curr_val, parameter_info)
                    
                    # Mostrar métricas
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric(f"Atual", f"{curr_val:.2f} {parameter_info['unit']}")
                    col_b.metric(f"Máximo", f"{max_val:.2f} {parameter_info['unit']}")
                    col_c.metric(f"Médio", f"{mean_val:.2f} {parameter_info['unit']}")
                    
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
                        daily_forecast = forecast_data.groupby('date')['value'].mean().reset_index()
                        
                        for i, row in daily_forecast.iterrows():
                            day_cat, day_color = air_quality_category(row['value'], parameter_info)
                            st.markdown(f"""
                            <div style="padding:5px; border-radius:3px; background-color:{day_color}; color:white; margin:2px 0;">
                            <b>{row['date'].strftime('%d/%m/%Y')}:</b> {parameter} médio previsto: {row['value']:.2f} {parameter_info['unit']} - {day_cat}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Exportar dados
                st.subheader("💾 Exportar Dados")
                csv = df_combined.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Baixar Dados (CSV)",
                    data=csv,
                    file_name=f"{parameter}_data_{city}_{start_date}_to_{end_date}.csv",
                    mime="text/csv",
                )

with col2:
    if st.button("📑 Gerar Relatório Completo", type="primary"):
        # Gerar relatório completo com todos os parâmetros
        generate_complete_report()

# Função para gerar mapas de comparação entre cidades
def compare_cities():
    st.subheader("🗺️ Comparação entre Municípios")
    
    # Selecionar cidades para comparação
    cities_to_compare = st.multiselect("Selecione municípios para comparar", 
                                      list(cities.keys()), 
                                      default=[city])
    
    if len(cities_to_compare) < 1:
        st.warning("Selecione pelo menos um município para análise.")
        return
    
    # Selecionar parâmetro
    param_to_compare = st.selectbox("Parâmetro para comparação", 
                                 list(air_quality_parameters.keys()))
    
    param_info = air_quality_parameters[param_to_compare]
    var_name = param_info["variable"]
    
    # Preparar para coleta de dados
    comparison_data = {}
    
    with st.spinner(f"Coletando dados de {param_to_compare} para {len(cities_to_compare)} municípios..."):
        for city_name in cities_to_compare:
            city_lat, city_lon = cities[city_name]
            
            # Format dates and times
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Preparar request para API
            request = {
                'variable': [var_name],
                'date': f'{start_date_str}/{end_date_str}',
                'time': ['00:00', '12:00'],  # Simplificado para a comparação
                'leadtime_hour': ['0'],
                'type': ['forecast'],
                'format': 'netcdf',
                'area': [city_lat + 1, city_lon - 1, city_lat - 1, city_lon + 1]  # Área pequena em torno da cidade
            }
            
            filename = f'{var_name}_{city_name}_compare.nc'
            
            try:
                # Baixar dados
                client.retrieve("cams-global-atmospheric-composition-forecasts", request).download(filename)
                ds = xr.open_dataset(filename)
                
                # Extrair série temporal
                df = extract_point_timeseries(ds, city_lat, city_lon, var_name=var_name)
                
                if not df.empty:
                    comparison_data[city_name] = df
                    
            except Exception as e:
                st.error(f"Erro ao obter dados para {city_name}: {str(e)}")
    
    # Se temos dados coletados, criar visualizações
    if comparison_data:
        # Criar dataframe consolidado
        all_cities_df = pd.DataFrame(columns=['time'])
        
        for city_name, df in comparison_data.items():
            city_df = df.copy()
            city_df = city_df.rename(columns={'value': city_name})
            
            if all_cities_df.empty:
                all_cities_df = city_df[['time', city_name]]
            else:
                all_cities_df = pd.merge_asof(all_cities_df, city_df[['time', city_name]], 
                                          on='time', direction='nearest')
        
        # Gráfico de comparação
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for city_name in comparison_data.keys():
            ax.plot(all_cities_df['time'], all_cities_df[city_name], 
                   marker='o', linestyle='-', label=city_name)
        
        # Formatar gráfico
        ax.set_title(f'Comparação de {param_to_compare} entre Municípios', fontsize=16)
        ax.set_xlabel('Data/Hora', fontsize=12)
        ax.set_ylabel(f'{param_to_compare} ({param_info["unit"]})', fontsize=12)
        
        # Adicionar faixas de qualidade do ar
        ax.axhspan(0, param_info["threshold_good"], alpha=0.2, color='green', label='Boa')
        ax.axhspan(param_info["threshold_good"], param_info["threshold_moderate"], alpha=0.2, color='yellow', label='Moderada')
        ax.axhspan(param_info["threshold_moderate"], param_info["threshold_unhealthy"], alpha=0.2, color='orange', label='Insalubre')
        ax.axhspan(param_info["threshold_unhealthy"], param_info["threshold_unhealthy"]*2, alpha=0.2, color='red', label='Perigosa')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Tabela de valores médios
        st.subheader("Valores Médios por Município")
        
        mean_values = {}
        current_values = {}
        max_values = {}
        
        for city_name in comparison_data.keys():
            mean_values[city_name] = all_cities_df[city_name].mean()
            current_values[city_name] = all_cities_df[city_name].iloc[-1] if len(all_cities_df) > 0 else 0
            max_values[city_name] = all_cities_df[city_name].max()
        
        # Criar dataframe para a tabela
        table_data = pd.DataFrame({
            'Município': list(comparison_data.keys()),
            f'Valor Atual ({param_info["unit"]})': [current_values[city] for city in comparison_data.keys()],
            f'Média ({param_info["unit"]})': [mean_values[city] for city in comparison_data.keys()],
            f'Máximo ({param_info["unit"]})': [max_values[city] for city in comparison_data.keys()],
            'Classificação': [air_quality_category(current_values[city], param_info)[0] for city in comparison_data.keys()]
        })
        
        # Ordenar por valor atual
        table_data = table_data.sort_values(f'Valor Atual ({param_info["unit"]})', ascending=False)
        
        # Exibir tabela
        st.dataframe(table_data)
        
        # Mostrar no mapa
        st.subheader("Mapa de Calor da Qualidade do Ar")
        
        # Criar mapa com as cidades e valores atuais
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
        ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
        
        # Adicionar pontos para cada cidade
        for city_name, df in comparison_data.items():
            city_lat, city_lon = cities[city_name]
            value = current_values[city_name]
            category, color = air_quality_category(value, param_info)
            
            # Tamanho do marcador proporcional ao valor
            size = 100 + (value / param_info["threshold_unhealthy"]) * 200
            size = min(500, max(100, size))  # Limitar tamanho
            
            ax.scatter(city_lon, city_lat, s=size, c=color, alpha=0.7, 
                      edgecolor='black', linewidth=1, transform=ccrs.PlateCarree())
            
            # Adicionar nome da cidade
            ax.text(city_lon, city_lat, f"{city_name}\n{value:.2f}", fontsize=9, 
                   ha='center', va='center', transform=ccrs.PlateCarree(),
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # Adicionar grid
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Zoom para mostrar todas as cidades
        city_lats = [cities[city][0] for city in cities_to_compare]
        city_lons = [cities[city][1] for city in cities_to_compare]
        
        lat_min, lat_max = min(city_lats) - 1, max(city_lats) + 1
        lon_min, lon_max = min(city_lons) - 1, max(city_lons) + 1
        
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Título
        ax.set_title(f'Mapa de {param_to_compare} em Municípios de MS', fontsize=16)
        
        # Adicionar legenda
        handles = []
        for desc, color in [('Boa', 'green'), ('Moderada', 'yellow'), 
                            ('Insalubre', 'orange'), ('Perigosa', 'red')]:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=color, markersize=10, label=desc))
        
        ax.legend(handles=handles, loc='lower left')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Exportar dados de comparação
        csv = all_cities_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Baixar Dados de Comparação (CSV)",
            data=csv,
            file_name=f"comparacao_{param_to_compare}_municipios.csv",
            mime="text/csv",
        )
    else:
        st.warning("Não foi possível obter dados para comparação.")

# Adicionar opção para comparar cidades
with st.expander("🔍 Comparar Municípios"):
    compare_cities()

# Adicionar informações na parte inferior
st.markdown("---")
st.markdown("""
### ℹ️ Sobre os dados
- **Fonte**: Copernicus Atmosphere Monitoring Service (CAMS)
- **Resolução temporal**: 3 horas
- **Atualização**: Diária

### 📖 Guia de interpretação:
- **Qualidade do Ar - Boa**: 
  - CO: < 4.4 ppm
  - NO₂: < 53 ppb
  - O₃: < 54 ppb
  - SO₂: < 35 ppb
  - AOD: < 0.1
- **Qualidade do Ar - Moderada**: 
  - CO: 4.4-9.4 ppm
  - NO₂: 53-100 ppb
  - O₃: 54-70 ppb
  - SO₂: 35-75 ppb
  - AOD: 0.1-0.2
- **Qualidade do Ar - Insalubre para grupos sensíveis**: 
  - CO: 9.4-12.4 ppm
  - NO₂: 100-360 ppb
  - O₃: 70-85 ppb
  - SO₂: 75-185 ppb
  - AOD: 0.2-0.5
- **Qualidade do Ar - Perigosa**: 
  - CO: > 12.4 ppm
  - NO₂: > 360 ppb
  - O₃: > 85 ppb
  - SO₂: > 185 ppb
  - AOD: > 0.5

Desenvolvido para monitoramento da qualidade do ar no estado de Mato Grosso do Sul - Brasil.
""")
