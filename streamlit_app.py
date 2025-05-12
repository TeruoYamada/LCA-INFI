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

# Botão para iniciar análise (movido para a sidebar para maior visibilidade)
if st.sidebar.button("🎞️ Gerar Análise Completa", type="primary"):

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
def predict_future_aod(df, days=5):  # Aumentado para 5 dias
    """Gera uma previsão simples de AOD baseada nos dados históricos."""
    if len(df) < 3:  # Precisa de pelo menos 3 pontos para uma previsão mínima
        return pd.DataFrame(columns=['time', 'aod', 'type'])
    
    # Preparar dados para regressão
    df_hist = df.copy()
    df_hist['time_numeric'] = (df_hist['time'] - df_hist['time'].min()).dt.total_seconds()
    
    # Modelo de regressão linear simples
    X = df_hist['time_numeric'].values.reshape(-1, 1)
    y = df_hist['aod'].values
    model = LinearRegression()
    model.fit(X, y)
    
    # Gerar pontos futuros
    last_time = df_hist['time'].max()
    future_times = [last_time + timedelta(hours=i*6) for i in range(1, days*4+1)]  # 4 pontos por dia (6h)
    future_time_numeric = [(t - df_hist['time'].min()).total_seconds() for t in future_times]
    
    # Prever valores
    future_aod = model.predict(np.array(future_time_numeric).reshape(-1, 1))
    
    # Limitar valores previstos (AOD não pode ser negativo)
    future_aod = np.maximum(future_aod, 0)
    
    # Criar DataFrame com previsão
    df_pred = pd.DataFrame({
        'time': future_times,
        'aod': future_aod,
        'type': 'forecast'
    })
    
    # Adicionar indicador aos dados históricos
    df_hist['type'] = 'historical'
    
    # Combinar histórico e previsão
    result = pd.concat([df_hist[['time', 'aod', 'type']], df_pred], ignore_index=True)
    return result

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
