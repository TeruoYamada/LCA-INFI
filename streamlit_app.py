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

# Configuração inicial da página
st.set_page_config(layout="wide", page_title="Monitor PM2.5/PM10 - MS", page_icon="🌍")

# ✅ Carregar autenticação a partir do secrets.toml
try:
    ads_url = st.secrets["ads"]["url"]
    ads_key = st.secrets["ads"]["key"]
    client = cdsapi.Client(url=ads_url, key=ads_key)
except Exception as e:
    st.error("❌ Erro ao carregar as credenciais do CDS API. Verifique seu secrets.toml.")
    st.stop()

# Dicionário de coordenadas dos municípios de MS
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

# Função para criar shapefile simplificado
def create_fallback_shapefile():
    """Cria um shapefile simplificado caso o oficial falhe"""
    from shapely.geometry import Polygon
    
    municipalities_data = []
    for city_name, (lat, lon) in cities.items():
        # Criar um polígono aproximado (quadrado) ao redor de cada cidade
        buffer_size = 0.15  # Aproximadamente 15km
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

# Função para carregar shapefiles dos municípios
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

# Função para calcular IQA (Índice de Qualidade do Ar)
def calculate_aqi(pm25, pm10):
    """
    Calcula o Índice de Qualidade do Ar baseado em PM2.5 e PM10.
    Usa os padrões da EPA adaptados para o Brasil.
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

# Função para extrair série temporal de PM2.5 e PM10
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
                    
                    # Converter de kg/m³ para μg/m³ se necessário
                    if pm25_val < 1:  # Provavelmente em kg/m³
                        pm25_val *= 1e9  # kg/m³ para μg/m³
                        pm10_val *= 1e9
                    
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
                
                # Converter unidades se necessário
                if pm25_val < 1:
                    pm25_val *= 1e9
                    pm10_val *= 1e9
                
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
        aqi_values = df.apply(lambda row: calculate_aqi(row['pm25'], row['pm10']), axis=1)
        df['aqi'] = aqi_values.apply(lambda x: x[0])
        df['aqi_category'] = aqi_values.apply(lambda x: x[1])
        df['aqi_color'] = aqi_values.apply(lambda x: x[2])
        
        return df
    else:
        return pd.DataFrame(columns=['time', 'pm25', 'pm10', 'aqi', 'aqi_category'])

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

# Função para criar mapa com contexto estadual
def create_enhanced_map_with_context(ds, pm25_var, city, lat_center, lon_center, ms_shapes, frame_idx=0, figsize=(14, 8)):
    """Cria um mapa com contexto estadual e foco no município selecionado."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.1)
    
    # Mapa de contexto: Mato Grosso do Sul
    ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax0.set_extent([-58.5, -50.5, -24.5, -17.0], crs=ccrs.PlateCarree())
    
    # Adicionar características do mapa
    ax0.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.8, edgecolor='gray')
    ax0.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax0.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5)
    
    # Adicionar shapefile do MS
    if ms_shapes is not None and not ms_shapes.empty:
        ms_shapes.boundary.plot(ax=ax0, color='gray', linewidth=0.5, transform=ccrs.PlateCarree())
        
        # Destacar o município selecionado
        selected = ms_shapes[ms_shapes['NM_MUN'].str.upper() == city.upper()]
        if not selected.empty:
            selected.plot(ax=ax0, facecolor='red', edgecolor='darkred', linewidth=2.0, 
                         alpha=0.5, transform=ccrs.PlateCarree())
    
    ax0.set_title('Mato Grosso do Sul - Contexto', fontsize=12, pad=10)
    
    # Mapa detalhado: Município selecionado
    ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    
    # Definir extensão com buffer ao redor do município
    buffer = 0.8
    ax1.set_extent([lon_center - buffer, lon_center + buffer, 
                   lat_center - buffer, lat_center + buffer], crs=ccrs.PlateCarree())
    
    # Adicionar características do mapa
    ax1.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.8, edgecolor='gray')
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5)
    
    # Plotar dados de PM2.5
    try:
        da = ds[pm25_var]
        if 'forecast_reference_time' in da.dims and 'forecast_period' in da.dims:
            frame = da.isel(forecast_reference_time=0, forecast_period=frame_idx).values
            frame_time = pd.to_datetime(ds.forecast_reference_time.values[0]) + pd.to_timedelta(
                ds.forecast_period.values[frame_idx], unit='h')
        else:
            time_dims = [d for d in da.dims if 'time' in d or 'forecast' in d]
            td = time_dims[0] if time_dims else da.dims[0]
            frame = da.isel({td: frame_idx}).values
            frame_time = pd.to_datetime(da[td].values[frame_idx])
        
        # Converter unidades se necessário
        if np.nanmax(frame) < 1:
            frame = frame * 1e9  # Converter de kg/m³ para μg/m³
        
        vmin, vmax = 0, np.nanpercentile(frame, 95)
        im = ax1.pcolormesh(ds.longitude, ds.latitude, frame, cmap='YlOrRd',
                           vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), alpha=0.8)
    except Exception as e:
        st.warning(f"Erro ao plotar dados de PM2.5: {str(e)}")
        im = None
    
    # Adicionar contorno do município selecionado
    if ms_shapes is not None and not ms_shapes.empty:
        selected = ms_shapes[ms_shapes['NM_MUN'].str.upper() == city.upper()]
        if not selected.empty:
            selected.boundary.plot(ax=ax1, color='red', linewidth=3.0, transform=ccrs.PlateCarree())
    
    # Adicionar ponto central do município
    ax1.plot(lon_center, lat_center, marker='o', markersize=10,
             markeredgecolor='white', markerfacecolor='blue', transform=ccrs.PlateCarree())
    
    # Adicionar grid e labels
    gl = ax1.gridlines(draw_labels=True, linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    
    # Título principal
    fig.suptitle(f"Monitoramento de PM2.5 - {city.upper()}", fontsize=16, fontweight='bold', y=0.95)
    
    # Subtítulo com data/hora
    if 'frame_time' in locals():
        ax1.set_title(f"{frame_time.strftime('%d/%m/%Y %H:%M UTC')}", fontsize=10)
    
    # Adicionar barra de cores
    if im is not None:
        cax = fig.add_axes([0.82, 0.15, 0.02, 0.3])
        cb = plt.colorbar(im, cax=cax)
        cb.set_label('PM2.5 (μg/m³)', fontsize=10)
    
    # Adicionar legenda informativa
    ax0.text(0.5, -0.1, "Mapa de contexto mostrando a localização\ndo município selecionado em Mato Grosso do Sul", 
             transform=ax0.transAxes, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig

# Função principal para análise de PM2.5 e PM10
def generate_pm_analysis():
    dataset = "cams-global-atmospheric-composition-forecasts"
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Gerar lista de horas para a requisição
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
    
    # Área de interesse centrada no município com buffer
    buffer = 1.5
    city_bounds = {
        'north': lat_center + buffer,
        'south': lat_center - buffer,
        'east': lon_center + buffer,
        'west': lon_center - buffer
    }
    
    # Requisição com PM2.5 e PM10 diretos do CAMS
    request = {
        'variable': [
            'particulate_matter_2.5um',  # PM2.5
            'particulate_matter_10um'    # PM10
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
        
        # Criar mapa estático com contexto estadual
        with st.spinner('🗺️ Criando mapa contextualizado...'):
            enhanced_map_fig = create_enhanced_map_with_context(
                ds, pm25_var, city, lat_center, lon_center, ms_shapes, frame_idx=0
            )
            
            # Salvar mapa estático
            static_map_filename = f'Enhanced_Map_{city}_{start_date}.png'
            enhanced_map_fig.savefig(static_map_filename, dpi=300, bbox_inches='tight')
            plt.close(enhanced_map_fig)
        
        # Criar animação para PM2.5
        da_pm25 = ds[pm25_var]
        
        # Converter unidades se necessário
        if da_pm25.max().values < 1:  # Provavelmente em kg/m³
            da_pm25 = da_pm25 * 1e9  # Converter para μg/m³
        
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
        
        vmin, vmax = float(da_pm25.min().values), float(da_pm25.max().values)
        vmin = max(0, vmin - 5)
        vmax = min(200, vmax + 10)
        
        # Criar figura para animação PM2.5
        fig = plt.figure(figsize=(12, 10))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Adicionar features do mapa
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.8, edgecolor='gray')
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5)
        
        # Grid
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Definir extensão do mapa
        ax.set_extent([city_bounds['west'], city_bounds['east'], 
                      city_bounds['south'], city_bounds['north']], 
                     crs=ccrs.PlateCarree())
        
        # Adicionar contorno do município
        if ms_shapes is not None and not ms_shapes.empty:
            selected = ms_shapes[ms_shapes['NM_MUN'].str.upper() == city.upper()]
            if not selected.empty:
                selected.boundary.plot(ax=ax, color='red', linewidth=2.5, transform=ccrs.PlateCarree())
        
        # Marcar o município selecionado
        ax.plot(lon_center, lat_center, 'ro', markersize=10, transform=ccrs.PlateCarree(), 
                markeredgecolor='white', markeredgewidth=2)
        
        # Obter primeiro frame
        if 'forecast_period' in da_pm25.dims and 'forecast_reference_time' in da_pm25.dims:
            first_frame_data = da_pm25.isel(forecast_period=0, forecast_reference_time=0).values
            first_frame_time = pd.to_datetime(ds.forecast_reference_time.values[0])
        else:
            first_frame_data = da_pm25.isel({time_dim: 0}).values
            first_frame_time = pd.to_datetime(da_pm25[time_dim].values[0])
        
        # Criar mapa de cores
        im = ax.pcolormesh(ds.longitude, ds.latitude, first_frame_data, 
                         cmap='YlOrRd', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        
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
                    fp_idx = min(i, len(da_pm25.forecast_period)-1)
                    frt_idx = min(0, len(da_pm25.forecast_reference_time)-1)
                    
                    frame_data = da_pm25.isel(forecast_period=fp_idx, forecast_reference_time=frt_idx).values
                    frame_time = pd.to_datetime(ds.forecast_reference_time.values[frt_idx]) + pd.to_timedelta(
                        da_pm25.forecast_period.values[fp_idx], unit='h')
                else:
                    t_idx = min(i, len(da_pm25[time_dim])-1)
                    frame_data = da_pm25.isel({time_dim: t_idx}).values
                    frame_time = pd.to_datetime(da_pm25[time_dim].values[t_idx])
                
                im.set_array(frame_data.ravel())
                title.set_text(f'PM2.5 - {city}\n{frame_time.strftime("%d/%m/%Y %H:%M UTC")}')
                
                return [im, title]
            except Exception as e:
                return [im, title]
        
        # Limitar número de frames
        actual_frames = min(frames, 24)
        
        # Criar animação
        ani = animation.FuncAnimation(fig, animate, frames=actual_frames, 
                                     interval=animation_speed, blit=True)
        
        # Salvar animação
        gif_filename = f'PM25_{city}_{start_date}_to_{end_date}.gif'
        
        with st.spinner('💾 Salvando animação...'):
            ani.save(gif_filename, writer=animation.PillowWriter(fps=2))
        
        plt.close(fig)
        
        return {
            'animation': gif_filename,
            'static_map': static_map_filename,
            'timeseries': df_timeseries,
            'forecast': df_forecast,
            'dataset': ds,
            'pm25_var': pm25_var,
            'pm10_var': pm10_var
        }
    
    except Exception as e:
        st.error(f"❌ Erro ao processar os dados: {str(e)}")
        return None

# Títulos e introdução
st.title("🌍 Monitoramento PM2.5 e PM10 - Mato Grosso do Sul")
st.markdown("""
### Sistema Integrado de Monitoramento da Qualidade do Ar

Este aplicativo monitora diretamente as concentrações de Material Particulado (PM2.5 e PM10) 
para todos os municípios de Mato Grosso do Sul usando dados diretos do CAMS.

**Características desta versão:**
- 📊 Dados diretos de PM2.5 e PM10 do CAMS (sem conversão de AOD)
- 🎯 Visualização centralizada no município selecionado
- 📈 Índice de Qualidade do Ar (IQA) calculado
- 🔮 Previsões para os próximos 5 dias
- 🗺️ Mapas com contexto estadual e foco municipal
""")

# Carregar shapefiles dos municípios
with st.spinner("Carregando shapes dos municípios..."):
    ms_shapes = load_ms_municipalities()

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Seleção de cidade
available_cities = sorted(list(cities.keys()))
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

# Informações sobre dados diretos
st.sidebar.info("📊 **Dados Diretos CAMS**\nEste sistema utiliza concentrações de PM2.5 e PM10 medidas diretamente pelos sensores do CAMS, sem conversão de AOD.")

# Botão principal
st.markdown("### 🚀 Iniciar Análise Completa")
st.markdown(f"Clique no botão abaixo para gerar análise de PM2.5 e PM10 centralizada em **{city}**.")

if st.button("🎯 Gerar Análise de Qualidade do Ar", type="primary", use_container_width=True):
    try:
        results = generate_pm_analysis()
        
        if results:
            # Criar abas
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Análise do Município", 
                "⚠️ Alerta de Qualidade do Ar", 
                f"🗺️ Mapa de {city}",
                "📈 Análise Detalhada PM"
            ])
            
            # Aba do Mapa
            with tab3:
                st.subheader(f"🗺️ Mapa Contextualizado - {city}")
                st.image(results['static_map'], caption=f"Distribuição de PM2.5 em {city} com contexto estadual")
                
                st.subheader(f"🎬 Animação PM2.5 - {city}")
                st.image(results['animation'], caption=f"Evolução temporal do PM2.5 em {city} ({start_date} a {end_date})")
                
                col1, col2 = st.columns(2)
                with col1:
                    with open(results['static_map'], "rb") as file:
                        btn = st.download_button(
                            label="⬇️ Baixar Mapa Estático (PNG)",
                            data=file,
                            file_name=f"Mapa_PM25_{city}_{start_date}.png",
                            mime="image/png"
                        )
                with col2:
                    with open(results['animation'], "rb") as file:
                        btn = st.download_button(
                            label="⬇️ Baixar Animação (GIF)",
                            data=file,
                            file_name=f"Animacao_PM25_{city}_{start_date}_to_{end_date}.gif",
                            mime="image/gif"
                        )
                
                # Informações sobre o mapa
                with st.expander("ℹ️ Informações Técnicas dos Mapas"):
                    st.markdown("""
                    ### Metodologia de Mapeamento
                    
                    **Dados Utilizados:**
                    - Fonte: CAMS (Copernicus Atmosphere Monitoring Service)
                    - Variável: particulate_matter_2.5um (PM2.5 direta)
                    - Resolução espacial: ~0.4° x 0.4° (~44 km)
                    - Resolução temporal: 3 horas
                    
                    **Shapefiles:**
                    - Municípios de MS: IBGE 2022
                    - Projeção: PlateCarree (Geographic)
                    - Sistema de coordenadas: EPSG:4326
                    
                    **Visualização:**
                    - Mapa estadual: Localização contextual
                    - Mapa local: Concentrações detalhadas
                    - Escala de cores: YlOrRd (Amarelo-Laranja-Vermelho)
                    - Níveis de qualidade baseados em padrões EPA/OMS
                    
                    **Interpretação:**
                    - 🟢 Verde: PM2.5 < 12 μg/m³ (Boa qualidade)
                    - 🟡 Amarelo: PM2.5 12-35 μg/m³ (Moderada)
                    - 🟠 Laranja: PM2.5 35-55 μg/m³ (Insalubre para sensíveis)
                    - 🔴 Vermelho: PM2.5 > 55 μg/m³ (Insalubre)
                    """)
            
            # Aba de Análise do Município
            with tab1:
                st.subheader(f"📊 Análise Detalhada - {city}")
                
                df_combined = results['forecast']
                
                # Separar dados históricos e previsão
                hist_data = df_combined[df_combined['type'] == 'historical']
                forecast_data = df_combined[df_combined['type'] == 'forecast']
                
                if not hist_data.empty:
                    # Estatísticas atuais
                    curr_pm25 = hist_data['pm25'].iloc[-1]
                    curr_pm10 = hist_data['pm10'].iloc[-1]
                    curr_aqi = hist_data['aqi'].iloc[-1]
                    curr_category = hist_data['aqi_category'].iloc[-1]
                    curr_color = hist_data['aqi_color'].iloc[-1]
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("PM2.5 Atual", f"{curr_pm25:.1f} μg/m³")
                    col2.metric("PM10 Atual", f"{curr_pm10:.1f} μg/m³")
                    col3.metric("IQA Atual", f"{curr_aqi:.0f}", curr_category)
                    
                    # Categoria de qualidade
                    st.markdown(f"""
                    <div style="padding:15px; border-radius:10px; background-color:{curr_color}; 
                    color:white; text-align:center; margin:10px 0;">
                    <h3 style="margin:0;">Qualidade do Ar Atual</h3>
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
                
                # Gráficos de série temporal
                if not df_combined.empty:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                    
                    # Gráfico PM2.5
                    ax1.plot(hist_data['time'], hist_data['pm25'], 
                           'o-', color='darkblue', label='PM2.5 Observado', markersize=4)
                    ax1.plot(forecast_data['time'], forecast_data['pm25'], 
                           'x--', color='red', label='PM2.5 Previsto', markersize=4)
                    ax1.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Limite OMS (25 μg/m³)')
                    ax1.set_ylabel('PM2.5 (μg/m³)', fontsize=12)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    ax1.set_title('Material Particulado PM2.5', fontsize=14)
                    
                    # Gráfico PM10
                    ax2.plot(hist_data['time'], hist_data['pm10'], 
                           'o-', color='brown', label='PM10 Observado', markersize=4)
                    ax2.plot(forecast_data['time'], forecast_data['pm10'], 
                           'x--', color='darkred', label='PM10 Previsto', markersize=4)
                    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Limite OMS (50 μg/m³)')
                    ax2.set_ylabel('PM10 (μg/m³)', fontsize=12)
                    ax2.set_xlabel('Data/Hora', fontsize=12)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.set_title('Material Particulado PM10', fontsize=14)
                    
                    # Formatar datas
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
                    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Exportar dados
                    csv = df_combined.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Baixar Dados Completos (CSV)",
                        data=csv,
                        file_name=f"PM_data_{city}_{start_date}_to_{end_date}.csv",
                        mime="text/csv",
                    )
            
# Aba de Alertas
with tab2:
    st.subheader("⚠️ Sistema de Alerta de Qualidade do Ar")
    
    # Análise da situação atual
    if not df_combined.empty:
        current_data = df_combined[df_combined['type'] == 'historical']
        if not current_data.empty:
            latest_data = current_data.iloc[-1]
            
            # Determinar nível de alerta
            alert_level = "BAIXO"
            alert_color = "green"
            recommendations = []
            
            if latest_data['aqi'] > 100:
                alert_level = "MODERADO"
                alert_color = "orange"
                recommendations.append("Pessoas sensíveis devem reduzir atividades ao ar livre")
            if latest_data['aqi'] > 150:
                alert_level = "ALTO"
                alert_color = "red"
                recommendations.append("Toda a população deve evitar atividades ao ar livre prolongadas")
            if latest_data['aqi'] > 200:
                alert_level = "MUITO ALTO"
                alert_color = "purple"
                recommendations.append("Evitar qualquer atividade ao ar livre")
            
            # Display do alerta
            st.markdown(f"""
            <div style="padding:20px; border-radius:10px; background-color:{alert_color}; 
            color:white; text-align:center; margin:10px 0;">
            <h2 style="margin:0;">NÍVEL DE ALERTA: {alert_level}</h2>
            <h3 style="margin:5px 0;">IQA: {latest_data['aqi']:.0f} - {latest_data['aqi_category']}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Detalhes do alerta
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Concentrações Atuais")
                st.metric("PM2.5", f"{latest_data['pm25']:.1f} μg/m³")
                st.metric("PM10", f"{latest_data['pm10']:.1f} μg/m³")
                
                # Comparação com padrões OMS
                st.subheader("📏 Comparação com Padrões OMS")
                pm25_status = "✅ Dentro" if latest_data['pm25'] <= 25 else "⚠️ Acima"
                pm10_status = "✅ Dentro" if latest_data['pm10'] <= 50 else "⚠️ Acima"
                
                st.write(f"**PM2.5**: {pm25_status} do limite (25 μg/m³)")
                st.write(f"**PM10**: {pm10_status} do limite (50 μg/m³)")
            
            with col2:
                st.subheader("📈 Tendência")
                if len(current_data) > 1:
                    pm25_trend = current_data['pm25'].iloc[-1] - current_data['pm25'].iloc[-2]
                    pm10_trend = current_data['pm10'].iloc[-1] - current_data['pm10'].iloc[-2]
                    
                    trend_icon_25 = "↗️" if pm25_trend > 0 else "↘️" if pm25_trend < 0 else "➡️"
                    trend_icon_10 = "↗️" if pm10_trend > 0 else "↘️" if pm10_trend < 0 else "➡️"
                    
                    st.metric("Tendência PM2.5", f"{pm25_trend:+.1f} μg/m³", delta=None, 
                             help=f"{trend_icon_25} Variação desde a última medição")
                    st.metric("Tendência PM10", f"{pm10_trend:+.1f} μg/m³", delta=None,
                             help=f"{trend_icon_10} Variação desde a última medição")
                
                st.subheader("⏰ Última Atualização")
                st.write(latest_data['time'].strftime("%d/%m/%Y %H:%M"))
            
            # Recomendações
            st.subheader("💡 Recomendações")
            if not recommendations:
                st.success("✅ Condições favoráveis. Mantenha atividades normais.")
            else:
                for rec in recommendations:
                    st.warning(rec)
            
            # Informações adicionais para grupos sensíveis
            if latest_data['aqi'] > 100:
                st.info("""
                **Grupos sensíveis** incluem:
                - Crianças e idosos
                - Pessoas com doenças respiratórias ou cardíacas
                - Gestantes
                - Indivíduos que praticam atividades ao ar livre
                """)
        
        # Previsão de alertas futuros
        if not forecast_data.empty:
            st.subheader("🔮 Previsão de Alertas para os Próximos Dias")
            
            # Agrupar por dia
            forecast_data['date'] = forecast_data['time'].dt.date
            daily_forecast = forecast_data.groupby('date').agg({
                'aqi': 'max',
                'aqi_category': lambda x: x.iloc[x.values.argmax()],
                'pm25': 'max',
                'pm10': 'max'
            }).reset_index()
            
            for _, row in daily_forecast.iterrows():
                aqi_color = 'green' if row['aqi'] <= 50 else 'yellow' if row['aqi'] <= 100 else 'orange' if row['aqi'] <= 150 else 'red'
                st.markdown(f"""
                <div style="padding:10px; border-radius:5px; background-color:{aqi_color}; 
                color:white; margin:5px 0;">
                <b>{row['date'].strftime('%d/%m')}:</b> IQA {row['aqi']:.0f} - {row['aqi_category']}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Dados insuficientes para gerar alertas.")

# Aba de análise detalhada
with tab4:
    st.subheader("📈 Análise Detalhada de Material Particulado")
    
    if not df_combined.empty:
        # Separar dados históricos e previsões
        hist_data = df_combined[df_combined['type'] == 'historical']
        forecast_data = df_combined[df_combined['type'] == 'forecast']
        
        # Estatísticas descritivas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Estatísticas Descritivas - PM2.5")
            if not hist_data.empty:
                pm25_stats = hist_data['pm25'].describe()
                st.write(f"Média: {pm25_stats['mean']:.2f} μg/m³")
                st.write(f"Mediana: {pm25_stats['50%']:.2f} μg/m³")
                st.write(f"Máximo: {pm25_stats['max']:.2f} μg/m³")
                st.write(f"Mínimo: {pm25_stats['min']:.2f} μg/m³")
                st.write(f"Desvio Padrão: {pm25_stats['std']:.2f} μg/m³")
        
        with col2:
            st.subheader("📊 Estatísticas Descritivas - PM10")
            if not hist_data.empty:
                pm10_stats = hist_data['pm10'].describe()
                st.write(f"Média: {pm10_stats['mean']:.2f} μg/m³")
                st.write(f"Mediana: {pm10_stats['50%']:.2f} μg/m³")
                st.write(f"Máximo: {pm10_stats['max']:.2f} μg/m³")
                st.write(f"Mínimo: {pm10_stats['min']:.2f} μg/m³")
                st.write(f"Desvio Padrão: {pm10_stats['std']:.2f} μg/m³")
        
        # Gráficos de distribuição
        st.subheader("📈 Distribuição de Concentrações")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        if not hist_data.empty:
            # Histograma PM2.5
            ax1.hist(hist_data['pm25'], bins=15, alpha=0.7, color='darkblue', edgecolor='black')
            ax1.axvline(hist_data['pm25'].mean(), color='red', linestyle='--', label=f'Média: {hist_data["pm25"].mean():.1f}')
            ax1.axvline(25, color='orange', linestyle=':', label='Limite OMS: 25 μg/m³')
            ax1.set_xlabel('PM2.5 (μg/m³)')
            ax1.set_ylabel('Frequência')
            ax1.set_title('Distribuição de PM2.5')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Histograma PM10
            ax2.hist(hist_data['pm10'], bins=15, alpha=0.7, color='brown', edgecolor='black')
            ax2.axvline(hist_data['pm10'].mean(), color='red', linestyle='--', label=f'Média: {hist_data["pm10"].mean():.1f}')
            ax2.axvline(50, color='orange', linestyle=':', label='Limite OMS: 50 μg/m³')
            ax2.set_xlabel('PM10 (μg/m³)')
            ax2.set_ylabel('Frequência')
            ax2.set_title('Distribuição de PM10')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Análise de correlação
        if len(hist_data) > 2:
            st.subheader("🔍 Correlação entre PM2.5 e PM10")
            
            correlation = hist_data['pm25'].corr(hist_data['pm10'])
            st.metric("Coeficiente de Correlação", f"{correlation:.3f}")
            
            # Gráfico de dispersão
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(hist_data['pm25'], hist_data['pm10'], alpha=0.6)
            
            # Linha de tendência
            z = np.polyfit(hist_data['pm25'], hist_data['pm10'], 1)
            p = np.poly1d(z)
            ax.plot(hist_data['pm25'], p(hist_data['pm25']), "r--", alpha=0.8)
            
            ax.set_xlabel('PM2.5 (μg/m³)')
            ax.set_ylabel('PM10 (μg/m³)')
            ax.set_title(f'Correlação PM2.5 vs PM10 (R = {correlation:.3f})')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Análise da razão PM2.5/PM10
            st.subheader("⚖️ Razão PM2.5/PM10")
            
            pm_ratio = hist_data['pm25'] / hist_data['pm10']
            avg_ratio = pm_ratio.mean()
            
            st.metric("Razão Média PM2.5/PM10", f"{avg_ratio:.2f}")
            
            # Interpretação da razão
            if avg_ratio > 0.6:
                st.info("""
                **Interpretação:** Razão alta (>0.6) sugere predominância de fontes antropogênicas:
                - Emissões veiculares
                - Queima de combustíveis fósseis
                - Processos industriais
                """)
            elif avg_ratio > 0.4:
                st.info("""
                **Interpretação:** Razão moderada (0.4-0.6) sugere mistura de fontes:
                - Combinação de fontes naturais e antropogênicas
                - Condições atmosféricas variadas
                """)
            else:
                st.info("""
                **Interpretação:** Razão baixa (<0.4) sugere predominância de fontes naturais:
                - Poeira do solo
                - Partículas de origem marinha
                - Material biológico
                """)
        
        # Análise temporal detalhada
        st.subheader("⏰ Variação Temporal")
        
        if not hist_data.empty:
            hist_data['hour'] = hist_data['time'].dt.hour
            hourly_avg = hist_data.groupby('hour').agg({
                'pm25': 'mean',
                'pm10': 'mean'
            }).reset_index()
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(hourly_avg['hour'], hourly_avg['pm25'], 'o-', label='PM2.5', color='darkblue')
            ax.plot(hourly_avg['hour'], hourly_avg['pm10'], 's-', label='PM10', color='brown')
            
            ax.set_xlabel('Hora do Dia')
            ax.set_ylabel('Concentração Média (μg/m³)')
            ax.set_title('Variação Horária Média das Concentrações')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(0, 24, 3))
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Explicação dos padrões horários
            st.info("""
            **Padrões horários típicos:**
            - **Manhã (6-9h):** Pico devido às emissões veiculares e condições meteorológicas
            - **Meio-dia (12-14h):** Redução devido à dispersão por convecção térmica
            - **Tarde/noite (17-20h):** Segundo pico devido ao tráfego e estabilização atmosférica
            - **Madrugada (0-5h):** Valores geralmente mais baixos
            """)
        
        # Comparação com padrões internacionais
        st.subheader("🌍 Comparação com Padrões Internacionais")
        
        standards_data = {
            'Organização': ['OMS', 'EPA', 'CONAMA', 'UE'],
            'PM2.5 (24h)': [25, 35, 60, 25],
            'PM10 (24h)': [50, 150, 150, 50]
        }
        
        standards_df = pd.DataFrame(standards_data)
        
        if not hist_data.empty:
            # Adicionar colunas de comparação
            latest_pm25 = hist_data['pm25'].iloc[-1]
            latest_pm10 = hist_data['pm10'].iloc[-1]
            
            standards_df['Status PM2.5'] = standards_df['PM2.5 (24h)'].apply(
                lambda x: '✅' if latest_pm25 <= x else '⚠️'
            )
            standards_df['Status PM10'] = standards_df['PM10 (24h)'].apply(
                lambda x: '✅' if latest_pm10 <= x else '⚠️'
            )
        
        st.dataframe(standards_df, use_container_width=True)
        
        # Informações adicionais
        with st.expander("📋 Legenda e Informações Adicionais"):
            st.markdown("""
            **Legenda:**
            - ✅: Dentro do limite
            - ⚠️: Acima do limite
            
            **Notas:**
            - **OMS:** Organização Mundial da Saúde
            - **EPA:** Agência de Proteção Ambiental dos EUA
            - **CONAMA:** Conselho Nacional do Meio Ambiente (Brasil)
            - **UE:** União Europeia
            
            **Observação:** Os limites são para médias de 24 horas. Valores instantâneos podem variar.
            """)
    else:
        st.warning("Dados insuficientes para análise detalhada.")
            
    except Exception as e:
        st.error(f"❌ Erro durante a análise: {str(e)}")

# Rodapé informativo
st.markdown("---")
st.markdown("""
### ℹ️ Informações Importantes

**Sobre os Dados Diretos:**
- As concentrações de PM2.5/PM10 são obtidas diretamente do CAMS, sem conversões
- Dados calibrados e validados continuamente com estações de monitoramento
- Precisão superior aos métodos de conversão de AOD

**Dados Fornecidos por:**
- CAMS (Copernicus Atmosphere Monitoring Service) - União Europeia
- Processamento: Sistema desenvolvido para monitoramento ambiental de MS

**Desenvolvido para:** Monitoramento da Qualidade do Ar em Mato Grosso do Sul

**Aviso:** Este sistema é uma ferramenta de apoio à decisão e não substitui monitoramento oficial.
""")
