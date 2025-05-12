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

# Carregar autenticação a partir do secrets.toml
try:
    ads_url = st.secrets["ads"]["url"]
    ads_key = st.secrets["ads"]["key"]
    client = cdsapi.Client(url=ads_url, key=ads_key)
except Exception as e:
    st.error("❌ Erro ao carregar as credenciais do CDS API. Verifique seu secrets.toml.")
    st.stop()

# Função para baixar shapefile dos municípios de MS (melhorada)
@st.cache_data
def load_ms_municipalities():
    try:
        # URL para um shapefile de municípios do MS
        url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/" + \
              "municipio_2022/UFs/MS/MS_Municipios_2022.zip"
        
        try:
            gdf = gpd.read_file(url)
            return gdf
        except:
            # Adicionar gráfico de barras comparando as 10 cidades mais poluídas
            st.subheader("📊 Comparação das 10 Cidades mais Poluídas")
            
            # Criar gráfico de barras
            fig, ax = plt.figure(figsize=(12, 6)), plt.subplot(1, 1, 1)
            
            # Limitar para top 10
            top10 = alerts_df.head(10).copy()
            
            # Definir cores baseadas na categoria
            colors = []
            for cat in top10['categoria']:
                if cat == 'Boa':
                    colors.append('green')
                elif cat == 'Moderada':
                    colors.append('yellow')
                elif cat == 'Insalubre':
                    colors.append('orange')
                else:  # Perigosa
                    colors.append('red')
            
            # Criar barras
            bars = ax.bar(top10['cidade'], top10['aod_maximo'], color=colors)
            
            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontweight='bold')
            
            # Estilizar gráfico
            ax.set_title('10 Municípios com Maiores Níveis de AOD Previstos', fontsize=14)
            ax.set_xlabel('Município', fontsize=12)
            ax.set_ylabel('AOD Máximo Previsto', fontsize=12)
            ax.set_ylim(0, max(top10['aod_maximo']) * 1.2)  # Dar espaço para os rótulos
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Mostrar gráfico
            st.pyplot(fig)
            
            # Adicionar exportação de dados
            st.subheader("💾 Exportar Dados")
            
            # Preparar CSV para download
            csv = alerts_df.to_csv(index=False).encode('utf-8')
            
            # Botão para download
            st.download_button(
                label="⬇️ Baixar Tabela Completa (CSV)",
                data=csv,
                file_name=f"AOD_previsao_municipios_MS_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
            
            # Adicionar alertas específicos para cidades em situação perigosa
            dangerous_cities = alerts_df[alerts_df['categoria'] == 'Perigosa']
            
            if not dangerous_cities.empty:
                st.subheader("⚠️ ALERTA DE RISCO!")
                st.markdown(f"""
                <div style="padding:15px; border-radius:5px; background-color:red; color:white; margin:10px 0;">
                <h3>⚠️ {len(dangerous_cities)} municípios com níveis PERIGOSOS de poluição previstos!</h3>
                <p>Os seguintes municípios apresentam previsão de níveis perigosos de AOD (acima de 0.5):</p>
                <ul>
                {"".join([f"<li><b>{row['cidade']}</b>: AOD {row['aod_maximo']:.3f} em {row['data']} às {row['hora']}</li>" for _, row in dangerous_cities.iterrows()])}
                </ul>
                <p><b>Recomendação:</b> Pessoas nestes municípios devem evitar atividades ao ar livre, especialmente 
                indivíduos com condições respiratórias pré-existentes, idosos e crianças.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Adicionar mapa interativo com todos os municípios
            st.subheader("🌐 Mapa Interativo - Previsão por Município")
            
            try:
                # Criar GeoDataFrame para visualização
                ms_shapes = load_ms_municipalities()
                
                if not ms_shapes.empty:
                    # Mesclar dados de previsão com shapes
                    merged_gdf = ms_shapes.merge(alerts_df, left_on='NM_MUN', right_on='cidade', how='left')
                    
                    # Converter para formato adequado para st.map
                    map_data = pd.DataFrame({
                        'latitude': [cities_coords[city][0] for city in alerts_df['cidade'] if city in cities_coords],
                        'longitude': [cities_coords[city][1] for city in alerts_df['cidade'] if city in cities_coords],
                        'aod': alerts_df['aod_maximo'],
                        'city': alerts_df['cidade'],
                        'date': alerts_df['data'],
                        'category': alerts_df['categoria']
                    })
                    
                    # Mostrar mapa
                    st.map(map_data, latitude='latitude', longitude='longitude', color='category', size='aod')
                else:
                    st.warning("Não foi possível carregar os shapes dos municípios para o mapa interativo.")
            
            except Exception as e:
                st.warning(f"Não foi possível criar o mapa interativo: {str(e)}")
             Fallback: criar geodataframe com todos os municípios de MS
            # Lista completa dos 79 municípios de MS com aproximações de coordenadas
            data = {
                'NM_MUN': [
                    'Água Clara', 'Alcinópolis', 'Amambai', 'Anastácio', 'Anaurilândia',
                    'Angélica', 'Antônio João', 'Aparecida do Taboado', 'Aquidauana', 'Aral Moreira',
                    'Bandeirantes', 'Bataguassu', 'Batayporã', 'Bela Vista', 'Bodoquena',
                    'Bonito', 'Brasilândia', 'Caarapó', 'Camapuã', 'Campo Grande',
                    'Caracol', 'Cassilândia', 'Chapadão do Sul', 'Corguinho', 'Coronel Sapucaia',
                    'Corumbá', 'Costa Rica', 'Coxim', 'Deodápolis', 'Dois Irmãos do Buriti',
                    'Douradina', 'Dourados', 'Eldorado', 'Fátima do Sul', 'Figueirão',
                    'Glória de Dourados', 'Guia Lopes da Laguna', 'Iguatemi', 'Inocência', 'Itaporã',
                    'Itaquiraí', 'Ivinhema', 'Japorã', 'Jaraguari', 'Jardim',
                    'Jateí', 'Juti', 'Ladário', 'Laguna Carapã', 'Maracaju',
                    'Miranda', 'Mundo Novo', 'Naviraí', 'Nioaque', 'Nova Alvorada do Sul',
                    'Nova Andradina', 'Novo Horizonte do Sul', 'Paraíso das Águas', 'Paranaíba', 'Paranhos',
                    'Pedro Gomes', 'Ponta Porã', 'Porto Murtinho', 'Ribas do Rio Pardo', 'Rio Brilhante',
                    'Rio Negro', 'Rio Verde de Mato Grosso', 'Rochedo', 'Santa Rita do Pardo', 'São Gabriel do Oeste',
                    'Selvíria', 'Sete Quedas', 'Sidrolândia', 'Sonora', 'Tacuru',
                    'Taquarussu', 'Terenos', 'Três Lagoas', 'Vicentina'
                ],
                'geometry': [
                    # Aqui vão as geometrias simplificadas para cada município
                    # Por simplicidade, estou usando pontos com buffer como aproximação
                    # Na implementação real, deveria usar os polígonos corretos dos municípios
                ]
            }
            
            # Adicionando coordenadas aproximadas para cada município usando buffer
            # Essas são aproximações - em uma implementação real, usaria as coordenadas reais
            coordinates = [
                (-52.8941, -20.4653), (-53.7041, -18.3215), (-55.2246, -23.1022), (-55.8099, -20.4813), (-52.7179, -22.1855),
                (-54.0675, -22.1527), (-55.9507, -22.1923), (-51.0966, -20.0873), (-55.7950, -20.4697), (-55.6384, -22.9346),
                (-54.3585, -19.9091), (-52.4220, -21.7153), (-53.2708, -22.2940), (-56.5261, -22.1064), (-56.7151, -20.5375),
                (-56.4836, -21.1275), (-52.0377, -21.2554), (-54.8224, -22.6365), (-54.0413, -19.5344), (-54.6201, -20.4697),
                (-57.0290, -22.0280), (-51.7348, -19.1119), (-52.6265, -18.7907), (-54.8308, -19.8341), (-55.7407, -23.2624),
                (-57.6510, -19.0082), (-53.1355, -18.5425), (-54.7687, -18.5122), (-54.1663, -22.2827), (-55.2991, -20.7233),
                (-54.6139, -22.0407), (-54.8120, -22.2231), (-54.2834, -23.7938), (-54.5162, -22.3789), (-53.6450, -18.6782),
                (-54.2316, -22.8057), (-55.3945, -21.4521), (-54.5665, -23.6802), (-51.9293, -19.7355), (-54.7941, -22.0771),
                (-54.1867, -23.4692), (-53.8127, -22.3055), (-54.4037, -23.8949), (-54.3991, -20.1391), (-56.1508, -21.4744),
                (-54.3023, -22.4821), (-54.6060, -22.8596), (-57.5967, -19.0087), (-54.6468, -22.5466), (-54.9612, -21.6131),
                (-56.3801, -20.2379), (-54.2775, -23.9415), (-54.1949, -23.0624), (-55.8313, -21.1423), (-54.3839, -21.4649),
                (-53.3419, -22.2320), (-53.8002, -22.6679), (-52.9695, -19.0518), (-51.1866, -20.0008), (-55.4291, -23.8940),
                (-54.5511, -18.0869), (-55.7271, -22.5334), (-57.8835, -21.6983), (-53.7589, -20.4444), (-54.5432, -21.8028),
                (-54.9860, -19.4510), (-54.8382, -18.9188), (-54.8920, -19.9513), (-53.3083, -21.1408), (-54.5608, -19.3944),
                (-51.4188, -20.3672), (-54.7143, -23.9656), (-54.9696, -20.9154), (-54.7548, -17.5683), (-55.0137, -23.6323),
                (-53.3521, -22.4993), (-54.8643, -20.4712), (-51.7005, -20.7849), (-54.4149, -22.4099)
            ]
            
            # Criar as geometrias para cada município
            geometries = [gpd.points_from_xy([lon], [lat])[0].buffer(0.2) for lon, lat in coordinates]
            data['geometry'] = geometries
            
            gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
            return gdf
    except Exception as e:
        st.warning(f"Não foi possível carregar os shapes dos municípios: {str(e)}")
        # Retornar DataFrame vazio com estrutura esperada
        return gpd.GeoDataFrame(columns=['NM_MUN', 'geometry'], crs="EPSG:4326")

# Dicionário com coordenadas de todas as cidades do MS
def get_ms_cities_coordinates():
    # Usar os dados dos municípios para extrair coordenadas
    ms_shapes = load_ms_municipalities()
    
    if ms_shapes.empty:
        # Fallback para coordenadas padrão caso os shapes não estejam disponíveis
        return {
            "Campo Grande": [-20.4697, -54.6201],
            "Dourados": [-22.2231, -54.8120],
            "Três Lagoas": [-20.7849, -51.7005],
            "Corumbá": [-19.0082, -57.6510],
            "Ponta Porã": [-22.5334, -55.7271]
        }
    
    # Criar dicionário de coordenadas a partir do centroide de cada município
    cities = {}
    for _, row in ms_shapes.iterrows():
        try:
            centroid = row.geometry.centroid
            cities[row['NM_MUN']] = [centroid.y, centroid.x]  # latitude, longitude
        except:
            # Em caso de erro, usar o centroide do envelope (bounding box)
            try:
                bbox = row.geometry.bounds
                lat = (bbox[1] + bbox[3]) / 2
                lon = (bbox[0] + bbox[2]) / 2
                cities[row['NM_MUN']] = [lat, lon]
            except:
                # Ignorar se não conseguir extrair coordenadas
                pass
    
    return cities

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
def predict_future_aod(df, days=5):
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

# Nova função para gerar análise de AOD para todas as cidades
def generate_ms_cities_aod_forecast(days=5):
    """Gera previsão de AOD para todas as cidades de MS nos próximos dias."""
    
    # Obter coordenadas de todos os municípios
    cities_coords = get_ms_cities_coordinates()
    
    if not cities_coords:
        st.error("Não foi possível obter coordenadas dos municípios.")
        return None
    
    # Configurar datas para a análise
    start_date = datetime.today() - timedelta(days=2)
    end_date = datetime.today() + timedelta(days=days)
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Definir região para abranger todo o estado de MS
    # Coordenadas aproximadas dos limites de MS
    ms_bounds = {
        "north": -17.0,  # Latitude Norte
        "south": -24.0,  # Latitude Sul
        "west": -58.0,   # Longitude Oeste
        "east": -51.0    # Longitude Leste
    }
    
    # Preparar request para API
    hours = ['00:00', '06:00', '12:00', '18:00']  # Reduzir resolução temporal para economizar tempo
    
    request = {
        'variable': ['total_aerosol_optical_depth_550nm'],
        'date': f'{start_date_str}/{end_date_str}',
        'time': hours,
        'leadtime_hour': ['0', '24', '48', '72', '96', '120'],  # Previsões de até 5 dias
        'type': ['forecast'],
        'format': 'netcdf',
        'area': [ms_bounds["north"], ms_bounds["west"], 
                ms_bounds["south"], ms_bounds["east"]]
    }
    
    filename = f'AOD550_MS_forecast_{start_date.strftime("%Y%m%d")}.nc'
    
    try:
        with st.spinner('📥 Baixando dados do CAMS para todo o estado...'):
            client.retrieve("cams-global-atmospheric-composition-forecasts", request).download(filename)
        
        ds = xr.open_dataset(filename)
        
        # Verificar variáveis disponíveis
        variable_names = list(ds.data_vars)
        aod_var = next((var for var in variable_names if 'aod' in var.lower()), variable_names[0])
        
        # Inicializar dicionário para armazenar resultados
        cities_forecasts = {}
        
        with st.spinner("⏳ Processando previsões para todos os municípios..."):
            progress_bar = st.progress(0)
            
            # Para cada cidade, extrair série temporal e fazer previsão
            for i, (city, coords) in enumerate(cities_coords.items()):
                try:
                    # Atualizar progresso
                    progress = i / len(cities_coords)
                    progress_bar.progress(progress)
                    
                    # Extrair série temporal
                    lat, lon = coords
                    df_timeseries = extract_point_timeseries(ds, lat, lon, var_name=aod_var)
                    
                    if not df_timeseries.empty:
                        # Fazer previsão
                        df_forecast = predict_future_aod(df_timeseries, days=days)
                        
                        # Armazenar resultado
                        cities_forecasts[city] = df_forecast
                except Exception as e:
                    st.warning(f"Erro ao processar {city}: {str(e)}")
            
            # Finalizar barra de progresso
            progress_bar.progress(1.0)
        
        # Processar resultados para gerar tabela de alerta
        pollution_alerts = []
        
        for city, forecast_df in cities_forecasts.items():
            # Filtrar apenas as previsões (não dados históricos)
            future_data = forecast_df[forecast_df['type'] == 'forecast']
            
            if not future_data.empty:
                # Encontrar o valor máximo de AOD previsto
                max_aod = future_data['aod'].max()
                
                # Determinar qual dia terá o maior valor de AOD
                max_aod_idx = future_data['aod'].idxmax()
                max_aod_day = future_data.loc[max_aod_idx, 'time'].strftime('%d/%m/%Y')
                max_aod_time = future_data.loc[max_aod_idx, 'time'].strftime('%H:%M')
                
                # Adicionar à lista de alertas
                pollution_alerts.append({
                    'cidade': city,
                    'aod_maximo': max_aod,
                    'data': max_aod_day,
                    'hora': max_aod_time,
                    'categoria': categorize_aod(max_aod)
                })
        
        # Criar DataFrame com os alertas
        alerts_df = pd.DataFrame(pollution_alerts)
        
        # Ordenar por AOD máximo (decrescente)
        alerts_df = alerts_df.sort_values('aod_maximo', ascending=False).reset_index(drop=True)
        
        return {
            'alerts': alerts_df,
            'dataset': ds,
            'forecasts': cities_forecasts
        }
    
    except Exception as e:
        st.error(f"❌ Erro ao processar os dados: {str(e)}")
        st.write("Detalhes da requisição:")
        st.write(request)
        return None

# Função para categorizar valor de AOD
def categorize_aod(value):
    if value < 0.1:
        return "Boa"
    elif value < 0.2:
        return "Moderada"
    elif value < 0.5:
        return "Insalubre"
    else:
        return "Perigosa"

# Função para colorir célula baseado no valor de AOD
def color_aod_cell(val):
    if val < 0.1:
        return 'background-color: green; color: white'
    elif val < 0.2:
        return 'background-color: yellow; color: black'
    elif val < 0.5:
        return 'background-color: orange; color: black'
    else:
        return 'background-color: red; color: white'

# Função para gerar mapa de calor do estado com previsão de AOD
def generate_state_heatmap(dataset, forecast_day=1):
    """Gera um mapa de calor do estado para um dia específico da previsão."""
    try:
        # Identificar variável de AOD
        variable_names = list(dataset.data_vars)
        aod_var = next((var for var in variable_names if 'aod' in var.lower()), variable_names[0])
        
        # Identificar a estrutura do dataset
        if 'forecast_reference_time' in dataset[aod_var].dims and 'forecast_period' in dataset[aod_var].dims:
            # Para datasets com tempo de referência e períodos de previsão
            # Selecionar o tempo de referência mais recente
            ref_time_idx = -1  # último tempo de referência
            
            # Encontrar o período correspondente ao dia da previsão
            # (assumindo que os períodos são em horas e queremos o dia completo)
            target_period = forecast_day * 24  # converter dias em horas
            
            # Encontrar o período mais próximo
            periods = dataset.forecast_period.values
            period_idx = np.abs(periods - target_period).argmin()
            
            # Selecionar o frame de dados
            data = dataset[aod_var].isel(
                forecast_reference_time=ref_time_idx,
                forecast_period=period_idx
            ).values
            
            # Obter o timestamp real deste frame
            ref_time = pd.to_datetime(dataset.forecast_reference_time.values[ref_time_idx])
            period = dataset.forecast_period.values[period_idx]
            timestamp = ref_time + pd.to_timedelta(period, unit='h')
        else:
            # Para datasets com uma única dimensão temporal
            time_dim = next((dim for dim in dataset[aod_var].dims if 'time' in dim), None)
            
            if time_dim:
                # Calcular índice para o dia da previsão
                # (assumindo que os dados estão ordenados cronologicamente)
                target_idx = min(forecast_day * 4, len(dataset[time_dim]) - 1)  # 4 pontos por dia
                data = dataset[aod_var].isel({time_dim: target_idx}).values
                timestamp = pd.to_datetime(dataset[time_dim].values[target_idx])
            else:
                raise ValueError("Não foi possível identificar a dimensão temporal nos dados")
        
        # Criar figura
        fig = plt.figure(figsize=(12, 10))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Adicionar features básicas
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
        ax.add_feature(cfeature.STATES.with_scale('10m'), linestyle=':')
        
        # Adicionar grid
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Obter limites de MS para ajustar o mapa
        ms_bounds = {
            "north": -17.0,  # Latitude Norte
            "south": -24.0,  # Latitude Sul
            "west": -58.0,   # Longitude Oeste
            "east": -51.0    # Longitude Leste
        }
        
        # Definir extensão do mapa
        ax.set_extent([ms_bounds["west"], ms_bounds["east"], 
                      ms_bounds["south"], ms_bounds["north"]], 
                     crs=ccrs.PlateCarree())
        
        # Determinar range de cores
        vmin, vmax = float(dataset[aod_var].min().values), float(dataset[aod_var].max().values)
        vmin = max(0, vmin - 0.05)
        vmax = min(2, vmax + 0.05)  # AOD geralmente não ultrapassa 2
        
        # Criar mapa de calor
        im = ax.pcolormesh(dataset.longitude, dataset.latitude, data, 
                         cmap='YlOrRd', vmin=vmin, vmax=vmax)
        
        # Adicionar barra de cores
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('AOD 550nm')
        
        # Adicionar título
        day_str = timestamp.strftime('%d/%m/%Y %H:%M')
        ax.set_title(f'Previsão de AOD 550nm para Mato Grosso do Sul - {day_str}', fontsize=14)
        
        # Adicionar borders dos municípios se disponíveis
        ms_shapes = load_ms_municipalities()
        if not ms_shapes.empty:
            ms_shapes.boundary.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.7)
        
        # Adicionar nomes das principais cidades
        cities_coords = get_ms_cities_coordinates()
        cities_to_label = ['Campo Grande', 'Dourados', 'Três Lagoas', 'Corumbá', 'Ponta Porã']
        
        for city in cities_to_label:
            if city in cities_coords:
                lat, lon = cities_coords[city]
                ax.text(lon, lat, city, fontsize=10, fontweight='bold', 
                       ha='center', va='center', color='black',
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                       transform=ccrs.PlateCarree(), zorder=4)
        
        plt.tight_layout()
        
        # Salvar como imagem temporária
        map_filename = f'MS_AOD_map_day{forecast_day}.png'
        plt.savefig(map_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return map_filename
    
    except Exception as e:
        st.error(f"Erro ao gerar mapa de calor: {str(e)}")
        return None

# Função para gerar alerta de poluição
def generate_pollution_alert():
    st.title("🚨 Alerta de Poluição Atmosférica - Mato Grosso do Sul")
    st.markdown("""
    Este painel apresenta previsões de Profundidade Óptica de Aerossóis (AOD) a 550nm 
    para os municípios de Mato Grosso do Sul nos próximos 5 dias. Os dados são obtidos 
    em tempo real do CAMS (Copernicus Atmosphere Monitoring Service).
    """)
    
    # Adicionar botão para gerar análise
    if st.button("🔍 Gerar Alerta de Poluição", type="primary"):
        # Executar análise e obter resultados
        with st.spinner("⚙️ Gerando análise completa para todos os municípios..."):
            results = generate_ms_cities_aod_forecast(days=5)
        
        if results:
            alerts_df = results['alerts']
            
            # Cabeçalho
            st.header("🌫️ Previsão de Poluição Atmosférica")
            st.markdown(f"Análise realizada em: **{datetime.now().strftime('%d/%m/%Y %H:%M')}**")
            
            # Adicionar mapa do estado para o dia mais crítico (dia 1)
            st.subheader("🗺️ Mapa de Previsão de AOD para o Estado")
            
            # Criar tabs para diferentes dias
            map_tabs = st.tabs([f"Dia {i+1}" for i in range(5)])
            
            # Gerar mapas para cada dia
            for i, tab in enumerate(map_tabs):
                with tab:
                    map_file = generate_state_heatmap(results['dataset'], forecast_day=i+1)
                    if map_file:
                        st.image(map_file, caption=f"Previsão de AOD para Dia {i+1}")
                    else:
                        st.warning("Não foi possível gerar o mapa para este dia.")
            
            # Exibir tabela com TOP 20 cidades mais poluídas
            st.subheader("🔝 TOP 20 Municípios - Maiores Níveis de Poluição Previstos")
            
            # Limitar para top 20
            top_cities = alerts_df.head(20).copy()
            
            # Adicionar estilo às células baseado nos valores de AOD
            def format_aod_table(df):
                # Formatando valores numéricos
                formatted_df = df.copy()
                formatted_df['aod_maximo'] = formatted_df['aod_maximo'].apply(lambda x: f"{x:.3f}")
                
                # Aplicar estilo condicional baseado na categoria
                styled = formatted_df.style.applymap(
                    lambda _: 'background-color: green; color: white', 
                    subset=pd.IndexSlice[formatted_df[formatted_df['categoria'] == 'Boa'].index, ['categoria']]
                ).applymap(
                    lambda _: 'background-color: yellow; color: black', 
                    subset=pd.IndexSlice[formatted_df[formatted_df['categoria'] == 'Moderada'].index, ['categoria']]
                ).applymap(
                    lambda _: 'background-color: orange; color: black', 
                    subset=pd.IndexSlice[formatted_df[formatted_df['categoria'] == 'Insalubre'].index, ['categoria']]
                ).applymap(
                    lambda _: 'background-color: red; color: white', 
                    subset=pd.IndexSlice[formatted_df[formatted_df['categoria'] == 'Perigosa'].index, ['categoria']]
                )
                
                return styled
            
            # Renomear colunas para exibição
            top_cities = top_cities.rename(columns={
                'cidade': 'Município',
                'aod_maximo': 'AOD Máximo',
                'data': 'Data',
                'hora': 'Hora',
                'categoria': 'Categoria'
            })
            
            # Exibir tabela estilizada
            st.dataframe(format_aod_table(top_cities), use_container_width=True)
            
            #
