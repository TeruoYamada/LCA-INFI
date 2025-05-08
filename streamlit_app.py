import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import requests
import io
from sklearn.linear_model import LinearRegression
from PIL import Image
import base64
import time
import cdsapi

# Configuração da página
st.set_page_config(
    page_title="Sistema Integrado de Qualidade do Ar - MS",
    page_icon="🌬️",
    layout="wide"
)

# Título e introdução
st.title("🌬️ Sistema Integrado de Monitoramento da Qualidade do Ar - Mato Grosso do Sul")
st.markdown("""
Este aplicativo permite monitorar e analisar a qualidade do ar nos municípios de Mato Grosso do Sul, 
integrando dados do Serviço de Monitoramento da Atmosfera Copernicus (CAMS) para múltiplos poluentes 
atmosféricos (MP10, MP2.5, O3, NO2, SO2, CO) e aerossóis (AOD).
Os dados são classificados segundo os padrões da Resolução CONAMA nº 491/2018 e da OMS (2021).
""")

# Função para carregar os municípios de MS
@st.cache_data
def load_ms_municipalities():
    try:
        # Tentativa de carregar shapefile online
        url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/MS/MS_Municipios_2022.zip"
        
        try:
            gdf = gpd.read_file(url)
            return gdf
        except Exception as e:
            st.warning(f"Erro ao carregar shapefile online: {str(e)}")
            # Fallback: criar geodataframe simplificado
            data = {
                'NM_MUN': [
                    'Campo Grande', 'Dourados', 'Três Lagoas', 'Corumbá', 'Ponta Porã',
                    'Naviraí', 'Nova Andradina', 'Aquidauana', 'Maracaju', 'Paranaíba',
                    'Sidrolândia', 'Coxim', 'Amambai', 'Rio Brilhante', 'Chapadão do Sul'
                ],
                'geometry': [
                    gpd.points_from_xy([-54.6201], [-20.4697])[0].buffer(0.2),
                    gpd.points_from_xy([-54.8120], [-22.2231])[0].buffer(0.2),
                    gpd.points_from_xy([-51.7005], [-20.7849])[0].buffer(0.2),
                    gpd.points_from_xy([-57.6510], [-19.0082])[0].buffer(0.2),
                    gpd.points_from_xy([-55.7271], [-22.5334])[0].buffer(0.2),
                    gpd.points_from_xy([-54.1994], [-23.0624])[0].buffer(0.2),
                    gpd.points_from_xy([-53.3435], [-22.2384])[0].buffer(0.2),
                    gpd.points_from_xy([-55.7879], [-20.4697])[0].buffer(0.2),
                    gpd.points_from_xy([-55.1678], [-21.6407])[0].buffer(0.2),
                    gpd.points_from_xy([-51.1909], [-19.6746])[0].buffer(0.2),
                    gpd.points_from_xy([-54.9692], [-20.9330])[0].buffer(0.2),
                    gpd.points_from_xy([-54.7605], [-18.5067])[0].buffer(0.2),
                    gpd.points_from_xy([-55.2253], [-23.1058])[0].buffer(0.2),
                    gpd.points_from_xy([-54.5426], [-21.8033])[0].buffer(0.2),
                    gpd.points_from_xy([-52.6276], [-18.7908])[0].buffer(0.2)
                ]
            }
            gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
            return gdf
    except Exception as e:
        st.error(f"Não foi possível carregar os municípios: {str(e)}")
        return gpd.GeoDataFrame(columns=['NM_MUN', 'geometry'], crs="EPSG:4326")

# Dicionário com coordenadas das cidades do MS
cities_coords = {
    "Campo Grande": [-20.4697, -54.6201],
    "Dourados": [-22.2231, -54.8120],
    "Três Lagoas": [-20.7849, -51.7005],
    "Corumbá": [-19.0082, -57.6510],
    "Ponta Porã": [-22.5334, -55.7271],
    "Naviraí": [-23.0624, -54.1994],
    "Nova Andradina": [-22.2384, -53.3435],
    "Aquidauana": [-20.4697, -55.7879],
    "Maracaju": [-21.6407, -55.1678],
    "Paranaíba": [-19.6746, -51.1909],
    "Sidrolândia": [-20.9330, -54.9692],
    "Coxim": [-18.5067, -54.7605],
    "Amambai": [-23.1058, -55.2253],
    "Rio Brilhante": [-21.8033, -54.5426],
    "Chapadão do Sul": [-18.7908, -52.6276]
}

# Função para obter dados do CAMS
@st.cache_data(ttl=3600)  # Cache válido por 1 hora
def get_cams_data(municipalities, start_date, end_date):
    """
    Obtém dados reais de qualidade do ar do Copernicus Atmosphere Monitoring Service (CAMS)
    para os municípios selecionados no período especificado.
    """
    # Lista de poluentes a serem buscados
    pollutants = {
        'MP10': 'particulate_matter_10um',
        'MP2.5': 'particulate_matter_2.5um',
        'O3': 'ozone',
        'NO2': 'nitrogen_dioxide',
        'SO2': 'sulphur_dioxide',
        'CO': 'carbon_monoxide',
        'AOD': 'total_aerosol_optical_depth'
    }
    
    # Cliente CDS API (é necessário ter credenciais configuradas)
    try:
        # Configuração do cliente CAMS com st.secrets
        ads_url = st.secrets["ads"]["url"]
        ads_key = st.secrets["ads"]["key"]
        c = cdsapi.Client(url=ads_url, key=ads_key)
    except Exception as e:
        st.error(f"Erro ao inicializar cliente CAMS: {str(e)}")
        st.info("Usando dados simulados como fallback. Para usar dados reais, configure as credenciais do CAMS.")
        # Fallback para dados simulados se não conseguir conectar ao CAMS
        return _get_simulated_air_quality_data(municipalities, start_date, end_date)
    
    # Criar DataFrame com datas no período
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Lista para armazenar os dados
    data_list = []
    
    # Definir área para MS (aproximadamente)
    area = [
        -18.0, -59.0, -24.0, -50.0,  # [north, west, south, east]
    ]
    
    try:
        # Fazer a requisição ao CAMS para os dados do período
        with st.spinner("Buscando dados do CAMS..."):
            # Formato de data para CAMS
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Requisitar dados do CAMS
            result = c.retrieve(
                'cams-global-atmospheric-composition-forecasts',
                {
                    'date': f'{start_date_str}/{end_date_str}',
                    'type': 'forecast',
                    'format': 'netcdf',
                    'variable': list(pollutants.values()),
                    'time': [
                        '00:00', '03:00', '06:00',
                        '09:00', '12:00', '15:00',
                        '18:00', '21:00',
                    ],
                    'leadtime_hour': '0',
                    'area': area,
                    'model_level': '137',  # Superfície
                }
            )
            
            # Processar arquivo NetCDF
            import xarray as xr
            with result.download() as buff:
                ds = xr.open_dataset(buff)
            
            # Para cada município, extrair dados na coordenada mais próxima
            for municipality in municipalities:
                # Obter coordenadas do município
                lat, lon = cities_coords.get(municipality, [-20.4697, -54.6201])  # Padrão: Campo Grande
                
                # Encontrar o ponto de grade mais próximo
                municipality_data = ds.sel(latitude=lat, longitude=lon, method='nearest')
                
                # Para cada data no período, obter médias diárias
                for date in date_range:
                    date_data = municipality_data.sel(time=date.strftime('%Y-%m-%d'), method='nearest')
                    
                    # Extrair valores médios diários para cada poluente
                    pollutant_values = {}
                    for pollutant_name, pollutant_var in pollutants.items():
                        if pollutant_var in date_data:
                            # Converter para unidades corretas
                            value = date_data[pollutant_var].mean().values.item()
                            
                            # Conversões específicas para cada poluente (simplificado)
                            if pollutant_name in ['MP10', 'MP2.5']:
                                value = value * 1e9  # Converter de kg/m³ para μg/m³
                            elif pollutant_name in ['O3', 'NO2', 'SO2']:
                                value = value * 1e9  # Converter de mol/mol para μg/m³ (aproximação)
                            elif pollutant_name == 'CO':
                                value = value * 1e6  # Converter para ppm
                            
                            pollutant_values[pollutant_name] = max(0, value)  # Garantir valores não-negativos
                        else:
                            # Valor padrão se não disponível
                            pollutant_values[pollutant_name] = 0
                    
                    # Adicionar à lista de dados
                    data_list.append({
                        'Município': municipality,
                        'Data': date,
                        'Latitude': lat,
                        'Longitude': lon,
                        **pollutant_values
                    })
    except Exception as e:
        st.error(f"Erro ao obter dados do CAMS: {str(e)}")
        st.info("Usando dados simulados como fallback. Para usar dados reais, configure as credenciais do CAMS.")
        # Fallback para dados simulados
        return _get_simulated_air_quality_data(municipalities, start_date, end_date)
    
    # Criar DataFrame
    df = pd.DataFrame(data_list)
    return df

# Função de fallback para simular dados
def _get_simulated_air_quality_data(municipalities, start_date, end_date):
    """
    Simula dados de qualidade do ar como fallback quando não é possível obter dados reais.
    """
    # Lista de poluentes
    pollutants = ['MP10', 'MP2.5', 'O3', 'NO2', 'SO2', 'CO', 'AOD']
    
    # Criar DataFrame com datas no período
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Lista para armazenar os dados
    data_list = []
    
    # Para cada município e data, gerar dados simulados
    for municipality in municipalities:
        # Seed baseado no nome do município para gerar dados consistentes
        seed = sum(ord(c) for c in municipality)
        np.random.seed(seed)
        
        # Obter coordenadas do município
        lat, lon = cities_coords.get(municipality, [-20.4697, -54.6201])  # Padrão: Campo Grande
        
        # Gerar base para o município (valores médios)
        # Valores ajustados para serem mais realistas para o MS
        base_values = {
            'MP10': np.random.uniform(20, 60),  # μg/m³
            'MP2.5': np.random.uniform(10, 30),  # μg/m³
            'O3': np.random.uniform(40, 100),    # μg/m³
            'NO2': np.random.uniform(20, 60),    # μg/m³
            'SO2': np.random.uniform(10, 40),    # μg/m³
            'CO': np.random.uniform(1, 5),      # ppm
            'AOD': np.random.uniform(0.05, 0.3)  # adimensional
        }
        
        # Ajustar valores baseados na localização (simulando efeito urbano/rural)
        # Campo Grande e Dourados (mais urbanizadas) têm valores ligeiramente maiores
        if municipality in ["Campo Grande", "Dourados"]:
            for pollutant in ['MP10', 'MP2.5', 'NO2', 'CO']:
                base_values[pollutant] *= 1.2
        
        # Simular efeito sazonal (estação seca vs. úmida)
        # Mato Grosso do Sul tem estação seca de maio a setembro
        is_dry_season = lambda d: d.month >= 5 and d.month <= 9
        
        # Tendência ao longo do tempo
        trend_factor = np.random.uniform(0.8, 1.2, size=len(date_range))
        
        for i, date in enumerate(date_range):
            # Aplicar efeito sazonal
            seasonal_factor = 1.5 if is_dry_season(date) else 1.0
            
            # Aumentar especialmente MP10, MP2.5 e AOD durante estação seca (queimadas)
            seasonal_values = {p: base_values[p] for p in pollutants}
            if is_dry_season(date):
                seasonal_values['MP10'] *= 1.8
                seasonal_values['MP2.5'] *= 1.7
                seasonal_values['AOD'] *= 2.0
            
            # Aplicar tendência e variação diária
            daily_values = {
                pollutant: max(0, seasonal_values[pollutant] * trend_factor[i] * 
                             np.random.uniform(0.8, 1.2) * seasonal_factor)
                for pollutant in pollutants
            }
            
            # Ajustes adicionais para aumentar a variabilidade entre municípios
            # Maior urbanização = mais poluição de origem veicular (NO2, CO)
            urban_factor = 1.0 + abs(np.sin(seed * 0.1)) * 0.5
            daily_values['NO2'] *= urban_factor
            daily_values['CO'] *= urban_factor
            
            # Adicionar à lista de dados
            data_list.append({
                'Município': municipality,
                'Data': date,
                'Latitude': lat,
                'Longitude': lon,
                **daily_values
            })
    
    # Criar DataFrame
    df = pd.DataFrame(data_list)
    return df

# Função para obter previsão da qualidade do ar
def get_air_quality_forecast(municipalities, days=5):
    """
    Obtém previsão da qualidade do ar para os próximos dias
    """
    today = datetime.now().date()
    forecast_end = today + timedelta(days=days)
    
    # Tentar obter dados de previsão do CAMS
    with st.spinner(f"Obtendo previsão para os próximos {days} dias..."):
        forecast_data = get_cams_data(municipalities, today, forecast_end)
    
    return forecast_data

# Função para classificar a qualidade do ar conforme CONAMA 491/2018
def classify_air_quality_conama(df):
    """
    Classifica a qualidade do ar conforme a Resolução CONAMA nº 491/2018
    """
    # Limites definidos pela Resolução CONAMA nº 491/2018 (valores simplificados)
    limits_conama = {
        'MP10': [50, 100, 150, 250, 250],  # μg/m³ - média de 24h
        'MP2.5': [25, 50, 75, 125, 125],   # μg/m³ - média de 24h
        'O3': [100, 130, 160, 200, 200],   # μg/m³ - média de 8h
        'NO2': [200, 240, 320, 1130, 1130], # μg/m³ - média de 1h
        'SO2': [20, 40, 365, 800, 800],    # μg/m³ - média de 24h
        'CO': [9, 11, 13, 15, 15],         # ppm - média de 8h
        'AOD': [0.1, 0.2, 0.3, 0.5, 0.5]   # adimensional - valores aproximados
    }
    
    # Categorias de qualidade
    categories = ['Boa', 'Moderada', 'Ruim', 'Muito Ruim', 'Péssima']
    colors = ['#00ccff', '#009933', '#ffff00', '#ff9933', '#ff0000', '#990000']
    
    # Criar cópia para não modificar o original
    result_df = df.copy()
    
    # Adicionar classificação para cada poluente
    for pollutant, limits in limits_conama.items():
        category_col = f'Categoria_{pollutant}'
        result_df[category_col] = 'Boa'
        
        for i, limit in enumerate(limits):
            if i == 0:
                # Boa: até o primeiro limite
                result_df.loc[result_df[pollutant] <= limit, category_col] = categories[0]
            elif i < len(limits) - 1:
                # Categorias intermediárias
                result_df.loc[(result_df[pollutant] > limits[i-1]) & 
                            (result_df[pollutant] <= limit), 
                            category_col] = categories[i]
            else:
                # Última categoria: acima do penúltimo limite
                result_df.loc[result_df[pollutant] > limits[i-1], 
                            category_col] = categories[i]
    
    # Determinar categoria geral (pior caso)
    category_cols = [f'Categoria_{p}' for p in limits_conama.keys()]
    
    # Função para obter o índice da categoria
    def category_index(cat):
        try:
            return categories.index(cat)
        except ValueError:
            return 0
    
    # Aplicar função para encontrar a pior categoria
    result_df['Categoria_Geral'] = result_df[category_cols].apply(
        lambda row: categories[max(category_index(cat) for cat in row)], axis=1
    )
    
    # Adicionar coluna de cor
    result_df['Cor'] = result_df['Categoria_Geral'].apply(
        lambda cat: colors[categories.index(cat)]
    )
    
    return result_df

# Função para classificar a qualidade do ar conforme OMS
def classify_air_quality_who(df):
    """
    Classifica a qualidade do ar conforme as diretrizes da OMS (2021)
    """
    # Limites definidos pela OMS (2021) - valores simplificados
    limits_who = {
        'MP10': [15, 45, 75, 150, 150],    # μg/m³ - média de 24h
        'MP2.5': [5, 15, 25, 50, 50],      # μg/m³ - média de 24h
        'O3': [60, 100, 140, 180, 180],    # μg/m³ - média de 8h
        'NO2': [25, 50, 100, 200, 200],    # μg/m³ - média de 24h
        'SO2': [40, 80, 160, 350, 350],    # μg/m³ - média de 24h
        'CO': [4, 9, 13, 15, 15],          # ppm - média de 8h
        'AOD': [0.05, 0.1, 0.2, 0.4, 0.4]  # adimensional - valores aproximados
    }
    
    # Categorias de qualidade
    categories = ['Boa', 'Moderada', 'Ruim', 'Muito Ruim', 'Péssima']
    colors = ['#00ccff', '#009933', '#ffff00', '#ff9933', '#ff0000', '#990000']
    
    # Criar cópia para não modificar o original
    result_df = df.copy()
    
    # Adicionar classificação para cada poluente
    for pollutant, limits in limits_who.items():
        category_col = f'Categoria_WHO_{pollutant}'
        result_df[category_col] = 'Boa'
        
        for i, limit in enumerate(limits):
            if i == 0:
                # Boa: até o primeiro limite
                result_df.loc[result_df[pollutant] <= limit, category_col] = categories[0]
            elif i < len(limits) - 1:
                # Categorias intermediárias
                result_df.loc[(result_df[pollutant] > limits[i-1]) & 
                            (result_df[pollutant] <= limit), 
                            category_col] = categories[i]
            else:
                # Última categoria: acima do penúltimo limite
                result_df.loc[result_df[pollutant] > limits[i-1], 
                            category_col] = categories[i]
    
    # Determinar categoria geral (pior caso)
    category_cols = [f'Categoria_WHO_{p}' for p in limits_who.keys()]
    
    # Função para obter o índice da categoria
    def category_index(cat):
        try:
            return categories.index(cat)
        except ValueError:
            return 0
    
    # Aplicar função para encontrar a pior categoria
    result_df['Categoria_Geral_WHO'] = result_df[category_cols].apply(
        lambda row: categories[max(category_index(cat) for cat in row)], axis=1
    )
    
    # Adicionar coluna de cor
    result_df['Cor_WHO'] = result_df['Categoria_Geral_WHO'].apply(
        lambda cat: colors[categories.index(cat)]
    )
    
    return result_df

# Função para criar gráfico da série temporal
def create_time_series_plot(df, municipality, standard='CONAMA'):
    """
    Cria um gráfico interativo de série temporal para o município selecionado
    """
    # Filtrar dados para o município
    mun_data = df[df['Município'] == municipality].sort_values('Data')
    
    # Selecionar as colunas de categoria e cor apropriadas
    if standard == 'CONAMA':
        cat_col = 'Categoria_Geral'
        color_col = 'Cor'
    else:  # WHO
        cat_col = 'Categoria_Geral_WHO'
        color_col = 'Cor_WHO'
    
    # Criar subplots - um para cada poluente
    pollutants = ['MP10', 'MP2.5', 'O3', 'NO2', 'SO2', 'CO', 'AOD']
    fig = make_subplots(rows=4, cols=2, subplot_titles=pollutants + ['Qualidade Geral'],
                       shared_xaxes=True, vertical_spacing=0.1)
    
    # Adicionar gráficos para cada poluente
    for i, pollutant in enumerate(pollutants):
        row = i // 2 + 1
        col = i % 2 + 1
        
        # Pular se ultrapassar o número de linhas
        if row > 4:
            continue
            
        # Selecionar coluna de categoria específica do poluente
        poll_cat_col = f'Categoria_{pollutant}' if standard=='CONAMA' else f'Categoria_WHO_{pollutant}'
        
        # Adicionar série temporal
        fig.add_trace(
            go.Scatter(
                x=mun_data['Data'],
                y=mun_data[pollutant],
                mode='lines+markers',
                name=pollutant,
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate=f"{pollutant}: %{{y:.1f}}<br>Data: %{{x|%d/%m/%Y}}<br>Categoria: {mun_data[poll_cat_col]}"
            ),
            row=row, col=col
        )
        
        # Adicionar limites (simplificado - apenas o primeiro limite para "Boa")
        if standard == 'CONAMA':
            limits = {
                'MP10': 50, 'MP2.5': 25, 'O3': 100,
                'NO2': 200, 'SO2': 20, 'CO': 9, 'AOD': 0.1
            }
        else:  # WHO
            limits = {
                'MP10': 15, 'MP2.5': 5, 'O3': 60,
                'NO2': 25, 'SO2': 40, 'CO': 4, 'AOD': 0.05
            }
        
        # Adicionar linha de limite
        fig.add_trace(
            go.Scatter(
                x=[mun_data['Data'].min(), mun_data['Data'].max()],
                y=[limits[pollutant], limits[pollutant]],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                name=f"Limite {pollutant}",
                showlegend=False
            ),
            row=row, col=col
        )
    
    # Adicionar gráfico de categoria geral (último painel)
    # Criar mapeamento numérico para categorias
    category_map = {'Boa': 0, 'Moderada': 1, 'Ruim': 2, 'Muito Ruim': 3, 'Péssima': 4}
    mun_data['cat_numeric'] = mun_data[cat_col].map(category_map)
    
    # Cores para as categorias
    cat_colors = ['#00ccff', '#009933', '#ffff00', '#ff9933', '#ff0000']
    
    # Adicionar gráfico de barras para categoria geral
    fig.add_trace(
        go.Bar(
            x=mun_data['Data'],
            y=mun_data['cat_numeric'],
            marker=dict(
                color=mun_data[cat_col].map(lambda c: cat_colors[category_map[c]]),
                line=dict(width=0)
            ),
            hovertemplate="Data: %{x|%d/%m/%Y}<br>Qualidade: " + mun_data[cat_col],
            name="Qualidade Geral"
        ),
        row=4, col=2
    )
    
    # Configurar eixo y para categorias
    fig.update_yaxes(
        tickvals=[0, 1, 2, 3, 4],
        ticktext=['Boa', 'Moderada', 'Ruim', 'Muito Ruim', 'Péssima'],
        row=4, col=2
    )
    
    # Atualizar layout
    fig.update_layout(
        title=f"Série Temporal de Poluentes - {municipality} (Padrão: {standard})",
        height=900,
        width=1000,
        showlegend=False,
        template="plotly_white"
    )
    
    # Atualizar títulos dos eixos
    units = {
        'MP10': 'μg/m³',
        'MP2.5': 'μg/m³',
        'O3': 'μg/m³',
        'NO2': 'μg/m³',
        'SO2': 'μg/m³',
        'CO': 'ppm',
        'AOD': 'adimensional'
    }
    
    for i, pollutant in enumerate(pollutants):
        if i >= 7:  # Pular se for além do número de poluentes
            continue
            
        row = i // 2 + 1
        col = i % 2 + 1
        
        if row <= 4:  # Verificar se está dentro do range
            fig.update_yaxes(title_text=units[pollutant], row=row, col=col)
            
            if row == 4:  # Última linha
                fig.update_xaxes(title_text="Data", row=row, col=col)
    
    return fig

# Função para gerar mapa de qualidade do ar
def create_air_quality_map(df, gdf, date, pollutant='Categoria_Geral', standard='CONAMA'):
    """
    Cria um mapa interativo com a qualidade do ar para todos os municípios em uma data específica
    Usando uma abordagem com go.Scattermapbox para maior estabilidade
    """
    # Filtrar dados para a data
    date_data = df[df['Data'] == date]
    
# Determinar se estamos visualizando um poluente específico ou a categoria geral
    is_category = pollutant in ['Categoria_Geral', 'Categoria_Geral_WHO']
    
    # Selecionar a coluna apropriada dependendo do padrão
    if is_category:
        if standard == 'CONAMA':
            value_col = 'Categoria_Geral'
            color_col = 'Cor'
        else:  # WHO
            value_col = 'Categoria_Geral_WHO'
            color_col = 'Cor_WHO'
    else:
        # Se for um poluente específico, usar o nome diretamente
        value_col = pollutant
        
        # E selecionar a coluna de categoria apropriada
        if standard == 'CONAMA':
            cat_col = f'Categoria_{pollutant}'
        else:  # WHO
            cat_col = f'Categoria_WHO_{pollutant}'
    
    # Mesclar dados com o geodataframe
    map_data = gdf.merge(date_data, left_on='NM_MUN', right_on='Município', how='inner')
    
    # Verificar se há dados para criar o mapa
    if map_data.empty:
        st.warning(f"Não há dados de qualidade do ar disponíveis para {date.strftime('%d/%m/%Y')}")
        return None
    
    # Criar uma figura com scatter_mapbox
    fig = go.Figure()
    
    # Definir cores para as categorias (se estamos visualizando categorias)
    color_map = {
        'Boa': '#00ccff',
        'Moderada': '#009933',
        'Ruim': '#ffff00',
        'Muito Ruim': '#ff9933',
        'Péssima': '#ff0000'
    }
    
    # Escala de cores para valores numéricos (se estamos visualizando poluentes específicos)
    if not is_category:
        # Determinar faixa de valores para o poluente
        vmin, vmax = map_data[value_col].min(), map_data[value_col].max()
        
        # Colorscale específico para cada tipo de poluente
        colorscales = {
            'MP10': 'Reds',
            'MP2.5': 'Oranges',
            'O3': 'Blues',
            'NO2': 'Purples',
            'SO2': 'Greens',
            'CO': 'Greys',
            'AOD': 'YlOrBr'
        }
        colorscale = colorscales.get(value_col, 'Viridis')
    
    # Adicionar pontos para cada município
    for idx, row in map_data.iterrows():
        # Obter o centroide da geometria para posicionar o ponto
        try:
            # Se a geometria for um polígono, extrair o centroide
            if hasattr(row.geometry, 'centroid'):
                lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
            # Se for um ponto (caso do fallback), usar as coordenadas diretamente
            elif hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                lon, lat = row.geometry.x, row.geometry.y
            else:
                # Usar coordenadas do dicionário como fallback
                if row['Município'] in cities_coords:
                    lat, lon = cities_coords[row['Município']]
                else:
                    # Usar as coordenadas do dataframe
                    lat, lon = row['Latitude'], row['Longitude']
            
            # Determinar a cor do marcador
            if is_category:
                # Para categorias, usar o mapa de cores pré-definido
                category = row[value_col]
                marker_color = color_map.get(category, '#cccccc')
                
                # Texto para hover
                hover_text = f"<b>{row['Município']}</b><br>" + \
                            f"Qualidade do Ar: {category}<br>" + \
                            f"MP10: {row['MP10']:.2f} μg/m³<br>" + \
                            f"MP2.5: {row['MP2.5']:.2f} μg/m³<br>" + \
                            f"O3: {row['O3']:.2f} μg/m³<br>" + \
                            f"NO2: {row['NO2']:.2f} μg/m³<br>" + \
                            f"SO2: {row['SO2']:.2f} μg/m³<br>" + \
                            f"CO: {row['CO']:.2f} ppm<br>" + \
                            f"AOD: {row['AOD']:.2f}"
            else:
                # Para poluentes específicos, usar o valor numérico
                value = row[value_col]
                
                # Normalizar valor entre 0 e 1 para coloração
                norm_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                
                # Escolher cor baseada no valor normalizado
                marker_color = px.colors.sample_colorscale(
                    colorscale, [norm_value])[0]
                
                # Texto para hover
                hover_text = f"<b>{row['Município']}</b><br>" + \
                            f"{value_col}: {value:.2f}<br>" + \
                            f"Categoria: {row[cat_col]}"
            
            # Adicionar marker
            fig.add_trace(go.Scattermapbox(
                lat=[lat],
                lon=[lon],
                mode='markers',
                marker=dict(
                    size=15,
                    color=marker_color,
                    opacity=0.8
                ),
                text=row['Município'],
                hoverinfo='text',
                hovertext=hover_text,
                name=row['Município']
            ))
        except Exception as e:
            st.warning(f"Erro ao processar município {row['Município']}: {str(e)}")
            continue
    
    # Configurar o layout do mapa
    fig.update_layout(
        title=f"Mapa de {pollutant if not is_category else 'Qualidade do Ar'} - {date.strftime('%d/%m/%Y')} (Padrão: {standard})",
        mapbox=dict(
            style="carto-positron",
            zoom=5,
            center={"lat": -20.5, "lon": -54.6},
        ),
        height=600,
        margin={"r":0,"t":50,"l":0,"b":0},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Adicionar legenda manualmente
    if is_category:
        # Legenda para categorias
        legend_categories = list(color_map.keys())
        for i, category in enumerate(legend_categories):
            fig.add_trace(go.Scattermapbox(
                lat=[None],
                lon=[None],
                mode='markers',
                marker=dict(size=10, color=color_map[category]),
                name=category,
                showlegend=True
            ))
    else:
        # Legenda para valores numéricos (simplificada)
        # Adicionar alguns pontos de referência na legenda
        values = map_data[value_col]
        breaks = np.linspace(vmin, vmax, 5)
        
        # Obter unidade para o poluente
        units = {
            'MP10': 'μg/m³',
            'MP2.5': 'μg/m³',
            'O3': 'μg/m³',
            'NO2': 'μg/m³',
            'SO2': 'μg/m³',
            'CO': 'ppm',
            'AOD': ''
        }
        unit = units.get(value_col, '')
        
        for i, val in enumerate(breaks):
            norm_val = i / (len(breaks) - 1)
            color = px.colors.sample_colorscale(colorscale, [norm_val])[0]
            
            fig.add_trace(go.Scattermapbox(
                lat=[None],
                lon=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=f"{val:.2f} {unit}",
                showlegend=True
            ))
    
    return fig

# Função para criar animação de poluentes
def create_pollutant_animation(df, gdf, start_date, end_date, pollutant, standard='CONAMA'):
    """
    Cria uma animação avançada da evolução espacial de um poluente específico
    """
    # Filtrar dados para o período
    period_data = df[(df['Data'] >= start_date) & (df['Data'] <= end_date)]
    
    # Obter datas únicas no período
    dates = sorted(period_data['Data'].unique())
    
    # Verificar se há dados suficientes
    if len(dates) < 2:
        st.warning("Período muito curto para criação de animação. Selecione um período mais longo.")
        return None
    
    # Limitar a 7 dias para performance
    if len(dates) > 7:
        st.warning(f"Muitas datas selecionadas. Mostrando apenas os primeiros 7 dias de {len(dates)} disponíveis.")
        dates = dates[:7]
    
    # Criar um mapa para cada data
    animation_frames = []
    
    # Determinar range dos valores para consistência na escala de cores
    if pollutant != 'Categoria_Geral' and pollutant != 'Categoria_Geral_WHO':
        vmin = period_data[pollutant].min()
        vmax = period_data[pollutant].max()
    else:
        vmin = vmax = None
    
    # Definir colorscale específico para cada tipo de poluente
    colorscales = {
        'MP10': 'Reds',
        'MP2.5': 'Oranges',
        'O3': 'Blues',
        'NO2': 'Purples',
        'SO2': 'Greens',
        'CO': 'Greys',
        'AOD': 'YlOrBr',
        'Categoria_Geral': None,
        'Categoria_Geral_WHO': None
    }
    
    # Criar figura base
    fig = go.Figure()
    
    # Para cada data, criar um frame
    for date in dates:
        with st.spinner(f"Processando animação para {date.strftime('%d/%m/%Y')}..."):
            # Filtrar dados para esta data
            date_data = period_data[period_data['Data'] == date]
            
            # Criar frame para esta data
            frame_data = []
            
            # Se estamos visualizando categorias
            is_category = pollutant in ['Categoria_Geral', 'Categoria_Geral_WHO']
            
            if is_category:
                cat_col = pollutant
                color_map = {
                    'Boa': '#00ccff',
                    'Moderada': '#009933',
                    'Ruim': '#ffff00',
                    'Muito Ruim': '#ff9933',
                    'Péssima': '#ff0000'
                }
                
                # Para cada município, criar um marker
                for _, row in date_data.iterrows():
                    lat, lon = row['Latitude'], row['Longitude']
                    category = row[cat_col]
                    marker_color = color_map.get(category, '#cccccc')
                    
                    # Texto para hover
                    hover_text = f"<b>{row['Município']}</b><br>" + \
                                f"Qualidade do Ar: {category}<br>" + \
                                f"Data: {date.strftime('%d/%m/%Y')}<br>" + \
                                f"MP10: {row['MP10']:.2f} μg/m³<br>" + \
                                f"MP2.5: {row['MP2.5']:.2f} μg/m³<br>" + \
                                f"O3: {row['O3']:.2f} μg/m³<br>" + \
                                f"NO2: {row['NO2']:.2f} μg/m³<br>" + \
                                f"SO2: {row['SO2']:.2f} μg/m³<br>" + \
                                f"CO: {row['CO']:.2f} ppm<br>" + \
                                f"AOD: {row['AOD']:.2f}"
                    
                    # Adicionar marker
                    frame_data.append(
                        go.Scattermapbox(
                            lat=[lat],
                            lon=[lon],
                            mode='markers',
                            marker=dict(
                                size=15,
                                color=marker_color,
                                opacity=0.8
                            ),
                            text=row['Município'],
                            hoverinfo='text',
                            hovertext=hover_text,
                            name=row['Município']
                        )
                    )
            else:
                # Para visualização de poluente específico
                # Usar interpolação espacial para criar uma superfície contínua
                from scipy.interpolate import griddata
                
                # Extrair coordenadas e valores
                lats = date_data['Latitude'].values
                lons = date_data['Longitude'].values
                values = date_data[pollutant].values
                
                # Criar grid para interpolação
                grid_size = 100
                lat_min, lat_max = min(lats) - 0.5, max(lats) + 0.5
                lon_min, lon_max = min(lons) - 0.5, max(lons) + 0.5
                grid_lats = np.linspace(lat_min, lat_max, grid_size)
                grid_lons = np.linspace(lon_min, lon_max, grid_size)
                grid_lon, grid_lat = np.meshgrid(grid_lons, grid_lats)
                
                # Interpolar valores
                grid_values = griddata((lons, lats), values, (grid_lon, grid_lat), method='cubic')
                
                # Criar Contour
                colorscale = colorscales[pollutant]
                
                # Adicionar contorno
                frame_data.append(
                    go.Contourmapbox(
                        z=grid_values,
                        lat=grid_lats,
                        lon=grid_lons,
                        colorscale=colorscale,
                        zmin=vmin,
                        zmax=vmax,
                        colorbar=dict(
                            title=pollutant,
                            titleside="right",
                            titlefont=dict(size=10),
                            tickfont=dict(size=8)
                        ),
                        hoverinfo='z',
                        name=f"{pollutant} - {date.strftime('%d/%m/%Y')}"
                    )
                )
                
                # Adicionar pontos para cidades
                frame_data.append(
                    go.Scattermapbox(
                        lat=lats,
                        lon=lons,
                        mode='markers+text',
                        marker=dict(
                            size=10,
                            color='white',
                            opacity=0.7
                        ),
                        text=date_data['Município'],
                        textposition="top center",
                        hoverinfo='text',
                        hovertext=[f"{mun}: {val:.2f}" for mun, val in zip(date_data['Município'], values)],
                        name="Municípios"
                    )
                )
            
            # Adicionar frame
            animation_frames.append(
                go.Frame(
                    data=frame_data,
                    name=date.strftime('%d/%m/%Y')
                )
            )
    
    # Configurar figura principal
    # Usar o primeiro frame como dados iniciais
    fig.add_traces(animation_frames[0].data)
    
    # Adicionar frames à figura
    fig.frames = animation_frames
    
    # Configurar sliders
    sliders = [{
        'active': 0,
        'steps': [
            {
                'method': 'animate',
                'args': [
                    [frame.name],
                    {
                        'mode': 'immediate',
                        'frame': {'duration': 500, 'redraw': True},
                        'transition': {'duration': 300}
                    }
                ],
                'label': frame.name
            }
            for frame in animation_frames
        ],
        'x': 0.1,
        'len': 0.9,
        'y': 0,
        'yanchor': 'top',
        'pad': {'t': 50, 'b': 10},
        'currentvalue': {
            'visible': True,
            'prefix': 'Data: ',
            'xanchor': 'right',
            'font': {'size': 16, 'color': '#666'}
        }
    }]
    
    # Adicionar botões de play/pause
    updatemenus = [{
        'type': 'buttons',
        'buttons': [
            {
                'args': [
                    None,
                    {
                        'frame': {'duration': 500, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                    }
                ],
                'label': '▶️ Play',
                'method': 'animate'
            },
            {
                'args': [
                    [None],
                    {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }
                ],
                'label': '⏸️ Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 10},
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }]
    
    # Configurar layout
    title = f"Evolução de {pollutant if pollutant not in ['Categoria_Geral', 'Categoria_Geral_WHO'] else 'Qualidade do Ar'}"
    title += f" ({start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')})"
    
    fig.update_layout(
        title=title,
        mapbox=dict(
            style="carto-positron",
            zoom=5,
            center={"lat": -20.5, "lon": -54.6},
        ),
        height=700,
        margin={"r":0,"t":50,"l":0,"b":0},
        updatemenus=updatemenus,
        sliders=sliders
    )
    
    return fig

# Função para gerar relatório de qualidade do ar
def generate_air_quality_report(df, municipality, start_date, end_date, standard='CONAMA'):
    """
    Gera um relatório de qualidade do ar para um município específico
    """
    # Filtrar dados para o município e período
    mun_data = df[(df['Município'] == municipality) & 
                 (df['Data'] >= start_date) & 
                 (df['Data'] <= end_date)].sort_values('Data')
    
    # Selecionar as colunas de categoria e cor apropriadas
    if standard == 'CONAMA':
        cat_col = 'Categoria_Geral'
        color_col = 'Cor'
        prefix = ''
    else:  # WHO
        cat_col = 'Categoria_Geral_WHO'
        color_col = 'Cor_WHO'
        prefix = 'WHO_'
    
    # Calcular estatísticas
    stats = {}
    pollutants = ['MP10', 'MP2.5', 'O3', 'NO2', 'SO2', 'CO', 'AOD']
    
    for pollutant in pollutants:
        stats[pollutant] = {
            'Média': mun_data[pollutant].mean(),
            'Máximo': mun_data[pollutant].max(),
            'Mínimo': mun_data[pollutant].min(),
            'Desvio Padrão': mun_data[pollutant].std()
        }
    
    # Contar ocorrências de cada categoria
    cat_counts = mun_data[cat_col].value_counts().to_dict()
    
    # Identificar o poluente mais crítico
    worst_days = {}
    for pollutant in pollutants:
        cat_col_poll = f'Categoria_{prefix}{pollutant}'
        if mun_data.empty:
            continue
            
        worst_idx = mun_data[cat_col_poll].map(
            {'Boa': 0, 'Moderada': 1, 'Ruim': 2, 'Muito Ruim': 3, 'Péssima': 4}
        ).idxmax()
        
        if not pd.isna(worst_idx):
            worst_day = mun_data.loc[worst_idx]
            worst_days[pollutant] = {
                'Data': worst_day['Data'],
                'Valor': worst_day[pollutant],
                'Categoria': worst_day[cat_col_poll]
            }
    
    # Criar o relatório
    report = {
        'Município': municipality,
        'Período': {
            'Início': start_date,
            'Fim': end_date,
            'Total de Dias': (end_date - start_date).days + 1
        },
        'Padrão': 'CONAMA 491/2018' if standard == 'CONAMA' else 'OMS (2021)',
        'Estatísticas': stats,
        'Categorias': cat_counts,
        'Dias Críticos': worst_days
    }
    
    return report

# Carregar municípios
with st.spinner("Carregando shapes dos municípios..."):
    ms_municipalities = load_ms_municipalities()
    municipalities_list = sorted(ms_municipalities['NM_MUN'].unique().tolist())

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Seleção de municípios
selected_municipalities = st.sidebar.multiselect(
    "Selecione os municípios",
    options=municipalities_list,
    default=[municipalities_list[0]] if municipalities_list else []
)

# Período de análise
st.sidebar.subheader("Período de Análise")
start_date = st.sidebar.date_input(
    "Data de Início",
    datetime.now() - timedelta(days=30)
)
end_date = st.sidebar.date_input(
    "Data Final",
    datetime.now()
)

# Padrão de qualidade do ar
standard = st.sidebar.radio(
    "Padrão de Qualidade do Ar",
    options=["CONAMA", "OMS"],
    help="CONAMA: Resolução nº 491/2018 | OMS: Diretrizes da OMS (2021)"
)

# Verificar seleções
if not selected_municipalities:
    st.warning("⚠️ Por favor, selecione pelo menos um município.")
    st.stop()

if start_date > end_date:
    st.error("❌ A data de início deve ser anterior à data final.")
    st.stop()

# Obter dados de qualidade do ar
with st.spinner("🔄 Carregando dados de qualidade do ar do CAMS..."):
    air_data = get_cams_data(selected_municipalities, start_date, end_date)
    
    # Classificar qualidade do ar
    air_data = classify_air_quality_conama(air_data)
    air_data = classify_air_quality_who(air_data)

# Layout principal - usar abas
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Visão Geral", 
    "📈 Séries Temporais", 
    "🗺️ Mapa de Qualidade do Ar",
    "🎬 Animação de Concentração",
    "📝 Relatórios",
    "🔮 Previsão"
])

with tab1:
    st.header("📊 Visão Geral da Qualidade do Ar")
    
    # Mostrar estatísticas gerais
    st.subheader("Estatísticas do Período")
    
    # Criar métricas por município
    for municipality in selected_municipalities:
        st.markdown(f"### 🏙️ {municipality}")
        
        # Filtrar dados para o município
        mun_data = air_data[air_data['Município'] == municipality]
        
        # Mostrar categoria predominante
        if standard == "CONAMA":
            cat_counts = mun_data['Categoria_Geral'].value_counts()
            predominant_cat = cat_counts.index[0] if not cat_counts.empty else "N/A"
            cat_color = mun_data[mun_data['Categoria_Geral'] == predominant_cat]['Cor'].iloc[0] if not mun_data.empty else "#CCCCCC"
        else:  # WHO
            cat_counts = mun_data['Categoria_Geral_WHO'].value_counts()
            predominant_cat = cat_counts.index[0] if not cat_counts.empty else "N/A"
            cat_color = mun_data[mun_data['Categoria_Geral_WHO'] == predominant_cat]['Cor_WHO'].iloc[0] if not mun_data.empty else "#CCCCCC"
        
        # Criar colunas para métricas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="padding:10px; border-radius:5px; background-color:{cat_color}; color:white; text-align:center; margin:5px 0;">
            <h4 style="margin:0;">Qualidade Predominante: {predominant_cat}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Mostrar pior dia
            if standard == "CONAMA":
                worst_day_idx = mun_data['Categoria_Geral'].map({
                    'Boa': 0, 'Moderada': 1, 'Ruim': 2, 'Muito Ruim': 3, 'Péssima': 4
                }).idxmax()
            else:  # WHO
                worst_day_idx = mun_data['Categoria_Geral_WHO'].map({
                    'Boa': 0, 'Moderada': 1, 'Ruim': 2, 'Muito Ruim': 3, 'Péssima': 4
                }).idxmax()
            
            if not pd.isna(worst_day_idx) and not mun_data.empty:
                worst_day = mun_data.loc[worst_day_idx]
                worst_date = worst_day['Data'].strftime('%d/%m/%Y')
                worst_cat = worst_day['Categoria_Geral'] if standard == "CONAMA" else worst_day['Categoria_Geral_WHO']
                st.metric("Pior Dia", f"{worst_date} ({worst_cat})")
            else:
                st.metric("Pior Dia", "N/A")
        
        with col3:
            # Calcular média dos poluentes
            pollutants = ['MP10', 'MP2.5', 'O3', 'NO2', 'SO2', 'CO', 'AOD']
            avg_values = {p: mun_data[p].mean() for p in pollutants}
            # Encontrar o poluente com maior valor em relação ao limite
            if standard == "CONAMA":
                limits = {
                    'MP10': 50, 'MP2.5': 25, 'O3': 100,
                    'NO2': 200, 'SO2': 20, 'CO': 9, 'AOD': 0.1
                }
            else:  # WHO
                limits = {
                    'MP10': 15, 'MP2.5': 5, 'O3': 60,
                    'NO2': 25, 'SO2': 40, 'CO': 4, 'AOD': 0.05
                }
            
            ratios = {p: avg_values[p] / limits[p] for p in pollutants}
            worst_pollutant = max(ratios, key=ratios.get)
            st.metric("Poluente Crítico", f"{worst_pollutant} ({avg_values[worst_pollutant]:.1f})")
        
        # Mostrar distribuição de categorias
        st.subheader(f"Distribuição de Qualidade do Ar - {municipality}")
        
        if standard == "CONAMA":
            cat_col = 'Categoria_Geral'
        else:  # WHO
            cat_col = 'Categoria_Geral_WHO'
        
        categories = ['Boa', 'Moderada', 'Ruim', 'Muito Ruim', 'Péssima']
        colors = ['#00ccff', '#009933', '#ffff00', '#ff9933', '#ff0000']
        
        # Contar ocorrências de cada categoria
        cat_data = mun_data[cat_col].value_counts().reindex(categories, fill_value=0)
        
        # Criar gráfico
        fig = px.bar(
            x=cat_data.index,
            y=cat_data.values,
            labels={'x': 'Categoria', 'y': 'Número de Dias'},
            title=f"Distribuição de Categorias de Qualidade do Ar - {municipality} (Padrão: {standard})"
        )
        
        # Definir cores
        fig.update_traces(marker_color=colors)
        fig.update_layout(xaxis_title="Categoria", yaxis_title="Número de Dias")
        
        st.plotly_chart(fig)
        
        # Adicionar separador
        st.markdown("---")

with tab2:
    st.header("📈 Séries Temporais")
    
    # Selecionar município para análise detalhada
    selected_mun = st.selectbox(
        "Selecione um município para análise detalhada",
        options=selected_municipalities
    )
    
    # Criar gráfico de série temporal
    with st.spinner("Gerando série temporal..."):
        fig = create_time_series_plot(air_data, selected_mun, standard)
        st.plotly_chart(fig, use_container_width=True)
    
    # Resumo dos poluentes
    st.subheader("Resumo dos Poluentes")
    
    # Filtrar dados para o município
    mun_data = air_data[air_data['Município'] == selected_mun]
    
    # Criar tabela de resumo
    pollutants = ['MP10', 'MP2.5', 'O3', 'NO2', 'SO2', 'CO', 'AOD']
    summary_data = []
    
    for pollutant in pollutants:
        # Obter estatísticas
        mean_val = mun_data[pollutant].mean()
        max_val = mun_data[pollutant].max()
        min_val = mun_data[pollutant].min()
        std_val = mun_data[pollutant].std()
        
        # Obter limites
        if standard == "CONAMA":
            limit = {
                'MP10': 50, 'MP2.5': 25, 'O3': 100,
                'NO2': 200, 'SO2': 20, 'CO': 9, 'AOD': 0.1
            }[pollutant]
            
            cat_col = f'Categoria_{pollutant}'
        else:  # WHO
            limit = {
                'MP10': 15, 'MP2.5': 5, 'O3': 60,
                'NO2': 25, 'SO2': 40, 'CO': 4, 'AOD': 0.05
            }[pollutant]
            
            cat_col = f'Categoria_WHO_{pollutant}'
        
        # Calcular excedências
        exceedances = (mun_data[pollutant] > limit).sum()
        exceedance_pct = exceedances / len(mun_data) * 100
        
        # Categoria predominante
        cat_counts = mun_data[cat_col].value_counts()
        predominant_cat = cat_counts.index[0] if not cat_counts.empty else "N/A"
        
        # Adicionar unidades
        units = {
            'MP10': 'μg/m³', 'MP2.5': 'μg/m³', 'O3': 'μg/m³',
            'NO2': 'μg/m³', 'SO2': 'μg/m³', 'CO': 'ppm', 'AOD': ''
        }[pollutant]
        
        # Adicionar à lista
        summary_data.append({
            'Poluente': pollutant,
            'Média': f"{mean_val:.2f} {units}",
            'Máximo': f"{max_val:.2f} {units}",
            'Mínimo': f"{min_val:.2f} {units}",
            'Desvio Padrão': f"{std_val:.2f}",
            'Excedências': f"{exceedances} dias ({exceedance_pct:.1f}%)",
            'Categoria Predominante': predominant_cat
        })
    
    # Criar dataframe
    summary_df = pd.DataFrame(summary_data)
    
    # Mostrar tabela
    st.dataframe(summary_df, use_container_width=True)
    
    # Opção para download dos dados
    st.download_button(
        label="⬇️ Baixar Dados Completos (CSV)",
        data=mun_data.to_csv(index=False).encode('utf-8'),
        file_name=f"qualidade_ar_{selected_mun}_{start_date}_a_{end_date}.csv",
        mime="text/csv"
    )

with tab3:
    st.header("🗺️ Mapa de Qualidade do Ar")
    
    # Selecionar data e poluente para o mapa
    col1, col2 = st.columns(2)
    
    with col1:
        selected_date = st.date_input(
            "Selecione uma data para visualizar o mapa",
            value=pd.to_datetime(air_data['Data'].min()).date()
        )
    
    with col2:
        # Opções: Categoria geral ou poluente específico
        pollutant_options = ['Categoria_Geral', 'MP10', 'MP2.5', 'O3', 'NO2', 'SO2', 'CO', 'AOD']
        selected_pollutant = st.selectbox(
            "Selecione o parâmetro a visualizar",
            options=pollutant_options,
            format_func=lambda x: "Qualidade Geral" if x == "Categoria_Geral" else x
        )
    
    # Converter para datetime
    selected_datetime = pd.to_datetime(selected_date)
    
    # Verificar se há dados para a data
    if not air_data[air_data['Data'].dt.date == selected_date].empty:
        # Criar mapa
        with st.spinner("Gerando mapa..."):
            fig = create_air_quality_map(air_data, ms_municipalities, selected_datetime, selected_pollutant, standard)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Adicionar legenda
                if selected_pollutant == 'Categoria_Geral':
                    st.markdown("""
                    ### Legenda de Qualidade do Ar
                    - 🔵 **Boa**: Qualidade do ar satisfatória, com mínimo ou nenhum risco à saúde.
                    - 🟢 **Moderada**: Qualidade do ar aceitável, mas pode haver risco para pessoas muito sensíveis.
                    - 🟡 **Ruim**: Membros de grupos sensíveis podem ter efeitos na saúde.
                    - 🟠 **Muito Ruim**: Todos podem começar a sentir efeitos na saúde, grupos sensíveis podem ter efeitos mais graves.
                    - 🔴 **Péssima**: Alerta de saúde. Toda a população pode ter riscos de saúde mais sérios.
                    """)
                else:
                    # Informações sobre o poluente selecionado
                    pollutant_info = {
                        'MP10': """
                        **Material Particulado (MP10)** - Partículas inaláveis com diâmetro menor que 10 micrômetros.
                        - **Fontes**: Queimadas, construção civil, ressuspensão de poeira, indústrias.
                        - **Efeitos na saúde**: Agravamento de asma, diminuição da função pulmonar, aumento de doenças respiratórias.
                        - **Limite CONAMA**: 50 μg/m³ (média de 24h)
                        - **Limite OMS**: 15 μg/m³ (média de 24h)
                        """,
                        'MP2.5': """
                        **Material Particulado Fino (MP2.5)** - Partículas inaláveis com diâmetro menor que 2,5 micrômetros.
                        - **Fontes**: Queimadas, veículos a diesel, processos industriais, formação secundária na atmosfera.
                        - **Efeitos na saúde**: Penetra profundamente nos pulmões e na corrente sanguínea, causando problemas respiratórios e cardiovasculares.
                        - **Limite CONAMA**: 25 μg/m³ (média de 24h)
                        - **Limite OMS**: 5 μg/m³ (média de 24h)
                        """,
                        'O3': """
                        **Ozônio (O3)** - Poluente secundário formado por reações fotoquímicas.
                        - **Fontes**: Formado pela reação de NOx e COVs na presença de luz solar.
                        - **Efeitos na saúde**: Irritação nos olhos e vias respiratórias, redução da função pulmonar, agravamento de asma.
                        - **Limite CONAMA**: 100 μg/m³ (média de 8h)
                        - **Limite OMS**: 60 μg/m³ (média de 8h)
                        """,
                        'NO2': """
                        **Dióxido de Nitrogênio (NO2)** - Gás poluente primário e precursor de outros poluentes.
                        - **Fontes**: Veículos automotores, usinas termelétricas, indústrias.
                        - **Efeitos na saúde**: Irritação do sistema respiratório, redução da capacidade pulmonar, aumento de infecções respiratórias.
                        - **Limite CONAMA**: 200 μg/m³ (média de 1h)
                        - **Limite OMS**: 25 μg/m³ (média de 24h)
                        """,
                        'SO2': """
                        **Dióxido de Enxofre (SO2)** - Gás poluente primário.
                        - **Fontes**: Queima de combustíveis com enxofre, termelétricas, indústrias.
                        - **Efeitos na saúde**: Irritação dos olhos, nariz e garganta, agravamento de doenças respiratórias, especialmente asma.
                        - **Limite CONAMA**: 20 μg/m³ (média de 24h)
                        - **Limite OMS**: 40 μg/m³ (média de 24h)
                        """,
                        'CO': """
                        **Monóxido de Carbono (CO)** - Gás tóxico sem cor ou odor.
                        - **Fontes**: Veículos, queimadas, combustão incompleta.
                        - **Efeitos na saúde**: Reduz a capacidade do sangue de transportar oxigênio, causando dores de cabeça, tonturas e até morte.
                        - **Limite CONAMA**: 9 ppm (média de 8h)
                        - **Limite OMS**: 4 ppm (média de 8h)
                        """,
                        'AOD': """
                        **Profundidade Óptica de Aerossóis (AOD)** - Medida da quantidade de luz bloqueada por partículas suspensas.
                        - **Fontes**: Queimadas, poeira, spray marinho, emissões urbanas e industriais.
                        - **Significado**: Valores mais altos indicam maior concentração de aerossóis e pior qualidade do ar.
                        - **Interpretação**:
                          - AOD < 0.1: Céu limpo, boa qualidade do ar
                          - AOD 0.1-0.3: Qualidade moderada
                          - AOD 0.3-0.5: Qualidade ruim
                          - AOD > 0.5: Qualidade muito ruim a péssima
                        """
                    }
                    
                    st.markdown(f"### Informações sobre {selected_pollutant}")
                    st.markdown(pollutant_info.get(selected_pollutant, ""))
    else:
        st.warning(f"Não há dados disponíveis para a data {selected_date.strftime('%d/%m/%Y')}.")

with tab4:
    st.header("🎬 Animação de Concentração de Poluentes")
    
    # Seleção de período e poluente para animação
    st.subheader("Configurações da Animação")
    
    col1, col2 = st.columns(2)
    
    with col1:
        anim_start_date = st.date_input(
            "Data inicial da animação",
            value=start_date,
            key="anim_start_date"
        )
        
        anim_end_date = st.date_input(
            "Data final da animação",
            value=min(start_date + timedelta(days=6), end_date),
            key="anim_end_date"
        )
    
    with col2:
        # Selecionar poluente para animação
        anim_pollutant = st.selectbox(
            "Parâmetro a animar",
            options=['Categoria_Geral', 'MP10', 'MP2.5', 'O3', 'NO2', 'SO2', 'CO', 'AOD'],
            format_func=lambda x: "Qualidade Geral" if x == "Categoria_Geral" else x,
            key="anim_pollutant"
        )
    
    # Botão para gerar animação
    if st.button("🎬 Gerar Animação de Concentração", type="primary"):
        if (anim_end_date - anim_start_date).days > 14:
            st.warning("Por favor, selecione um período de no máximo 14 dias para a animação.")
        elif anim_start_date > anim_end_date:
            st.error("A data inicial deve ser anterior à data final.")
        else:
            # Gerar animação avançada
            with st.spinner("Gerando animação de concentração..."):
                animation_fig = create_pollutant_animation(
                    air_data, 
                    ms_municipalities, 
                    pd.to_datetime(anim_start_date), 
                    pd.to_datetime(anim_end_date), 
                    anim_pollutant, 
                    standard
                )
            
            if animation_fig:
                st.plotly_chart(animation_fig, use_container_width=True)
                
                st.success("Animação gerada com sucesso! Use os controles abaixo da visualização para reproduzir a animação.")
                
                # Explicações sobre a animação
                st.markdown("""
                ### Como usar a animação:
                
                - Use o botão **▶️ Play** para iniciar a animação automática
                - Use o botão **⏸️ Pause** para pausar a animação
                - Você também pode arrastar o slider para visualizar uma data específica
                - Passe o mouse sobre os pontos ou áreas para ver detalhes
                
                Esta animação mostra a evolução espacial da concentração do poluente selecionado ao longo do tempo.
                Para poluentes específicos, é utilizada interpolação espacial para criar uma visualização contínua das concentrações.
                """)
            else:
                st.warning("Não foi possível gerar a animação. Verifique os dados e tente novamente.")

with tab5:
    st.header("📝 Relatórios de Qualidade do Ar")
    
    # Selecionar município para relatório
    report_mun = st.selectbox(
        "Selecione um município para gerar relatório",
        options=selected_municipalities,
        key="report_municipality"
    )
    
    # Gerar relatório
    if st.button("🔍 Gerar Relatório Detalhado"):
        with st.spinner("Gerando relatório..."):
            report = generate_air_quality_report(air_data, report_mun, start_date, end_date, standard)
            
            # Mostrar relatório
            st.subheader(f"Relatório de Qualidade do Ar - {report_mun}")
            st.markdown(f"""
            ### Informações Gerais
            - **Município**: {report['Município']}
            - **Período**: {report['Período']['Início'].strftime('%d/%m/%Y')} a {report['Período']['Fim'].strftime('%d/%m/%Y')} ({report['Período']['Total de Dias']} dias)
            - **Padrão Utilizado**: {report['Padrão']}
            """)
            
            # Mostrar distribuição de categorias
            st.subheader("Distribuição de Categorias")
            
            # Criar gráfico de pizza
            categories = ['Boa', 'Moderada', 'Ruim', 'Muito Ruim', 'Péssima']
            colors = ['#00ccff', '#009933', '#ffff00', '#ff9933', '#ff0000']
            
            cat_values = []
            for cat in categories:
                cat_values.append(report['Categorias'].get(cat, 0))
            
            fig = px.pie(
                values=cat_values,
                names=categories,
                title=f"Distribuição de Categorias - {report_mun}",
                color_discrete_sequence=colors
            )
            st.plotly_chart(fig)
            
            # Estatísticas dos poluentes
            st.subheader("Estatísticas por Poluente")
            
            # Criar colunas para cada poluente
            cols = st.columns(3)
            pollutants = list(report['Estatísticas'].keys())
            
            for i, pollutant in enumerate(pollutants):
                col_idx = i % 3
                with cols[col_idx]:
                    stats = report['Estatísticas'][pollutant]
                    
                    # Obter unidade
                    unit = 'ppm' if pollutant == 'CO' else ('adimensional' if pollutant == 'AOD' else 'μg/m³')
                    
                    st.markdown(f"""
                    #### {pollutant}
                    - **Média**: {stats['Média']:.2f} {unit}
                    - **Máximo**: {stats['Máximo']:.2f} {unit}
                    - **Mínimo**: {stats['Mínimo']:.2f} {unit}
                    - **Desvio Padrão**: {stats['Desvio Padrão']:.2f}
                    """)
                    
                    # Adicionar dia crítico se disponível
                    if pollutant in report['Dias Críticos']:
                        critical_day = report['Dias Críticos'][pollutant]
                        st.markdown(f"""
                        **Dia Crítico**: {critical_day['Data'].strftime('%d/%m/%Y')}  
                        **Valor**: {critical_day['Valor']:.2f} {unit}  
                        **Categoria**: {critical_day['Categoria']}
                        """)
            
            # Observações finais
            st.subheader("Recomendações e Observações")
            
            # Determinar recomendações com base na categoria predominante
            predominant_cat = max(report['Categorias'], key=report['Categorias'].get) if report['Categorias'] else "N/A"
            
            recommendations = {
                'Boa': """
                A qualidade do ar está satisfatória. Continue monitorando, mas não são necessárias medidas específicas.
                """,
                'Moderada': """
                A qualidade do ar está aceitável, mas pode haver riscos para pessoas muito sensíveis.
                - Pessoas com doenças respiratórias ou cardíacas devem limitar esforços prolongados ao ar livre.
                - Recomenda-se continuar o monitoramento regular da qualidade do ar.
                """,
                'Ruim': """
                A qualidade do ar apresenta riscos para grupos sensíveis.
                - Pessoas com doenças respiratórias ou cardíacas, idosos e crianças devem evitar esforços prolongados ao ar livre.
                - Recomenda-se intensificar o monitoramento e implementar medidas de controle de emissões.
                - Considerar campanhas de conscientização sobre a qualidade do ar.
                """,
                'Muito Ruim': """
                A qualidade do ar está insalubre e pode afetar toda a população.
                - Todos devem limitar atividades ao ar livre, especialmente grupos sensíveis.
                - Recomenda-se implementar medidas de controle de emissões de forma urgente.
                - Considerar a emissão de alertas de saúde pública.
                - Intensificar a fiscalização de fontes poluidoras.
                """,
                'Péssima': """
                A qualidade do ar está perigosa para a saúde.
                - Todos devem evitar atividades ao ar livre.
                - Recomenda-se implementar medidas emergenciais de controle de poluição.
                - Emitir alertas de saúde pública e considerar a suspensão de atividades em escolas e locais públicos.
                - Implementar rodízio de veículos se aplicável.
                - Suspender atividades industriais com altas emissões temporariamente.
                """
            }
            
            st.markdown(recommendations.get(predominant_cat, "Não há recomendações disponíveis."))
            
            # Opção para download do relatório em formato JSON
            report_json = json.dumps(
                {
                    'Município': report['Município'],
                    'Período': {
                        'Início': report['Período']['Início'].strftime('%d/%m/%Y'),
                        'Fim': report['Período']['Fim'].strftime('%d/%m/%Y'),
                        'Total de Dias': report['Período']['Total de Dias']
                    },
                    'Padrão': report['Padrão'],
                    'Categorias': report['Categorias'],
                    'Estatísticas': {
                        p: {k: float(v) for k, v in stats.items()} 
                        for p, stats in report['Estatísticas'].items()
                    },
                    'Dias Críticos': {
                        p: {
                            'Data': day['Data'].strftime('%d/%m/%Y'),
                            'Valor': float(day['Valor']),
                            'Categoria': day['Categoria']
                        } 
                        for p, day in report['Dias Críticos'].items()
                    },
                    'Recomendações': recommendations.get(predominant_cat, "Não há recomendações disponíveis.")
                },
                ensure_ascii=False,
                indent=4
            )
            
            st.download_button(
                label="⬇️ Baixar Relatório (JSON)",
                data=report_json.encode('utf-8'),
                file_name=f"relatorio_{report_mun}_{start_date}_a_{end_date}.json",
                mime="application/json"
            )
            
            # Opção para download do relatório em PDF (simulação)
            st.markdown("""
            Para gerar relatórios em PDF, seria necessário implementar uma biblioteca adicional como ReportLab ou pdfkit.
            Esta funcionalidade pode ser adicionada em versões futuras do aplicativo.
            """)

with tab6:
    st.header("🔮 Previsão da Qualidade do Ar")
    
    # Selecionar município para previsão
    forecast_mun = st.selectbox(
        "Selecione um município para previsão",
        options=selected_municipalities,
        key="forecast_municipality"
    )
    
    # Selecionar número de dias para previsão
    forecast_days = st.slider(
        "Número de dias para previsão",
        min_value=1,
        max_value=7,
        value=5,
        key="forecast_days"
    )
    
    # Botão para gerar previsão
    if st.button("🔮 Gerar Previsão", key="generate_forecast"):
        with st.spinner("Gerando previsão de qualidade do ar..."):
            # Obter dados de previsão
            forecast_data = get_air_quality_forecast([forecast_mun], forecast_days)
            
            if forecast_data is not None and not forecast_data.empty:
                # Classificar dados de previsão
                forecast_data = classify_air_quality_conama(forecast_data)
                forecast_data = classify_air_quality_who(forecast_data)
                
                # Filtrar para o município selecionado
                mun_forecast = forecast_data[forecast_data['Município'] == forecast_mun]
                
                # Mostrar previsão em formato de cards
                st.subheader(f"Previsão para {forecast_mun} - Próximos {forecast_days} dias")
                
                # Seletor de padrão para previsão
                forecast_standard = st.radio(
                    "Padrão para previsão",
                    options=["CONAMA", "OMS"],
                    horizontal=True,
                    key="forecast_standard"
                )
                
                # Selecionar coluna de categoria
                cat_col = 'Categoria_Geral' if forecast_standard == 'CONAMA' else 'Categoria_Geral_WHO'
                
                # Criar cards para cada dia
                cols = st.columns(min(forecast_days, 5))
                
                for i, (_, row) in enumerate(mun_forecast.iterrows()):
                    if i >= len(cols):
                        break
                        
                    with cols[i]:
                        date = row['Data'].strftime('%d/%m/%Y')
                        category = row[cat_col]
                        
                        # Cores para as categorias
                        cat_colors = {
                            'Boa': '#00ccff',
                            'Moderada': '#009933',
                            'Ruim': '#ffff00',
                            'Muito Ruim': '#ff9933',
                            'Péssima': '#ff0000'
                        }
                        
                        color = cat_colors.get(category, '#cccccc')
                        
                        # Criar card
                        st.markdown(f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: {color}; color: {'black' if category in ['Boa', 'Moderada', 'Ruim'] else 'white'}; text-align: center;">
                            <h4>{date}</h4>
                            <h3>{category}</h3>
                            <p>MP10: {row['MP10']:.1f} μg/m³</p>
                            <p>MP2.5: {row['MP2.5']:.1f} μg/m³</p>
                            <p>O3: {row['O3']:.1f} μg/m³</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Gráfico de previsão
                st.subheader("Gráfico de Previsão")
                
                # Criar figura para previsão
                fig = go.Figure()
                
                # Adicionar linha para cada poluente
                pollutants = ['MP10', 'MP2.5', 'O3', 'NO2', 'SO2', 'CO']
                colors = px.colors.qualitative.Plotly
                
                for i, pollutant in enumerate(pollutants):
                    fig.add_trace(go.Scatter(
                        x=mun_forecast['Data'],
                        y=mun_forecast[pollutant],
                        mode='lines+markers',
                        name=pollutant,
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=8)
                    ))
                
                # Adicionar linha para categoria
                cat_map = {'Boa': 0, 'Moderada': 1, 'Ruim': 2, 'Muito Ruim': 3, 'Péssima': 4}
                mun_forecast['cat_num'] = mun_forecast[cat_col].map(cat_map)
                
                # Adicionar eixo secundário para categorias
                fig.add_trace(go.Scatter(
                    x=mun_forecast['Data'],
                    y=mun_forecast['cat_num'],
                    mode='lines+markers',
                    name='Qualidade do Ar',
                    line=dict(color='black', width=3, dash='dot'),
                    marker=dict(
                        size=12,
                        color=mun_forecast[cat_col].map(lambda c: cat_colors.get(c, '#cccccc'))
                    ),
                    yaxis='y2'
                ))
                
                # Atualizar layout
                fig.update_layout(
                    title=f"Previsão de Qualidade do Ar - {forecast_mun}",
                    xaxis=dict(title='Data'),
                    yaxis=dict(title='Concentração'),
                    yaxis2=dict(
                        title='Categoria',
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black'),
                        overlaying='y',
                        side='right',
                        tickvals=[0, 1, 2, 3, 4],
                        ticktext=['Boa', 'Moderada', 'Ruim', 'Muito Ruim', 'Péssima']
                    ),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Adicionar texto explicativo
                st.markdown("""
                ### Sobre a Previsão
                
                Esta previsão é baseada nos dados do Serviço de Monitoramento da Atmosfera Copernicus (CAMS), 
                um serviço que fornece dados contínuos de qualidade do ar e composição atmosférica.
                
                Os dados são processados de acordo com os padrões selecionados (CONAMA ou OMS) e mostram 
                a evolução prevista dos principais poluentes e da qualidade geral do ar para os próximos dias.
                
                **Observação:** As previsões têm maior precisão para os primeiros dias e podem estar sujeitas 
                a variações dependendo de condições meteorológicas não previstas e eventos como queimadas.
                """)
                
                # Adicionar botão para download dos dados de previsão
                csv = mun_forecast.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Baixar Dados da Previsão (CSV)",
                    data=csv,
                    file_name=f"previsao_{forecast_mun}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.error("Não foi possível obter dados de previsão. Tente novamente mais tarde.")

# Adicionar informações na parte inferior
st.markdown("---")
st.markdown("""
### 🌍 Dados do Copernicus Atmosphere Monitoring Service (CAMS)

Este aplicativo utiliza dados do Serviço de Monitoramento da Atmosfera Copernicus (CAMS), uma 
iniciativa da União Europeia que fornece informações contínuas sobre a qualidade do ar e a 
composição atmosférica em escala global e regional.

**Informações sobre os dados:**
- MP10 e MP2.5: Material particulado com diâmetro inferior a 10 e 2,5 micrômetros, respectivamente.
- O3: Ozônio, um poluente secundário formado por reações fotoquímicas.
- NO2: Dióxido de nitrogênio, principalmente de fontes de combustão como veículos.
- SO2: Dióxido de enxofre, de fontes industriais e queima de combustíveis com enxofre.
- CO: Monóxido de carbono, de combustão incompleta.
- AOD: Profundidade Óptica de Aerossóis, uma medida da quantidade de aerossóis na atmosfera.

Para mais informações sobre o CAMS, visite [https://atmosphere.copernicus.eu/](https://atmosphere.copernicus.eu/)

### 📊 Referências de Qualidade do Ar

#### Resolução CONAMA nº 491/2018
A Resolução CONAMA nº 491/2018 estabelece os padrões de qualidade do ar no Brasil, definindo limites para concentrações de poluentes atmosféricos.

**Poluentes Regulamentados:**
- MP10 (Partículas Inaláveis): 50 μg/m³ (média de 24h)
- MP2.5 (Partículas Inaláveis Finas): 25 μg/m³ (média de 24h)
- O3 (Ozônio): 100 μg/m³ (média de 8h)
- NO2 (Dióxido de Nitrogênio): 200 μg/m³ (média de 1h)
- SO2 (Dióxido de Enxofre): 20 μg/m³ (média de 24h)
- CO (Monóxido de Carbono): 9 ppm (média de 8h)

#### Diretrizes da OMS (2021)
A Organização Mundial da Saúde (OMS) atualizou suas diretrizes de qualidade do ar em 2021, estabelecendo limites mais restritivos.

**Limites Recomendados:**
- MP10 (Partículas Inaláveis): 15 μg/m³ (média de 24h)
- MP2.5 (Partículas Inaláveis Finas): 5 μg/m³ (média de 24h)
- O3 (Ozônio): 60 μg/m³ (média de 8h)
- NO2 (Dióxido de Nitrogênio): 25 μg/m³ (média de 24h)
- SO2 (Dióxido de Enxofre): 40 μg/m³ (média de 24h)
- CO (Monóxido de Carbono): 4 ppm (média de 8h)

### 🏥 Efeitos na Saúde

- **Boa**: Qualidade do ar satisfatória, com mínimo ou nenhum risco à saúde.
- **Moderada**: Qualidade do ar aceitável, mas pode haver risco para pessoas muito sensíveis.
- **Ruim**: Membros de grupos sensíveis podem ter efeitos na saúde.
- **Muito Ruim**: Todos podem começar a sentir efeitos na saúde, grupos sensíveis podem ter efeitos mais graves.
- **Péssima**: Alerta de saúde. Toda a população pode ter riscos de saúde mais sérios.

### 💨 Fontes Comuns de Poluição Atmosférica em Mato Grosso do Sul

- **Queimadas**: Principal fonte de poluição atmosférica no estado, especialmente durante a estação seca.
- **Veículos**: Emissões de veículos em centros urbanos como Campo Grande e Dourados.
- **Indústrias**: Processamento de cana-de-açúcar, frigoríficos, papel e celulose.
- **Agropecuária**: Poeira de solo, ressuspensão e emissões de implementos agrícolas.
- **Atividades mineradoras**: Principalmente na região de Corumbá.

---

Sistema desenvolvido para monitoramento da qualidade do ar no estado de Mato Grosso do Sul - Brasil.
Utiliza dados do Serviço de Monitoramento da Atmosfera Copernicus (CAMS).
""")

# Adicionar créditos e versão
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🔄 Versão 2.0
**Novas funcionalidades:**
- Integração com dados reais do CAMS
- Previsão da qualidade do ar
- Animação avançada de concentração de poluentes
- Interpolação espacial para visualização contínua
""")

# Informações sobre o CAMS na sidebar
st.sidebar.markdown("---")
st.sidebar.info("""
**Dados fornecidos por:**  
[Copernicus Atmosphere Monitoring Service (CAMS)](https://atmosphere.copernicus.eu/)
""")
