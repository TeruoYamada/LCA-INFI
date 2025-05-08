import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.dates as mdates
import requests
import io
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# Configuração da página
st.set_page_config(
    page_title="Qualidade do Ar - MS",
    page_icon="🌬️",
    layout="wide"
)

# Título e introdução
st.title("🌬️ Sistema de Monitoramento da Qualidade do Ar - Mato Grosso do Sul")
st.markdown("""
Este aplicativo permite visualizar, analisar e gerar relatórios sobre a qualidade do ar
nos municípios de Mato Grosso do Sul, seguindo os padrões da Resolução CONAMA nº 491/2018 e da OMS.
""")

# Função para carregar os municípios de MS
@st.cache_data
def load_ms_municipalities():
    try:
        # URL para o shapefile dos municípios do MS
        url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/MS/MS_Municipios_2022.zip"
        
        try:
            gdf = gpd.read_file(url)
            return gdf
        except Exception as e:
            st.warning(f"Erro ao carregar shapefile: {str(e)}")
            # Criar um geodataframe simplificado com alguns municípios
            data = {
                'NM_MUN': ['Campo Grande', 'Dourados', 'Três Lagoas', 'Corumbá', 'Ponta Porã',
                          'Naviraí', 'Nova Andradina', 'Aquidauana', 'Maracaju', 'Paranaíba'],
                'geometry': [
                    gpd.points_from_xy([-54.6201], [-20.4697])[0].buffer(0.2),
                    gpd.points_from_xy([-54.812], [-22.2231])[0].buffer(0.2),
                    gpd.points_from_xy([-51.7005], [-20.7849])[0].buffer(0.2),
                    gpd.points_from_xy([-57.651], [-19.0082])[0].buffer(0.2),
                    gpd.points_from_xy([-55.7271], [-22.5334])[0].buffer(0.2),
                    gpd.points_from_xy([-54.1994], [-23.0624])[0].buffer(0.2),
                    gpd.points_from_xy([-53.3435], [-22.2384])[0].buffer(0.2),
                    gpd.points_from_xy([-55.7879], [-20.4697])[0].buffer(0.2),
                    gpd.points_from_xy([-55.1678], [-21.6407])[0].buffer(0.2),
                    gpd.points_from_xy([-51.1909], [-19.6746])[0].buffer(0.2)
                ]
            }
            gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
            return gdf
    except Exception as e:
        st.error(f"Não foi possível carregar os municípios: {str(e)}")
        return gpd.GeoDataFrame(columns=['NM_MUN', 'geometry'], crs="EPSG:4326")

# Função para simular/obter dados de qualidade do ar
def get_air_quality_data(municipalities, start_date, end_date):
    """
    Simula ou obtém dados de qualidade do ar para os municípios selecionados
    no período especificado. Em um ambiente de produção, substituir por
    chamadas a APIs reais de qualidade do ar.
    """
    # Lista de poluentes
    pollutants = ['MP10', 'MP2.5', 'O3', 'NO2', 'SO2', 'CO']
    
    # Criar DataFrame com datas no período
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Lista para armazenar os dados
    data_list = []
    
    # Para cada município e data, gerar dados simulados
    for municipality in municipalities:
        # Seed baseado no nome do município para gerar dados consistentes
        seed = sum(ord(c) for c in municipality)
        np.random.seed(seed)
        
        # Gerar base para o município (valores médios)
        base_values = {
            'MP10': np.random.uniform(20, 60),
            'MP2.5': np.random.uniform(10, 30),
            'O3': np.random.uniform(40, 100),
            'NO2': np.random.uniform(20, 60),
            'SO2': np.random.uniform(10, 40),
            'CO': np.random.uniform(1, 5)
        }
        
        # Tendência ao longo do tempo (para simular variação sazonal)
        trend_factor = np.random.uniform(0.8, 1.2, size=len(date_range))
        
        for i, date in enumerate(date_range):
            # Aplicar tendência e variação diária
            daily_values = {
                pollutant: max(0, base_values[pollutant] * trend_factor[i] * 
                             np.random.uniform(0.8, 1.2))
                for pollutant in pollutants
            }
            
            # Adicionar à lista de dados
            data_list.append({
                'Município': municipality,
                'Data': date,
                **daily_values
            })
    
    # Criar DataFrame
    df = pd.DataFrame(data_list)
    return df

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
        'CO': [9, 11, 13, 15, 15]          # ppm - média de 8h
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
        'CO': [4, 9, 13, 15, 15]           # ppm - média de 8h
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
    pollutants = ['MP10', 'MP2.5', 'O3', 'NO2', 'SO2', 'CO']
    fig = make_subplots(rows=3, cols=2, subplot_titles=pollutants,
                       shared_xaxes=True, vertical_spacing=0.1)
    
    # Adicionar gráficos para cada poluente
    for i, pollutant in enumerate(pollutants):
        row = i // 2 + 1
        col = i % 2 + 1
        
        # Adicionar série temporal
        fig.add_trace(
            go.Scatter(
                x=mun_data['Data'],
                y=mun_data[pollutant],
                mode='lines+markers',
                name=pollutant,
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate=f"{pollutant}: %{{y:.1f}}<br>Data: %{{x|%d/%m/%Y}}<br>Categoria: {mun_data[f'Categoria_{pollutant}' if standard=='CONAMA' else f'Categoria_WHO_{pollutant}']}"
            ),
            row=row, col=col
        )
        
        # Adicionar limites (simplificado - apenas o primeiro limite para "Boa")
        if standard == 'CONAMA':
            limits = {
                'MP10': 50,
                'MP2.5': 25,
                'O3': 100,
                'NO2': 200,
                'SO2': 20,
                'CO': 9
            }
        else:  # WHO
            limits = {
                'MP10': 15,
                'MP2.5': 5,
                'O3': 60,
                'NO2': 25,
                'SO2': 40,
                'CO': 4
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
    
    # Atualizar layout
    fig.update_layout(
        title=f"Série Temporal de Poluentes - {municipality} (Padrão: {standard})",
        height=800,
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
        'CO': 'ppm'
    }
    
    for i, pollutant in enumerate(pollutants):
        row = i // 2 + 1
        col = i % 2 + 1
        fig.update_yaxes(title_text=units[pollutant], row=row, col=col)
        
        if row == 3:  # Última linha
            fig.update_xaxes(title_text="Data", row=row, col=col)
    
    return fig

# Função para gerar mapa de qualidade do ar
def create_air_quality_map(df, gdf, date, standard='CONAMA'):
    """
    Cria um mapa interativo com a qualidade do ar para todos os municípios em uma data específica
    Usando uma abordagem alternativa com go.Figure para evitar problemas com px.choropleth_mapbox
    """
    # Filtrar dados para a data
    date_data = df[df['Data'] == date]
    
    # Selecionar as colunas de categoria e cor apropriadas
    if standard == 'CONAMA':
        cat_col = 'Categoria_Geral'
        color_col = 'Cor'
    else:  # WHO
        cat_col = 'Categoria_Geral_WHO'
        color_col = 'Cor_WHO'
    
    # Mesclar dados com o geodataframe
    map_data = gdf.merge(date_data, left_on='NM_MUN', right_on='Município', how='inner')
    
    # Verificar se há dados para criar o mapa
    if map_data.empty:
        st.warning(f"Não há dados de qualidade do ar disponíveis para {date.strftime('%d/%m/%Y')}")
        return None
    
    # Criar uma figura básica com scatter_mapbox (alternativa mais estável)
    fig = go.Figure()
    
    # Cores para as categorias
    color_map = {
        'Boa': '#00ccff',
        'Moderada': '#009933',
        'Ruim': '#ffff00',
        'Muito Ruim': '#ff9933',
        'Péssima': '#ff0000'
    }
    
    # Adicionar pontos para cada município
    for idx, row in map_data.iterrows():
        # Obter a categoria de qualidade do ar e a cor correspondente
        category = row[cat_col]
        color = color_map.get(category, '#cccccc')
        
        # Obter o centroide da geometria para posicionar o ponto
        try:
            # Se a geometria for um polígono, extrair o centroide
            if hasattr(row.geometry, 'centroid'):
                lon, lat = row.geometry.centroid.x, row.geometry.centroid.y
            # Se for um ponto (caso do fallback), usar as coordenadas diretamente
            elif hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                lon, lat = row.geometry.x, row.geometry.y
            else:
                # Usar coordenadas extraídas dos dados simulados como fallback
                city_coords = {
                    'Campo Grande': [-20.4697, -54.6201],
                    'Dourados': [-22.2231, -54.812],
                    'Três Lagoas': [-20.7849, -51.7005],
                    'Corumbá': [-19.0082, -57.651],
                    'Ponta Porã': [-22.5334, -55.7271],
                    'Naviraí': [-23.0624, -54.1994],
                    'Nova Andradina': [-22.2384, -53.3435],
                    'Aquidauana': [-20.4697, -55.7879],
                    'Maracaju': [-21.6407, -55.1678],
                    'Paranaíba': [-19.6746, -51.1909]
                }
                if row['Município'] in city_coords:
                    lat, lon = city_coords[row['Município']]
                else:
                    # Skip if we can't determine coordinates
                    continue
            
            # Criar texto para hover
            hover_text = f"<b>{row['Município']}</b><br>" + \
                        f"Qualidade do Ar: {category}<br>" + \
                        f"MP10: {row['MP10']:.2f} μg/m³<br>" + \
                        f"MP2.5: {row['MP2.5']:.2f} μg/m³<br>" + \
                        f"O3: {row['O3']:.2f} μg/m³<br>" + \
                        f"NO2: {row['NO2']:.2f} μg/m³<br>" + \
                        f"SO2: {row['SO2']:.2f} μg/m³<br>" + \
                        f"CO: {row['CO']:.2f} ppm"
            
            # Adicionar marker
            fig.add_trace(go.Scattermapbox(
                lat=[lat],
                lon=[lon],
                mode='markers',
                marker=dict(
                    size=15,
                    color=color,
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
        title=f"Mapa de Qualidade do Ar - {date.strftime('%d/%m/%Y')} (Padrão: {standard})",
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
    pollutants = ['MP10', 'MP2.5', 'O3', 'NO2', 'SO2', 'CO']
    
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
with st.spinner("🔄 Carregando dados de qualidade do ar..."):
    air_data = get_air_quality_data(selected_municipalities, start_date, end_date)
    
    # Classificar qualidade do ar
    if standard == "CONAMA":
        air_data = classify_air_quality_conama(air_data)
    else:  # WHO
        air_data = classify_air_quality_who(air_data)

# Layout principal - usar abas
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visão Geral", 
    "📈 Séries Temporais", 
    "🗺️ Mapa de Qualidade do Ar",
    "📝 Relatórios"
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
            pollutants = ['MP10', 'MP2.5', 'O3', 'NO2', 'SO2', 'CO']
            avg_values = {p: mun_data[p].mean() for p in pollutants}
            # Encontrar o poluente com maior valor em relação ao limite
            if standard == "CONAMA":
                limits = {
                    'MP10': 50, 'MP2.5': 25, 'O3': 100,
                    'NO2': 200, 'SO2': 20, 'CO': 9
                }
            else:  # WHO
                limits = {
                    'MP10': 15, 'MP2.5': 5, 'O3': 60,
                    'NO2': 25, 'SO2': 40, 'CO': 4
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
    pollutants = ['MP10', 'MP2.5', 'O3', 'NO2', 'SO2', 'CO']
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
                'NO2': 200, 'SO2': 20, 'CO': 9
            }[pollutant]
            
            cat_col = f'Categoria_{pollutant}'
        else:  # WHO
            limit = {
                'MP10': 15, 'MP2.5': 5, 'O3': 60,
                'NO2': 25, 'SO2': 40, 'CO': 4
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
            'NO2': 'μg/m³', 'SO2': 'μg/m³', 'CO': 'ppm'
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
    
    # Selecionar data para o mapa
    selected_date = st.date_input(
        "Selecione uma data para visualizar o mapa",
        value=pd.to_datetime(air_data['Data'].min()).date()
    )
    
    # Converter para datetime
    selected_datetime = pd.to_datetime(selected_date)
    
    # Verificar se há dados para a data
    if not air_data[air_data['Data'].dt.date == selected_date].empty:
        # Criar mapa
        with st.spinner("Gerando mapa..."):
            fig = create_air_quality_map(air_data, ms_municipalities, selected_datetime, standard)
            st.plotly_chart(fig, use_container_width=True)
            
            # Adicionar legenda
            st.markdown("""
            ### Legenda de Qualidade do Ar
            - 🔵 **Boa**: Qualidade do ar satisfatória, com mínimo ou nenhum risco à saúde.
            - 🟢 **Moderada**: Qualidade do ar aceitável, mas pode haver risco para pessoas muito sensíveis.
            - 🟡 **Ruim**: Membros de grupos sensíveis podem ter efeitos na saúde.
            - 🟠 **Muito Ruim**: Todos podem começar a sentir efeitos na saúde, grupos sensíveis podem ter efeitos mais graves.
            - 🔴 **Péssima**: Alerta de saúde. Toda a população pode ter riscos de saúde mais sérios.
            """)
    else:
        st.warning(f"Não há dados disponíveis para a data {selected_date.strftime('%d/%m/%Y')}.")

with tab4:
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
                    unit = 'ppm' if pollutant == 'CO' else 'μg/m³'
                    
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

# Adicionar informações na parte inferior
st.markdown("---")
st.markdown("""
### ℹ️ Sobre os Padrões de Qualidade do Ar

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
- CO (Monóxido de Carbono): 4 ppm (média de 24h)

### 🏥 Efeitos na Saúde

- **Boa**: Qualidade do ar satisfatória, com mínimo ou nenhum risco à saúde.
- **Moderada**: Qualidade do ar aceitável, mas pode haver risco para pessoas muito sensíveis.
- **Ruim**: Membros de grupos sensíveis podem ter efeitos na saúde.
- **Muito Ruim**: Todos podem começar a sentir efeitos na saúde, grupos sensíveis podem ter efeitos mais graves.
- **Péssima**: Alerta de saúde. Toda a população pode ter riscos de saúde mais sérios.

---

Desenvolvido para monitoramento da qualidade do ar no estado de Mato Grosso do Sul - Brasil.
""")
