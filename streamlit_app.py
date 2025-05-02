import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cdsapi
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import animation
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from sklearn.linear_model import LinearRegression
import io

# Configuração inicial da página
st.set_page_config(layout="wide", page_title="Qualidade do Ar - MS")

# Carregar autenticação a partir do secrets.toml
try:
    ads_url = st.secrets["ads"]["url"]
    ads_key = st.secrets["ads"]["key"]
    client = cdsapi.Client(url=ads_url, key=ads_key)
except Exception as e:
    st.error("❌ Erro ao carregar as credenciais do CDS API. Verifique seu secrets.toml.")
    st.warning("Para uso local, crie um arquivo .streamlit/secrets.toml com suas credenciais")
    client = None

# Dicionário com padrões de qualidade do ar segundo a Resolução CONAMA nº 491/2018
# Valores em μg/m³, exceto CO que está em ppm
CONAMA_STANDARDS = {
    'MP10': {
        'Boa': 50,
        'Moderada': 100,
        'Ruim': 150,
        'Muito Ruim': 250,
        'Péssima': 250
    },
    'MP2.5': {
        'Boa': 25,
        'Moderada': 50,
        'Ruim': 75,
        'Muito Ruim': 125,
        'Péssima': 125
    },
    'O3': {
        'Boa': 100,
        'Moderada': 130,
        'Ruim': 160,
        'Muito Ruim': 200,
        'Péssima': 200
    },
    'NO2': {
        'Boa': 200,
        'Moderada': 240,
        'Ruim': 320,
        'Muito Ruim': 1130,
        'Péssima': 1130
    },
    'SO2': {
        'Boa': 20,
        'Moderada': 40,
        'Ruim': 365,
        'Muito Ruim': 800,
        'Péssima': 800
    },
    'CO': {
        'Boa': 9,
        'Moderada': 11,
        'Ruim': 13,
        'Muito Ruim': 15,
        'Péssima': 15
    }
}

# Dicionário com padrões da OMS (2021) - Valores em μg/m³, exceto CO que está em ppm
WHO_STANDARDS = {
    'MP10': {
        'Diário': 45,
        'Anual': 15
    },
    'MP2.5': {
        'Diário': 15,
        'Anual': 5
    },
    'O3': {
        'Pico de temporada': 60,
        '8 horas': 100
    },
    'NO2': {
        'Diário': 25,
        'Anual': 10
    },
    'SO2': {
        'Diário': 40
    },
    'CO': {
        '24 horas': 4
    }
}

# Cores para categorias de qualidade do ar
QUALITY_COLORS = {
    'Boa': '#00e400',  # Verde
    'Moderada': '#ffff00',  # Amarelo
    'Ruim': '#ff7e00',  # Laranja
    'Muito Ruim': '#ff0000',  # Vermelho
    'Péssima': '#99004c'  # Roxo
}

# Dicionário com algumas cidades do MS
cities = {
    "Campo Grande": [-20.4697, -54.6201],
    "Dourados": [-22.2231, -54.812],
    "Três Lagoas": [-20.7849, -51.7005],
    "Corumbá": [-19.0082, -57.651],
    "Ponta Porã": [-22.5334, -55.7271],
    "Naviraí": [-23.0613, -54.1995],
    "Nova Andradina": [-22.2384, -53.3449],
    "Aquidauana": [-20.4666, -55.7868],
    "Coxim": [-18.5013, -54.7603],
    "Bonito": [-21.1261, -56.4836]
}

# Função para carregar shapefile dos municípios de MS
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
        except Exception as e:
            # Fallback: criar geodataframe simplificado com alguns municípios
            data = {
                'NM_MUN': list(cities.keys()),
                'geometry': [
                    gpd.points_from_xy([lon], [lat])[0].buffer(0.2)
                    for city, (lat, lon) in cities.items()
                ]
            }
            gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
            st.warning(f"Usando geometrias simplificadas para os municípios: {str(e)}")
            return gdf
    except Exception as e:
        st.warning(f"Não foi possível carregar os shapes dos municípios: {str(e)}")
        # Retornar DataFrame vazio com estrutura esperada
        return gpd.GeoDataFrame(columns=['NM_MUN', 'geometry'], crs="EPSG:4326")

# Função para classificar a qualidade do ar baseada nos padrões CONAMA
def classify_air_quality(pollutant, value):
    """Classifica a qualidade do ar para um poluente baseado no valor e padrões do CONAMA"""
    if pollutant not in CONAMA_STANDARDS:
        return "Não classificado", "#808080"  # Cinza para não classificado
    
    standards = CONAMA_STANDARDS[pollutant]
    
    if value <= standards['Boa']:
        return "Boa", QUALITY_COLORS['Boa']
    elif value <= standards['Moderada']:
        return "Moderada", QUALITY_COLORS['Moderada']
    elif value <= standards['Ruim']:
        return "Ruim", QUALITY_COLORS['Ruim']
    elif value <= standards['Muito Ruim']:
        return "Muito Ruim", QUALITY_COLORS['Muito Ruim']
    else:
        return "Péssima", QUALITY_COLORS['Péssima']

# Função para comparar com padrões da OMS
def compare_who_standards(pollutant, value, period='Diário'):
    """Compara o valor do poluente com os padrões da OMS"""
    if pollutant not in WHO_STANDARDS:
        return "Sem padrão OMS", False
    
    standards = WHO_STANDARDS[pollutant]
    
    if period not in standards:
        available_periods = list(standards.keys())
        if not available_periods:
            return "Sem padrão OMS", False
        period = available_periods[0]  # Usar o primeiro período disponível
    
    limit = standards[period]
    complies = value <= limit
    
    return f"{'Atende' if complies else 'Não atende'} (Limite: {limit})", complies

# Função para gerar séries temporais sintéticas de poluentes
def generate_synthetic_data(city, start_date, end_date, pollutants):
    """Gera dados sintéticos para simulação de poluentes atmosféricos"""
    # Criar range de datas
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Criar DataFrame base
    df = pd.DataFrame({'datetime': date_range})
    
    # Seed baseado no nome da cidade para ter padrões consistentes por cidade
    city_seed = sum(ord(c) for c in city)
    np.random.seed(city_seed)
    
    # Configurações por poluente com base em padrões típicos
    pollutant_config = {
        'MP10': {'base': 30, 'daily_var': 15, 'noise': 10, 'trend': 0.05},
        'MP2.5': {'base': 12, 'daily_var': 7, 'noise': 5, 'trend': 0.02},
        'O3': {'base': 50, 'daily_var': 30, 'noise': 15, 'trend': 0.1},
        'NO2': {'base': 100, 'daily_var': 60, 'noise': 30, 'trend': 0.15},
        'SO2': {'base': 10, 'daily_var': 5, 'noise': 3, 'trend': 0.01},
        'CO': {'base': 5, 'daily_var': 2, 'noise': 1, 'trend': 0.005}
    }
    
    # Adicionar colunas de poluentes
    for pollutant in pollutants:
        if pollutant in pollutant_config:
            config = pollutant_config[pollutant]
            
            # Componente de hora do dia (ciclo diário)
            hour_of_day = np.array([h.hour for h in date_range])
            daily_pattern = np.sin(hour_of_day * 2 * np.pi / 24) * config['daily_var']
            
            # Componente de tendência (longo prazo)
            days_from_start = (df['datetime'] - df['datetime'].min()).dt.total_seconds() / (24 * 3600)
            trend = days_from_start.values * config['trend']
            
            # Componente aleatório
            noise = np.random.normal(0, config['noise'], len(df))
            
            # Juntar componentes
            values = config['base'] + daily_pattern + trend + noise
            
            # Garantir que não há valores negativos
            values = np.maximum(values, 0)
            
            # Adicionar ao DataFrame
            df[pollutant] = values
    
    # Incluir eventos especiais (picos de poluição) em datas aleatórias
    n_events = (end_date - start_date).days // 3  # Aproximadamente um evento a cada 3 dias
    for _ in range(n_events):
        event_day = np.random.randint(0, len(df))
        event_duration = np.random.randint(6, 24)  # Duração de 6 a 24 horas
        
        # Escolher um poluente aleatório para o evento
        event_pollutant = np.random.choice(pollutants)
        
        # Multiplicador para o evento
        multiplier = np.random.uniform(1.5, 3.0)
        
        # Aplicar evento
        start_idx = max(0, event_day - event_duration // 2)
        end_idx = min(len(df), start_idx + event_duration)
        
        # Aplicar aumento gradual e diminuição gradual
        for i in range(start_idx, end_idx):
            position = (i - start_idx) / event_duration
            intensity = np.sin(position * np.pi) * (multiplier - 1) + 1
            df.loc[i, event_pollutant] *= intensity
    
    return df

# Função para calcular estatísticas por poluente
def calculate_pollutant_stats(df, pollutants):
    """Calcula estatísticas básicas para cada poluente"""
    stats = {}
    
    for pollutant in pollutants:
        if pollutant in df.columns:
            current = df[pollutant].iloc[-1] if not df.empty else 0
            avg = df[pollutant].mean() if not df.empty else 0
            max_val = df[pollutant].max() if not df.empty else 0
            
            # Classificar qualidade atual
            quality, color = classify_air_quality(pollutant, current)
            
            # Comparar com OMS
            who_comparison, who_complies = compare_who_standards(pollutant, current)
            
            stats[pollutant] = {
                'current': current,
                'average': avg,
                'maximum': max_val,
                'quality': quality,
                'color': color,
                'who_comparison': who_comparison,
                'who_complies': who_complies
            }
    
    return stats

# Função para gerar previsão de poluentes
def predict_pollutant_trends(df, pollutants, days_ahead=3):
    """Gera previsão simples baseada em regressão linear para os próximos dias"""
    if df.empty or len(df) < 24:  # Precisa de pelo menos 24 horas de dados
        return None
    
    # Preparar DataFrame para resultado
    future_dates = pd.date_range(
        start=df['datetime'].max() + timedelta(hours=1),
        periods=days_ahead * 24,
        freq='H'
    )
    forecast_df = pd.DataFrame({'datetime': future_dates})
    
    # Para cada poluente, criar um modelo simples
    for pollutant in pollutants:
        if pollutant in df.columns:
            # Preparar dados para o modelo
            X = np.array(range(len(df))).reshape(-1, 1)
            y = df[pollutant].values
            
            # Criar e treinar modelo
            model = LinearRegression()
            model.fit(X, y)
            
            # Prever valores futuros
            X_future = np.array(range(len(df), len(df) + len(future_dates))).reshape(-1, 1)
            y_future = model.predict(X_future)
            
            # Adicionar componente de hora do dia
            hour_of_day = np.array([d.hour for d in future_dates])
            daily_pattern = np.sin(hour_of_day * 2 * np.pi / 24) * df[pollutant].std() * 0.5
            
            # Combinar tendência com padrão diário
            forecast_df[pollutant] = np.maximum(0, y_future + daily_pattern)
    
    # Adicionar coluna de tipo para diferenciar dados históricos de previsões
    df_hist = df.copy()
    df_hist['type'] = 'historical'
    forecast_df['type'] = 'forecast'
    
    # Combinar dados históricos e previsão
    combined_df = pd.concat([df_hist, forecast_df], ignore_index=True)
    
    return combined_df

# Função para visualizar séries temporais de poluentes
def plot_pollutant_timeseries(df, pollutant, city_name):
    """Cria gráfico de série temporal para um poluente"""
    if df.empty or pollutant not in df.columns:
        return None
    
    # Separar dados históricos e previsão
    hist_data = df[df['type'] == 'historical'] if 'type' in df.columns else df
    forecast_data = df[df['type'] == 'forecast'] if 'type' in df.columns else pd.DataFrame()
    
    # Criar figura
    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 1, 1)
    
    # Plotar dados históricos
    ax.plot(hist_data['datetime'], hist_data[pollutant], 
           color='blue', marker='o', markersize=3, linestyle='-', label='Observado')
    
    # Plotar dados de previsão se disponíveis
    if not forecast_data.empty:
        ax.plot(forecast_data['datetime'], forecast_data[pollutant], 
               color='red', marker='x', markersize=3, linestyle='--', label='Previsão')
    
    # Adicionar linhas horizontais para os padrões CONAMA
    if pollutant in CONAMA_STANDARDS:
        for category, value in CONAMA_STANDARDS[pollutant].items():
            ax.axhline(y=value, color=QUALITY_COLORS.get(category, 'gray'), 
                      linestyle='--', alpha=0.7, label=f'Limite {category}')
    
    # Adicionar linha para o padrão OMS
    if pollutant in WHO_STANDARDS and 'Diário' in WHO_STANDARDS[pollutant]:
        who_value = WHO_STANDARDS[pollutant]['Diário']
        ax.axhline(y=who_value, color='black', linestyle='-', alpha=0.7, label=f'OMS (Diário)')
    
    # Configurar gráfico
    ax.set_title(f'{pollutant} em {city_name}', fontsize=14)
    ax.set_xlabel('Data/Hora', fontsize=12)
    
    # Definir unidade correta para o poluente
    if pollutant == 'CO':
        ax.set_ylabel(f'{pollutant} (ppm)', fontsize=12)
    else:
        ax.set_ylabel(f'{pollutant} (μg/m³)', fontsize=12)
    
    # Formatar eixo x
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
    plt.xticks(rotation=45)
    
    # Adicionar legenda e grid
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

# Função para criar dashboard de todos os poluentes
def plot_pollutants_dashboard(df, pollutants, city_name):
    """Cria um dashboard com todos os poluentes usando Plotly"""
    if df.empty:
        return None
    
    # Criar subplots
    n_plots = len(pollutants)
    rows = (n_plots + 1) // 2  # Arredondamento para cima
    cols = min(2, n_plots)
    
    fig = make_subplots(rows=rows, cols=cols, 
                        subplot_titles=[f"{p}" for p in pollutants])
    
    # Contador para posição dos subplots
    plot_idx = 0
    
    for pollutant in pollutants:
        if pollutant in df.columns:
            # Determinar posição do subplot
            row = (plot_idx // cols) + 1
            col = (plot_idx % cols) + 1
            plot_idx += 1
            
            # Separar dados históricos e previsão
            hist_data = df[df['type'] == 'historical'] if 'type' in df.columns else df
            forecast_data = df[df['type'] == 'forecast'] if 'type' in df.columns else pd.DataFrame()
            
            # Adicionar traço para dados históricos
            fig.add_trace(
                go.Scatter(
                    x=hist_data['datetime'], 
                    y=hist_data[pollutant],
                    mode='lines+markers',
                    name=f'{pollutant} observado',
                    line=dict(color='blue')
                ),
                row=row, col=col
            )
            
            # Adicionar traço para previsão se disponível
            if not forecast_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_data['datetime'], 
                        y=forecast_data[pollutant],
                        mode='lines+markers',
                        name=f'{pollutant} previsto',
                        line=dict(color='red', dash='dash')
                    ),
                    row=row, col=col
                )
            
            # Adicionar linhas horizontais para os padrões CONAMA
            if pollutant in CONAMA_STANDARDS:
                for category, value in CONAMA_STANDARDS[pollutant].items():
                    fig.add_trace(
                        go.Scatter(
                            x=[df['datetime'].min(), df['datetime'].max()],
                            y=[value, value],
                            mode='lines',
                            name=f'{category} ({value})',
                            line=dict(color=QUALITY_COLORS.get(category, 'gray'), dash='dash'),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
            
            # Adicionar linha para padrão OMS
            if pollutant in WHO_STANDARDS and 'Diário' in WHO_STANDARDS[pollutant]:
                who_value = WHO_STANDARDS[pollutant]['Diário']
                fig.add_trace(
                    go.Scatter(
                        x=[df['datetime'].min(), df['datetime'].max()],
                        y=[who_value, who_value],
                        mode='lines',
                        name=f'OMS ({who_value})',
                        line=dict(color='black'),
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            # Definir unidade correta para o eixo Y
            y_title = 'ppm' if pollutant == 'CO' else 'μg/m³'
            fig.update_yaxes(title_text=y_title, row=row, col=col)
    
    # Atualizar layout
    fig.update_layout(
        title=f'Monitoramento de Poluentes em {city_name}',
        height=300 * rows,
        width=1000,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Função para gerar relatório de qualidade do ar
def generate_air_quality_report(df, pollutants, city_name):
    """Gera um relatório completo da qualidade do ar"""
    if df.empty:
        return "Não há dados suficientes para gerar um relatório."
    
    # Calcular estatísticas
    stats = calculate_pollutant_stats(df, pollutants)
    
    # Determinar qualidade geral (pior classificação entre os poluentes)
    quality_ranks = {
        'Boa': 1,
        'Moderada': 2,
        'Ruim': 3,
        'Muito Ruim': 4,
        'Péssima': 5
    }
    
    overall_quality = "Boa"
    overall_rank = 0
    
    for pollutant, data in stats.items():
        quality = data['quality']
        rank = quality_ranks.get(quality, 0)
        if rank > overall_rank:
            overall_rank = rank
            overall_quality = quality
    
    # Obter data e hora atual
    current_time = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    # Construir relatório
    report = f"""# Relatório de Qualidade do Ar - {city_name}

**Data/Hora:** {current_time}

## Classificação Geral: {overall_quality}

### Resumo por Poluente:

"""
    
    for pollutant, data in stats.items():
        unit = "ppm" if pollutant == "CO" else "μg/m³"
        report += f"""
#### {pollutant}:
- **Valor atual:** {data['current']:.2f} {unit}
- **Média:** {data['average']:.2f} {unit}
- **Máximo:** {data['maximum']:.2f} {unit}
- **Classificação:** {data['quality']}
- **Comparação com OMS:** {data['who_comparison']}
"""
    
    report += """
## Interpretação:

- **Boa:** Qualidade do ar satisfatória, baixo potencial de danos à saúde.
- **Moderada:** Pessoas de grupos sensíveis podem apresentar sintomas.
- **Ruim:** Toda população pode apresentar sintomas como tosse seca, cansaço, ardor nos olhos, nariz e garganta.
- **Muito Ruim:** Toda população pode apresentar agravamento dos sintomas e efeitos mais sérios à saúde.
- **Péssima:** Sérios riscos à saúde. Toda população pode apresentar sintomas graves.

## Recomendações:

"""
    
    # Adicionar recomendações baseadas na qualidade geral
    if overall_quality == "Boa":
        report += "- Ideal para atividades ao ar livre.\n"
        report += "- Não há restrições para a população em geral.\n"
    elif overall_quality == "Moderada":
        report += "- Pessoas de grupos sensíveis (crianças, idosos e pessoas com doenças respiratórias e cardíacas) devem reduzir esforços físicos pesados ao ar livre.\n"
        report += "- População em geral pode realizar atividades ao ar livre.\n"
    elif overall_quality == "Ruim":
        report += "- Pessoas de grupos sensíveis devem evitar atividades ao ar livre.\n"
        report += "- População em geral deve reduzir atividades ao ar livre prolongadas ou intensas.\n"
    elif overall_quality == "Muito Ruim":
        report += "- Pessoas de grupos sensíveis devem permanecer em casa.\n"
        report += "- População em geral deve evitar atividades ao ar livre.\n"
        report += "- Se possível, permanecer em ambientes fechados.\n"
    else:  # Péssima
        report += "- Toda a população deve permanecer em casa, mantendo as janelas fechadas.\n"
        report += "- Suspender atividades físicas ao ar livre.\n"
        report += "- Buscar orientação médica em caso de sintomas respiratórios ou cardíacos.\n"
    
    return report

# Função para gerar mapa espacial de qualidade do ar
def generate_air_quality_map(pollutant):
    """Gera um mapa espacial para um poluente específico usando dados sintéticos"""
    # Criar DataFrame com cidades e valores sintéticos para o poluente
    cities_data = []
    for city_name, (lat, lon) in cities.items():
        # Gerar valor sintético baseado no nome da cidade (para consistência)
        city_seed = sum(ord(c) for c in city_name)
        np.random.seed(city_seed)
        
        # Ajustar valores de acordo com o poluente
        if pollutant == 'MP10':
            value = np.random.normal(40, 15)
        elif pollutant == 'MP2.5':
            value = np.random.normal(15, 7)
        elif pollutant == 'O3':
            value = np.random.normal(80, 20)
        elif pollutant == 'NO2':
            value = np.random.normal(100, 40)
        elif pollutant == 'SO2':
            value = np.random.normal(15, 8)
        elif pollutant == 'CO':
            value = np.random.normal(4, 2)
        else:
            value = np.random.normal(50, 20)
        
        # Garantir valores não negativos
        value = max(0, value)
        
        # Classificar qualidade
        quality, color = classify_air_quality(pollutant, value)
        
        cities_data.append({
            'city': city_name,
            'lat': lat,
            'lon': lon,
            'value': value,
            'quality': quality,
            'color': color
        })
    
    # Criar DataFrame
    df_map = pd.DataFrame(cities_data)
    
    # Criar mapa com Plotly
    fig = px.scatter_mapbox(df_map, 
                            lat='lat', 
                            lon='lon', 
                            color='quality',
                            color_discrete_map={q: c for q, c in QUALITY_COLORS.items()},
                            size='value',
                            size_max=20,
                            hover_name='city',
                            hover_data={
                                'value': True,
                                'quality': True,
                                'lat': False,
                                'lon': False
                            },
                            zoom=5.5,
                            center={"lat": -21.0, "lon": -55.0},
                            title=f'Mapa de {pollutant} em Mato Grosso do Sul')
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":50,"l":0,"b":0},
        legend_title_text='Qualidade do Ar'
    )
    
    # Adicionar rótulos das cidades
    for i, row in df_map.iterrows():
        fig.add_annotation(
            lat=row['lat'],
            lon=row['lon'],
            text=row['city'],
            showarrow=False,
            yshift=15,
            font=dict(size=10, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=2,
            opacity=0.8
        )
    
    return fig

# Função principal para gerar interface do usuário
# Função principal para gerar interface do usuário
def main():
    # Títulos e introdução
    st.title("🌬️ Monitoramento de Qualidade do Ar - Mato Grosso do Sul")
    st.markdown("""
    Este aplicativo permite monitorar e analisar a qualidade do ar em diferentes municípios de Mato Grosso do Sul,
    com foco nos principais poluentes atmosféricos: MP10, MP2.5, O3, NO2, SO2 e CO.
    """)
    
    # Barra lateral para configurações
    st.sidebar.header("⚙️ Configurações")
    
    # Seleção de cidade
    city_name = st.sidebar.selectbox("Selecione um município:", list(cities.keys()))
    
    # Lista de poluentes para análise
    pollutant_options = ['MP10', 'MP2.5', 'O3', 'NO2', 'SO2', 'CO']
    selected_pollutants = st.sidebar.multiselect(
        "Selecione os poluentes para análise:",
        options=pollutant_options,
        default=['MP10', 'MP2.5', 'O3']
    )
    
    # Se nenhum poluente for selecionado, usar padrões
    if not selected_pollutants:
        selected_pollutants = ['MP10', 'MP2.5', 'O3']
        st.sidebar.warning("Nenhum poluente selecionado. Usando poluentes padrão.")
    
    # Configurações de data
    st.sidebar.subheader("Período de Análise")
    
    # Definir datas padrão: 7 dias atrás até hoje
    default_end_date = datetime.now()
    default_start_date = default_end_date - timedelta(days=7)
    
    start_date = st.sidebar.date_input(
        "Data inicial:",
        value=default_start_date,
        max_value=default_end_date
    )
    
    end_date = st.sidebar.date_input(
        "Data final:",
        value=default_end_date,
        min_value=start_date,
        max_value=default_end_date
    )
    
    # Converter para datetime para processamento
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Verificar se o período é válido
    if start_datetime >= end_datetime:
        st.sidebar.error("A data inicial deve ser anterior à data final.")
        return
    
    # Opções de visualização
    st.sidebar.subheader("Opções de Visualização")
    show_forecast = st.sidebar.checkbox("Mostrar previsão futura", value=True)
    forecast_days = st.sidebar.slider("Dias de previsão:", 1, 7, 3) if show_forecast else 0
    
    # Obter os dados
    with st.spinner("Gerando dados para análise..."):
        # Gerar dados sintéticos
        df = generate_synthetic_data(
            city_name, 
            start_datetime, 
            end_datetime, 
            selected_pollutants
        )
        
        # Adicionar previsões se solicitado
        if show_forecast and forecast_days > 0:
            df = predict_pollutant_trends(df, selected_pollutants, days_ahead=forecast_days)
    
    # Criar abas para diferentes visualizações
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Visão Geral", 
        "🔍 Análise Detalhada", 
        "🗺️ Mapa Espacial", 
        "📊 Comparativo", 
        "📝 Relatório"
    ])
    
    # Tab 1: Visão Geral
    with tab1:
        # Mostrar indicadores atuais
        st.subheader(f"Qualidade do Ar Atual em {city_name}")
        
        # Calcular estatísticas
        stats = calculate_pollutant_stats(df, selected_pollutants)
        
        # Determinar pior qualidade para destaque
        worst_quality = "Boa"
        quality_rank = {
            "Boa": 1,
            "Moderada": 2,
            "Ruim": 3,
            "Muito Ruim": 4,
            "Péssima": 5
        }
        worst_rank = 0
        worst_pollutant = None
        
        for pollutant, data in stats.items():
            current_rank = quality_rank.get(data['quality'], 0)
            if current_rank > worst_rank:
                worst_rank = current_rank
                worst_quality = data['quality']
                worst_pollutant = pollutant
        
        # Mostrar classificação geral
        col1, col2 = st.columns([1, 3])
        
        with col1:
            quality_color = QUALITY_COLORS.get(worst_quality, "#808080")
            st.markdown(
                f"""
                <div style="
                    background-color: {quality_color}; 
                    padding: 20px; 
                    border-radius: 10px; 
                    text-align: center;
                    color: {'black' if worst_quality in ['Boa', 'Moderada'] else 'white'};
                    font-weight: bold;
                    font-size: 24px;
                ">
                {worst_quality}
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            if worst_pollutant:
                st.caption(f"Determinado pelo poluente {worst_pollutant}")
        
        with col2:
            # Criar colunas para cada poluente
            num_pollutants = len(selected_pollutants)
            
            if num_pollutants > 0:
                cols = st.columns(num_pollutants)
                
                for i, pollutant in enumerate(selected_pollutants):
                    if pollutant in stats:
                        with cols[i]:
                            data = stats[pollutant]
                            
                            # Definir unidade
                            unit = "ppm" if pollutant == "CO" else "μg/m³"
                            
                            st.metric(
                                label=pollutant,
                                value=f"{data['current']:.1f} {unit}",
                                delta=f"{data['quality']}"
                            )
                            
                            # Adicionar indicador visual de qualidade
                            st.markdown(
                                f"""
                                <div style="
                                    height: 10px; 
                                    background-color: {data['color']}; 
                                    width: 100%; 
                                    border-radius: 5px;
                                "></div>
                                """, 
                                unsafe_allow_html=True
                            )
        
        # Mostrar gráfico de todos os poluentes
        st.subheader("Evolução Temporal dos Poluentes")
        
        dashboard_fig = plot_pollutants_dashboard(df, selected_pollutants, city_name)
        if dashboard_fig:
            st.plotly_chart(dashboard_fig, use_container_width=True)
        else:
            st.warning("Não há dados suficientes para gerar o gráfico.")
    
    # Tab 2: Análise Detalhada
    with tab2:
        st.subheader(f"Análise Detalhada por Poluente em {city_name}")
        
        # Selecionar poluente para análise detalhada
        selected_pollutant = st.selectbox(
            "Selecione um poluente para análise detalhada:",
            options=selected_pollutants,
            key="detailed_pollutant"
        )
        
        # Mostrar gráfico detalhado
        if selected_pollutant:
            # Gráfico de série temporal
            st.markdown(f"### Série Temporal - {selected_pollutant}")
            
            timeseries_fig = plot_pollutant_timeseries(df, selected_pollutant, city_name)
            if timeseries_fig:
                st.pyplot(timeseries_fig)
            else:
                st.warning("Não há dados suficientes para gerar o gráfico.")
            
            # Estatísticas detalhadas
            if selected_pollutant in stats:
                st.markdown("### Estatísticas")
                
                data = stats[selected_pollutant]
                unit = "ppm" if selected_pollutant == "CO" else "μg/m³"
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Valor Atual", f"{data['current']:.2f} {unit}")
                with col2:
                    st.metric("Média", f"{data['average']:.2f} {unit}")
                with col3:
                    st.metric("Máximo", f"{data['maximum']:.2f} {unit}")
                
                # Tabela de limites
                st.markdown("### Limites Regulatórios")
                
                if selected_pollutant in CONAMA_STANDARDS:
                    limits_data = {
                        "Categoria": list(CONAMA_STANDARDS[selected_pollutant].keys()),
                        "Valor Limite (CONAMA)": list(CONAMA_STANDARDS[selected_pollutant].values())
                    }
                    
                    if selected_pollutant in WHO_STANDARDS:
                        for period, value in WHO_STANDARDS[selected_pollutant].items():
                            limits_data[f"OMS ({period})"] = [value] * len(limits_data["Categoria"])
                    
                    limits_df = pd.DataFrame(limits_data)
                    
                    # Adicionar unidade à coluna de valores
                    for col in limits_df.columns:
                        if "Valor" in col:
                            limits_df[col] = limits_df[col].astype(str) + f" {unit}"
                    
                    st.dataframe(limits_df, use_container_width=True)
                else:
                    st.info(f"Não há limites definidos para {selected_pollutant} nos padrões consultados.")
    
    # Tab 3: Mapa Espacial
    with tab3:
        st.subheader("Distribuição Espacial da Qualidade do Ar em MS")
        
        # Selecionar poluente para o mapa
        map_pollutant = st.selectbox(
            "Selecione um poluente para visualização espacial:",
            options=pollutant_options,
            key="map_pollutant"
        )
        
        # Gerar e mostrar mapa
        map_fig = generate_air_quality_map(map_pollutant)
        st.plotly_chart(map_fig, use_container_width=True)
        
        # Adicionar shapes dos municípios
        st.markdown("### Mapa dos Municípios")
        
        # Carregar shapefile dos municípios
        municipalities = load_ms_municipalities()
        
        if not municipalities.empty:
            # Criar mapa com geopandas
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            municipalities.plot(ax=ax, color='lightgray', edgecolor='black')
            
            # Adicionar pontos para as cidades
            city_points = pd.DataFrame({
                'city': list(cities.keys()),
                'lat': [lat for lat, _ in cities.values()],
                'lon': [lon for _, lon in cities.values()]
            })
            
            ax.scatter(city_points['lon'], city_points['lat'], color='red', s=50)
            
            for _, row in city_points.iterrows():
                ax.annotate(row['city'], (row['lon'], row['lat']), 
                           xytext=(5, 5), textcoords='offset points')
            
            ax.set_title("Municípios de Mato Grosso do Sul")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            
            st.pyplot(fig)
        else:
            st.warning("Não foi possível carregar os shapes dos municípios.")
    
    # Tab 4: Comparativo
    with tab4:
        st.subheader("Comparativo entre Municípios")
        
        # Selecionar cidades para comparação
        compare_cities = st.multiselect(
            "Selecione municípios para comparação:",
            options=list(cities.keys()),
            default=[city_name]
        )
        
        # Selecionar poluente para comparação
        compare_pollutant = st.selectbox(
            "Selecione um poluente para comparação:",
            options=pollutant_options,
            key="compare_pollutant"
        )
        
        if compare_cities and compare_pollutant:
            # Gerar dados para cada cidade
            comparison_data = []
            
            for city in compare_cities:
                city_df = generate_synthetic_data(
                    city, 
                    start_datetime, 
                    end_datetime, 
                    [compare_pollutant]
                )
                
                # Calcular média diária
                city_df['date'] = city_df['datetime'].dt.date
                daily_avg = city_df.groupby('date')[compare_pollutant].mean().reset_index()
                
                for _, row in daily_avg.iterrows():
                    comparison_data.append({
                        'Município': city,
                        'Data': row['date'],
                        'Valor': row[compare_pollutant]
                    })
            
            # Criar DataFrame de comparação
            compare_df = pd.DataFrame(comparison_data)
            
            # Plotar gráfico comparativo
            fig = px.line(
                compare_df, 
                x='Data', 
                y='Valor', 
                color='Município',
                title=f'Comparação de {compare_pollutant} entre Municípios',
                labels={'Valor': f'{compare_pollutant} ({"ppm" if compare_pollutant == "CO" else "μg/m³"})'}
            )
            
            # Adicionar linha para o padrão OMS
            if compare_pollutant in WHO_STANDARDS and 'Diário' in WHO_STANDARDS[compare_pollutant]:
                who_value = WHO_STANDARDS[compare_pollutant]['Diário']
                fig.add_hline(
                    y=who_value, 
                    line_dash="dash", 
                    line_color="black",
                    annotation_text=f"Padrão OMS ({who_value})"
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabela comparativa de estatísticas
            st.markdown("### Estatísticas Comparativas")
            
            stats_data = []
            for city in compare_cities:
                city_df = generate_synthetic_data(
                    city, 
                    start_datetime, 
                    end_datetime, 
                    [compare_pollutant]
                )
                
                city_stats = calculate_pollutant_stats(city_df, [compare_pollutant])
                
                if compare_pollutant in city_stats:
                    data = city_stats[compare_pollutant]
                    stats_data.append({
                        'Município': city,
                        'Média': f"{data['average']:.2f}",
                        'Máximo': f"{data['maximum']:.2f}",
                        'Qualidade': data['quality'],
                        'Atende OMS': "Sim" if data['who_complies'] else "Não"
                    })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
    
    # Tab 5: Relatório
    with tab5:
        st.subheader(f"Relatório de Qualidade do Ar para {city_name}")
        
        # Gerar relatório
        report = generate_air_quality_report(df, selected_pollutants, city_name)
        
        # Exibir relatório
        st.markdown(report)
        
        # Opção para download
        report_filename = f"relatorio_qualidade_ar_{city_name}_{datetime.now().strftime('%Y%m%d')}.md"
        
        st.download_button(
            label="📥 Download do Relatório",
            data=report,
            file_name=report_filename,
            mime="text/markdown"
        )
        
        # Adicionar opção para visualização resumida
        st.subheader("Resumo Visual")
        
        # Criar tabela resumida
        if stats:
            # Determinar cores para cada célula com base na qualidade
            quality_data = {pollutant: data['quality'] for pollutant, data in stats.items()}
            color_data = {pollutant: data['color'] for pollutant, data in stats.items()}
            
            # Criar tabela HTML estilizada
            html_table = """
            <table style="width:100%; border-collapse: collapse; margin-top: 20px;">
                <tr>
                    <th style="text-align: left; padding: 12px; background-color: #f2f2f2;">Poluente</th>
                    <th style="text-align: center; padding: 12px; background-color: #f2f2f2;">Valor Atual</th>
                    <th style="text-align: center; padding: 12px; background-color: #f2f2f2;">Qualidade</th>
                </tr>
            """
            
            for pollutant, data in stats.items():
                unit = "ppm" if pollutant == "CO" else "μg/m³"
                text_color = "black" if data['quality'] in ['Boa', 'Moderada'] else "white"
                
                html_table += f"""
                <tr>
                    <td style="text-align: left; padding: 12px; border-bottom: 1px solid #ddd;">{pollutant}</td>
                    <td style="text-align: center; padding: 12px; border-bottom: 1px solid #ddd;">{data['current']:.2f} {unit}</td>
                    <td style="text-align: center; padding: 12px; border-bottom: 1px solid #ddd; background-color: {data['color']}; color: {text_color};">{data['quality']}</td>
                </tr>
                """
            
            html_table += "</table>"
            
            # Exibir tabela
            st.markdown(html_table, unsafe_allow_html=True)

# Executar aplicação
if __name__ == "__main__":
    main()

    
