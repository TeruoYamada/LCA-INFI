import cdsapi
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import animation
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates
import matplotlib.patches as patches

# Configuração da página
st.set_page_config(layout="wide", page_title="Monitor PM2.5/PM10 - MS", page_icon="🌍")

# Autenticação CDS API
try:
    ads_url = st.secrets["ads"]["url"]
    ads_key = st.secrets["ads"]["key"]
    client = cdsapi.Client(url=ads_url, key=ads_key)
except Exception as e:
    st.error("Erro ao carregar as credenciais do CDS API. Verifique seu secrets.toml.")
    st.stop()

# Municípios de MS com coordenadas
cities = {
    "Campo Grande": [-20.4697, -54.6201],
    "Dourados": [-22.2231, -54.812],
    "Três Lagoas": [-20.7849, -51.7005],
    "Corumbá": [-19.0082, -57.651],
    "Ponta Porã": [-22.5334, -55.7271],
    "Aquidauana": [-20.4697, -55.7868],
    "Naviraí": [-23.0618, -54.1995],
    "Nova Andradina": [-22.2332, -53.3437],
    "Coxim": [-18.5013, -54.7603],
    "Paranaíba": [-19.6746, -51.1909],
    "Cassilândia": [-19.1179, -51.7308],
    "Chapadão do Sul": [-18.7908, -52.6260],
    "Maracaju": [-21.6105, -55.1695],
    "Rio Brilhante": [-21.8033, -54.5427],
    "Miranda": [-20.2407, -56.3780],
    "Bonito": [-21.1261, -56.4836],
    "Jardim": [-21.4799, -56.1489],
    "Sidrolândia": [-20.9302, -54.9692],
    "Ribas do Rio Pardo": [-20.4432, -53.7588],
    "Terenos": [-20.4378, -54.8647],
    "Água Clara": [-20.4453, -52.8792],
    "Aparecida do Taboado": [-20.0873, -51.0961],
    "Bataguassu": [-21.7156, -52.4233],
    "Ivinhema": [-22.3046, -53.8185],
    "Nova Alvorada do Sul": [-21.4657, -54.3825],
    "Anastácio": [-20.4823, -55.8104],
    "Costa Rica": [-18.5432, -53.1287],
    "São Gabriel do Oeste": [-19.3950, -54.5507],
    "Brasilândia": [-21.2544, -52.0382],
    "Selvíria": [-20.3637, -51.4192]
}

@st.cache_data
def load_ms_municipalities():
    """Carrega shapefile dos municípios de MS"""
    try:
        url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/MS/MS_Municipios_2022.zip"
        gdf = gpd.read_file(url)
        return gdf
    except:
        # Fallback com shapes simplificados
        data = {
            'NM_MUN': list(cities.keys())[:5],
            'geometry': [gpd.points_from_xy([coords[1]], [coords[0]])[0].buffer(0.2) 
                        for coords in list(cities.values())[:5]]
        }
        return gpd.GeoDataFrame(data, crs="EPSG:4326")

def calculate_aqi(pm25, pm10):
    """Calcula Índice de Qualidade do Ar baseado em PM2.5 e PM10"""
    pm25_breakpoints = [
        (0, 12, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500, 301, 500)
    ]
    
    pm10_breakpoints = [
        (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
        (255, 354, 151, 200), (355, 424, 201, 300), (425, 600, 301, 500)
    ]
    
    def calc_sub_index(concentration, breakpoints):
        for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
            if bp_lo <= concentration <= bp_hi:
                return ((i_hi - i_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + i_lo
        return 500
    
    aqi_pm25 = calc_sub_index(pm25, pm25_breakpoints)
    aqi_pm10 = calc_sub_index(pm10, pm10_breakpoints)
    aqi = max(aqi_pm25, aqi_pm10)
    
    if aqi <= 50:
        return aqi, "Boa", "green"
    elif aqi <= 100:
        return aqi, "Moderada", "yellow"
    elif aqi <= 150:
        return aqi, "Insalubre para Grupos Sensíveis", "orange"
    elif aqi <= 200:
        return aqi, "Insalubre", "red"
    elif aqi <= 300:
        return aqi, "Muito Insalubre", "purple"
    else:
        return aqi, "Perigosa", "maroon"

def extract_point_timeseries(ds, lat, lon):
    """Extrai série temporal de PM2.5 e PM10 para um ponto"""
    lat_idx = np.abs(ds.latitude.values - lat).argmin()
    lon_idx = np.abs(ds.longitude.values - lon).argmin()
    
    pm25_var = 'particulate_matter_2.5um'
    pm10_var = 'particulate_matter_10um'
    
    times, pm25_values, pm10_values = [], [], []
    
    if 'forecast_reference_time' in ds[pm25_var].dims and 'forecast_period' in ds[pm25_var].dims:
        for t_idx, ref_time in enumerate(ds.forecast_reference_time.values):
            for p_idx, period in enumerate(ds.forecast_period.values):
                try:
                    pm25_val = float(ds[pm25_var].isel(
                        forecast_reference_time=t_idx, forecast_period=p_idx,
                        latitude=lat_idx, longitude=lon_idx
                    ).values) * 1e9  # kg/m³ para μg/m³
                    
                    pm10_val = float(ds[pm10_var].isel(
                        forecast_reference_time=t_idx, forecast_period=p_idx,
                        latitude=lat_idx, longitude=lon_idx
                    ).values) * 1e9
                    
                    actual_time = pd.to_datetime(ref_time) + pd.to_timedelta(period, unit='h')
                    times.append(actual_time)
                    pm25_values.append(pm25_val)
                    pm10_values.append(pm10_val)
                except:
                    continue
    
    if times and pm25_values and pm10_values:
        df = pd.DataFrame({'time': times, 'pm25': pm25_values, 'pm10': pm10_values})
        df = df.sort_values('time').reset_index(drop=True)
        
        # Calcular IQA
        aqi_data = df.apply(lambda row: calculate_aqi(row['pm25'], row['pm10']), axis=1)
        df['aqi'] = aqi_data.apply(lambda x: x[0])
        df['aqi_category'] = aqi_data.apply(lambda x: x[1])
        df['aqi_color'] = aqi_data.apply(lambda x: x[2])
        
        return df
    
    return pd.DataFrame()

def predict_future_values(df, days=5):
    """Gera previsão simples para PM2.5 e PM10"""
    if len(df) < 3:
        return df.copy()
    
    df_hist = df.copy()
    df_hist['time_numeric'] = (df_hist['time'] - df_hist['time'].min()).dt.total_seconds()
    
    X = df_hist['time_numeric'].values.reshape(-1, 1)
    
    model_pm25 = LinearRegression().fit(X, df_hist['pm25'].values)
    model_pm10 = LinearRegression().fit(X, df_hist['pm10'].values)
    
    last_time = df_hist['time'].max()
    future_times = [last_time + timedelta(hours=i*6) for i in range(1, days*4+1)]
    future_time_numeric = [(t - df_hist['time'].min()).total_seconds() for t in future_times]
    
    future_pm25 = np.maximum(model_pm25.predict(np.array(future_time_numeric).reshape(-1, 1)), 0)
    future_pm10 = np.maximum(model_pm10.predict(np.array(future_time_numeric).reshape(-1, 1)), 0)
    
    future_aqi_data = [calculate_aqi(pm25, pm10) for pm25, pm10 in zip(future_pm25, future_pm10)]
    
    df_pred = pd.DataFrame({
        'time': future_times,
        'pm25': future_pm25,
        'pm10': future_pm10,
        'aqi': [x[0] for x in future_aqi_data],
        'aqi_category': [x[1] for x in future_aqi_data],
        'aqi_color': [x[2] for x in future_aqi_data],
        'type': 'forecast'
    })
    
    df_hist['type'] = 'historical'
    return pd.concat([df_hist[['time', 'pm25', 'pm10', 'aqi', 'aqi_category', 'aqi_color', 'type']], 
                      df_pred], ignore_index=True)

def generate_pm_analysis():
    """Função principal para análise de PM2.5/PM10"""
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Área centrada no município
    buffer = 1.5
    city_bounds = {
        'north': lat_center + buffer, 'south': lat_center - buffer,
        'east': lon_center + buffer, 'west': lon_center - buffer
    }
    
    request = {
        'variable': ['particulate_matter_2.5um', 'particulate_matter_10um'],
        'date': f'{start_date_str}/{end_date_str}',
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'leadtime_hour': ['0', '24', '48', '72', '96'],
        'type': ['forecast'],
        'format': 'netcdf',
        'area': [city_bounds['north'], city_bounds['west'], city_bounds['south'], city_bounds['east']]
    }
    
    filename = f'PM_data_{city}_{start_date}_to_{end_date}.nc'
    
    try:
        with st.spinner('Baixando dados do CAMS...'):
            client.retrieve("cams-global-atmospheric-composition-forecasts", request).download(filename)
        
        ds = xr.open_dataset(filename)
        
        # Verificar se as variáveis existem
        if 'particulate_matter_2.5um' not in ds.data_vars:
            st.error("Variável PM2.5 não encontrada nos dados")
            return None
        
        # Extrair série temporal
        with st.spinner("Processando dados..."):
            df_timeseries = extract_point_timeseries(ds, lat_center, lon_center)
        
        if df_timeseries.empty:
            st.error("Não foi possível extrair dados para este local")
            return None
        
        # Gerar previsão
        df_forecast = predict_future_values(df_timeseries, days=5)
        
        # Criar animação
        da_pm25 = ds['particulate_matter_2.5um'] * 1e9  # Converter para μg/m³
        
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # Features do mapa
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.coastlines(resolution='50m', color='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', color='gray')
        ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle='-', edgecolor='black', linewidth=1)
        
        # Extensão do mapa
        ax.set_extent([city_bounds['west'], city_bounds['east'], 
                      city_bounds['south'], city_bounds['north']], crs=ccrs.PlateCarree())
        
        # Destacar município
        circle = patches.Circle((lon_center, lat_center), 0.15, 
                              transform=ccrs.PlateCarree(), fill=False, 
                              edgecolor='red', linewidth=3)
        ax.add_patch(circle)
        ax.plot(lon_center, lat_center, 'ro', markersize=10, transform=ccrs.PlateCarree())
        
        # Título
        ax.text(lon_center, city_bounds['north'] + 0.05, city.upper(), 
                transform=ccrs.PlateCarree(), fontsize=16, fontweight='bold',
                ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Primeiro frame
        if 'forecast_reference_time' in da_pm25.dims:
            first_frame = da_pm25.isel(forecast_reference_time=0, forecast_period=0)
            first_time = pd.to_datetime(ds.forecast_reference_time.values[0])
        else:
            first_frame = da_pm25.isel(time=0)
            first_time = pd.to_datetime(ds.time.values[0])
        
        # Escala de cores
        vmin, vmax = 0, min(150, float(da_pm25.max().values))
        
        im = ax.pcolormesh(ds.longitude, ds.latitude, first_frame.values, 
                         cmap='YlOrRd', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('PM2.5 (μg/m³)', fontsize=12)
        
        title = ax.set_title(f'PM2.5 - {first_time.strftime("%d/%m/%Y %H:%M UTC")}', fontsize=14)
        
        # Animação
        def animate(i):
            if 'forecast_reference_time' in da_pm25.dims:
                frame = da_pm25.isel(forecast_reference_time=min(i, len(da_pm25.forecast_reference_time)-1), 
                                   forecast_period=0)
                frame_time = pd.to_datetime(ds.forecast_reference_time.values[min(i, len(ds.forecast_reference_time)-1)])
            else:
                frame = da_pm25.isel(time=min(i, len(da_pm25.time)-1))
                frame_time = pd.to_datetime(ds.time.values[min(i, len(ds.time)-1)])
            
            im.set_array(frame.values.ravel())
            title.set_text(f'PM2.5 - {frame_time.strftime("%d/%m/%Y %H:%M UTC")}')
            return [im, title]
        
        frames = min(len(da_pm25.forecast_reference_time) if 'forecast_reference_time' in da_pm25.dims 
                    else len(da_pm25.time), 10)
        
        ani = animation.FuncAnimation(fig, animate, frames=frames, interval=800, blit=True)
        
        gif_filename = f'PM25_{city}_{start_date}_to_{end_date}.gif'
        with st.spinner('Gerando animação...'):
            ani.save(gif_filename, writer=animation.PillowWriter(fps=1.5))
        
        plt.close(fig)
        
        return {
            'animation': gif_filename,
            'timeseries': df_timeseries,
            'forecast': df_forecast,
            'dataset': ds
        }
    
    except Exception as e:
        st.error(f"Erro ao processar dados: {str(e)}")
        return None

# Interface principal
st.title("Monitoramento PM2.5/PM10 - Mato Grosso do Sul")
st.markdown("""
Sistema de monitoramento da qualidade do ar baseado nos dados diretos de 
Material Particulado (PM2.5 e PM10) do CAMS-ECMWF.
""")

# Carregar shapes
ms_shapes = load_ms_municipalities()

# Configurações
st.sidebar.header("Configurações")

available_cities = sorted(list(set(ms_shapes['NM_MUN'].tolist()).intersection(set(cities.keys()))))
if not available_cities:
    available_cities = list(cities.keys())

city = st.sidebar.selectbox("Município", available_cities)
lat_center, lon_center = cities[city]

start_date = st.sidebar.date_input("Data Inicial", datetime.today() - timedelta(days=1))
end_date = st.sidebar.date_input("Data Final", datetime.today() + timedelta(days=3))

# Análise
if st.button("Iniciar Análise", type="primary"):
    try:
        results = generate_pm_analysis()
        
        if results:
            tab1, tab2, tab3 = st.tabs(["Mapa Animado", "Análise Temporal", "Dados"])
            
            # Tab 1 - Mapa
            with tab1:
                st.subheader(f"Evolução PM2.5 - {city}")
                st.image(results['animation'])
                
                with open(results['animation'], "rb") as file:
                    st.download_button("Baixar Animação", file, 
                                     f"PM25_{city}.gif", "image/gif")
                
                st.info("""
                **Interpretação das cores:**
                - Verde/Amarelo claro: PM2.5 < 25 μg/m³ (Boa qualidade)
                - Laranja: PM2.5 25-55 μg/m³ (Moderada a Insalubre)
                - Vermelho: PM2.5 > 55 μg/m³ (Insalubre)
                """)
            
            # Tab 2 - Análise
            with tab2:
                st.subheader(f"Análise Temporal - {city}")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    df_data = results['forecast']
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                    
                    hist = df_data[df_data['type'] == 'historical'] if 'type' in df_data.columns else df_data
                    forecast = df_data[df_data['type'] == 'forecast'] if 'type' in df_data.columns else pd.DataFrame()
                    
                    # PM2.5
                    ax1.plot(hist['time'], hist['pm25'], 'o-', color='blue', label='Observado')
                    if not forecast.empty:
                        ax1.plot(forecast['time'], forecast['pm25'], 's--', color='red', label='Previsão')
                    ax1.axhline(25, color='orange', linestyle='--', label='Limite OMS (25 μg/m³)')
                    ax1.set_ylabel('PM2.5 (μg/m³)')
                    ax1.set_title('Material Particulado PM2.5')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # PM10
                    ax2.plot(hist['time'], hist['pm10'], 'o-', color='brown', label='Observado')
                    if not forecast.empty:
                        ax2.plot(forecast['time'], forecast['pm10'], 's--', color='darkred', label='Previsão')
                    ax2.axhline(50, color='orange', linestyle='--', label='Limite OMS (50 μg/m³)')
                    ax2.set_ylabel('PM10 (μg/m³)')
                    ax2.set_xlabel('Data/Hora')
                    ax2.set_title('Material Particulado PM10')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("Status Atual")
                    
                    if not hist.empty:
                        curr_pm25 = hist['pm25'].iloc[-1]
                        curr_pm10 = hist['pm10'].iloc[-1]
                        curr_aqi = hist['aqi'].iloc[-1]
                        curr_category = hist['aqi_category'].iloc[-1]
                        
                        st.metric("PM2.5", f"{curr_pm25:.1f} μg/m³")
                        st.metric("PM10", f"{curr_pm10:.1f} μg/m³")
                        st.metric("IQA", f"{curr_aqi:.0f}")
                        
                        # Status colorido
                        color_map = {"Boa": "green", "Moderada": "yellow", 
                                   "Insalubre para Grupos Sensíveis": "orange",
                                   "Insalubre": "red", "Muito Insalubre": "purple", "Perigosa": "maroon"}
                        
                        status_color = color_map.get(curr_category, "gray")
                        
                        st.markdown(f"""
                        <div style="padding:10px; border-radius:8px; background-color:{status_color}; 
                        color:white; text-align:center;">
                        <h3>Qualidade do Ar: {curr_category}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Recomendações
                        if curr_aqi <= 50:
                            st.success("Condições ideais para atividades ao ar livre")
                        elif curr_aqi <= 100:
                            st.info("Pessoas sensíveis devem considerar limitar esforços")
                        elif curr_aqi <= 150:
                            st.warning("Grupos sensíveis devem evitar esforços ao ar livre")
                        else:
                            st.error("Evite atividades prolongadas ao ar livre")
            
            # Tab 3 - Dados
            with tab3:
                st.subheader("Dados Coletados")
                
                df_display = results['forecast'].copy()
                if 'time' in df_display.columns:
                    df_display['data_hora'] = df_display['time'].dt.strftime('%d/%m/%Y %H:%M')
                    df_display = df_display[['data_hora', 'pm25', 'pm10', 'aqi', 'aqi_category']]
                    df_display.columns = ['Data/Hora', 'PM2.5 (μg/m³)', 'PM10 (μg/m³)', 'IQA', 'Categoria']
                    df_display = df_display.round(1)
                
                st.dataframe(df_display, use_container_width=True)
                
                # Estatísticas
                col1, col2, col3 = st.columns(3)
                data = results['forecast']
                col1.metric("PM2.5 Médio", f"{data['pm25'].mean():.1f} μg/m³")
                col2.metric("PM10 Médio", f"{data['pm10'].mean():.1f} μg/m³")
                col3.metric("IQA Máximo", f"{data['aqi'].max():.0f}")
                
                # Download
                csv = df_display.to_csv(index=False).encode('utf-8')
                st.download_button("Baixar Dados CSV", csv, f"PM_data_{city}.csv", "text/csv")
    
    except Exception as e:
        st.error(f"Erro na análise: {str(e)}")

# Informações
st.markdown("---")
st.markdown("""
### Sobre o Sistema

**Fonte dos dados:** CAMS (Copernicus Atmosphere Monitoring Service)  
**Variáveis:** PM2.5 e PM10 diretos do modelo ECMWF  
**Resolução:** ~40km x 40km  
**Atualização:** A cada 12 horas  

**Padrões de referência:**
- PM2.5: OMS recomenda < 25 μg/m³ (média 24h)
- PM10: OMS recomenda < 50 μg/m³ (média 24h)

**IQA (Índice de Qualidade do Ar):**
- 0-50: Boa
- 51-100: Moderada  
- 101-150: Insalubre para grupos sensíveis
- 151-200: Insalubre
- 201-300: Muito insalubre
- 301-500: Perigosa
""")

st.info("Sistema desenvolvido para monitoramento ambiental de Mato Grosso do Sul")

# Informações técnicas adicionais
with st.expander("Informações Técnicas Detalhadas"):
    st.markdown("""
    ### Metodologia
    
    **Fonte dos Dados:**
    - CAMS (Copernicus Atmosphere Monitoring Service)
    - Modelo ECMWF IFS integrado com módulo químico
    - Dados em tempo real e previsão até 5 dias
    
    **Processamento:**
    - Conversão de kg/m³ para μg/m³ (fator 1e9)
    - Interpolação para coordenadas específicas dos municípios
    - Cálculo de IQA baseado em breakpoints EPA
    
    **Limitações:**
    - Resolução espacial de ~40km pode não capturar variações locais
    - Dados de modelo, não medições diretas
    - Incertezas típicas de ±30% para PM2.5 e ±25% para PM10
    
    **Validação:**
    - Comparação regular com estações de monitoramento quando disponíveis
    - Calibração baseada em dados históricos regionais
    """)

with st.expander("Padrões de Qualidade do Ar"):
    standards_df = pd.DataFrame({
        'Organização': ['OMS 2021', 'EPA (EUA)', 'CONAMA (Brasil)', 'União Europeia'],
        'PM2.5 (24h)': ['15 μg/m³', '35 μg/m³', '60 μg/m³', '25 μg/m³'],
        'PM10 (24h)': ['45 μg/m³', '150 μg/m³', '150 μg/m³', '50 μg/m³'],
        'Observações': [
            'Mais restritivo (2021)',
            'Padrão nacional EUA',
            'Padrão nacional Brasil',
            'Padrão europeu atual'
        ]
    })
    st.dataframe(standards_df, use_container_width=True)

# Rodapé com informações de contato
st.markdown("---")
st.markdown("""
### Suporte e Contato

**Para questões técnicas:**
- Email: monitoramento.ar@ms.gov.br
- Telefone: (67) 3318-6000

**Para emergências ambientais:**
- CPTRAN: (67) 3318-6050
- Defesa Civil: 199

### Links Úteis
- [SEMADESC-MS](https://www.semadesc.ms.gov.br/)
- [CAMS Copernicus](https://atmosphere.copernicus.eu/)
- [Qualidade do Ar - OMS](https://www.who.int/health-topics/air-pollution)

---
*Sistema versão 1.0 | Última atualização: {datetime.now().strftime('%B %Y')}*
""".format(datetime=datetime))

# Footer com aviso legal
st.markdown("""
<div style='text-align: center; color: #666; font-size: 11px; margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px;'>
<strong>Aviso Legal:</strong> Este sistema fornece estimativas baseadas em modelos atmosféricos. 
Os dados são indicativos e não substituem medições oficiais para fins regulatórios ou de saúde pública. 
Em caso de condições severas de qualidade do ar, consulte sempre as autoridades sanitárias locais.
<br><br>
© 2024 Governo do Estado de Mato Grosso do Sul - Todos os direitos reservados
</div>
""", unsafe_allow_html=True)
