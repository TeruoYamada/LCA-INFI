import cdsapi
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import animation
import streamlit as st
from datetime import datetime, timedelta
import os

# ✅ Carregar autenticação a partir do secrets.toml
try:
    ads_url = st.secrets["ads"]["url"]
    ads_key = st.secrets["ads"]["key"]
    client = cdsapi.Client(url=ads_url, key=ads_key)
except Exception as e:
    st.error("❌ Erro ao carregar as credenciais do CDS API. Verifique seu secrets.toml.")
    st.stop()

# 🎯 Dicionário com algumas cidades do MS
cities = {
    "Campo Grande": [-20.4697, -54.6201],
    "Dourados": [-22.2231, -54.812],
    "Três Lagoas": [-20.7849, -51.7005],
    "Corumbá": [-19.0082, -57.651],
    "Ponta Porã": [-22.5334, -55.7271]
}

st.title("🌀 AOD Animation Generator (CAMs 550nm - Mato Grosso do Sul)")

city = st.selectbox("Selecione a cidade", list(cities.keys()))
lat_center, lon_center = cities[city]

start_date = st.date_input("Data de Início", datetime.today())
end_date = st.date_input("Data Final", datetime.today())
start_hour = st.selectbox("Horário Inicial", list(range(0, 24, 3)), format_func=lambda x: f"{x:02d}:00")
end_hour = st.selectbox("Horário Final", list(range(0, 24, 3)), format_func=lambda x: f"{x:02d}:00")

if st.button("🎞️ Gerar Animação"):
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
    
    # If no hours were added, use default
    if not hours:
        hours = ['00:00', '03:00', '06:00', '09:00', '12:00']
    
    request = {
        'variable': ['total_aerosol_optical_depth_550nm'],  # Corrected variable name
        'date': f'{start_date_str}/{end_date_str}',
        'time': hours,
        'leadtime_hour': ['0'],  # Reduzido para apenas uma previsão para simplificar
        'type': ['forecast'],
        'format': 'netcdf',
        'area': [lat_center + 5, lon_center - 5, lat_center - 5, lon_center + 5]
    }
    
    filename = f'AOD550_{city}_{start_date}_to_{end_date}.nc'
    
    try:
        with st.spinner('📥 Baixando dados do CAMS...'):
            client.retrieve(dataset, request).download(filename)
        
        ds = xr.open_dataset(filename)
        
        # Check the actual variable name in the dataset
        variable_names = list(ds.data_vars)
        st.write(f"Variáveis disponíveis: {variable_names}")
        
        # Use a variável 'aod550' encontrada nos dados
        aod_var = 'aod550'
        
        st.write(f"Usando variável: {aod_var}")
        da = ds[aod_var]
        
        # Check dimensions
        st.write(f"Dimensões: {da.dims}")
        
        # Identificar as dimensões corretas para iterar
        if 'forecast_reference_time' in da.dims:
            time_dim = 'forecast_reference_time'
        else:
            time_dim = list(da.dims)[0]  # Usa a primeira dimensão como tempo se não encontrar
        
        frames = len(da[time_dim])
        st.write(f"✅ Total de frames disponíveis: {frames}")
        
        if frames < 1:
            st.error("Erro: Dados insuficientes para animação.")
        else:
            vmin, vmax = float(da.min().values), float(da.max().values)
            
            # Adicionar um buffer para melhor visualização
            vmin = max(0, vmin - 0.05)
            vmax = min(2, vmax + 0.05)  # AOD geralmente não ultrapassa 2
            
            fig = plt.figure(figsize=(10, 5))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
            ax.set_extent([lon_center - 5, lon_center + 5, lat_center - 5, lat_center + 5], crs=ccrs.PlateCarree())
            
            # Obter o primeiro frame para inicializar o plot
            if 'forecast_period' in da.dims:
                # Se tiver combinação de forecast_period e forecast_reference_time
                if len(da.forecast_period) > 0 and len(da.forecast_reference_time) > 0:
                    da_frame = da.isel(forecast_period=0, forecast_reference_time=0).values
                    time_value = str(ds.forecast_reference_time.values[0])
                else:
                    # Se alguma dimensão tiver tamanho zero, use indexação dinâmica
                    first_frame_coords = {dim: 0 for dim in da.dims if len(da[dim]) > 0}
                    da_frame = da.isel(**first_frame_coords).values
                    time_value = "Indisponível"
            else:
                # Caso simples: apenas uma dimensão de tempo
                da_frame = da.isel({time_dim: 0}).values
                time_value = str(da[time_dim].values[0])
            
            # Garantir que da_frame tenha 2 dimensões para pcolormesh
            if len(da_frame.shape) != 2:
                st.error(f"Erro: Formato de dados inesperado. Shape: {da_frame.shape}")
                st.stop()
            
            im = ax.pcolormesh(ds.longitude, ds.latitude, da_frame, cmap='YlOrRd', vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label('AOD 550nm')
            
            def animate(i):
                try:
                    if 'forecast_period' in da.dims:
                        if len(da.forecast_reference_time) > i:
                            frame_data = da.isel(forecast_period=0, forecast_reference_time=i).values
                            time_value = str(ds.forecast_reference_time.values[i])
                        else:
                            # Caso em que não há frames suficientes na dimensão reference_time
                            frame_coords = {dim: min(i, len(da[dim])-1) if dim == 'forecast_reference_time' else 0 
                                        for dim in da.dims if dim in ['forecast_period', 'forecast_reference_time']}
                            frame_data = da.isel(**frame_coords).values
                            time_value = "Frame " + str(i+1)
                    else:
                        frame_data = da.isel({time_dim: min(i, len(da[time_dim])-1)}).values
                        if len(da[time_dim]) > i:
                            time_value = str(da[time_dim].values[i])
                        else:
                            time_value = "Frame " + str(i+1)
                    
                    # Atualizar dados do plot
                    im.set_array(frame_data.ravel())
                    ax.set_title(f'AOD 550nm em {city} - {time_value[:16]}', fontsize=12)
                    return [im]
                except Exception as e:
                    st.error(f"Erro no frame {i}: {str(e)}")
                    return [im]
            
            # Ajustar número de frames se necessário
            actual_frames = min(frames, 10)  # Limitar a 10 frames para evitar problemas
            
            ani = animation.FuncAnimation(fig, animate, frames=actual_frames, interval=500, blit=True)
            gif_filename = f'AOD550_{city}_{start_date}_to_{end_date}.gif'
            
            with st.spinner('💾 Salvando animação...'):
                ani.save(gif_filename, writer=animation.PillowWriter(fps=2))
            
            st.success(f"🎉 Animação salva como {gif_filename}")
            st.image(gif_filename)
            
            # Limpar arquivo temporário
            plt.close(fig)
    
    except Exception as e:
        st.error(f"❌ Erro ao processar os dados: {str(e)}")
        st.write("Detalhes da requisição:")
        st.write(request)
