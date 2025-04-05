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
    while current_hour != end_hour + 3:
        hours.append(f"{current_hour:02d}:00")
        current_hour = (current_hour + 3) % 24
        if current_hour == 0 and end_hour != 21:  # Break to avoid infinite loop if end < start
            break
    
    # If no hours were added, use default
    if not hours:
        hours = ['00:00', '03:00', '06:00', '09:00', '12:00']
    
    request = {
        'variable': ['total_aerosol_optical_depth_550nm'],  # Corrected variable name
        'date': f'{start_date_str}/{end_date_str}',
        'time': hours,
        'leadtime_hour': ['0', '3', '6'],  # Common leadtime hours
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
        
        # Try to find the AOD variable
        aod_var = None
        for var in variable_names:
            if 'aod' in var.lower() or 'optical_depth' in var.lower():
                aod_var = var
                break
        
        if not aod_var:
            aod_var = variable_names[0]  # Just use the first variable if AOD not found
        
        st.write(f"Usando variável: {aod_var}")
        da = ds[aod_var]
        
        # Check dimensions
        st.write(f"Dimensões: {da.dims}")
        
        # Adapt to actual dimensions in the dataset
        time_dim = None
        for dim in da.dims:
            if 'time' in dim or 'forecast' in dim:
                time_dim = dim
                break
        
        if not time_dim:
            st.error("Não foi possível identificar a dimensão temporal nos dados.")
            st.stop()
        
        frames = len(da[time_dim])
        st.write(f"✅ Total de frames disponíveis: {frames}")
        
        if frames < 2:
            st.error("Erro: Dados insuficientes para animação.")
        else:
            vmin, vmax = float(da.min()), float(da.max())
            fig = plt.figure(figsize=(10, 5))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
            ax.set_extent([lon_center - 5, lon_center + 5, lat_center - 5, lat_center + 5], crs=ccrs.PlateCarree())
            
            # Handle different dimension structures
            if 'forecast_period' in da.dims and 'forecast_reference_time' in da.dims:
                da_frame = da.isel(forecast_period=0, forecast_reference_time=0).values
            else:
                # Get first frame based on time dimension
                da_frame = da.isel({time_dim: 0}).values
            
            im = ax.pcolormesh(ds.longitude, ds.latitude, da_frame, cmap='YlOrRd', vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label('AOD 550nm')
            
            def animate(i):
                if 'forecast_period' in da.dims and 'forecast_reference_time' in da.dims:
                    frame_data = da.isel(forecast_period=0, forecast_reference_time=i).values
                    time_value = str(ds.forecast_reference_time.values[i])
                else:
                    frame_data = da.isel({time_dim: i}).values
                    time_value = str(da[time_dim].values[i])
                
                im.set_array(frame_data.ravel())
                ax.set_title(f'AOD 550nm em {city} - {time_value[:16]}', fontsize=12)
            
            ani = animation.FuncAnimation(fig, animate, frames=frames, interval=300)
            gif_filename = f'AOD550_{city}_{start_date}_to_{end_date}.gif'
            ani.save(gif_filename, writer=animation.PillowWriter(fps=5))
            
            st.success(f"🎉 Animação salva como {gif_filename}")
            st.image(gif_filename)
    
    except Exception as e:
        st.error(f"❌ Erro ao baixar os dados: {str(e)}")
        st.write("Detalhes da requisição:")
        st.write(request)
