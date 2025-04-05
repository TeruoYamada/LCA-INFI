import cdsapi
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import animation
import streamlit as st
from datetime import datetime
import os

# 🎯 Carregar autenticação do secrets
ads_url = st.secrets["ads"]["url"]
ads_key = st.secrets["ads"]["key"]
client = cdsapi.Client(url=ads_url, key=ads_key)

# Dicionário com algumas cidades do MS
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
start_time = st.time_input("Horário Inicial", datetime.strptime("00:00", "%H:%M").time())
end_time = st.time_input("Horário Final", datetime.strptime("12:00", "%H:%M").time())

if st.button("🎞️ Gerar Animação"):
    dataset = "cams-global-atmospheric-composition-forecasts"
    request = {
        'variable': ['aerosol_optical_depth_550nm'],
        'model_level': ['1'],  # ou remova se não for necessário
        'date': f'{start_date}/{end_date}',
        'time': [start_time.strftime("%H:%M"), end_time.strftime("%H:%M")],
        'leadtime_hour': ['0'],
        'type': ['forecast'],
        'format': 'netcdf',
        'area': [lat_center + 5, lon_center - 5, lat_center - 5, lon_center + 5]  # +-5 graus em torno da cidade
    }

    filename = f'AOD550_{city}_{start_date}_to_{end_date}.nc'
    
    try:
        with st.spinner('📥 Baixando dados do CAMS...'):
            client.retrieve(dataset, request).download(filename)

        ds = xr.open_dataset(filename)
        da = ds['aod550']

        frames = len(ds.forecast_reference_time)
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

            da_frame = da.isel(forecast_period=0, forecast_reference_time=0).values
            im = ax.pcolormesh(ds.longitude, ds.latitude, da_frame, cmap='YlOrRd', vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label('AOD 550nm')

            def animate(i):
                frame_data = da.isel(forecast_period=0, forecast_reference_time=i).values
                im.set_array(frame_data.ravel())
                ax.set_title(f'AOD 550nm em {city} - {str(ds.forecast_reference_time.values[i])[:16]}', fontsize=12)

            ani = animation.FuncAnimation(fig, animate, frames=frames, interval=300)
            gif_filename = f'AOD550_{city}_{start_date}_to_{end_date}.gif'
            ani.save(gif_filename, writer=animation.PillowWriter(fps=5))

            st.success(f"🎉 Animação salva como {gif_filename}")
            st.image(gif_filename)

    except Exception as e:
        st.error(f"❌ Erro ao baixar os dados: {str(e)}")
