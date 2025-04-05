import cdsapi
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import animation
import streamlit as st
from datetime import datetime
import os

st.title("AOD Animation Generator (CAMs 550nm)")

start_date = st.date_input("Data de Início", datetime.today())
end_date = st.date_input("Data Final", datetime.today())
start_time = st.time_input("Horário Inicial", datetime.strptime("00:00", "%H:%M").time())
end_time = st.time_input("Horário Final", datetime.strptime("12:00", "%H:%M").time())

if st.button("Gerar Animação"):
    dataset = "cams-global-atmospheric-composition-forecasts"
    request = {
        'variable': ['organic_matter_aerosol_optical_depth_550nm'],
        'date': [f'{start_date}/{end_date}'],
        'time': [start_time.strftime("%H:%M"), end_time.strftime("%H:%M")],
        'leadtime_hour': ['0'],
        'type': ['forecast'],
        'data_format': 'netcdf',
        'area': [80, -150, 25, -50]
    }

    filename = f'OAOD_{start_date}_to_{end_date}.nc'
    
    with st.spinner('Baixando dados do CAMS...'):
        client = cdsapi.Client()
        client.retrieve(dataset, request).download(filename)

    if not os.path.exists(filename):
        st.error("Erro: O arquivo não foi baixado corretamente.")
    else:
        ds = xr.open_dataset(filename)
        if 'forecast_reference_time' not in ds:
            st.error("Erro: forecast_reference_time não encontrado no dataset.")
        else:
            da = ds['omaod550']
            frames = len(ds.forecast_reference_time)
            st.write(f"Total de frames disponíveis: {frames}")

            if frames < 2:
                st.error("Erro: Dados insuficientes para animação.")
            else:
                vmin, vmax = float(da.min()), float(da.max())
                fig = plt.figure(figsize=(10, 5))
                ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
                ax.coastlines()
                ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
                ax.set_title(f'Organic Matter AOD at 550nm, {start_date} to {end_date}', fontsize=12)

                da_frame = da.isel(forecast_period=0, forecast_reference_time=0).values
                im = ax.pcolormesh(ds.longitude, ds.latitude, da_frame, cmap='YlOrRd', vmin=vmin, vmax=vmax)
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                cbar.set_label('Organic Matter AOD at 550nm')

                def animate(i):
                    frame_data = da.isel(forecast_period=0, forecast_reference_time=i).values
                    im.set_array(im.norm(frame_data))
                    ax.set_title(f'Organic Matter AOD at 550nm, {str(ds.forecast_reference_time.values[i])[:16]}', fontsize=12)

                ani = animation.FuncAnimation(fig, animate, frames=frames, interval=300)
                gif_filename = f'OAOD_{start_date}_to_{end_date}.gif'
                ani.save(gif_filename, writer=animation.PillowWriter(fps=5))
                st.success(f"Animação salva como {gif_filename}")
                st.image(gif_filename)




