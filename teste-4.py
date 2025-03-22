import cdsapi
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import animation
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import os

DATADIR = '.'

def download_and_animate():
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    start_time = start_time_entry.get()
    end_time = end_time_entry.get()
    
    dataset = "cams-global-atmospheric-composition-forecasts"
    request = {
        'variable': ['organic_matter_aerosol_optical_depth_550nm'],
        'date': [f'{start_date}/{end_date}'],
        'time': [start_time, end_time],
        'leadtime_hour': ['0'],
        'type': ['forecast'],
        'data_format': 'netcdf',
        'area': [80, -150, 25, -50]
    }
    
    client = cdsapi.Client()
    filename = f'{DATADIR}/OAOD_{start_date}_to_{end_date}.nc'
    
    client.retrieve(dataset, request).download(filename)
    
    if not os.path.exists(filename):
        status_label.config(text="Erro: O arquivo não foi baixado corretamente.")
        return
    
    ds = xr.open_dataset(filename)
    print(ds)  # Verificar a estrutura do dataset

    if 'forecast_reference_time' not in ds:
        status_label.config(text="Erro: forecast_reference_time não encontrado no dataset.")
        return

    da = ds['omaod550']  # AOD na banda de 550nm
    
    # Verificar se há mais de um frame
    frames = len(ds.forecast_reference_time)
    print(f"Total de frames disponíveis: {frames}")
    if frames < 2:
        status_label.config(text="Erro: Dados insuficientes para animação.")
        return
    
    vmin, vmax = float(da.min()), float(da.max())  # Ajustando escala de cores
    
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax.set_title(f'Organic Matter AOD at 550nm, {start_date} to {end_date}', fontsize=12)
    ax.coastlines(color='black')
    
    da_frame = da.isel(forecast_period=0, forecast_reference_time=0).values
    im = ax.pcolormesh(ds.longitude, ds.latitude, da_frame, cmap='YlOrRd', vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Organic Matter AOD at 550nm')
    
    def animate(i):
        frame_data = da.isel(forecast_period=0, forecast_reference_time=i).values
        im.set_array(im.norm(frame_data))  # Ajuste para evitar problemas na animação
        ax.set_title(f'Organic Matter AOD at 550nm, {str(ds.forecast_reference_time.values[i])[:16]}', fontsize=12)
    
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=300)
    
    gif_filename = f'{DATADIR}/OAOD_{start_date}_to_{end_date}.gif'
    ani.save(gif_filename, writer=animation.PillowWriter(fps=5))
    
    status_label.config(text=f"Animação salva como {gif_filename}")

# Criando interface gráfica (GUI)
root = tk.Tk()
root.title("AOD Animation Generator")

tk.Label(root, text="Data de Início (YYYY-MM-DD):").pack()
start_date_entry = ttk.Entry(root)
start_date_entry.pack()
start_date_entry.insert(0, datetime.today().strftime('%Y-%m-%d'))

tk.Label(root, text="Data Final (YYYY-MM-DD):").pack()
end_date_entry = ttk.Entry(root)
end_date_entry.pack()
end_date_entry.insert(0, datetime.today().strftime('%Y-%m-%d'))

tk.Label(root, text="Horário Inicial (HH:MM):").pack()
start_time_entry = ttk.Entry(root)
start_time_entry.pack()
start_time_entry.insert(0, "00:00")

tk.Label(root, text="Horário Final (HH:MM):").pack()
end_time_entry = ttk.Entry(root)
end_time_entry.pack()
end_time_entry.insert(0, "12:00")

ttk.Button(root, text="Gerar Animação", command=download_and_animate).pack()
status_label = tk.Label(root, text="")
status_label.pack()

root.mainloop()




