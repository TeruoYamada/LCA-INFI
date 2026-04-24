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
from scipy import stats
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ============================================================
#  CREDENCIAIS — lidas de .streamlit/secrets.toml
#
#  Estrutura esperada no secrets.toml:
#
#  [ads]
#  url = "https://ads.atmosphere.copernicus.eu/api/v2"
#  key = "sua-chave-cams"
#
#  [email]
#  remetente    = "seuemail@gmail.com"
#  senha_app    = "xxxx xxxx xxxx xxxx"   # Senha de App do Gmail (16 chars)
#  destinatario = "destinatario@gmail.com"
# ============================================================
try:
    EMAIL_REMETENTE    = st.secrets["email"]["remetente"]
    EMAIL_SENHA_APP    = st.secrets["email"]["senha_app"]
    EMAIL_DESTINATARIO = st.secrets["email"]["destinatario"]
    _email_configurado = True
except KeyError:
    _email_configurado = False


def enviar_alerta_email(cidade, aqi, categoria, pm25, pm10, data_critica, cidades_criticas_df=None):
    """Envia alerta por e-mail quando IQA > 100. Retorna True se bem-sucedido."""
    if not _email_configurado:
        return False

    try:
        msg            = MIMEMultipart("alternative")
        msg["Subject"] = f"ALERTA Qualidade do Ar - {cidade} | IQA: {aqi:.0f} ({categoria})"
        msg["From"]    = EMAIL_REMETENTE
        msg["To"]      = EMAIL_DESTINATARIO

        if aqi <= 150:
            cor_fundo, cor_texto = "#ff7e00", "#ffffff"
        elif aqi <= 200:
            cor_fundo, cor_texto = "#ff0000", "#ffffff"
        elif aqi <= 300:
            cor_fundo, cor_texto = "#8f3f97", "#ffffff"
        else:
            cor_fundo, cor_texto = "#7e0023", "#ffffff"

        recomendacao = (
            "Grupos sensíveis (crianças, idosos, asmáticos) devem evitar esforços ao ar livre."
            if aqi <= 150 else
            "Evite esforços prolongados ao ar livre. Mantenha janelas fechadas."
            if aqi <= 200 else
            "EVITE TODAS as atividades ao ar livre. Procure ambientes fechados com filtração de ar."
        )

        tabela_cidades = ""
        if cidades_criticas_df is not None and not cidades_criticas_df.empty:
            rows = "".join(
                f"<tr>"
                f"<td style='padding:6px 12px;border-bottom:1px solid #eee;'>{r['cidade']}</td>"
                f"<td style='padding:6px 12px;border-bottom:1px solid #eee;text-align:center;'>{r['pm25_max']:.1f}</td>"
                f"<td style='padding:6px 12px;border-bottom:1px solid #eee;text-align:center;'>{r['pm10_max']:.1f}</td>"
                f"<td style='padding:6px 12px;border-bottom:1px solid #eee;text-align:center;font-weight:bold;'>{r['aqi_max']:.0f}</td>"
                f"<td style='padding:6px 12px;border-bottom:1px solid #eee;'>{r['categoria']}</td>"
                f"</tr>"
                for _, r in cidades_criticas_df.head(10).iterrows()
            )
            tabela_cidades = f"""
            <h3 style='color:#333;margin-top:28px;'>Municípios em Alerta (Top 10)</h3>
            <table style='border-collapse:collapse;width:100%;font-size:13px;'>
              <thead><tr style='background:#f0f0f0;'>
                <th style='padding:8px 12px;text-align:left;'>Município</th>
                <th style='padding:8px 12px;'>PM2.5 Máx</th>
                <th style='padding:8px 12px;'>PM10 Máx</th>
                <th style='padding:8px 12px;'>IQA Máx</th>
                <th style='padding:8px 12px;text-align:left;'>Categoria</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>"""

        html = f"""
        <html><body style='font-family:Arial,sans-serif;background:#f5f5f5;padding:20px;'>
          <div style='max-width:680px;margin:auto;background:#fff;border-radius:12px;
                      box-shadow:0 2px 12px rgba(0,0,0,.1);overflow:hidden;'>
            <div style='background:{cor_fundo};padding:28px 32px;text-align:center;'>
              <h1 style='color:{cor_texto};margin:0;font-size:24px;'>🚨 ALERTA DE QUALIDADE DO AR</h1>
              <p style='color:{cor_texto};margin:8px 0 0;font-size:16px;opacity:.9;'>
                Mato Grosso do Sul — Monitor PM2.5 / PM10
              </p>
            </div>
            <div style='padding:28px 32px;'>
              <h2 style='color:#222;margin-top:0;'>{cidade}</h2>
              <table style='border-collapse:collapse;width:100%;font-size:15px;margin-bottom:20px;'>
                <tr>
                  <td style='padding:10px 16px;background:#fff8e1;font-weight:bold;color:#e65100;'>IQA</td>
                  <td style='padding:10px 16px;font-size:22px;font-weight:bold;color:{cor_fundo};'>
                    {aqi:.0f} — {categoria}</td>
                </tr>
                <tr>
                  <td style='padding:10px 16px;color:#555;'>PM2.5</td>
                  <td style='padding:10px 16px;font-weight:bold;'>{pm25:.1f} μg/m³
                    <span style='color:#888;font-size:12px;'>(OMS 24h: 25 μg/m³)</span></td>
                </tr>
                <tr>
                  <td style='padding:10px 16px;color:#555;'>PM10</td>
                  <td style='padding:10px 16px;font-weight:bold;'>{pm10:.1f} μg/m³
                    <span style='color:#888;font-size:12px;'>(OMS 24h: 50 μg/m³)</span></td>
                </tr>
                <tr>
                  <td style='padding:10px 16px;color:#555;'>Data Crítica</td>
                  <td style='padding:10px 16px;'>{data_critica}</td>
                </tr>
              </table>
              <div style='background:#fff3cd;border-left:4px solid #ff9800;
                          padding:14px 18px;border-radius:6px;margin-bottom:20px;'>
                <strong>⚠ Recomendações:</strong><br>{recomendacao}
              </div>
              {tabela_cidades}
              <p style='color:#999;font-size:11px;margin-top:28px;border-top:1px solid #eee;padding-top:14px;'>
                Alerta gerado automaticamente pelo <strong>Monitor PM2.5/PM10 — MS</strong>.<br>
                Dados: CAMS (Copernicus) · {datetime.now().strftime("%d/%m/%Y %H:%M")}
              </p>
            </div>
          </div>
        </body></html>"""

        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(EMAIL_REMETENTE, EMAIL_SENHA_APP)
            srv.sendmail(EMAIL_REMETENTE, EMAIL_DESTINATARIO, msg.as_string())
        return True

    except Exception as e:
        print(f"Erro ao enviar e-mail: {e}")
        return False


def create_pm_animation(ds, pm_var, city, lat_center, lon_center, ms_shapes, start_date, pm_type="PM2.5"):
    da_pm = ds[pm_var]
    if da_pm.max().values < 1e-6:   da_pm = da_pm * 1e9
    elif da_pm.max().values < 1e-3: da_pm = da_pm * 1e6
    elif da_pm.max().values > 1000: da_pm = da_pm / 1000

    time_dims = [d for d in da_pm.dims if 'time' in d or 'forecast' in d]
    time_dim  = 'forecast_reference_time' if 'forecast_reference_time' in da_pm.dims else time_dims[0]
    frames    = len(da_pm[time_dim])
    if frames < 1:
        st.error("Dados insuficientes para animação."); return None

    vmin, vmax = float(da_pm.min().values), float(da_pm.max().values)
    if pm_type == "PM2.5":
        vmin, vmax, cmap = max(0, vmin-5), min(200, vmax+10), 'YlOrRd'
    else:
        vmin, vmax, cmap = max(0, vmin-5), min(300, vmax+15), 'Oranges'

    fig = plt.figure(figsize=(14, 10))
    ax  = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,  facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', color='gray')
    ax.add_feature(cfeature.STATES.with_scale('50m'),  linestyle='-', edgecolor='black', linewidth=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False; gl.right_labels = False
    ax.set_extent([-58.5, -50.5, -24.5, -17.0], crs=ccrs.PlateCarree())

    if ms_shapes is not None and not ms_shapes.empty:
        try:
            ms_shapes.boundary.plot(ax=ax, color='black', linewidth=0.6,
                                    transform=ccrs.PlateCarree(), alpha=0.7)
            sel = ms_shapes[ms_shapes['NM_MUN'].str.upper() == city.upper()]
            if not sel.empty:
                sel.boundary.plot(ax=ax, color='red', linewidth=3.0, transform=ccrs.PlateCarree())
                sel.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=3.0,
                         transform=ccrs.PlateCarree(), alpha=0.8)
        except Exception as e:
            print(f"Erro shapefile: {e}")

    ax.plot(lon_center, lat_center, 'ro', markersize=12, transform=ccrs.PlateCarree(),
            markeredgecolor='white', markeredgewidth=2, zorder=10)

    if 'forecast_period' in da_pm.dims and 'forecast_reference_time' in da_pm.dims:
        ffd  = da_pm.isel(forecast_period=0, forecast_reference_time=0).values
        fft  = pd.to_datetime(ds.forecast_reference_time.values[0])
    else:
        ffd  = da_pm.isel({time_dim: 0}).values
        fft  = pd.to_datetime(da_pm[time_dim].values[0])

    im    = ax.pcolormesh(ds.longitude, ds.latitude, ffd, cmap=cmap, vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree(), alpha=0.8)
    cbar  = plt.colorbar(im, fraction=0.046, pad=0.04, orientation='horizontal')
    cbar.set_label(f'{pm_type} (μg/m³)', fontsize=12, weight='bold')
    title = ax.set_title(f'{pm_type} - {city}\n{fft.strftime("%d/%m/%Y %H:%M UTC")}',
                         fontsize=16, pad=20, weight='bold')
    lim_txt = 'Limites: OMS=25 μg/m³, EPA=35 μg/m³' if pm_type == "PM2.5" else 'Limites: OMS=50 μg/m³, EPA=150 μg/m³'
    ax.text(0.02, 0.02, lim_txt, transform=ax.transAxes, fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='orange'),
            verticalalignment='bottom')

    def animate(i):
        try:
            if 'forecast_period' in da_pm.dims and 'forecast_reference_time' in da_pm.dims:
                fd = da_pm.isel(forecast_period=min(0,len(da_pm.forecast_period)-1),
                                forecast_reference_time=min(i,len(da_pm.forecast_reference_time)-1)).values
                ft = pd.to_datetime(ds.forecast_reference_time.values[min(i,len(da_pm.forecast_reference_time)-1)])
            else:
                fd = da_pm.isel({time_dim: min(i,len(da_pm[time_dim])-1)}).values
                ft = pd.to_datetime(da_pm[time_dim].values[min(i,len(da_pm[time_dim])-1)])
            im.set_array(fd.ravel())
            title.set_text(f'{pm_type} - {city}\n{ft.strftime("%d/%m/%Y %H:%M UTC")}')
            return [im, title]
        except Exception as e:
            print(f"Frame {i}: {e}"); return [im, title]

    return fig, animate, min(frames, 20)


def extract_pm_timeseries(ds, lat, lon, pm25_var, pm10_var):
    lat_idx = np.abs(ds.latitude.values - lat).argmin()
    lon_idx = np.abs(ds.longitude.values - lon).argmin()
    times, pm25v, pm10v = [], [], []

    def norm(v):
        if v < 1e-6: return v * 1e9
        if v < 1e-3: return v * 1e6
        if v > 1000: return v / 1000
        return v

    if 'forecast_reference_time' in ds.dims and 'forecast_period' in ds.dims:
        for ti, rt in enumerate(ds.forecast_reference_time.values):
            for pi, per in enumerate(ds.forecast_period.values):
                try:
                    v25 = norm(float(ds[pm25_var].isel(forecast_reference_time=ti,forecast_period=pi,latitude=lat_idx,longitude=lon_idx).values))
                    v10 = norm(float(ds[pm10_var].isel(forecast_reference_time=ti,forecast_period=pi,latitude=lat_idx,longitude=lon_idx).values))
                    times.append(pd.to_datetime(rt) + pd.to_timedelta(per, unit='h'))
                    pm25v.append(v25); pm10v.append(v10)
                except: continue
    elif any(d in ds.dims for d in ['time','forecast_reference_time']):
        td = next(d for d in ds.dims if d in ['time','forecast_reference_time'])
        for ti in range(len(ds[td])):
            try:
                v25 = norm(float(ds[pm25_var].isel({td:ti,'latitude':lat_idx,'longitude':lon_idx}).values))
                v10 = norm(float(ds[pm10_var].isel({td:ti,'latitude':lat_idx,'longitude':lon_idx}).values))
                times.append(pd.to_datetime(ds[td].isel({td:ti}).values))
                pm25v.append(v25); pm10v.append(v10)
            except: continue

    if not times:
        return pd.DataFrame(columns=['time','pm25','pm10','aqi','aqi_category'])

    df = pd.DataFrame({'time':times,'pm25':pm25v,'pm10':pm10v}).sort_values('time').reset_index(drop=True)
    aq = df.apply(lambda r: calculate_aqi(r['pm25'],r['pm10']), axis=1)
    df['aqi'] = aq.apply(lambda x: x[0])
    df['aqi_category'] = aq.apply(lambda x: x[1])
    df['aqi_color']    = aq.apply(lambda x: x[2])
    return df


def calculate_aqi(pm25, pm10):
    def sub(c, bp):
        for lo,hi,ilo,ihi in bp:
            if lo <= c <= hi: return ((ihi-ilo)/(hi-lo))*(c-lo)+ilo
        return 500
    aqi = max(
        sub(pm25,[(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500,301,500)]),
        sub(pm10,[(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,600,301,500)])
    )
    if aqi<=50:  return aqi,"Boa","green"
    if aqi<=100: return aqi,"Moderada","yellow"
    if aqi<=150: return aqi,"Insalubre para Grupos Sensíveis","orange"
    if aqi<=200: return aqi,"Insalubre","red"
    if aqi<=300: return aqi,"Muito Insalubre","purple"
    return aqi,"Perigosa","maroon"


def predict_future_values(df, days=5, max_date=None):
    if len(df) < 3:
        return pd.DataFrame(columns=['time','pm25','pm10','aqi','type'])
    dh = df.copy()
    dh['tn'] = (dh['time'] - dh['time'].min()).dt.total_seconds()
    X  = dh['tn'].values.reshape(-1,1)
    m25 = LinearRegression().fit(X, dh['pm25'].values)
    m10 = LinearRegression().fit(X, dh['pm10'].values)

    last = dh['time'].max()
    cands = [last + timedelta(hours=i*6) for i in range(1, days*4+1)
             if timedelta(hours=i*6) <= timedelta(days=days)]
    if max_date is not None:
        mx    = pd.Timestamp(max_date).replace(hour=23,minute=59,second=59)
        cands = [t for t in cands if t <= mx]
    if not cands:
        dh['type'] = 'historical'
        return dh[['time','pm25','pm10','aqi','aqi_category','aqi_color','type']]

    ftn  = [(t - dh['time'].min()).total_seconds() for t in cands]
    fp25 = np.maximum(m25.predict(np.array(ftn).reshape(-1,1)), 0)
    fp10 = np.maximum(m10.predict(np.array(ftn).reshape(-1,1)), 0)
    fa,fc,fco = [],[],[]
    for v25,v10 in zip(fp25,fp10):
        a,c,co = calculate_aqi(v25,v10); fa.append(a); fc.append(c); fco.append(co)

    dh['type'] = 'historical'
    return pd.concat([
        dh[['time','pm25','pm10','aqi','aqi_category','aqi_color','type']],
        pd.DataFrame({'time':cands,'pm25':fp25,'pm10':fp10,'aqi':fa,'aqi_category':fc,'aqi_color':fco,'type':'forecast'})
    ], ignore_index=True)


def analyze_all_cities(ds, pm25_var, pm10_var, cities_dict, end_date=None):
    res, pb, st_txt = [], st.progress(0), st.empty()
    for i,(name,coords) in enumerate(cities_dict.items()):
        pb.progress((i+1)/len(cities_dict)); st_txt.text(f"Analisando {name}... ({i+1}/{len(cities_dict)})")
        df_ts = extract_pm_timeseries(ds, coords[0], coords[1], pm25_var, pm10_var)
        if not df_ts.empty:
            fo = predict_future_values(df_ts, days=5, max_date=end_date)
            fo = fo[fo['type']=='forecast']
            if not fo.empty:
                idx = fo['aqi'].idxmax()
                res.append({'cidade':name,'pm25_max':fo['pm25'].max(),'pm10_max':fo['pm10'].max(),
                            'aqi_max':fo['aqi'].max(),'data_max':fo.loc[idx,'time'],'categoria':fo.loc[idx,'aqi_category']})
    pb.empty(); st_txt.empty()
    if res:
        df = pd.DataFrame(res).sort_values('aqi_max',ascending=False).reset_index(drop=True)
        df['pm25_max']=df['pm25_max'].round(1); df['pm10_max']=df['pm10_max'].round(1)
        df['aqi_max']=df['aqi_max'].round(0); df['data_max']=df['data_max'].dt.strftime('%d/%m/%Y %H:%M')
        return df
    return pd.DataFrame(columns=['cidade','pm25_max','pm10_max','aqi_max','data_max','categoria'])


def get_aqi_color(v):
    if v<=50:  return '#00e400'
    if v<=100: return '#ffff00'
    if v<=150: return '#ff7e00'
    if v<=200: return '#ff0000'
    if v<=300: return '#8f3f97'
    return '#7e0023'

def style_aqi_table(df):
    def s(row):
        bg = get_aqi_color(row['IQA Máx']); tc = 'white' if row['IQA Máx']>100 else 'black'
        return [f'background-color:{bg};color:{tc}']*len(row)
    return df.style.apply(s, axis=1)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Monitor PM2.5/PM10 - MS", page_icon="🌍")

try:
    client = cdsapi.Client(url=st.secrets["ads"]["url"], key=st.secrets["ads"]["key"])
except Exception:
    st.error("Erro ao carregar credenciais CAMS. Verifique secrets.toml → seção [ads]."); st.stop()


@st.cache_data
def load_ms_municipalities():
    url = ("https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/"
           "malhas_municipais/municipio_2022/UFs/MS/MS_Municipios_2022.zip")
    try:
        gdf = gpd.read_file(url)
        if 'NM_MUN' not in gdf.columns:
            for alt in ['NM_MUNICIP','NOME']:
                if alt in gdf.columns: gdf['NM_MUN'] = gdf[alt]; break
        return gdf
    except Exception as e:
        st.warning(f"Shapefile IBGE indisponível: {e}"); return create_fallback_shapefile()


cities = {
    "Água Clara":[-20.4453,-52.8792],"Alcinópolis":[-18.3255,-53.7042],"Amambai":[-23.1058,-55.2253],
    "Anastácio":[-20.4823,-55.8104],"Anaurilândia":[-22.1852,-52.7191],"Angélica":[-22.1527,-53.7708],
    "Antônio João":[-22.1927,-55.9511],"Aparecida do Taboado":[-20.0873,-51.0961],
    "Aquidauana":[-20.4697,-55.7868],"Aral Moreira":[-22.9384,-55.6331],
    "Bandeirantes":[-19.9279,-54.3581],"Bataguassu":[-21.7156,-52.4233],"Batayporã":[-22.2947,-53.2705],
    "Bela Vista":[-22.1073,-56.5263],"Bodoquena":[-20.5372,-56.7138],"Bonito":[-21.1261,-56.4836],
    "Brasilândia":[-21.2544,-52.0382],"Caarapó":[-22.6368,-54.8209],"Camapuã":[-19.5302,-54.0431],
    "Campo Grande":[-20.4697,-54.6201],"Caracol":[-22.0112,-57.0278],"Cassilândia":[-19.1179,-51.7308],
    "Chapadão do Sul":[-18.7908,-52.626],"Corguinho":[-19.8243,-54.8281],"Coronel Sapucaia":[-23.2724,-55.5278],
    "Corumbá":[-19.0082,-57.651],"Costa Rica":[-18.5432,-53.1287],"Coxim":[-18.5013,-54.7603],
    "Deodápolis":[-22.2789,-54.1583],"Dois Irmãos do Buriti":[-20.6845,-55.2915],
    "Douradina":[-22.043,-54.6158],"Dourados":[-22.2231,-54.812],"Eldorado":[-23.7868,-54.2836],
    "Fátima do Sul":[-22.3789,-54.5131],"Figueirão":[-18.6782,-53.638],"Glória de Dourados":[-22.4136,-54.2336],
    "Guia Lopes da Laguna":[-21.4583,-56.1117],"Iguatemi":[-23.6835,-54.5635],"Inocência":[-19.7276,-51.9281],
    "Itaporã":[-22.075,-54.7933],"Itaquiraí":[-23.4779,-54.1873],"Ivinhema":[-22.3046,-53.8185],
    "Japorã":[-23.8903,-54.4059],"Jaraguari":[-20.1386,-54.3996],"Jardim":[-21.4799,-56.1489],
    "Jateí":[-22.4806,-54.3078],"Juti":[-22.8596,-54.606],"Ladário":[-19.009,-57.5973],
    "Laguna Carapã":[-22.5448,-55.1502],"Maracaju":[-21.6105,-55.1695],"Miranda":[-20.2407,-56.378],
    "Mundo Novo":[-23.9355,-54.2807],"Naviraí":[-23.0618,-54.1995],"Nioaque":[-21.1419,-55.8296],
    "Nova Alvorada do Sul":[-21.4657,-54.3825],"Nova Andradina":[-22.2332,-53.3437],
    "Novo Horizonte do Sul":[-22.6693,-53.8601],"Paraíso das Águas":[-19.0218,-53.0116],
    "Paranaíba":[-19.6746,-51.1909],"Paranhos":[-23.8905,-55.4289],"Pedro Gomes":[-18.0996,-54.5507],
    "Ponta Porã":[-22.5334,-55.7271],"Porto Murtinho":[-21.6981,-57.8825],"Ribas do Rio Pardo":[-20.4432,-53.7588],
    "Rio Brilhante":[-21.8033,-54.5427],"Rio Negro":[-19.4473,-54.9859],"Rio Verde de Mato Grosso":[-18.9249,-54.8434],
    "Rochedo":[-19.9566,-54.894],"Santa Rita do Pardo":[-21.3016,-52.8333],"São Gabriel do Oeste":[-19.395,-54.5507],
    "Selvíria":[-20.3637,-51.4192],"Sete Quedas":[-23.971,-55.0396],"Sidrolândia":[-20.9302,-54.9692],
    "Sonora":[-17.5698,-54.7551],"Tacuru":[-23.6361,-55.0141],"Taquarussu":[-22.4898,-53.3519],
    "Terenos":[-20.4378,-54.8647],"Três Lagoas":[-20.7849,-51.7005],"Vicentina":[-22.4098,-54.4415]
}

def create_fallback_shapefile():
    from shapely.geometry import Polygon
    data = [{'NM_MUN':n,'geometry':Polygon([(lon-0.15,lat-0.15),(lon+0.15,lat-0.15),
             (lon+0.15,lat+0.15),(lon-0.15,lat+0.15),(lon-0.15,lat-0.15)])}
            for n,(lat,lon) in cities.items()]
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


# ------ Interface ------
st.title("🌍 Monitoramento PM2.5 e PM10 - Mato Grosso do Sul")
st.markdown("**Dados CAMS · Previsão +5 dias · Alerta automático por e-mail**")

with st.spinner("Carregando shapes municipais..."):
    ms_shapes = load_ms_municipalities()

st.sidebar.header("Configurações")
available = sorted(set(ms_shapes['NM_MUN'].tolist()) & set(cities.keys())) or list(cities.keys())
default_i = available.index("Campo Grande") if "Campo Grande" in available else 0
city      = st.sidebar.selectbox("Município", available, index=default_i)
lat_center, lon_center = cities[city]

st.sidebar.subheader("Período")
start_date = st.sidebar.date_input("Data de Início", datetime.today() - timedelta(days=2))
end_date   = start_date + timedelta(days=5)
st.sidebar.info(f"📅 +5 dias fixo\n{start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}")

all_hours  = list(range(0,24,3))
start_hour = st.sidebar.selectbox("Horário Inicial", all_hours, format_func=lambda x: f"{x:02d}:00")
end_hour   = st.sidebar.selectbox("Horário Final",   all_hours, index=len(all_hours)-1,
                                   format_func=lambda x: f"{x:02d}:00")

with st.sidebar.expander("Visualização"):
    animation_speed     = st.slider("Velocidade animação (ms)", 200, 1000, 500)
    show_pm10_animation = st.checkbox("Animação PM10", value=True)

if _email_configurado:
    st.sidebar.success(f"📧 Alertas ativos\n→ {EMAIL_DESTINATARIO}")
else:
    st.sidebar.error("📧 E-mail não configurado\nAdicione [email] no secrets.toml")


def generate_pm_analysis():
    hours = []
    cur   = start_hour
    while True:
        hours.append(f"{cur:02d}:00")
        if cur == end_hour: break
        cur = (cur+3) % 24
        if cur == start_hour: break
    if not hours:
        hours = ['00:00','03:00','06:00','09:00','12:00','15:00','18:00','21:00']

    request  = {
        'variable':      ['particulate_matter_2.5um','particulate_matter_10um'],
        'date':          f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
        'time':          hours,
        'leadtime_hour': ['0','24','48','72','96','120'],
        'type':          ['forecast'],
        'format':        'netcdf',
        'area':          [-17.0,-58.5,-24.5,-50.5]
    }
    fname = f'PM_{city}_{start_date}_to_{end_date}.nc'

    try:
        with st.spinner('Baixando dados CAMS...'):
            client.retrieve("cams-global-atmospheric-composition-forecasts", request).download(fname)

        ds = xr.open_dataset(fname)
        dv = list(ds.data_vars)
        pm25_var = next((v for v in dv if 'pm2p5' in v.lower() or '2.5' in v), None)
        pm10_var = next((v for v in dv if 'pm10'  in v.lower() or '10um' in v), None)
        if not pm25_var or not pm10_var:
            st.error("Variáveis PM não encontradas."); st.write(dv); return None

        with st.spinner("Extraindo série temporal..."):
            df_ts = extract_pm_timeseries(ds, lat_center, lon_center, pm25_var, pm10_var)
        if df_ts.empty: st.error("Sem dados para este local."); return None

        with st.spinner("Gerando previsões..."):
            df_fc = predict_future_values(df_ts, days=5, max_date=end_date)

        with st.spinner("Animação PM2.5..."):
            r25 = create_pm_animation(ds,pm25_var,city,lat_center,lon_center,ms_shapes,start_date,"PM2.5")
            if r25 is None: return None
            f25,a25,fr25 = r25
            ani25 = animation.FuncAnimation(f25,a25,frames=fr25,interval=animation_speed,blit=True)
            gif25 = f'PM25_{city}_{start_date}.gif'
            ani25.save(gif25, writer=animation.PillowWriter(fps=2)); plt.close(f25)

        gif10 = None
        if show_pm10_animation:
            with st.spinner("Animação PM10..."):
                r10 = create_pm_animation(ds,pm10_var,city,lat_center,lon_center,ms_shapes,start_date,"PM10")
                if r10:
                    f10,a10,fr10 = r10
                    ani10 = animation.FuncAnimation(f10,a10,frames=fr10,interval=animation_speed,blit=True)
                    gif10 = f'PM10_{city}_{start_date}.gif'
                    ani10.save(gif10, writer=animation.PillowWriter(fps=2)); plt.close(f10)

        mx = ds[pm25_var].max().values
        if mx < 1e-6:   ds[pm25_var]*=1e9;  ds[pm10_var]*=1e9
        elif mx < 1e-3: ds[pm25_var]*=1e6;  ds[pm10_var]*=1e6
        elif mx > 1000: ds[pm25_var]/=1000; ds[pm10_var]/=1000

        top = None
        try:
            with st.spinner("Analisando todos os municípios..."):
                top = analyze_all_cities(ds, pm25_var, pm10_var, cities, end_date=end_date)
        except Exception as e:
            st.warning(f"Análise estadual parcial: {e}")
            top = pd.DataFrame(columns=['cidade','pm25_max','pm10_max','aqi_max','data_max','categoria'])

        # --- ALERTA E-MAIL ---
        fo        = df_fc[df_fc['type']=='forecast']
        aqi_max_p = fo['aqi'].max() if not fo.empty else 0
        if aqi_max_p > 100:
            idx_w = fo['aqi'].idxmax()
            dt_c  = fo.loc[idx_w,'time']
            dc    = dt_c.strftime('%d/%m/%Y %H:%M') if hasattr(dt_c,'strftime') else str(dt_c)
            crit  = top[top['aqi_max']>100] if top is not None and not top.empty else None
            with st.spinner("Enviando alerta por e-mail..."):
                ok = enviar_alerta_email(city, fo.loc[idx_w,'aqi'], fo.loc[idx_w,'aqi_category'],
                                         fo.loc[idx_w,'pm25'], fo.loc[idx_w,'pm10'], dc, crit)
            if ok:
                st.success(f"📧 Alerta enviado → {EMAIL_DESTINATARIO}  |  IQA: {fo.loc[idx_w,'aqi']:.0f} ({fo.loc[idx_w,'aqi_category']})")
            elif _email_configurado:
                st.warning("Falha ao enviar e-mail. Verifique as credenciais no secrets.toml.")
        # ---------------------

        return {'animation_pm25':gif25,'animation_pm10':gif10,'timeseries':df_ts,
                'forecast':df_fc,'dataset':ds,'pm25_var':pm25_var,'pm10_var':pm10_var,'top_pollution':top}

    except Exception as e:
        st.error(f"Erro: {e}"); import traceback; st.code(traceback.format_exc()); return None
    finally:
        if os.path.exists(fname):
            try: os.remove(fname)
            except: pass


st.markdown(f"Análise de **{city}** · {start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}")

if st.button("Gerar Análise de Qualidade do Ar", type="primary", use_container_width=True):
    try:
        results = generate_pm_analysis()
        if results:
            tab1, tab2, tab3 = st.tabs(["Análise do Município","Alerta Estadual",f"Animações - {city}"])

            with tab3:
                st.subheader(f"Animações — {city}")
                if os.path.exists(results['animation_pm25']):
                    st.markdown("#### PM2.5")
                    st.image(results['animation_pm25'], caption=f"PM2.5 · {city}")
                    with open(results['animation_pm25'],"rb") as f:
                        st.download_button("⬇ GIF PM2.5", f, file_name=f"PM25_{city}.gif", mime="image/gif")
                if results['animation_pm10'] and os.path.exists(results['animation_pm10']):
                    st.markdown("#### PM10")
                    st.image(results['animation_pm10'], caption=f"PM10 · {city}")
                    with open(results['animation_pm10'],"rb") as f:
                        st.download_button("⬇ GIF PM10", f, file_name=f"PM10_{city}.gif", mime="image/gif")

            with tab1:
                st.subheader(f"Análise — {city}")
                c1, c2 = st.columns([3,2])
                dc, hd, fd = results['forecast'], None, None
                hd = dc[dc['type']=='historical']
                fd = dc[dc['type']=='forecast']

                with c1:
                    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,10),sharex=True)
                    for ax,col,ch,cf,lim1,lim2,yl in [
                        (ax1,'pm25','darkblue','steelblue',25,35,'PM2.5 (μg/m³)'),
                        (ax2,'pm10','brown','sienna',50,150,'PM10 (μg/m³)')]:
                        if not hd.empty: ax.plot(hd['time'],hd[col],'o-',color=ch,label=f'{col.upper()} Obs.',markersize=6)
                        if not fd.empty: ax.plot(fd['time'],fd[col],'s--',color=cf,label=f'{col.upper()} Prev.',markersize=5,alpha=0.8)
                        ax.axhline(y=lim1,color='orange',linestyle='--',alpha=0.7,label=f'OMS ({lim1})')
                        ax.axhline(y=lim2,color='red',   linestyle='--',alpha=0.7,label=f'EPA ({lim2})')
                        ax.set_ylabel(yl,fontsize=11); ax.legend(); ax.grid(True,alpha=0.3)
                    if not hd.empty: ax3.plot(hd['time'],hd['aqi'],'o-',color='purple',label='IQA Obs.',markersize=6)
                    if not fd.empty: ax3.plot(fd['time'],fd['aqi'],'s--',color='mediumpurple',label='IQA Prev.',markersize=5,alpha=0.8)
                    for ylo,yhi,cor in [(0,50,'green'),(51,100,'yellow'),(101,150,'orange'),(151,200,'red')]:
                        ax3.axhspan(ylo,yhi,alpha=0.15,color=cor)
                    ax3.set_ylabel('IQA',fontsize=11); ax3.legend(); ax3.grid(True,alpha=0.3)
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
                    plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig)

                with c2:
                    if not hd.empty:
                        r = hd.iloc[-1]
                        ca,cb = st.columns(2)
                        ca.metric("PM2.5",f"{r['pm25']:.1f} μg/m³"); cb.metric("PM10",f"{r['pm10']:.1f} μg/m³")
                        st.metric("IQA",f"{r['aqi']:.0f}")
                        st.markdown(f"<div style='padding:15px;border-radius:10px;background:{r['aqi_color']};"
                                    f"color:white;text-align:center;'><h3 style='margin:0;'>Qualidade do Ar</h3>"
                                    f"<h2 style='margin:5px 0;'>{r['aqi_category']}</h2></div>",
                                    unsafe_allow_html=True)
                        v = r['aqi']
                        if v<=50:   st.success("Condições ideais para atividades ao ar livre")
                        elif v<=100: st.info("Pessoas sensíveis devem limitar esforços")
                        elif v<=150: st.warning("Grupos sensíveis: evite atividades ao ar livre")
                        elif v<=200: st.error("Evite esforços prolongados ao ar livre")
                        else:        st.error("Evite TODAS as atividades ao ar livre")
                        st.progress(min(r['pm25']/25,1.0)); st.caption(f"PM2.5: {r['pm25']:.1f}/25 μg/m³ (OMS)")
                        st.progress(min(r['pm10']/50,1.0)); st.caption(f"PM10: {r['pm10']:.1f}/50 μg/m³ (OMS)")
                    csv = dc.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇ CSV Completo", csv, file_name=f"PM_{city}_{start_date}.csv", mime="text/csv")

            with tab2:
                st.subheader("Alerta Estadual — Mato Grosso do Sul")
                tp = results.get('top_pollution')
                if tp is not None and not tp.empty:
                    t20 = tp.head(20)
                    crit,very = t20[t20['aqi_max']>100], t20[t20['aqi_max']>150]
                    ca,cb,cc = st.columns(3)
                    ca.metric("Cidades em Alerta",len(crit)); cb.metric("Condição Insalubre",len(very))
                    cc.metric("IQA Máx Previsto",f"{t20['aqi_max'].max():.0f}")
                    if len(crit)>0 and len(t20)>=3:
                        st.error(f"### {len(crit)} municípios em alerta até {end_date.strftime('%d/%m/%Y')}\n\n"
                                 f"1. **{t20.iloc[0]['cidade']}** — IQA {t20.iloc[0]['aqi_max']:.0f}\n"
                                 f"2. **{t20.iloc[1]['cidade']}** — IQA {t20.iloc[1]['aqi_max']:.0f}\n"
                                 f"3. **{t20.iloc[2]['cidade']}** — IQA {t20.iloc[2]['aqi_max']:.0f}")
                    st.dataframe(style_aqi_table(t20.rename(columns={
                        'cidade':'Município','pm25_max':'PM2.5 Máx (μg/m³)','pm10_max':'PM10 Máx (μg/m³)',
                        'aqi_max':'IQA Máx','data_max':'Data Crítica','categoria':'Categoria'})),
                        use_container_width=True)
                    fig,ax = plt.subplots(figsize=(14,7))
                    t10 = t20.head(10); x,w = np.arange(len(t10)),0.35
                    b1 = ax.bar(x-w/2,t10['pm25_max'],w,color='darkblue',alpha=0.8,label='PM2.5')
                    b2 = ax.bar(x+w/2,t10['pm10_max'],w,color='brown',   alpha=0.8,label='PM10')
                    ax.set_xticks(x); ax.set_xticklabels(t10['cidade'],rotation=45,ha='right')
                    ax.axhline(25,color='orange',linestyle='--',alpha=0.7,label='OMS PM2.5')
                    ax.axhline(50,color='red',   linestyle='--',alpha=0.7,label='OMS PM10')
                    ax.legend(); ax.grid(True,alpha=0.3)
                    ax.set_title(f'PM Máximo Previsto até {end_date.strftime("%d/%m/%Y")}')
                    for b,v in list(zip(b1,t10['pm25_max']))+list(zip(b2,t10['pm10_max'])):
                        ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{v:.0f}',ha='center',va='bottom',fontsize=9)
                    plt.tight_layout(); st.pyplot(fig)
                    st.download_button("⬇ CSV Alerta", t20.to_csv(index=False).encode('utf-8'),
                                       file_name=f"Alerta_MS_{start_date}.csv", mime="text/csv")
                else:
                    st.info("Dados estaduais não disponíveis.")

    except Exception as e:
        st.error(f"Erro: {e}"); import traceback; st.code(traceback.format_exc())


st.markdown("---")
with st.expander("ℹ️ Configuração e Suporte"):
    status_email = "✅ Configurado" if _email_configurado else "❌ Não configurado"
    st.markdown(f"""
    **Status e-mail:** {status_email}

    **Estrutura do `.streamlit/secrets.toml`:**
    ```toml
    [ads]
    url = "https://ads.atmosphere.copernicus.eu/api/v2"
    key = "sua-chave-cams"

    [email]
    remetente    = "seuemail@gmail.com"
    senha_app    = "xxxx xxxx xxxx xxxx"
    destinatario = "destinatario@gmail.com"
    ```
    > **Senha de App Gmail:** acesse myaccount.google.com/apppasswords
    > (requer verificação em 2 etapas ativa na conta)

    **Regra de alerta:** IQA previsto > 100 → e-mail enviado automaticamente
    **Resolução:** ~0.4° × 0.4° · Temporal: 3h · Previsão: +5 dias fixo
    """)
