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
from email.mime.base import MIMEBase
from email import encoders
import base64


try:
    EMAIL_REMETENTE     = st.secrets["email"]["remetente"]
    EMAIL_SENHA_APP     = st.secrets["email"]["senha_app"]
    # Suporte a múltiplos destinatários: pode ser string única ou lista
    _dest_raw           = st.secrets["email"]["destinatario"]
    if isinstance(_dest_raw, str):
        EMAIL_DESTINATARIOS = [e.strip() for e in _dest_raw.split(",") if e.strip()]
    else:
        EMAIL_DESTINATARIOS = list(_dest_raw)
    _email_configurado  = True
except KeyError:
    EMAIL_DESTINATARIOS = []
    _email_configurado  = False


def enviar_relatorio_email(
    cidade, df_timeseries, df_forecast, top_pollution_df,
    start_date, end_date,
    gif_pm25_path=None, gif_pm10_path=None
):
    if not _email_configurado:
        return False

    try:
        hist = df_timeseries
        fc   = df_forecast[df_forecast['type'] == 'forecast'] if 'type' in df_forecast.columns else pd.DataFrame()

        if not hist.empty:
            ultimo       = hist.iloc[-1]
            curr_pm25    = ultimo['pm25']
            curr_pm10    = ultimo['pm10']
            curr_aqi     = ultimo['aqi']
            curr_cat     = ultimo['aqi_category']
            curr_color   = ultimo['aqi_color']
            curr_time    = ultimo['time'].strftime('%d/%m/%Y %H:%M') if hasattr(ultimo['time'], 'strftime') else str(ultimo['time'])
        else:
            curr_pm25 = curr_pm10 = curr_aqi = 0
            curr_cat  = "N/D"
            curr_color = "gray"
            curr_time  = "N/D"

        if not fc.empty:
            idx_max     = fc['aqi'].idxmax()
            prev_pm25   = fc.loc[idx_max, 'pm25']
            prev_pm10   = fc.loc[idx_max, 'pm10']
            prev_aqi    = fc.loc[idx_max, 'aqi']
            prev_cat    = fc.loc[idx_max, 'aqi_category']
            prev_color  = fc.loc[idx_max, 'aqi_color']
            prev_time   = fc.loc[idx_max, 'time']
            prev_time_s = prev_time.strftime('%d/%m/%Y %H:%M') if hasattr(prev_time, 'strftime') else str(prev_time)
            aqi_max_periodo = fc['aqi'].max()
            pm25_med_prev   = fc['pm25'].mean()
            pm10_med_prev   = fc['pm10'].mean()
        else:
            prev_pm25 = prev_pm10 = prev_aqi = 0
            prev_cat  = "N/D"
            prev_color = "gray"
            prev_time_s = "N/D"
            aqi_max_periodo = 0
            pm25_med_prev = pm10_med_prev = 0

        def cor_faixa(aqi_val):
            if aqi_val <= 50:   return "#00c853", "#ffffff"
            if aqi_val <= 100:  return "#ffd600", "#333333"
            if aqi_val <= 150:  return "#ff7e00", "#ffffff"
            if aqi_val <= 200:  return "#ff0000", "#ffffff"
            if aqi_val <= 300:  return "#8f3f97", "#ffffff"
            return "#7e0023", "#ffffff"

        cor_h, txt_h = cor_faixa(aqi_max_periodo if aqi_max_periodo > curr_aqi else curr_aqi)

        def recomendacao(aqi_val):
            if aqi_val <= 50:
                return "✅ Condições ideais para atividades ao ar livre."
            if aqi_val <= 100:
                return "⚠️ Pessoas sensíveis devem considerar limitar esforços prolongados ao ar livre."
            if aqi_val <= 150:
                return "⚠️ Grupos sensíveis (crianças, idosos, asmáticos) devem evitar esforços ao ar livre."
            if aqi_val <= 200:
                return "🚨 Evite esforços prolongados ao ar livre. Mantenha janelas fechadas."
            return "🚨 EVITE TODAS as atividades ao ar livre. Procure ambientes com filtração de ar."

        rec_atual  = recomendacao(curr_aqi)
        rec_prev   = recomendacao(prev_aqi)

        def badge_aqi(aqi_val, cat):
            bg, tc = cor_faixa(aqi_val)
            return f"<span style='background:{bg};color:{tc};padding:2px 8px;border-radius:4px;font-weight:bold;font-size:12px;'>{aqi_val:.0f} – {cat}</span>"

        def rows_timeseries(df, tipo_label, cor_tipo):
            rows = ""
            for _, r in df.tail(10).iterrows():
                t = r['time'].strftime('%d/%m %H:%M') if hasattr(r['time'], 'strftime') else str(r['time'])
                badge = badge_aqi(r['aqi'], r['aqi_category'])
                rows += (f"<tr>"
                         f"<td style='padding:6px 10px;border-bottom:1px solid #eee;'>{t}</td>"
                         f"<td style='padding:6px 10px;border-bottom:1px solid #eee;text-align:center;"
                         f"color:{cor_tipo};font-weight:bold;'>{tipo_label}</td>"
                         f"<td style='padding:6px 10px;border-bottom:1px solid #eee;text-align:center;'>{r['pm25']:.1f}</td>"
                         f"<td style='padding:6px 10px;border-bottom:1px solid #eee;text-align:center;'>{r['pm10']:.1f}</td>"
                         f"<td style='padding:6px 10px;border-bottom:1px solid #eee;'>{badge}</td>"
                         f"</tr>")
            return rows

        rows_hist = rows_timeseries(hist, "Previsto", "#1565c0")
        rows_prev = rows_timeseries(fc,   "Previsão",  "#e65100") if not fc.empty else ""

        tabela_serie = f"""
        <h3 style='color:#333;margin-top:28px;border-bottom:2px solid #e0e0e0;padding-bottom:6px;'>
          📊 Série Temporal — {cidade}
        </h3>
        <table style='border-collapse:collapse;width:100%;font-size:13px;'>
          <thead>
            <tr style='background:#f5f5f5;'>
              <th style='padding:8px 10px;text-align:left;'>Data/Hora</th>
              <th style='padding:8px 10px;'>Tipo</th>
              <th style='padding:8px 10px;'>PM2.5 (μg/m³)</th>
              <th style='padding:8px 10px;'>PM10 (μg/m³)</th>
              <th style='padding:8px 10px;text-align:left;'>IQA</th>
            </tr>
          </thead>
          <tbody>{rows_hist}{rows_prev}</tbody>
        </table>
        <p style='font-size:11px;color:#999;margin-top:4px;'>
          * Dados CAMS: previsões até {end_date.strftime('%d/%m/%Y')}
        </p>"""

        tabela_estadual = ""
        n_alerta = 0
        if top_pollution_df is not None and not top_pollution_df.empty:
            crit_df = top_pollution_df[top_pollution_df['aqi_max'] > 100]
            n_alerta = len(crit_df)
            rows_est = ""
            for _, r in top_pollution_df.head(15).iterrows():
                bg, tc = cor_faixa(r['aqi_max'])
                destaque = "font-weight:bold;" if r['aqi_max'] > 150 else ""
                rows_est += (f"<tr>"
                             f"<td style='padding:6px 10px;border-bottom:1px solid #eee;{destaque}'>{r['cidade']}</td>"
                             f"<td style='padding:6px 10px;border-bottom:1px solid #eee;text-align:center;'>{r['pm25_max']:.1f}</td>"
                             f"<td style='padding:6px 10px;border-bottom:1px solid #eee;text-align:center;'>{r['pm10_max']:.1f}</td>"
                             f"<td style='padding:6px 10px;border-bottom:1px solid #eee;text-align:center;'>"
                             f"<span style='background:{bg};color:{tc};padding:2px 8px;border-radius:4px;"
                             f"font-weight:bold;font-size:12px;'>{r['aqi_max']:.0f}</span></td>"
                             f"<td style='padding:6px 10px;border-bottom:1px solid #eee;color:{bg};font-weight:bold;'>{r['categoria']}</td>"
                             f"<td style='padding:6px 10px;border-bottom:1px solid #eee;font-size:12px;color:#555;'>{r['data_max']}</td>"
                             f"</tr>")
            tabela_estadual = f"""
            <h3 style='color:#333;margin-top:32px;border-bottom:2px solid #e0e0e0;padding-bottom:6px;'>
              🗺️ Ranking Estadual — Municípios em Alerta (Top 15)
            </h3>
            <table style='border-collapse:collapse;width:100%;font-size:13px;'>
              <thead>
                <tr style='background:#f5f5f5;'>
                  <th style='padding:8px 10px;text-align:left;'>Município</th>
                  <th style='padding:8px 10px;'>PM2.5 Máx</th>
                  <th style='padding:8px 10px;'>PM10 Máx</th>
                  <th style='padding:8px 10px;'>IQA Máx</th>
                  <th style='padding:8px 10px;text-align:left;'>Categoria</th>
                  <th style='padding:8px 10px;text-align:left;'>Data Crítica</th>
                </tr>
              </thead>
              <tbody>{rows_est}</tbody>
            </table>"""

        def card(titulo, valor, unidade, cor_borda):
            return (f"<div style='flex:1;min-width:130px;background:#fff;border-radius:10px;"
                    f"border-left:5px solid {cor_borda};padding:14px 16px;box-shadow:0 1px 6px rgba(0,0,0,.08);'>"
                    f"<div style='font-size:12px;color:#777;'>{titulo}</div>"
                    f"<div style='font-size:22px;font-weight:bold;color:#222;'>{valor}"
                    f"<span style='font-size:12px;color:#999;margin-left:3px;'>{unidade}</span></div>"
                    f"</div>")

        cards_atual = (
            card("PM2.5 Previsto",      f"{curr_pm25:.1f}", "μg/m³", "#1565c0") +
            card("PM10 Previsto",       f"{curr_pm10:.1f}", "μg/m³", "#6d4c41") +
            card("IQA Atual",           f"{curr_aqi:.0f}",  "",       cor_h) +
            card("PM2.5 Prev. Máx",     f"{prev_pm25:.1f}", "μg/m³", "#e65100") +
            card("PM10 Prev. Máx",      f"{prev_pm10:.1f}", "μg/m³", "#bf360c") +
            card("IQA Prev. Máx",       f"{prev_aqi:.0f}",  "",       cor_h)
        )

        assunto = (f"📋 Relatório Qualidade do Ar — {cidade} | "
                   f"{start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}")

        alerta_header = ""
        if aqi_max_periodo > 100:
            alerta_header = f"""
            <div style='background:#fff3cd;border-left:5px solid #ff9800;padding:14px 18px;
                        border-radius:6px;margin-bottom:20px;'>
              <strong>🚨 ATENÇÃO:</strong> IQA previsto máximo de <strong>{aqi_max_periodo:.0f}</strong>
              em <strong>{cidade}</strong> até {end_date.strftime('%d/%m/%Y')} —
              categoria <strong>{prev_cat}</strong>.<br>
              {rec_prev}
            </div>"""

        gif_aviso = ""
        if gif_pm25_path or gif_pm10_path:
            gif_aviso = "<p style='color:#1565c0;font-size:13px;'>📎 Animações GIF anexadas a este e-mail.</p>"

        html = f"""
        <html>
        <body style='font-family:Arial,sans-serif;background:#f0f2f5;padding:20px;margin:0;'>
          <div style='max-width:720px;margin:auto;background:#fff;border-radius:14px;
                      box-shadow:0 4px 18px rgba(0,0,0,.12);overflow:hidden;'>

            <div style='background:{cor_h};padding:32px 36px;text-align:center;'>
              <h1 style='color:{txt_h};margin:0;font-size:26px;letter-spacing:.5px;'>
                🌍 Relatório de Qualidade do Ar
              </h1>
              <p style='color:{txt_h};margin:8px 0 0;font-size:15px;opacity:.92;'>
                Mato Grosso do Sul — Monitor PM2.5 / PM10 (CAMS)
              </p>
            </div>

            <div style='padding:32px 36px;'>

              <table style='width:100%;margin-bottom:20px;'>
                <tr>
                  <td style='font-size:14px;color:#555;'>
                    <strong>Município:</strong> {cidade}<br>
                    <strong>Período:</strong> {start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}<br>
                    <strong>Referência temporal:</strong> {curr_time}
                  </td>
                  <td style='text-align:right;'>
                    <div style='background:{cor_h};color:{txt_h};padding:10px 18px;border-radius:8px;
                                display:inline-block;font-size:13px;'>
                      <strong>Categoria atual:</strong><br>
                      <span style='font-size:18px;font-weight:bold;'>{curr_cat}</span>
                    </div>
                  </td>
                </tr>
              </table>

              {alerta_header}

              <div style='display:flex;flex-wrap:wrap;gap:10px;margin-bottom:24px;'>
                {cards_atual}
              </div>

              <div style='background:#e8f5e9;border-left:4px solid #43a047;
                          padding:12px 16px;border-radius:6px;margin-bottom:8px;font-size:13px;'>
                <strong>Condição prevista atual:</strong> {rec_atual}
              </div>
              <div style='background:#fff8e1;border-left:4px solid #ffa000;
                          padding:12px 16px;border-radius:6px;margin-bottom:24px;font-size:13px;'>
                <strong>Previsão (pior momento — {prev_time_s}):</strong> {rec_prev}
              </div>

              <h3 style='color:#333;border-bottom:2px solid #e0e0e0;padding-bottom:6px;margin-top:0;'>
                📏 Comparação com Padrões OMS (24h)
              </h3>
              <table style='width:100%;font-size:13px;margin-bottom:24px;'>
                <tr>
                  <td style='padding:6px 0;color:#555;'>PM2.5 previsto</td>
                  <td style='padding:6px 0;'>
                    <div style='background:#e0e0e0;border-radius:6px;height:14px;width:100%;position:relative;'>
                      <div style='background:#1565c0;border-radius:6px;height:14px;
                                  width:{min(curr_pm25/25*100,100):.0f}%;'></div>
                    </div>
                  </td>
                  <td style='padding:6px 0 6px 10px;white-space:nowrap;color:#222;font-weight:bold;'>
                    {curr_pm25:.1f} / 25 μg/m³
                  </td>
                </tr>
                <tr>
                  <td style='padding:6px 0;color:#555;'>PM10 previsto</td>
                  <td style='padding:6px 0;'>
                    <div style='background:#e0e0e0;border-radius:6px;height:14px;width:100%;'>
                      <div style='background:#6d4c41;border-radius:6px;height:14px;
                                  width:{min(curr_pm10/50*100,100):.0f}%;'></div>
                    </div>
                  </td>
                  <td style='padding:6px 0 6px 10px;white-space:nowrap;color:#222;font-weight:bold;'>
                    {curr_pm10:.1f} / 50 μg/m³
                  </td>
                </tr>
                <tr>
                  <td style='padding:6px 0;color:#555;'>PM2.5 prev. máx</td>
                  <td style='padding:6px 0;'>
                    <div style='background:#e0e0e0;border-radius:6px;height:14px;width:100%;'>
                      <div style='background:#e65100;border-radius:6px;height:14px;
                                  width:{min(prev_pm25/25*100,100):.0f}%;'></div>
                    </div>
                  </td>
                  <td style='padding:6px 0 6px 10px;white-space:nowrap;color:#222;font-weight:bold;'>
                    {prev_pm25:.1f} / 25 μg/m³
                  </td>
                </tr>
                <tr>
                  <td style='padding:6px 0;color:#555;'>PM10 prev. máx</td>
                  <td style='padding:6px 0;'>
                    <div style='background:#e0e0e0;border-radius:6px;height:14px;width:100%;'>
                      <div style='background:#bf360c;border-radius:6px;height:14px;
                                  width:{min(prev_pm10/50*100,100):.0f}%;'></div>
                    </div>
                  </td>
                  <td style='padding:6px 0 6px 10px;white-space:nowrap;color:#222;font-weight:bold;'>
                    {prev_pm10:.1f} / 50 μg/m³
                  </td>
                </tr>
              </table>

              <h3 style='color:#333;border-bottom:2px solid #e0e0e0;padding-bottom:6px;'>
                🎨 Escala do Índice de Qualidade do Ar (IQA)
              </h3>
              <table style='border-collapse:collapse;width:100%;font-size:12px;margin-bottom:24px;'>
                <tr>
                  <td style='background:#00e400;color:#333;padding:5px 10px;border-radius:4px 0 0 4px;font-weight:bold;'>0–50</td>
                  <td style='padding:5px 10px;border-bottom:1px solid #f0f0f0;'>Boa — Atividades ao ar livre sem restrições</td>
                </tr>
                <tr>
                  <td style='background:#ffd600;color:#333;padding:5px 10px;font-weight:bold;'>51–100</td>
                  <td style='padding:5px 10px;border-bottom:1px solid #f0f0f0;'>Moderada — Sensíveis devem limitar esforços</td>
                </tr>
                <tr>
                  <td style='background:#ff7e00;color:#fff;padding:5px 10px;font-weight:bold;'>101–150</td>
                  <td style='padding:5px 10px;border-bottom:1px solid #f0f0f0;'>Insalubre para Grupos Sensíveis</td>
                </tr>
                <tr>
                  <td style='background:#ff0000;color:#fff;padding:5px 10px;font-weight:bold;'>151–200</td>
                  <td style='padding:5px 10px;border-bottom:1px solid #f0f0f0;'>Insalubre — Evite esforços ao ar livre</td>
                </tr>
                <tr>
                  <td style='background:#8f3f97;color:#fff;padding:5px 10px;font-weight:bold;'>201–300</td>
                  <td style='padding:5px 10px;border-bottom:1px solid #f0f0f0;'>Muito Insalubre</td>
                </tr>
                <tr>
                  <td style='background:#7e0023;color:#fff;padding:5px 10px;border-radius:0 0 0 4px;font-weight:bold;'>&gt;300</td>
                  <td style='padding:5px 10px;'>Perigosa — Evite TODAS as atividades ao ar livre</td>
                </tr>
              </table>

              {tabela_serie}
              {tabela_estadual}

              {"" if fc.empty else f"""
              <h3 style='color:#333;margin-top:28px;border-bottom:2px solid #e0e0e0;padding-bottom:6px;'>
                📈 Estatísticas do Período de Previsão
              </h3>
              <table style='border-collapse:collapse;width:100%;font-size:13px;margin-bottom:20px;'>
                <tr style='background:#f5f5f5;'>
                  <th style='padding:8px 12px;text-align:left;'>Indicador</th>
                  <th style='padding:8px 12px;'>PM2.5</th>
                  <th style='padding:8px 12px;'>PM10</th>
                  <th style='padding:8px 12px;'>IQA</th>
                </tr>
                <tr>
                  <td style='padding:7px 12px;border-bottom:1px solid #eee;'>Mínimo</td>
                  <td style='padding:7px 12px;border-bottom:1px solid #eee;text-align:center;'>{fc['pm25'].min():.1f} μg/m³</td>
                  <td style='padding:7px 12px;border-bottom:1px solid #eee;text-align:center;'>{fc['pm10'].min():.1f} μg/m³</td>
                  <td style='padding:7px 12px;border-bottom:1px solid #eee;text-align:center;'>{fc['aqi'].min():.0f}</td>
                </tr>
                <tr>
                  <td style='padding:7px 12px;border-bottom:1px solid #eee;'>Médio</td>
                  <td style='padding:7px 12px;border-bottom:1px solid #eee;text-align:center;'>{fc['pm25'].mean():.1f} μg/m³</td>
                  <td style='padding:7px 12px;border-bottom:1px solid #eee;text-align:center;'>{fc['pm10'].mean():.1f} μg/m³</td>
                  <td style='padding:7px 12px;border-bottom:1px solid #eee;text-align:center;'>{fc['aqi'].mean():.0f}</td>
                </tr>
                <tr>
                  <td style='padding:7px 12px;'>Máximo</td>
                  <td style='padding:7px 12px;text-align:center;font-weight:bold;color:#e65100;'>{fc['pm25'].max():.1f} μg/m³</td>
                  <td style='padding:7px 12px;text-align:center;font-weight:bold;color:#bf360c;'>{fc['pm10'].max():.1f} μg/m³</td>
                  <td style='padding:7px 12px;text-align:center;font-weight:bold;'>{fc['aqi'].max():.0f}</td>
                </tr>
              </table>"""}

              {gif_aviso}

              {"" if n_alerta == 0 else f"""
              <div style='background:#ffebee;border-left:5px solid #c62828;padding:14px 18px;
                          border-radius:6px;margin-top:20px;font-size:13px;'>
                <strong>⚠️ Alerta Estadual:</strong> {n_alerta} município(s) com IQA previsto acima de 100
                até {end_date.strftime('%d/%m/%Y')} em Mato Grosso do Sul.
              </div>"""}

              <p style='color:#bbb;font-size:11px;margin-top:32px;border-top:1px solid #eee;
                        padding-top:14px;text-align:center;'>
                Relatório gerado automaticamente pelo <strong>Monitor PM2.5/PM10 — MS</strong>.<br>
                Dados: CAMS (Copernicus Atmosphere Monitoring Service) ·
                {datetime.now().strftime("%d/%m/%Y %H:%M")}<br>
                Resolução espacial: ~0.4° × 0.4° · Temporal: 3h
              </p>
            </div>
          </div>
        </body>
        </html>"""

        msg            = MIMEMultipart("mixed")
        msg["Subject"] = assunto
        msg["From"]    = EMAIL_REMETENTE
        msg["To"]      = ", ".join(EMAIL_DESTINATARIOS)

        alt_part = MIMEMultipart("alternative")
        alt_part.attach(MIMEText(html, "html"))
        msg.attach(alt_part)

        for gif_path, label in [(gif_pm25_path, "PM25"), (gif_pm10_path, "PM10")]:
            if gif_path and os.path.exists(gif_path):
                with open(gif_path, "rb") as f:
                    part = MIMEBase("image", "gif")
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", "attachment",
                                    filename=f"Animacao_{label}_{cidade}_{start_date.strftime('%Y%m%d')}.gif")
                    msg.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(EMAIL_REMETENTE, EMAIL_SENHA_APP)
            srv.sendmail(EMAIL_REMETENTE, EMAIL_DESTINATARIOS, msg.as_string())
        return True

    except Exception as e:
        print(f"Erro ao enviar e-mail: {e}")
        return False


def create_pm_animation(ds, pm_var, city, lat_center, lon_center, ms_shapes, start_date, pm_type="PM2.5"):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    import pandas as pd

    da_pm = ds[pm_var]

    if da_pm.max().values < 1e-6:
        da_pm = da_pm * 1e9
    elif da_pm.max().values < 1e-3:
        da_pm = da_pm * 1e6
    elif da_pm.max().values > 1000:
        da_pm = da_pm / 1000

    time_dims = [dim for dim in da_pm.dims if 'time' in dim or 'forecast' in dim]

    if 'forecast_reference_time' in da_pm.dims:
        time_dim = 'forecast_reference_time'
        frames = len(da_pm[time_dim])
    else:
        time_dim = time_dims[0]
        frames = len(da_pm[time_dim])

    if frames < 1:
        st.error("Erro: Dados insuficientes para animação.")
        return None

    vmin_raw = float(da_pm.min().values)
    vmax_raw = float(da_pm.max().values)

    # Limiar: valores abaixo de 1 μg/m³ aparecem em branco
    THRESHOLD = 1.0

    if pm_type == "PM2.5":
        vmin = THRESHOLD
        vmax = min(200, vmax_raw + 10)
        colormap_name = 'YlOrRd'
    else:
        vmin = THRESHOLD
        vmax = min(300, vmax_raw + 15)
        colormap_name = 'Oranges'

    # Colormap com branco para valores abaixo do limiar
    import copy
    cmap_obj = plt.cm.get_cmap(colormap_name).copy()
    cmap_obj.set_under('white')

    ms_extent = [-58.5, -50.5, -24.5, -17.0]

    fig = plt.figure(figsize=(14, 10))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.coastlines(resolution='50m', color='black', linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', color='gray')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linestyle='-', edgecolor='black', linewidth=1)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    ax.set_extent(ms_extent, crs=ccrs.PlateCarree())

    if ms_shapes is not None and not ms_shapes.empty:
        try:
            ms_shapes.boundary.plot(ax=ax, color='black', linewidth=0.6,
                                    transform=ccrs.PlateCarree(), alpha=0.7)
            selected_city = ms_shapes[ms_shapes['NM_MUN'].str.upper() == city.upper()]
            if not selected_city.empty:
                selected_city.boundary.plot(ax=ax, color='red', linewidth=3.0,
                                            transform=ccrs.PlateCarree())
                selected_city.plot(ax=ax, facecolor='none', edgecolor='red',
                                   linewidth=3.0, transform=ccrs.PlateCarree(), alpha=0.8)
        except Exception as e:
            print(f"Erro ao plotar shapefile: {e}")

    ax.plot(lon_center, lat_center, 'ro', markersize=12, transform=ccrs.PlateCarree(),
            markeredgecolor='white', markeredgewidth=2, zorder=10)

    if 'forecast_period' in da_pm.dims and 'forecast_reference_time' in da_pm.dims:
        first_frame_data = da_pm.isel(forecast_period=0, forecast_reference_time=0).values
        first_frame_time = pd.to_datetime(ds.forecast_reference_time.values[0])
    else:
        first_frame_data = da_pm.isel({time_dim: 0}).values
        first_frame_time = pd.to_datetime(da_pm[time_dim].values[0])

    # Mascarar valores abaixo do limiar
    masked_first = np.ma.masked_less(first_frame_data, THRESHOLD)

    im = ax.pcolormesh(ds.longitude, ds.latitude, masked_first,
                       cmap=cmap_obj, vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), alpha=0.8)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04, orientation='horizontal',
                        extend='min')  # extend='min' indica que < vmin é branco
    cbar.set_label(f'{pm_type} (μg/m³)', fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)

    title = ax.set_title(f'{pm_type} (previsto CAMS) - {city}\n{first_frame_time.strftime("%d/%m/%Y %H:%M UTC")}',
                         fontsize=16, pad=20, weight='bold')

    if pm_type == "PM2.5":
        ax.text(0.02, 0.02, 'Limites: OMS=25 μg/m³, EPA=35 μg/m³',
                transform=ax.transAxes, fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='orange'),
                verticalalignment='bottom')
    else:
        ax.text(0.02, 0.02, 'Limites: OMS=50 μg/m³, EPA=150 μg/m³',
                transform=ax.transAxes, fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='orange'),
                verticalalignment='bottom')

    def animate(i):
        try:
            frame_data = None
            frame_time = None

            if 'forecast_period' in da_pm.dims and 'forecast_reference_time' in da_pm.dims:
                fp_idx = min(0, len(da_pm.forecast_period) - 1)
                frt_idx = min(i, len(da_pm.forecast_reference_time) - 1)
                frame_data = da_pm.isel(forecast_period=fp_idx, forecast_reference_time=frt_idx).values
                frame_time = pd.to_datetime(ds.forecast_reference_time.values[frt_idx])
            else:
                t_idx = min(i, len(da_pm[time_dim]) - 1)
                frame_data = da_pm.isel({time_dim: t_idx}).values
                frame_time = pd.to_datetime(da_pm[time_dim].values[t_idx])

            # Mascarar abaixo do limiar em cada frame
            masked_frame = np.ma.masked_less(frame_data, THRESHOLD)
            im.set_array(masked_frame.ravel())
            title.set_text(f'{pm_type} (previsto CAMS) - {city}\n{frame_time.strftime("%d/%m/%Y %H:%M UTC")}')

            return [im, title]
        except Exception as e:
            print(f"Erro no frame {i}: {str(e)}")
            return [im, title]

    actual_frames = min(frames, 20)

    return fig, animate, actual_frames


def extract_pm_timeseries(ds, lat, lon, pm25_var, pm10_var):
    lat_idx = np.abs(ds.latitude.values - lat).argmin()
    lon_idx = np.abs(ds.longitude.values - lon).argmin()

    times = []
    pm25_values = []
    pm10_values = []

    time_dims = [dim for dim in ds.dims if 'time' in dim or 'forecast' in dim]

    if 'forecast_reference_time' in ds.dims and 'forecast_period' in ds.dims:
        for t_idx, ref_time in enumerate(ds.forecast_reference_time.values):
            for p_idx, period in enumerate(ds.forecast_period.values):
                try:
                    pm25_val = float(ds[pm25_var].isel(
                        forecast_reference_time=t_idx, forecast_period=p_idx,
                        latitude=lat_idx, longitude=lon_idx).values)
                    pm10_val = float(ds[pm10_var].isel(
                        forecast_reference_time=t_idx, forecast_period=p_idx,
                        latitude=lat_idx, longitude=lon_idx).values)

                    if pm25_val < 1e-6:
                        pm25_val *= 1e9; pm10_val *= 1e9
                    elif pm25_val < 1e-3:
                        pm25_val *= 1e6; pm10_val *= 1e6
                    elif pm25_val > 1000:
                        pm25_val /= 1000; pm10_val /= 1000

                    actual_time = pd.to_datetime(ref_time) + pd.to_timedelta(period, unit='h')
                    times.append(actual_time)
                    pm25_values.append(pm25_val)
                    pm10_values.append(pm10_val)
                except:
                    continue
    elif any(dim in ds.dims for dim in ['time', 'forecast_reference_time']):
        time_dim = next(dim for dim in ds.dims if dim in ['time', 'forecast_reference_time'])
        for t_idx in range(len(ds[time_dim])):
            try:
                pm25_val = float(ds[pm25_var].isel({time_dim: t_idx, 'latitude': lat_idx, 'longitude': lon_idx}).values)
                pm10_val = float(ds[pm10_var].isel({time_dim: t_idx, 'latitude': lat_idx, 'longitude': lon_idx}).values)

                if pm25_val < 1e-6:
                    pm25_val *= 1e9; pm10_val *= 1e9
                elif pm25_val < 1e-3:
                    pm25_val *= 1e6; pm10_val *= 1e6
                elif pm25_val > 1000:
                    pm25_val /= 1000; pm10_val /= 1000

                times.append(pd.to_datetime(ds[time_dim].isel({time_dim: t_idx}).values))
                pm25_values.append(pm25_val)
                pm10_values.append(pm10_val)
            except:
                continue

    if times and pm25_values and pm10_values:
        df = pd.DataFrame({'time': times, 'pm25': pm25_values, 'pm10': pm10_values})
        df = df.sort_values('time').reset_index(drop=True)

        aqi_values = df.apply(lambda row: calculate_aqi(row['pm25'], row['pm10']), axis=1)
        df['aqi'] = aqi_values.apply(lambda x: x[0])
        df['aqi_category'] = aqi_values.apply(lambda x: x[1])
        df['aqi_color'] = aqi_values.apply(lambda x: x[2])

        return df
    else:
        return pd.DataFrame(columns=['time', 'pm25', 'pm10', 'aqi', 'aqi_category'])


def calculate_aqi(pm25, pm10):
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

    aqi = max(calc_sub_index(pm25, pm25_breakpoints), calc_sub_index(pm10, pm10_breakpoints))

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


def predict_future_values(df, days=5, max_date=None):
    if len(df) < 3:
        return pd.DataFrame(columns=['time', 'pm25', 'pm10', 'aqi', 'type'])

    df_hist = df.copy()
    df_hist['time_numeric'] = (df_hist['time'] - df_hist['time'].min()).dt.total_seconds()

    X = df_hist['time_numeric'].values.reshape(-1, 1)

    model_pm25 = LinearRegression()
    model_pm25.fit(X, df_hist['pm25'].values)
    model_pm10 = LinearRegression()
    model_pm10.fit(X, df_hist['pm10'].values)

    last_time = df_hist['time'].max()

    forecast_horizon = timedelta(days=days)
    future_times_candidates = [
        last_time + timedelta(hours=i * 6)
        for i in range(1, days * 4 + 1)
        if timedelta(hours=i * 6) <= forecast_horizon
    ]

    if max_date is not None:
        max_dt = pd.Timestamp(max_date)
        max_dt = max_dt.replace(hour=23, minute=59, second=59)
        future_times = [t for t in future_times_candidates if t <= max_dt]
    else:
        future_times = future_times_candidates

    if not future_times:
        df_hist['type'] = 'historical'
        return df_hist[['time', 'pm25', 'pm10', 'aqi', 'aqi_category', 'aqi_color', 'type']]

    future_time_numeric = [(t - df_hist['time'].min()).total_seconds() for t in future_times]

    future_pm25 = np.maximum(model_pm25.predict(np.array(future_time_numeric).reshape(-1, 1)), 0)
    future_pm10 = np.maximum(model_pm10.predict(np.array(future_time_numeric).reshape(-1, 1)), 0)

    future_aqi, future_categories, future_colors = [], [], []
    for pm25, pm10 in zip(future_pm25, future_pm10):
        aqi, category, color = calculate_aqi(pm25, pm10)
        future_aqi.append(aqi)
        future_categories.append(category)
        future_colors.append(color)

    df_pred = pd.DataFrame({
        'time': future_times, 'pm25': future_pm25, 'pm10': future_pm10,
        'aqi': future_aqi, 'aqi_category': future_categories, 'aqi_color': future_colors,
        'type': 'forecast'
    })

    df_hist['type'] = 'historical'
    result = pd.concat([df_hist[['time', 'pm25', 'pm10', 'aqi', 'aqi_category', 'aqi_color', 'type']], df_pred], ignore_index=True)
    return result


def analyze_all_cities(ds, pm25_var, pm10_var, cities_dict, end_date=None):
    cities_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (city_name, coords) in enumerate(cities_dict.items()):
        progress_bar.progress((i + 1) / len(cities_dict))
        status_text.text(f"Analisando {city_name}... ({i+1}/{len(cities_dict)})")

        lat, lon = coords
        df_timeseries = extract_pm_timeseries(ds, lat, lon, pm25_var, pm10_var)

        if not df_timeseries.empty:
            df_forecast = predict_future_values(df_timeseries, days=5, max_date=end_date)
            forecast_only = df_forecast[df_forecast['type'] == 'forecast']

            if not forecast_only.empty:
                max_day_idx = forecast_only['aqi'].idxmax()
                cities_results.append({
                    'cidade': city_name,
                    'pm25_max': forecast_only['pm25'].max(),
                    'pm10_max': forecast_only['pm10'].max(),
                    'aqi_max': forecast_only['aqi'].max(),
                    'data_max': forecast_only.loc[max_day_idx, 'time'],
                    'categoria': forecast_only.loc[max_day_idx, 'aqi_category']
                })

    progress_bar.empty()
    status_text.empty()

    if cities_results:
        df_results = pd.DataFrame(cities_results).sort_values('aqi_max', ascending=False).reset_index(drop=True)
        df_results['pm25_max'] = df_results['pm25_max'].round(1)
        df_results['pm10_max'] = df_results['pm10_max'].round(1)
        df_results['aqi_max'] = df_results['aqi_max'].round(0)
        df_results['data_max'] = df_results['data_max'].dt.strftime('%d/%m/%Y %H:%M')
        return df_results
    else:
        return pd.DataFrame(columns=['cidade', 'pm25_max', 'pm10_max', 'aqi_max', 'data_max', 'categoria'])


def get_aqi_color(aqi_value):
    if aqi_value <= 50:    return '#00e400'
    elif aqi_value <= 100: return '#ffff00'
    elif aqi_value <= 150: return '#ff7e00'
    elif aqi_value <= 200: return '#ff0000'
    elif aqi_value <= 300: return '#8f3f97'
    else:                  return '#7e0023'

def style_aqi_table(df):
    def apply_styles(row):
        aqi = row['IQA Máx']
        bg_color = get_aqi_color(aqi)
        text_color = 'white' if aqi > 100 else 'black'
        return [f'background-color: {bg_color}; color: {text_color}'] * len(row)
    return df.style.apply(apply_styles, axis=1)


st.set_page_config(layout="wide", page_title="Monitor PM2.5/PM10 - MS", page_icon="🌍")

try:
    ads_url = st.secrets["ads"]["url"]
    ads_key = st.secrets["ads"]["key"]
    client = cdsapi.Client(url=ads_url, key=ads_key)
except Exception as e:
    st.error("Erro ao carregar as credenciais do CDS API. Verifique seu secrets.toml.")
    st.stop()


@st.cache_data
def load_ms_municipalities():
    try:
        url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/" + \
              "municipio_2022/UFs/MS/MS_Municipios_2022.zip"
        try:
            gdf = gpd.read_file(url)
            if 'NM_MUN' not in gdf.columns and 'NM_MUNICIP' in gdf.columns:
                gdf['NM_MUN'] = gdf['NM_MUNICIP']
            elif 'NM_MUN' not in gdf.columns and 'NOME' in gdf.columns:
                gdf['NM_MUN'] = gdf['NOME']
            return gdf
        except Exception as e:
            st.warning(f"Erro ao carregar shapefile oficial do IBGE: {e}")
            return create_fallback_shapefile()
    except Exception as e:
        st.warning(f"Não foi possível carregar os shapes dos municípios: {str(e)}")
        return create_fallback_shapefile()


cities = {
    "Água Clara": [-20.4453, -52.8792], "Alcinópolis": [-18.3255, -53.7042],
    "Amambai": [-23.1058, -55.2253], "Anastácio": [-20.4823, -55.8104],
    "Anaurilândia": [-22.1852, -52.7191], "Angélica": [-22.1527, -53.7708],
    "Antônio João": [-22.1927, -55.9511], "Aparecida do Taboado": [-20.0873, -51.0961],
    "Aquidauana": [-20.4697, -55.7868], "Aral Moreira": [-22.9384, -55.6331],
    "Bandeirantes": [-19.9279, -54.3581], "Bataguassu": [-21.7156, -52.4233],
    "Batayporã": [-22.2947, -53.2705], "Bela Vista": [-22.1073, -56.5263],
    "Bodoquena": [-20.5372, -56.7138], "Bonito": [-21.1261, -56.4836],
    "Brasilândia": [-21.2544, -52.0382], "Caarapó": [-22.6368, -54.8209],
    "Camapuã": [-19.5302, -54.0431], "Campo Grande": [-20.4697, -54.6201],
    "Caracol": [-22.0112, -57.0278], "Cassilândia": [-19.1179, -51.7308],
    "Chapadão do Sul": [-18.7908, -52.6260], "Corguinho": [-19.8243, -54.8281],
    "Coronel Sapucaia": [-23.2724, -55.5278], "Corumbá": [-19.0082, -57.651],
    "Costa Rica": [-18.5432, -53.1287], "Coxim": [-18.5013, -54.7603],
    "Deodápolis": [-22.2789, -54.1583], "Dois Irmãos do Buriti": [-20.6845, -55.2915],
    "Douradina": [-22.0430, -54.6158], "Dourados": [-22.2231, -54.812],
    "Eldorado": [-23.7868, -54.2836], "Fátima do Sul": [-22.3789, -54.5131],
    "Figueirão": [-18.6782, -53.6380], "Glória de Dourados": [-22.4136, -54.2336],
    "Guia Lopes da Laguna": [-21.4583, -56.1117], "Iguatemi": [-23.6835, -54.5635],
    "Inocência": [-19.7276, -51.9281], "Itaporã": [-22.0750, -54.7933],
    "Itaquiraí": [-23.4779, -54.1873], "Ivinhema": [-22.3046, -53.8185],
    "Japorã": [-23.8903, -54.4059], "Jaraguari": [-20.1386, -54.3996],
    "Jardim": [-21.4799, -56.1489], "Jateí": [-22.4806, -54.3078],
    "Juti": [-22.8596, -54.6060], "Ladário": [-19.0090, -57.5973],
    "Laguna Carapã": [-22.5448, -55.1502], "Maracaju": [-21.6105, -55.1695],
    "Miranda": [-20.2407, -56.3780], "Mundo Novo": [-23.9355, -54.2807],
    "Naviraí": [-23.0618, -54.1995], "Nioaque": [-21.1419, -55.8296],
    "Nova Alvorada do Sul": [-21.4657, -54.3825], "Nova Andradina": [-22.2332, -53.3437],
    "Novo Horizonte do Sul": [-22.6693, -53.8601], "Paraíso das Águas": [-19.0218, -53.0116],
    "Paranaíba": [-19.6746, -51.1909], "Paranhos": [-23.8905, -55.4289],
    "Pedro Gomes": [-18.0996, -54.5507], "Ponta Porã": [-22.5334, -55.7271],
    "Porto Murtinho": [-21.6981, -57.8825], "Ribas do Rio Pardo": [-20.4432, -53.7588],
    "Rio Brilhante": [-21.8033, -54.5427], "Rio Negro": [-19.4473, -54.9859],
    "Rio Verde de Mato Grosso": [-18.9249, -54.8434], "Rochedo": [-19.9566, -54.8940],
    "Santa Rita do Pardo": [-21.3016, -52.8333], "São Gabriel do Oeste": [-19.3950, -54.5507],
    "Selvíria": [-20.3637, -51.4192], "Sete Quedas": [-23.9710, -55.0396],
    "Sidrolândia": [-20.9302, -54.9692], "Sonora": [-17.5698, -54.7551],
    "Tacuru": [-23.6361, -55.0141], "Taquarussu": [-22.4898, -53.3519],
    "Terenos": [-20.4378, -54.8647], "Três Lagoas": [-20.7849, -51.7005],
    "Vicentina": [-22.4098, -54.4415]
}

def create_fallback_shapefile():
    from shapely.geometry import Polygon
    municipalities_data = []
    for city_name, (lat, lon) in cities.items():
        buffer_size = 0.15
        polygon = Polygon([
            (lon - buffer_size, lat - buffer_size), (lon + buffer_size, lat - buffer_size),
            (lon + buffer_size, lat + buffer_size), (lon - buffer_size, lat + buffer_size),
            (lon - buffer_size, lat - buffer_size)
        ])
        municipalities_data.append({'NM_MUN': city_name, 'geometry': polygon})
    return gpd.GeoDataFrame(municipalities_data, crs="EPSG:4326")


st.title("🌍 Monitoramento PM2.5 e PM10 - Mato Grosso do Sul")
st.markdown("""
### Sistema Integrado de Monitoramento da Qualidade do Ar

Este aplicativo apresenta as previsões de concentrações de Material Particulado (PM2.5 e PM10) 
para todos os municípios de Mato Grosso do Sul usando dados do modelo CAMS.

**Características desta versão:**
- Previsões de PM2.5 e PM10 do modelo CAMS (Copernicus)
- Visualização centralizada no município selecionado com contornos municipais
- Animações temporais para PM2.5 e PM10
- Índice de Qualidade do Ar (IQA) calculado
- Previsões limitadas à data final selecionada
""")


def generate_pm_analysis():
    dataset = "cams-global-atmospheric-composition-forecasts"

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    hours = []
    current_hour = start_hour
    while True:
        hours.append(f"{current_hour:02d}:00")
        if current_hour == end_hour:
            break
        current_hour = (current_hour + 3) % 24
        if current_hour == start_hour:
            break

    if not hours:
        hours = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']

    city_bounds = {'north': -17.0, 'south': -24.5, 'east': -50.5, 'west': -58.5}

    request = {
        'variable': ['particulate_matter_2.5um', 'particulate_matter_10um'],
        'date': f'{start_date_str}/{end_date_str}',
        'time': hours,
        'leadtime_hour': ['0', '24', '48', '72', '96', '120'],
        'type': ['forecast'],
        'format': 'netcdf',
        'area': [city_bounds['north'], city_bounds['west'], city_bounds['south'], city_bounds['east']]
    }

    filename = f'PM25_PM10_{city}_{start_date}_to_{end_date}.nc'

    try:
        with st.spinner('Baixando previsões de PM2.5 e PM10 do CAMS...'):
            client.retrieve(dataset, request).download(filename)

        ds = xr.open_dataset(filename)
        variable_names = list(ds.data_vars)
        pm25_var = next((var for var in variable_names if 'pm2p5' in var.lower() or '2.5' in var), None)
        pm10_var = next((var for var in variable_names if 'pm10' in var.lower() or '10um' in var), None)

        if not pm25_var or not pm10_var:
            st.error("Variáveis de PM2.5 ou PM10 não encontradas nos dados.")
            st.write("Variáveis disponíveis:", variable_names)
            return None

        with st.spinner("Extraindo previsões de PM para o município..."):
            df_timeseries = extract_pm_timeseries(ds, lat_center, lon_center, pm25_var, pm10_var)

        if df_timeseries.empty:
            st.error("Não foi possível extrair série temporal para este local.")
            return None

        with st.spinner("Gerando previsões estendidas..."):
            df_forecast = predict_future_values(df_timeseries, days=5, max_date=end_date)

        with st.spinner('Criando animação de PM2.5...'):
            animation_result = create_pm_animation(ds, pm25_var, city, lat_center, lon_center, ms_shapes, start_date, "PM2.5")
            if animation_result is None:
                return None
            fig_pm25, animate_pm25, frames_pm25 = animation_result
            ani_pm25 = animation.FuncAnimation(fig_pm25, animate_pm25, frames=frames_pm25, interval=animation_speed, blit=True)
            gif_filename_pm25 = f'PM25_{city}_{start_date}_to_{end_date}.gif'
            ani_pm25.save(gif_filename_pm25, writer=animation.PillowWriter(fps=2))
            plt.close(fig_pm25)

        gif_filename_pm10 = None
        if show_pm10_animation:
            with st.spinner('Criando animação de PM10...'):
                animation_result_pm10 = create_pm_animation(ds, pm10_var, city, lat_center, lon_center, ms_shapes, start_date, "PM10")
                if animation_result_pm10 is not None:
                    fig_pm10, animate_pm10, frames_pm10 = animation_result_pm10
                    ani_pm10 = animation.FuncAnimation(fig_pm10, animate_pm10, frames=frames_pm10, interval=animation_speed, blit=True)
                    gif_filename_pm10 = f'PM10_{city}_{start_date}_to_{end_date}.gif'
                    ani_pm10.save(gif_filename_pm10, writer=animation.PillowWriter(fps=2))
                    plt.close(fig_pm10)

        top_pollution_cities = None
        try:
            if ds[pm25_var].max().values < 1e-6:
                ds[pm25_var] = ds[pm25_var] * 1e9; ds[pm10_var] = ds[pm10_var] * 1e9
            elif ds[pm25_var].max().values < 1e-3:
                ds[pm25_var] = ds[pm25_var] * 1e6; ds[pm10_var] = ds[pm10_var] * 1e6
            elif ds[pm25_var].max().values > 1000:
                ds[pm25_var] = ds[pm25_var] / 1000; ds[pm10_var] = ds[pm10_var] / 1000

            with st.spinner("Analisando qualidade do ar em todos os municípios de MS..."):
                top_pollution_cities = analyze_all_cities(ds, pm25_var, pm10_var, cities, end_date=end_date)
        except Exception as e:
            st.warning(f"Não foi possível analisar todas as cidades: {str(e)}")
            top_pollution_cities = pd.DataFrame(columns=['cidade', 'pm25_max', 'pm10_max', 'aqi_max', 'data_max', 'categoria'])

        # Envio silencioso do relatório por e-mail (sem exibir status ao usuário)
        try:
            enviar_relatorio_email(
                cidade=city,
                df_timeseries=df_timeseries,
                df_forecast=df_forecast,
                top_pollution_df=top_pollution_cities,
                start_date=start_date,
                end_date=end_date,
                gif_pm25_path=gif_filename_pm25,
                gif_pm10_path=gif_filename_pm10
            )
        except Exception:
            pass

        return {
            'animation_pm25': gif_filename_pm25,
            'animation_pm10': gif_filename_pm10,
            'timeseries': df_timeseries,
            'forecast': df_forecast,
            'dataset': ds,
            'pm25_var': pm25_var,
            'pm10_var': pm10_var,
            'top_pollution': top_pollution_cities
        }

    except Exception as e:
        st.error(f"Erro ao processar os dados: {str(e)}")
        st.write("Detalhes da requisição:")
        st.write(request)
        return None
    finally:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except:
                pass


with st.spinner("Carregando shapes dos municípios..."):
    ms_shapes = load_ms_municipalities()

st.sidebar.header("Configurações")

available_cities = sorted(list(set(ms_shapes['NM_MUN'].tolist()).intersection(set(cities.keys()))))
if not available_cities:
    available_cities = list(cities.keys())

default_city_index = available_cities.index("Campo Grande") if "Campo Grande" in available_cities else 0
city = st.sidebar.selectbox("Selecione o município para análise detalhada", available_cities, index=default_city_index)
lat_center, lon_center = cities[city]

st.sidebar.subheader("Período de Análise")
start_date = st.sidebar.date_input("Data de Início", datetime.today() - timedelta(days=2))
end_date = st.sidebar.date_input("Data Final", datetime.today() + timedelta(days=5))

all_hours = list(range(0, 24, 3))
start_hour = st.sidebar.selectbox("Horário Inicial", all_hours, format_func=lambda x: f"{x:02d}:00")
end_hour = st.sidebar.selectbox("Horário Final", all_hours, index=len(all_hours)-1, format_func=lambda x: f"{x:02d}:00")

st.sidebar.subheader("Opções Avançadas")
with st.sidebar.expander("Configurações da Visualização"):
    animation_speed = st.slider("Velocidade da Animação (ms)", 200, 1000, 500)
    show_pm10_animation = st.checkbox("Gerar animação também para PM10", value=True)

st.sidebar.info("Previsões CAMS\nEste sistema utiliza previsões de PM2.5 e PM10 do modelo CAMS, com contornos municipais destacados.")

st.markdown("### Iniciar Análise Completa")
st.markdown(f"Clique no botão abaixo para gerar análise de PM2.5 e PM10 centralizada em **{city}** com contornos municipais.")

if st.button("Gerar Análise de Qualidade do Ar", type="primary", use_container_width=True):
    try:
        results = generate_pm_analysis()

        if results:
            tab1, tab2, tab3 = st.tabs([
                "Análise do Município",
                "Alerta de Qualidade do Ar",
                f"Animações - {city}"
            ])

            with tab3:
                st.subheader(f"Animações Temporais - {city}")

                st.markdown("### Evolução Temporal - PM2.5 (Previsto CAMS)")
                if os.path.exists(results['animation_pm25']):
                    st.image(results['animation_pm25'],
                             caption=f"Previsão temporal do PM2.5 em {city} com contornos municipais destacados ({start_date} a {end_date})")
                    with open(results['animation_pm25'], "rb") as file:
                        st.download_button(
                            label="Baixar Animação PM2.5 (GIF)",
                            data=file,
                            file_name=f"PM25_{city}_{start_date}_to_{end_date}.gif",
                            mime="image/gif"
                        )

                if results['animation_pm10'] and os.path.exists(results['animation_pm10']):
                    st.markdown("### Evolução Temporal - PM10 (Previsto CAMS)")
                    st.image(results['animation_pm10'],
                             caption=f"Previsão temporal do PM10 em {city} com contornos municipais destacados ({start_date} a {end_date})")
                    with open(results['animation_pm10'], "rb") as file:
                        st.download_button(
                            label="Baixar Animação PM10 (GIF)",
                            data=file,
                            file_name=f"PM10_{city}_{start_date}_to_{end_date}.gif",
                            mime="image/gif"
                        )

                st.info(f"""
                **Como interpretar as animações de {city}:**

                **Elementos do Mapa:**
                - Ponto vermelho: Localização exata de {city}
                - Contorno vermelho espesso: Limites do município de {city}
                - Linhas pretas finas: Contornos de todos os municípios de MS
                - Branco: concentração abaixo de 1 μg/m³ (limiar mínimo)
                - Cores: Intensidade das previsões de material particulado (CAMS)

                **Escala de Cores PM2.5 (Amarelo-Laranja-Vermelho):**
                - Branco: < 1 μg/m³ (abaixo do limiar)
                - Amarelo claro: 1–12 μg/m³ (Boa qualidade)
                - Amarelo/Laranja: 12–25 μg/m³ (Moderada)
                - Laranja: 25–35 μg/m³ (Limite OMS excedido)
                - Vermelho: > 35 μg/m³ (Insalubre)

                **Escala de Cores PM10 (Tons de Laranja):**
                - Branco: < 1 μg/m³ (abaixo do limiar)
                - Laranja claro: 1–25 μg/m³ (Boa qualidade)
                - Laranja médio: 25–50 μg/m³ (Moderada)
                - Laranja escuro: 50–150 μg/m³ (Limite OMS excedido)
                - Marrom: > 150 μg/m³ (Insalubre)
                """)

            with tab1:
                st.subheader(f"Análise Detalhada - {city}")

                col1, col2 = st.columns([3, 2])

                with col1:
                    df_combined = results['forecast']

                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

                    hist_data = df_combined[df_combined['type'] == 'historical']

                    # PM2.5 — apenas dados previstos pelo CAMS (sem linha de "previsto futuro")
                    if not hist_data.empty:
                        ax1.plot(hist_data['time'], hist_data['pm25'], 'o-', color='darkblue',
                                 label='PM2.5 Previsto (CAMS)', markersize=6)
                    ax1.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Limite OMS (25 μg/m³)')
                    ax1.axhline(y=35, color='red', linestyle='--', alpha=0.7, label='Limite EPA (35 μg/m³)')
                    ax1.set_ylabel('PM2.5 (μg/m³)', fontsize=12)
                    ax1.legend(); ax1.grid(True, alpha=0.3)
                    ax1.set_title('Material Particulado PM2.5 — Previsto CAMS', fontsize=14)

                    # PM10 — apenas dados previstos pelo CAMS
                    if not hist_data.empty:
                        ax2.plot(hist_data['time'], hist_data['pm10'], 'o-', color='brown',
                                 label='PM10 Previsto (CAMS)', markersize=6)
                    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Limite OMS (50 μg/m³)')
                    ax2.axhline(y=150, color='red', linestyle='--', alpha=0.7, label='Limite EPA (150 μg/m³)')
                    ax2.set_ylabel('PM10 (μg/m³)', fontsize=12)
                    ax2.legend(); ax2.grid(True, alpha=0.3)
                    ax2.set_title('Material Particulado PM10 — Previsto CAMS', fontsize=14)

                    # IQA — apenas dados previstos pelo CAMS
                    if not hist_data.empty:
                        ax3.plot(hist_data['time'], hist_data['aqi'], 'o-', color='purple',
                                 label='IQA Previsto (CAMS)', markersize=6)
                    ax3.axhspan(0, 50, alpha=0.2, color='green', label='Boa')
                    ax3.axhspan(51, 100, alpha=0.2, color='yellow', label='Moderada')
                    ax3.axhspan(101, 150, alpha=0.2, color='orange', label='Insalubre p/ Sensíveis')
                    ax3.axhspan(151, 200, alpha=0.2, color='red', label='Insalubre')
                    ax3.set_ylabel('IQA', fontsize=12)
                    ax3.set_xlabel('Data/Hora', fontsize=12)
                    ax3.legend(); ax3.grid(True, alpha=0.3)
                    ax3.set_title('Índice de Qualidade do Ar — Previsto CAMS', fontsize=14)
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

                with col2:
                    st.subheader("Estatísticas das Previsões")

                    if not hist_data.empty:
                        curr_pm25 = hist_data['pm25'].iloc[-1]
                        curr_pm10 = hist_data['pm10'].iloc[-1]
                        curr_aqi = hist_data['aqi'].iloc[-1]
                        curr_category = hist_data['aqi_category'].iloc[-1]
                        curr_color = hist_data['aqi_color'].iloc[-1]

                        col_a, col_b = st.columns(2)
                        col_a.metric("PM2.5 Previsto", f"{curr_pm25:.1f} μg/m³")
                        col_b.metric("PM10 Previsto", f"{curr_pm10:.1f} μg/m³")
                        st.metric("IQA", f"{curr_aqi:.0f}")

                        st.markdown(f"""
                        <div style="padding:15px; border-radius:10px; background-color:{curr_color}; 
                        color:white; text-align:center; margin:10px 0;">
                        <h3 style="margin:0;">Qualidade do Ar</h3>
                        <h2 style="margin:5px 0;">{curr_category}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                        st.subheader("Recomendações")
                        if curr_aqi <= 50:
                            st.success("Condições ideais para atividades ao ar livre")
                        elif curr_aqi <= 100:
                            st.info("Pessoas sensíveis devem considerar limitar esforços prolongados")
                        elif curr_aqi <= 150:
                            st.warning("Grupos sensíveis devem evitar esforços ao ar livre")
                        elif curr_aqi <= 200:
                            st.error("Evite esforços prolongados ao ar livre")
                        else:
                            st.error("Evite todas as atividades ao ar livre")

                        st.subheader("Comparação com Padrões OMS")
                        st.progress(min(curr_pm25 / 25, 1.0))
                        st.caption(f"PM2.5: {curr_pm25:.1f}/25 μg/m³ (Limite OMS 24h)")
                        st.progress(min(curr_pm10 / 50, 1.0))
                        st.caption(f"PM10: {curr_pm10:.1f}/50 μg/m³ (Limite OMS 24h)")

                    st.subheader("Exportar Dados")
                    csv = df_combined.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Baixar Dados Completos (CSV)",
                        data=csv,
                        file_name=f"PM_data_{city}_{start_date}_to_{end_date}.csv",
                        mime="text/csv",
                    )

            with tab2:
                st.subheader("Alerta de Qualidade do Ar - Mato Grosso do Sul")

                if 'top_pollution' in results and not results['top_pollution'].empty:
                    top_cities = results['top_pollution'].head(20)

                    critical_cities = top_cities[top_cities['aqi_max'] > 100]
                    very_critical = top_cities[top_cities['aqi_max'] > 150]

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Cidades em Alerta", len(critical_cities))
                    col2.metric("Condição Insalubre", len(very_critical))
                    col3.metric("IQA Máximo Previsto", f"{top_cities['aqi_max'].max():.0f}")

                    if len(critical_cities) > 0 and len(top_cities) >= 3:
                        st.error(f"""
                        ### ALERTA DE QUALIDADE DO AR

                        **{len(critical_cities)} municípios** com previsão de qualidade do ar 
                        inadequada até {end_date.strftime('%d/%m/%Y')}!

                        Municípios mais críticos:
                        1. **{top_cities.iloc[0]['cidade']}**: IQA {top_cities.iloc[0]['aqi_max']:.0f} - PM2.5: {top_cities.iloc[0]['pm25_max']:.1f} μg/m³
                        2. **{top_cities.iloc[1]['cidade']}**: IQA {top_cities.iloc[1]['aqi_max']:.0f} - PM2.5: {top_cities.iloc[1]['pm25_max']:.1f} μg/m³
                        3. **{top_cities.iloc[2]['cidade']}**: IQA {top_cities.iloc[2]['aqi_max']:.0f} - PM2.5: {top_cities.iloc[2]['pm25_max']:.1f} μg/m³
                        """)

                    st.markdown("### Ranking de Qualidade do Ar por Município (Previsões CAMS)")
                    top_cities_display = top_cities.rename(columns={
                        'cidade': 'Município', 'pm25_max': 'PM2.5 Máx (μg/m³)',
                        'pm10_max': 'PM10 Máx (μg/m³)', 'aqi_max': 'IQA Máx',
                        'data_max': 'Data Crítica', 'categoria': 'Categoria'
                    })
                    st.dataframe(style_aqi_table(top_cities_display), use_container_width=True)

                    st.subheader("Material Particulado Previsto — 10 Municípios Mais Críticos")
                    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
                    top10 = top_cities.head(10)
                    x_pos = np.arange(len(top10))
                    width = 0.35

                    bars1 = ax.bar(x_pos - width/2, top10['pm25_max'], width, color='darkblue', alpha=0.8, label='PM2.5 Previsto')
                    bars2 = ax.bar(x_pos + width/2, top10['pm10_max'], width, color='brown', alpha=0.8, label='PM10 Previsto')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(top10['cidade'], rotation=45, ha='right')
                    ax.set_ylabel('Concentração (μg/m³)', fontsize=12)
                    ax.set_title('PM2.5 e PM10 Máximos Previstos até ' + end_date.strftime('%d/%m/%Y') + ' (CAMS)', fontsize=14)
                    ax.axhline(y=25, color='orange', linestyle='--', alpha=0.7, label='Limite PM2.5 OMS (25 μg/m³)')
                    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Limite PM10 OMS (50 μg/m³)')
                    ax.legend(); ax.grid(True, alpha=0.3)

                    for bar, val in zip(bars1, top10['pm25_max']):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}', ha='center', va='bottom', fontsize=9)
                    for bar, val in zip(bars2, top10['pm10_max']):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}', ha='center', va='bottom', fontsize=9)

                    plt.tight_layout()
                    st.pyplot(fig)

                    csv_alert = top_cities.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Baixar Dados de Alerta (CSV)",
                        data=csv_alert,
                        file_name=f"Alerta_Qualidade_Ar_MS_{start_date}_to_{end_date}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("Dados de análise estadual não disponíveis. Mostrando apenas análise local.")

    except Exception as e:
        st.error(f"Erro durante a análise: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


st.markdown("---")
st.markdown("""
### Informações Importantes

**Sobre os Dados:**
- As previsões de PM2.5/PM10 são geradas pelo modelo CAMS (Copernicus Atmosphere Monitoring Service)
- Dados validados continuamente com estações de monitoramento globais
- Resolução espacial: ~0.4° × 0.4° (aprox. 44 km) · Resolução temporal: 3 horas

**Novidades desta versão:**
- Contornos municipais destacados nas animações
- Município selecionado evidenciado em vermelho
- Animações separadas para PM2.5 e PM10
- Previsões limitadas à data final escolhida pelo usuário
- Regiões com concentração < 1 μg/m³ exibidas em branco nos mapas

**Dados Fornecidos por:**
- CAMS (Copernicus Atmosphere Monitoring Service) — União Europeia

**Desenvolvido para:** Monitoramento da Qualidade do Ar em Mato Grosso do Sul
""")

with st.expander("Suporte e Informações Técnicas"):
    st.markdown(f"""
    ### Suporte Técnico

    **Parâmetros do Sistema:**
    - Resolução espacial: ~0.4° x 0.4° (aprox. 44 km)
    - Resolução temporal: 3 horas
    - Previsão: Até a data final selecionada pelo usuário
    - Variáveis principais: PM2.5 e PM10 (previsões CAMS)
    - Contornos: Municípios de MS com destaque do selecionado

    **Vantagens das Previsões CAMS:**
    - Calibração contínua com estações de superfície globais
    - Validação internacional rigorosa
    - Cobertura espacial completa para toda a região

    **Para Melhor Precisão:**
    - Use dados de múltiplos pontos temporais
    - Considere condições meteorológicas locais
    - Valide com medições locais quando disponível
    - Monitore tendências de longo prazo
    """)
