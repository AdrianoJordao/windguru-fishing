"""
    Translations (Portuguese / English) and small locale-aware formatters.
    Every key must exist in both languages (checked by tests/test_i18n.py).
"""

from __future__ import annotations

from datetime import datetime

LANGS = {"pt": "🇵🇹 PT", "en": "🇬🇧 EN"}
DEFAULT_LANG = "pt"

WEEKDAYS = {
    "pt": ("Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"),
    "en": ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
}

STRINGS = {
    "pt": {
        "page_title": "Previsão de Pesca Windguru",
        "title": "🎣 Previsão de Pesca Windguru",
        "subtitle": "Encontre os melhores dias para pescar com base nas condições meteorológicas",
        "lang_label": "Idioma",

        # Sidebar
        "sidebar_header": "⚙️ Configuração",
        "location": "📍 Local",
        "spot_name": "Nome do local",
        "spot_name_help": "Nome do seu local de pesca",
        "url": "URL do Windguru",
        "url_help": "URL completo do windguru.cz (ex.: https://www.windguru.cz/65000)",
        "tide_loc": "🌊 Local da maré (opcional)",
        "tide_loc_caption": "Deixe vazio para usar as coordenadas do local.",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "min": "Mín.",
        "max": "Máx.",
        "submit": "🎣 Ver Previsão",
        "group.waves": "🌊 Ondulação",
        "group.wind": "💨 Vento",
        "group.weather": "🌧️ Tempo",

        # Variables
        "var.wave_height_m.label": "Altura das ondas",
        "var.wave_height_m.short": "altura",
        "var.wave_height_m.help": "",
        "var.wave_period_s.label": "Período das ondas",
        "var.wave_period_s.short": "período",
        "var.wave_period_s.help": "",
        "var.wind_speed_kmh.label": "Velocidade do vento",
        "var.wind_speed_kmh.short": "vento",
        "var.wind_speed_kmh.help": "",
        "var.wind_gust_kmh.label": "Rajadas",
        "var.wind_gust_kmh.short": "rajadas",
        "var.wind_gust_kmh.help": "",
        "var.precipitation_mm.label": "Precipitação",
        "var.precipitation_mm.short": "chuva",
        "var.precipitation_mm.help": "Chuva por hora. Normalmente 0-2 mm/h é bom para pescar",
        "var.temperature_c.label": "Temperatura",
        "var.temperature_c.short": "temp.",
        "var.temperature_c.help": "",
        "var.cloud_cover_pct.label": "Nebulosidade",
        "var.cloud_cover_pct.short": "nuvens",
        "var.cloud_cover_pct.help": "Algumas nuvens podem melhorar a pesca",

        # Messages
        "warn_min_max": "⚠️ O mínimo é maior que o máximo em: {names}",
        "err_spot_id": "❌ Não foi possível encontrar o número do local Windguru em '{url}'.",
        "spinner": "🌊 A obter a previsão…",
        "err_fetch": "❌ Não foi possível obter a previsão. Tente novamente mais tarde.",
        "err_no_data": "❌ Sem dados de previsão. Verifique o URL e tente novamente.",
        "info_no_coords": "ℹ️ Este local não tem coordenadas, por isso não há marés. "
                          "Indique a latitude/longitude em *Local da maré* na barra lateral.",
        "warn_bad_coords": "⚠️ A latitude/longitude da maré não são números válidos. Foram ignoradas.",
        "warn_tides": "⚠️ Os dados de maré não estão disponíveis de momento.",
        "best_windows": "**Melhores períodos:** ",
        "no_windows": "Nenhum período de 2+ horas cumpre todos os critérios. Veja as horas amarelas no gráfico.",
        "caption_colors": "🟢 todas as condições dentro dos limites · 🟡 uma condição ligeiramente fora · "
                          "🔴 não recomendado. Clique num modelo na legenda para o esconder.",
        "caption_tides": "Marés: modelo marinho Open-Meteo em {lat:.3f}, {lon:.3f} (coordenadas {source}). "
                         "Horas aproximadas (±30 min); não usar para navegação.",
        "source.manual": "manuais",
        "source.windguru": "do Windguru",
        "source.builtin": "predefinidas",
        "compact_view": "📱 Vista compacta",
        "compact_help": "Gráfico de um dia de cada vez, mais fácil de ler no telemóvel",
        "show_models": "Mostrar modelos",
        "pick_day": "Dia",
        "week": "📅 Resumo da semana",
        "col.good_hours": "🟢 Horas boas",
        "col.best": "Melhor período",
        "all_day": "dia todo",
        "col.good_short": "🟢 h",
        "col.best_short": "Melhor",
        "col.tides": "Marés",
        "col.high": "Preia-mar",
        "col.low": "Baixa-mar",
        "caption_colors_compact": "🟢 todas as condições dentro dos limites · 🟡 uma condição ligeiramente fora · "
                                  "🔴 não recomendado. Linha preta: média dos modelos.",
        "download": "📥 Descarregar gráfico (HTML interativo)",
        "good_hours": "📋 Boas horas de pesca ({n})",
        "tide_table": "🌊 Tabela de marés",

        # Table columns and values
        "col.local_time": "Hora local",
        "col.status": "Estado",
        "col.day": "Dia",
        "col.time": "Hora",
        "col.tide": "Maré",
        "col.height": "Altura (cm)",
        "status.green": "🟢 Bom",
        "status.yellow": "🟡 Aceitável",
        "tide.High": "Preia-mar",
        "tide.Low": "Baixa-mar",

        # Chart
        "chart.title": "Previsão Windguru – {name}",
        "chart.mean": "Média (todos os modelos)",
        "chart.mean_short": "Média",
        "chart.tide_panel": "Maré (cm)",
        "chart.tide": "Maré",
        "chart.no_tide": "Sem dados de maré",

        # Welcome screen
        "welcome_info": "👈 Configure as condições de pesca na barra lateral e clique em '🎣 Ver Previsão' para começar!",
        "welcome_md": """
### Como usar:
1. **Indique o URL do Windguru** - Procure o seu local em windguru.cz e copie o URL
2. **Defina as condições ideais** - Escolha os limites aceitáveis de ondulação, vento e tempo
3. **Clique em Ver Previsão** - Obtenha uma previsão interativa com os períodos bons/maus a cores e as marés

### Cores:
- 🟢 **Verde**: Todas as condições dentro dos limites - perfeito para pescar!
- 🟡 **Amarelo**: Uma condição ligeiramente fora - ainda aceitável
- 🔴 **Vermelho**: Várias condições fora dos limites - não recomendado

### Variáveis meteorológicas:
- 🌧️ **Precipitação**: Chuva em mm por hora (normalmente 0-2 mm/h)
- 🌡️ **Temperatura**: Temperatura do ar em °C
- ☁️ **Nebulosidade**: Percentagem (algumas nuvens podem melhorar a pesca!)
- 🌊 **Marés**: Preia-mar e baixa-mar do local

### Locais populares:
- Sesimbra: `https://www.windguru.cz/65000`
- Cascais: `https://www.windguru.cz/574`
- Costa da Caparica: `https://www.windguru.cz/48963`
""",
        "footer": "Feito com ❤️ para pescadores por Adriano Jordão | Dados do [Windguru](https://www.windguru.cz) "
                  "· Marés do [Open-Meteo](https://open-meteo.com)",
    },

    "en": {
        "page_title": "Windguru Fishing Forecast",
        "title": "🎣 Windguru Fishing Forecast",
        "subtitle": "Find the best days for fishing based on weather conditions",
        "lang_label": "Language",

        # Sidebar
        "sidebar_header": "⚙️ Configuration",
        "location": "📍 Location",
        "spot_name": "Spot Name",
        "spot_name_help": "Name for your fishing spot",
        "url": "Windguru URL",
        "url_help": "Full URL from windguru.cz (e.g., https://www.windguru.cz/65000)",
        "tide_loc": "🌊 Tide location (optional)",
        "tide_loc_caption": "Leave empty to use the spot's coordinates.",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "min": "Min",
        "max": "Max",
        "submit": "🎣 Check Forecast",
        "group.waves": "🌊 Wave Conditions",
        "group.wind": "💨 Wind Conditions",
        "group.weather": "🌧️ Weather Conditions",

        # Variables
        "var.wave_height_m.label": "Wave Height",
        "var.wave_height_m.short": "Height",
        "var.wave_height_m.help": "",
        "var.wave_period_s.label": "Wave Period",
        "var.wave_period_s.short": "Period",
        "var.wave_period_s.help": "",
        "var.wind_speed_kmh.label": "Wind Speed",
        "var.wind_speed_kmh.short": "Speed",
        "var.wind_speed_kmh.help": "",
        "var.wind_gust_kmh.label": "Wind Gusts",
        "var.wind_gust_kmh.short": "Gust",
        "var.wind_gust_kmh.help": "",
        "var.precipitation_mm.label": "Precipitation",
        "var.precipitation_mm.short": "Rain",
        "var.precipitation_mm.help": "Rain per hour. Typically want 0-2 mm/h for good fishing",
        "var.temperature_c.label": "Temperature",
        "var.temperature_c.short": "Temp",
        "var.temperature_c.help": "",
        "var.cloud_cover_pct.label": "Cloud Cover",
        "var.cloud_cover_pct.short": "Clouds",
        "var.cloud_cover_pct.help": "Some clouds can improve fishing",

        # Messages
        "warn_min_max": "⚠️ Min is larger than max for: {names}",
        "err_spot_id": "❌ Could not find a Windguru spot number in '{url}'.",
        "spinner": "🌊 Fetching forecast…",
        "err_fetch": "❌ Could not fetch the forecast. Please try again later.",
        "err_no_data": "❌ No forecast data retrieved. Please check the URL and try again.",
        "info_no_coords": "ℹ️ No coordinates for this spot, so no tides. "
                          "Add a latitude/longitude under *Tide location* in the sidebar.",
        "warn_bad_coords": "⚠️ Tide latitude/longitude are not valid numbers. Ignoring them.",
        "warn_tides": "⚠️ Tide data is unavailable right now.",
        "best_windows": "**Best windows:** ",
        "no_windows": "No stretch of 2+ hours matches all your criteria. Check the yellow hours in the chart.",
        "caption_colors": "🟢 all conditions in range · 🟡 one condition slightly out · "
                          "🔴 not recommended. Click a model in the legend to hide it.",
        "caption_tides": "Tides: Open-Meteo marine model at {lat:.3f}, {lon:.3f} ({source} coordinates). "
                         "Times are approximate (±30 min); not for navigation.",
        "source.manual": "manual",
        "source.windguru": "Windguru",
        "source.builtin": "built-in",
        "compact_view": "📱 Compact view",
        "compact_help": "One day at a time, easier to read on a phone",
        "show_models": "Show models",
        "pick_day": "Day",
        "week": "📅 Week at a glance",
        "col.good_hours": "🟢 Good hours",
        "col.best": "Best stretch",
        "all_day": "all day",
        "col.good_short": "🟢 h",
        "col.best_short": "Best",
        "col.tides": "Tides",
        "col.high": "High tide",
        "col.low": "Low tide",
        "caption_colors_compact": "🟢 all conditions in range · 🟡 one condition slightly out · "
                                  "🔴 not recommended. Black line: mean of the models.",
        "download": "📥 Download Chart (interactive HTML)",
        "good_hours": "📋 Good Fishing Hours ({n})",
        "tide_table": "🌊 Tide Table",

        # Table columns and values
        "col.local_time": "Local time",
        "col.status": "Status",
        "col.day": "Day",
        "col.time": "Time",
        "col.tide": "Tide",
        "col.height": "Height (cm)",
        "status.green": "🟢 Good",
        "status.yellow": "🟡 Fair",
        "tide.High": "High",
        "tide.Low": "Low",

        # Chart
        "chart.title": "Windguru Forecast – {name}",
        "chart.mean": "Mean (all models)",
        "chart.mean_short": "Mean",
        "chart.tide_panel": "Tide (cm)",
        "chart.tide": "Tide",
        "chart.no_tide": "No tide data",

        # Welcome screen
        "welcome_info": "👈 Configure your fishing conditions in the sidebar and click '🎣 Check Forecast' to get started!",
        "welcome_md": """
### How to use:
1. **Enter your Windguru URL** - Find your spot on windguru.cz and copy the URL
2. **Set your ideal conditions** - Define acceptable ranges for waves, wind, and weather
3. **Click Check Forecast** - Get an interactive forecast with color-coded good/bad periods and tides

### Color coding:
- 🟢 **Green**: All conditions within your ranges - perfect for fishing!
- 🟡 **Yellow**: One condition slightly outside range - still manageable
- 🔴 **Red**: Multiple conditions outside range - not recommended

### Weather variables:
- 🌧️ **Precipitation**: Rain in mm per hour (typically want 0-2 mm/h)
- 🌡️ **Temperature**: Air temperature in °C
- ☁️ **Cloud Cover**: Percentage (some clouds can improve fishing!)
- 🌊 **Tides**: High and low tides for the spot

### Popular spots:
- Sesimbra: `https://www.windguru.cz/65000`
- Cascais: `https://www.windguru.cz/574`
- Costa da Caparica: `https://www.windguru.cz/48963`
""",
        "footer": "Made with ❤️ for fishermen by Adriano Jordão | Data from [Windguru](https://www.windguru.cz) "
                  "· Tides from [Open-Meteo](https://open-meteo.com)",
    },
}


def t(key: str, lang: str = DEFAULT_LANG, **kwargs) -> str:
    """ Translated text; falls back to Portuguese, then to the key itself """
    text = STRINGS.get(lang, {}).get(key)
    if text is None:
        text = STRINGS[DEFAULT_LANG].get(key, key)
    return text.format(**kwargs) if kwargs else text


def fmt_day(dt: datetime, lang: str = DEFAULT_LANG) -> str:
    """ 'Ter 24/11' / 'Tue 24/11' without depending on the system locale """
    names = WEEKDAYS.get(lang, WEEKDAYS[DEFAULT_LANG])
    return f"{names[dt.weekday()]} {dt:%d/%m}"


def fmt_day_time(dt: datetime, lang: str = DEFAULT_LANG) -> str:
    return f"{fmt_day(dt, lang)} {dt:%H:%M}"
