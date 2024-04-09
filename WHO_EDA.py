import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import branca
import branca.colormap as cm
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from streamlit_extras.add_vertical_space import add_vertical_space
import requests
import branca.colormap as cmp

geo_json_url = 'https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/world-countries.json'
response = requests.get(geo_json_url)
geo_json_data = response.json()


st.set_page_config(layout="wide", page_title="WHO EDA", page_icon="🌍")
grey = '#333333'
# Stworzenie mapy kolorystycznej
red = '#EE0000'
orange = '#EE7700'
yellow = '#EEEE00'
celadon = '#77EE00'
green = '#00EE00'
cmap_colors = [green, celadon, yellow, orange, red]
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)

@st.cache_data
def load_data():
    return pd.read_excel('D:/PUM/WHO_EDA/who_data_filled.xlsx')


def check_data_empty(data):
    if len(data) == 0:
        return True
    elif data['Latitude'].isna().sum() > 0 or data['Longitude'].isna().sum() > 0:
        return True
    else:
        return False

@st.cache_data
def map_cities(data, selected_pollutant):
    pol_min, pol_max = data[selected_pollutant].min(), data[selected_pollutant].max()
    if check_data_empty(data):
        st.warning("No measurements meet the selected criteria.", icon="⚠️")
        return folium.Map(location=[0,0 ], zoom_start=1.5, tiles="cartodbpositron")
    folium_map = folium.Map(location=[data['Latitude'].mean(), data['Latitude'].mean()], zoom_start=1.5,
                            tiles="cartodbpositron")
    for index, row in data.iterrows():
        country = row['WHO Country Name']
        city = row['City or Locality']
        year = row['Measurement Year']
        pol = row[selected_pollutant]
        location = row['Latitude'], row['Longitude']
        normalized_pol = (pol - pol_min) / (pol_max - pol_min)
        fill_color = mcolors.rgb2hex(custom_cmap(normalized_pol))
        popup_html = f"""
            <font size="4"><b>{city}</b>, <b>{country}</b><br> </font> <font size="3"> {selected_pollutant.split(" ")[0]}: </font> 
            <font color={fill_color} size="3">{pol}</font> <font size="3"><br>Year: {year}</font>
            """
        tooltip_html = f"""
                    <font size="2"> <b>{city}</b>, <b>{country}</b>: </font> 
                    <font color={fill_color} size="2">{pol}</font>
                    """
        folium.CircleMarker(location=location,
                            radius=4,
                            fill_color=fill_color,
                            color=grey,
                            stroke=True,
                            fill=True,
                            weight=1,
                            fill_opacity=0.8,
                            opacity=1,
                            tooltip=tooltip_html,
                            popup=popup_html).add_to(folium_map)
    if len(data) == 1:
        st.warning("There's only one measurement that meets the selected criteria.", icon="⚠️")
        return folium_map
    colormap = cm.LinearColormap(cmap_colors, vmin=pol_min, vmax=pol_max)
    colormap.caption = f'Measurements of {selected_pollutant}'
    colormap.add_to(folium_map)
    return folium_map


def map_countries(data, selected_pollutant):
    pol_min, pol_max = data[selected_pollutant].min(), data[selected_pollutant].max()
    map_data = data.copy()[['ISO3', selected_pollutant]]
    if check_data_empty(data):
        st.warning("No measurements meet the selected criteria.", icon="⚠️")
        return folium.Map(location=[0,0 ], zoom_start=1.5, tiles="cartodbpositron")
    folium_map = folium.Map(location=[data['Latitude'].mean(), data['Latitude'].mean()], zoom_start=1.5,
                            tiles="cartodbpositron")

    linear = cm.LinearColormap(cmap_colors, vmin=pol_min, vmax=pol_max)

    for feature in geo_json_data['features']:
        iso3 = feature['id']
        if iso3 in map_data['ISO3'].values:
            value = map_data.loc[map_data['ISO3'] == iso3, selected_pollutant].iloc[0]
            country = data.loc[data['ISO3'] == iso3, 'WHO Country Name'].iloc[0]
            fill_color = linear(value)
            popup_html = f"""
                <font size="4"><b>{country}</b><br> </font> <font size="3"> {selected_pollutant.split(" ")[0]}: </font> 
                <font color={mcolors.rgb2hex(fill_color)} size="3">{value:.2f}</font> <font size="3"></font>
                """

            tooltip_html = f"""
                        <font size="2"><b>{country}</b>: </font> 
                        <font color={mcolors.rgb2hex(fill_color)} size="2">{value:.2f}</font>
                        """
        else:
            country = feature['properties']['name']
            fill_color = '#808080'  # Szary kolor
            value = "No data"
            popup_html = f"""
                <font size="4"><b>{country}</b><br> </font> <font size="3"> {selected_pollutant.split(" ")[0]}: </font> 
                <font color={fill_color} size="3">{value}</font> <font size="3"></font>
                """

            tooltip_html = f"""
                        <font size="2"><b>{country}</b>: </font> 
                        <font color={fill_color} size="2">{value}</font>
                        """


        folium.GeoJson(
            feature,
            style_function=lambda x, fill_color=fill_color: {
                'fillColor': fill_color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7,
            },
            tooltip = tooltip_html,
            popup = folium.Popup(popup_html),
        ).add_to(folium_map)
    if len(map_data) == 1:
        st.warning("There's only one measurement that meets the selected criteria.", icon="⚠️")
        return folium_map
    linear.caption = f'Measurements of {selected_pollutant}'
    linear.add_to(folium_map)
    return folium_map


def map_regions(data, selected_pollutant):
    pol_min, pol_max = data[selected_pollutant].min(), data[selected_pollutant].max()
    map_data = data.copy()[['ISO3', selected_pollutant]]
    if check_data_empty(data):
        st.warning("No measurements meet the selected criteria.", icon="⚠️")
        return folium.Map(location=[0,0 ], zoom_start=1.5, tiles="cartodbpositron")
    folium_map = folium.Map(location=[data['Latitude'].mean(), data['Latitude'].mean()], zoom_start=1.5,
                            tiles="cartodbpositron")

    linear = cm.LinearColormap(cmap_colors, vmin=pol_min, vmax=pol_max)

    for feature in geo_json_data['features']:
        iso3 = feature['id']
        if iso3 in map_data['ISO3'].values:
            value = map_data.loc[map_data['ISO3'] == iso3, selected_pollutant].iloc[0]
            country = data.loc[data['ISO3'] == iso3, 'WHO Country Name'].iloc[0]
            region = data.loc[data['ISO3'] == iso3, 'WHO Region'].iloc[0]
            fill_color = linear(value)
            popup_html = f"""
                <font size="3">{country},</font><br> <font size="3"><b>{region}</b><br></font> <font size="3"> {selected_pollutant.split(" ")[0]}: </font> 
                <font color={mcolors.rgb2hex(fill_color)} size="3">{value:.2f}</font> <font size="3"></font>
                """

            tooltip_html = f"""
                        <font size="2"><b>{region}</b>: </font> 
                        <font color={mcolors.rgb2hex(fill_color)} size="2">{value:.2f}</font>
                        """
        else:
            country = feature['properties']['name']
            fill_color = '#808080'
            value = "No data"
            popup_html = f"""
                <font size="3"><b>{country}</b>,</font><br> </font> <font size="3"> {selected_pollutant.split(" ")[0]}: </font> 
                <font color={fill_color} size="3">{value}</font> <font size="3"></font>
                """

            tooltip_html = f"""
                        <font size="2"><b>{country}</b>: </font> 
                        <font color={fill_color} size="2">{value}</font>
                        """

        folium.GeoJson(
            feature,
            style_function=lambda x, fill_color=fill_color: {
                'fillColor': fill_color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7,
            },
            tooltip = tooltip_html,
            popup = folium.Popup(popup_html),
        ).add_to(folium_map)
    if len(map_data) == 1:
        st.warning("There's only one measurement that meets the selected criteria.", icon="⚠️")
    linear.caption = f'Measurements of {selected_pollutant}'
    linear.add_to(folium_map)
    return folium_map


def aggregate_by_region(data):
    columns_to_keep = ['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)', 'PM25 temporal coverage (%)',
                       'PM10 temporal coverage (%)', 'NO2 temporal coverage (%)', 'Latitude', 'Longitude']

    aggregated_data = data.groupby('WHO Region')[columns_to_keep].agg({
        'PM2.5 (μg/m3)': 'mean',
        'PM10 (μg/m3)': 'mean',
        'NO2 (μg/m3)': 'mean',
        'PM25 temporal coverage (%)': 'mean',
        'PM10 temporal coverage (%)': 'mean',
        'NO2 temporal coverage (%)': 'mean',
        'Latitude': 'mean',
        'Longitude': 'mean'
    }).reset_index()

    aggregated_data.columns = [col + '_aggregated' if col != 'WHO Region' else col for col in aggregated_data.columns]
    merged_data = pd.merge(data, aggregated_data, on='WHO Region', how='left')

    merged_data['PM2.5 (μg/m3)'] = merged_data['PM2.5 (μg/m3)_aggregated']
    merged_data['PM10 (μg/m3)'] = merged_data['PM10 (μg/m3)_aggregated']
    merged_data['NO2 (μg/m3)'] = merged_data['NO2 (μg/m3)_aggregated']
    merged_data['PM25 temporal coverage (%)'] = merged_data['PM25 temporal coverage (%)_aggregated']
    merged_data['PM10 temporal coverage (%)'] = merged_data['PM10 temporal coverage (%)_aggregated']
    merged_data['NO2 temporal coverage (%)'] = merged_data['NO2 temporal coverage (%)_aggregated']
    merged_data['Latitude'] = merged_data['Latitude_aggregated']
    merged_data['Longitude'] = merged_data['Longitude_aggregated']

    merged_data = merged_data.drop(columns=[col for col in merged_data.columns if col.endswith('_aggregated')])
    merged_data = aggregate_by_country(merged_data)
    return merged_data


def filter_form_regions(data):
    with st.form(f"filter_form_regions"):
        st.header(f"Filter data")
        # Wybór roku
        year_selection = ['Latest data from all locations'] + sorted(data['Measurement Year'].unique(), reverse=True)
        selected_year = st.selectbox("Measurement Year:", year_selection, key="year_regions",
                                     help="Choose a year of measurements.")
        # Wybór zmiennej
        selected_pollutant = st.selectbox("Pollutant Selection:", ["PM2.5 (μg/m3)", "PM10 (μg/m3)", "NO2 (μg/m3)"],
                                          key="variable_regions",
                                          help="Choose a pollutant variable to display on the map.")
        sorting_method = "Lowest pollution"
        st.form_submit_button('Confirm')
    filters = {
        "Measurement Year": selected_year,
    }
    filtered_data = data.copy()
    filtered_data = filtered_data[filtered_data[selected_pollutant].isna() == False]
    for column, value in filters.items():
        if value == 'Latest data from all locations':
            filtered_data = filtered_data.sort_values(by='Measurement Year', ascending=False)
            filtered_data = filtered_data.groupby(
                ['WHO Country Name', 'City or Locality', 'Latitude', 'Longitude']).first().reset_index()
        else:
            filtered_data = filtered_data[filtered_data[column] == value]
    aggregated_data = filtered_data.copy()
    aggregated_data = aggregate_by_country(aggregated_data)
    filtered_data = aggregate_by_region(filtered_data)
    if sorting_method == "Lowest pollution":
        filtered_data = filtered_data.sort_values(by=selected_pollutant, ascending=True)
        aggregated_data = aggregated_data.sort_values(by=selected_pollutant, ascending=True)
    return filtered_data, selected_pollutant, aggregated_data


def aggregate_by_country(data):
    columns_to_keep = ['WHO Region', 'ISO3', 'WHO Country Name', 'PM2.5 (μg/m3)', 'PM10 (μg/m3)',
                       'NO2 (μg/m3)', 'PM25 temporal coverage (%)', 'PM10 temporal coverage (%)',
                       'NO2 temporal coverage (%)', 'Latitude', 'Longitude']

    aggregated_data = data.groupby('WHO Country Name')[columns_to_keep].agg({
        'WHO Region': 'first',
        'ISO3': 'first',
        'PM2.5 (μg/m3)': 'mean',
        'PM10 (μg/m3)': 'mean',
        'NO2 (μg/m3)': 'mean',
        'PM25 temporal coverage (%)': 'mean',
        'PM10 temporal coverage (%)': 'mean',
        'NO2 temporal coverage (%)': 'mean',
        'Latitude': 'mean',
        'Longitude': 'mean'
    }).reset_index()
    return aggregated_data


def filter_form_countries(data):
    with st.form(f"filter_form_countries"):
        st.header(f"Filter data")
        # Wybór roku
        year_selection = ['Latest data from all locations'] + sorted(data['Measurement Year'].unique(), reverse=True)
        selected_year = st.selectbox("Measurement Year:", year_selection, key="year_countries",
                                     help="Choose a year of measurements.")
        # Wybór regionu
        region_selection = ['All regions'] + sorted(list(data['WHO Region'].unique()))
        selected_region = st.selectbox("Region:", region_selection, key="region_countries",
                                       help="Choose a region to filter locations.")
        # Wybór zmiennej
        selected_pollutant = st.selectbox("Pollutant Selection:", ["PM2.5 (μg/m3)", "PM10 (μg/m3)", "NO2 (μg/m3)"],
                                          key="variable_countries",
                                          help="Choose a pollutant variable to display on the map.")
        sorting_method = st.selectbox("Sorting Method", ["Highest pollution", "Lowest pollution"],
                                      key="sorting_method_countries")
        st.form_submit_button('Confirm')
    filters = {
        "Measurement Year": selected_year,
        "WHO Region": selected_region,
    }
    filtered_data = data.copy()
    filtered_data = filtered_data[filtered_data[selected_pollutant].isna() == False]
    for column, value in filters.items():
        if value == 'Latest data from all locations':
            filtered_data = filtered_data.sort_values(by='Measurement Year', ascending=False)
            filtered_data = filtered_data.groupby(
                ['WHO Country Name', 'City or Locality', 'Latitude', 'Longitude']).first().reset_index()
        elif value == 'All regions':
            filtered_data = filtered_data
        else:
            filtered_data = filtered_data[filtered_data[column] == value]
    aggregated_data = filtered_data.copy()
    filtered_data = aggregate_by_country(filtered_data)
    if sorting_method == "Highest pollution":
        filtered_data = filtered_data.sort_values(by=selected_pollutant, ascending=False)
        aggregated_data = aggregated_data.sort_values(by=selected_pollutant, ascending=False)
    elif sorting_method == "Lowest pollution":
        filtered_data = filtered_data.sort_values(by=selected_pollutant, ascending=True)
        aggregated_data = aggregated_data.sort_values(by=selected_pollutant, ascending=True)
    return filtered_data, selected_pollutant, aggregated_data


def filter_form_cities(data):
    with st.form(f"filter_form_cities"):
        st.header(f"Filter data")
        # Wybór roku
        year_selection = ['Latest data from all locations'] + sorted(data['Measurement Year'].unique(), reverse=True)
        selected_year = st.selectbox("Measurement Year:", year_selection, key="year_cities",
                                     help="Choose a year of measurements.")
        # Wybór regionu
        region_selection = ['All regions'] +  sorted(list(data['WHO Region'].unique()))
        selected_region = st.selectbox("Region:", region_selection, key="region_cities",
                                     help="Choose a region to filter locations.")

        # Wybór kraju
        country_selection = ['All countries'] + sorted(list(data['WHO Country Name'].unique()))
        selected_country = st.selectbox("Country:", country_selection, key="country_cities",
                                        help="Choose a country to filter locations.")
        # Wybór zmiennej
        selected_pollutant = st.selectbox("Pollutant Selection:", ["PM2.5 (μg/m3)", "PM10 (μg/m3)", "NO2 (μg/m3)"],
                                          key="variable_cities",
                                          help="Choose a pollutant variable to display on the map.")
        number_of_locations = st.number_input("Max. Number of Locations to Display", min_value=1, max_value=6905, value=30, key="number_of_locations_cities")
        sorting_method = st.selectbox("Sorting Method", ["Highest pollution", "Lowest pollution"], key="sorting_method_cities")
        st.form_submit_button('Confirm')
    filters = {
        "Measurement Year": selected_year,
        "WHO Region": selected_region,
        "WHO Country Name": selected_country,
    }
    filtered_data = data.copy()
    filtered_data = filtered_data[filtered_data[selected_pollutant].isna() == False]
    for column, value in filters.items():
        if value == 'Latest data from all locations':
            filtered_data = filtered_data.sort_values(by='Measurement Year', ascending=False)
            filtered_data = filtered_data.groupby(['WHO Country Name', 'City or Locality', 'Latitude', 'Longitude']).first().reset_index()
        elif value == 'All regions' or value == 'All countries':
            filtered_data = filtered_data
        else:
            filtered_data = filtered_data[filtered_data[column] == value]
    if sorting_method == "Highest pollution":
        filtered_data = filtered_data.sort_values(by=selected_pollutant, ascending=False)
    elif sorting_method == "Lowest pollution":
        filtered_data = filtered_data.sort_values(by=selected_pollutant, ascending=True)
    filtered_data = filtered_data.head(number_of_locations)
    return filtered_data, selected_pollutant


# Funkcja do wyświetlania podstawowych statystyk opisowych
def show_basic_stats(data):
    st.subheader("Basic Statistics")
    df = data.copy()
    df = df[['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)', 'PM25 temporal coverage (%)', 'PM10 temporal coverage (%)', 'NO2 temporal coverage (%)']]
    with st.container():
        st.dataframe(df.describe(), use_container_width=True)


# Wyświetlanie macierzy korelacji zmiennych
def show_correlation_matrix(data):
    plt.figure(figsize=(10, 8))
    df = data.copy()
    df = df[['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)', 'PM25 temporal coverage (%)', 'PM10 temporal coverage (%)',
             'NO2 temporal coverage (%)']]
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap=custom_cmap, fmt=".2f", linewidths=0.5,
                xticklabels=['PM2.5', 'PM10', 'NO2', 'PM25 TC', 'PM10 TC', 'NO2 TC'],
                yticklabels=['PM2.5', 'PM10', 'NO2', 'PM25 TC', 'PM10 TC', 'NO2 TC'])
    plt.title('Correlation Matrix of Pollutants and Temporal Coverage')
    st.pyplot(plt)


def show_pairplot(data):
    sns.set_palette(cmap_colors)
    df = data.copy()
    df = df[['PM2.5 (μg/m3)', 'PM10 (μg/m3)', 'NO2 (μg/m3)', 'PM25 temporal coverage (%)', 'PM10 temporal coverage (%)',
             'NO2 temporal coverage (%)']]
    pairplot = sns.pairplot(df)
    pairplot.fig.suptitle('Pairplot of Pollutants and Temporal Coverage', fontsize=16)
    pairplot.fig.subplots_adjust(top=0.95)
    temp_file = "pairplot.png"
    pairplot.savefig(temp_file)
    st.image(temp_file)


def show_bar_chart_cities(data, selected_pollutant):
    show_data = data.copy()
    if len(data) > 30:
        show_data = show_data.head(30)
    selected_pollutant_values = show_data[selected_pollutant]
    norm = Normalize(vmin=selected_pollutant_values.min(), vmax=selected_pollutant_values.max())
    plt.figure(figsize=(12, 6))
    colors = custom_cmap(norm(selected_pollutant_values))
    plt.bar(show_data['City or Locality']+ ', ' + show_data['ISO3'], selected_pollutant_values, color=colors)
    plt.xlabel('City', fontweight='bold')
    plt.ylabel('Pollution Level', fontweight='bold')
    plt.title(f'Pollution Levels for Each City ({selected_pollutant})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)


def show_bar_chart_countries(data, selected_pollutant):
    show_data = data.copy()
    if len(data) > 30:
        show_data = show_data.head(30)
    selected_pollutant_values = show_data[selected_pollutant]
    norm = Normalize(vmin=selected_pollutant_values.min(), vmax=selected_pollutant_values.max())
    plt.figure(figsize=(12, 6))
    colors = custom_cmap(norm(selected_pollutant_values))
    plt.bar(show_data['WHO Country Name']+ ' (' + show_data['ISO3'] + ')', selected_pollutant_values, color=colors)
    plt.xlabel('Country', fontweight='bold')
    plt.ylabel('Pollution Level', fontweight='bold')
    plt.title(f'Pollution Levels for Each Country ({selected_pollutant})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)


def show_bar_chart_regions(data, selected_pollutant):
    show_data = data.copy()
    selected_pollutant_values = show_data[selected_pollutant]
    norm = Normalize(vmin=selected_pollutant_values.min(), vmax=selected_pollutant_values.max())
    plt.figure(figsize=(12, 6))
    colors = custom_cmap(norm(selected_pollutant_values))
    plt.bar(show_data['WHO Region'], selected_pollutant_values, color=colors)
    plt.xlabel('Region', fontweight='bold')
    plt.ylabel('Pollution Level', fontweight='bold')
    plt.title(f'Pollution Levels for Each Region ({selected_pollutant})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)


def show_boxplot_countries(data, data2, selected_pollutant):
    show_data = data.copy()
    if len(data['WHO Country Name'].unique()) > 25:
        top_25_countries = data['WHO Country Name'].unique()[:25]
        show_data = show_data[show_data['WHO Country Name'].isin(top_25_countries)]
    sns.set_palette(cmap_colors)
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=show_data, x='ISO3', y=selected_pollutant)
    plt.title(f'Distribution of {selected_pollutant} Pollution Across the Countries')
    plt.xlabel('WHO Country Name', fontweight='bold')
    plt.ylabel(selected_pollutant, fontweight='bold')
    plt.xticks(rotation=45)
    st.pyplot(plt)


def show_boxplot_regions(data, selected_pollutant):
    show_data = data.copy()
    sns.set_palette(cmap_colors)
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=show_data, x='WHO Region', y=selected_pollutant)
    plt.title(f'Distribution of {selected_pollutant} Pollution Across WHO Regions')
    plt.xlabel('Region WHO', fontweight='bold')
    plt.ylabel(selected_pollutant, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)


def show_plots_cities(filtered_data, selected_pollutant):
    if not check_data_empty(filtered_data) and len(filtered_data) != 1:
        show_basic_stats(filtered_data)
        if len(filtered_data['City or Locality'].unique()) > 1:
            columns2 = st.columns([1, 5, 1])
            with columns2[1]:
                add_vertical_space(1)
                show_bar_chart_cities(filtered_data, selected_pollutant)
            columns3 = st.columns([1, 2, 1])
            with columns3[1]:
                add_vertical_space(1)
                show_correlation_matrix(filtered_data)
            columns4 = st.columns([1, 5, 1])
            with columns4[1]:
                add_vertical_space(2)
                show_pairplot(filtered_data)


def show_plots_countries(filtered_data, selected_pollutant, aggregated_data):
    if not check_data_empty(filtered_data) and len(filtered_data) != 1:
        show_basic_stats(filtered_data)
        if len(filtered_data['WHO Country Name'].unique()) > 1:
            columns2 = st.columns([1, 5, 1])
            with columns2[1]:
                add_vertical_space(1)
                show_bar_chart_countries(filtered_data, selected_pollutant)
            columns3 = st.columns([1, 2, 1])
            with columns3[1]:
                add_vertical_space(1)
                show_correlation_matrix(filtered_data)
            columns4 = st.columns([1, 5, 1])
            with columns4[1]:
                add_vertical_space(2)
                show_pairplot(filtered_data)
                add_vertical_space(1)
                show_boxplot_countries(aggregated_data, filtered_data, selected_pollutant)


def show_plots_regions(filtered_data, selected_pollutant, aggregated_data):
    if not check_data_empty(filtered_data) and len(filtered_data) != 1:
        show_basic_stats(filtered_data)
        columns2 = st.columns([1, 5, 1])
        with columns2[1]:
            add_vertical_space(1)
            show_bar_chart_regions(filtered_data, selected_pollutant)
        columns3 = st.columns([1, 2, 1])
        with columns3[1]:
            add_vertical_space(1)
            show_correlation_matrix(filtered_data)
        columns4 = st.columns([1, 5, 1])
        with columns4[1]:
            add_vertical_space(2)
            show_boxplot_regions(aggregated_data, selected_pollutant)


def main():
    st.title("Aplikacja do Eksploracyjnej Analizy Danych (EDA) dla danych zanieczyszczenia powietrza WHO")
    # Wczytanie danych
    who_data = load_data()
    # Wyświetlanie mapy
    tab1, tab2, tab3 = st.tabs(["Regions", "Countries", "Cities"])

    with tab1:
        columns = st.columns([2, 5])
        with columns[0]:
            filtered_data, selected_pollutant, aggregated_data = filter_form_regions(who_data)
        with columns[1]:
            folium_map = map_regions(filtered_data, selected_pollutant)
            folium_static(folium_map, width=900, height=500)
        show_plots_regions(filtered_data, selected_pollutant, aggregated_data)

    with tab2:
        columns = st.columns([2, 5])
        with columns[0]:
            filtered_data, selected_pollutant, aggregated_data = filter_form_countries(who_data)
        with columns[1]:
            folium_map = map_countries(filtered_data, selected_pollutant)
            folium_static(folium_map, width=900, height=500)
        show_plots_countries(filtered_data, selected_pollutant, aggregated_data)

    with tab3:
        columns = st.columns([2, 5])
        with columns[0]:
            filtered_data, selected_pollutant = filter_form_cities(who_data)
        with columns[1]:
            folium_map = map_cities(filtered_data, selected_pollutant)
            folium_static(folium_map, width=900, height=500)
        show_plots_cities(filtered_data, selected_pollutant)





# Uruchomienie aplikacji
if __name__ == "__main__":
    main()