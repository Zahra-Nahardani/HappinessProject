# import libraries
# streamlit run C:\Zahra\Uni_Verona\Programming\ProgrammingProject2\HappinessProject\HappinessReport.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        menu_title='Menu',
        options=['Home', 'Data Exploration', 'Data Analysis', 'Machine Learning'],
        icons=['house-door', 'table', 'pie-chart-fill', 'graph-up'],
        default_index=0
    )

if selected == 'Home':
    st.markdown("<h1 style='text-align: center;'>World Happiness Report</h1>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<h2 style='text-align: center;'>Programming for Data Science Final Project 2024</h2>",
                unsafe_allow_html=True)
    st.write("")
    st.write('<p style="text-align: center; font-size: 20px;">Zahra Nahardani</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 18px;"><a href="mailto:zahra.nahardani@studenti.univr.it">zahra.nahardani@studenti.univr.it</a></p>',
        unsafe_allow_html=True)

# Read the dataset
happiness_df = pd.read_csv('World-happiness-report-updated_2024.csv', encoding='ISO-8859-1', low_memory=False)
happiness_df_copy = happiness_df.copy()

# The number of missing values for each feature
missing_before_cleaning = happiness_df_copy.isnull().sum().reset_index()
missing_before_cleaning.columns = ['Feature', 'Missing Values']

# Remove missing values (with number of less than 60)
for feature in happiness_df_copy.columns:
    if happiness_df_copy[feature].isnull().sum() < 50:
        happiness_df_copy.dropna(subset=[feature], inplace=True)

# Replace missing values with mean
features_with_missing = ['Healthy life expectancy at birth', 'Perceptions of corruption']
for feature in features_with_missing:
    happiness_df_copy[feature].fillna(happiness_df_copy[feature].mean(), inplace=True)

# Number of missing values after cleaning data
missing_after_cleaning = happiness_df_copy.isnull().sum().reset_index()
missing_after_cleaning.columns = ['Feature', 'Missing Values']
########################################## Data Exploration

if selected == 'Data Exploration':
    st.subheader('About Dataset')
    st.write('The World Happiness Report is a significant survey that evaluates global happiness levels. Over the '
             'years, it has earned widespread recognition, prompting governments, organizations, and civil societies '
             'to integrate happiness metrics into their policy-making frameworks. Experts from diverse fields such as '
             'economics, psychology, survey analysis, national statistics, health, and public policy illustrate how '
             'well-being measures can be effectively utilized to gauge the progress of nations. The reports provide a '
             'comprehensive overview of global happiness today and explore how the emerging science of happiness accounts'
             ' for differences in happiness at both personal and national levels.')

    st.subheader('Display Dataset')

    # Show the shape of the DataFrame to get the number of rows and columns
    st.write('<b>Shape of Dataset:</b>', happiness_df_copy.shape, unsafe_allow_html=True)

    # Display the first and last 5 rows of the DataFrame
    option = st.selectbox('Select an option', ['First 5 rows of the dataset', 'Last 5 rows of the dataset'])
    if option == 'First 5 rows of the dataset':
        happiness_df_copy['year'] = happiness_df_copy['year'].astype(str)
        st.write(happiness_df_copy.head(5))
    elif option == 'Last 5 rows of the dataset':
        happiness_df_copy['year'] = happiness_df_copy['year'].astype(str)
        st.write(happiness_df_copy.tail(5))

    # Generate descriptive statistics of the numerical columns in the DataFrame.
    st.subheader('Dataset Describtion')
    st.write(happiness_df_copy.describe(), unsafe_allow_html=True)
    # Display information about the DataFrame
    happiness_df_copy.info()

    # Cleaning Data - Null Values Handling
    st.subheader('Data Cleaning')

    # display missing values before cleaning
    if st.button('Number of null values before cleaning'):
        st.table(missing_before_cleaning)

    # Number of missing values after cleaning data
    if st.button('Number of null values after cleaning'):
        st.table(missing_after_cleaning)

    # Average Healthy life expectancy at birth by country
    st.subheader('Average Healthy Life Expectancy at Birth by Country')
    st.write('Note: Healthy life expectancy at birth is the average number of years a newborn infant would live in good '
             'health, based on mortality rates and life expectancy at different ages.')
    countries = happiness_df_copy['Country name'].unique()
    selected_country = st.selectbox('Select a country', countries)
    if selected_country:
        avg_life_expectancy = happiness_df_copy[happiness_df_copy['Country name'] ==
                                                selected_country]['Healthy life expectancy at birth'].mean()
        st.write(f"The average Healthy life expectancy at birth in {selected_country} is {avg_life_expectancy:.2f} years.")

    # Healthy life expectancy at birth by country over the years
    if selected_country:
        life_expectancy_data = happiness_df_copy[happiness_df_copy['Country name'] == selected_country][
            ['year', 'Healthy life expectancy at birth']]
        st.write(life_expectancy_data)

########################################## Data Analysis

if selected == 'Data Analysis':

    # Bar plot for the top and bottom 5 countries by Healthy Life Expectancy
    st.subheader('Bar Chart')
    # Group by country
    avg_life_expectancy = happiness_df_copy.groupby('Country name')['Healthy life expectancy at birth'].mean().reset_index()
    avg_life_expectancy_sorted = avg_life_expectancy.sort_values(by='Healthy life expectancy at birth', ascending=False)
    top_5_countries = avg_life_expectancy_sorted.head(5)
    bottom_5_countries = avg_life_expectancy_sorted.tail(5)
    combined_countries = pd.concat([top_5_countries, bottom_5_countries])
    # Plotting
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Country name', y='Healthy life expectancy at birth', data=combined_countries, palette='viridis')
    plt.title('Top and Bottom 5 Countries by Healthy Life Expectancy at Birth', fontsize=16)
    plt.xlabel('Country')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Healthy Life Expectancy at Birth')
    st.pyplot()

    # Distribution of Healthy life expectancy
    st.subheader('Box Plot')
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(y=happiness_df_copy['Healthy life expectancy at birth'], ax=ax)
    ax.set_title('Box Plot of Healthy Life Expectancy at Birth', fontsize=16)
    ax.set_xlabel('Healthy Life Expectancy at Birth')
    st.pyplot(fig)

    st.write('In this plot, there are several outliers below the lower whisker. So, as the number of outliers (22) is '
             'relatively low compared to the total volume of the dataset (2239), it can be reasonable to remove them. '
             'As outliers can skew statistical measures and effect on analysis of data.')

# Remove outliers
# Calculate Q1, Q3, and IQR
Q1 = happiness_df_copy['Healthy life expectancy at birth'].quantile(0.25)
Q3 = happiness_df_copy['Healthy life expectancy at birth'].quantile(0.75)
IQR = Q3 - Q1
# lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = happiness_df_copy[(happiness_df_copy['Healthy life expectancy at birth'] < lower_bound) |
                             (happiness_df_copy['Healthy life expectancy at birth'] > upper_bound)]
num_outliers = outliers.shape[0]
# Filter data
filtered_happiness_df = happiness_df_copy[(happiness_df_copy['Healthy life expectancy at birth'] >= lower_bound) & (
        happiness_df_copy['Healthy life expectancy at birth'] <= upper_bound)]

if selected == 'Data Analysis':
    # Display outliers
    st.write('<b>Display outliers in Healthy Life Expectancy at Birth:</b>', unsafe_allow_html=True)
    st.write(outliers[['Country name', 'Healthy life expectancy at birth']])

    # Plotting the box plot again with filtered data
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(y=filtered_happiness_df['Healthy life expectancy at birth'], ax=ax, color='lightblue')
    ax.set_title('Box Plot of Healthy Life Expectancy at Birth (after removing outliers)', fontsize=16)
    ax.set_xlabel('Healthy Life Expectancy at Birth')
    st.pyplot(fig)

    # distribution of Ladder score and Logged GDP per capita
    st.subheader('Histogram')
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    sns.histplot(happiness_df_copy['Life Ladder'], kde=True, ax=axes[0], color='red')
    axes[0].set_title('Distribution of Life Ladder')
    axes[0].set_xlabel('Life Ladder')
    axes[0].set_ylabel('Frequency')

    sns.histplot(happiness_df_copy['Log GDP per capita'], kde=True, ax=axes[1], color='orange')
    axes[1].set_title('Distribution of Log GDP per Capita')
    axes[1].set_xlabel('Log GDP per Capita')
    axes[1].set_ylabel('Frequency')
    fig.suptitle('Distribution of Ladder Score and Logged GDP per Capita', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)
    st.write('In the histogram for the Life Ladder scores, the left tail of the distribution indicates that a few '
             'countries have very low scores. The presence of these low scores signifies that certain countries have'
             ' significantly lower happiness levels compared to the average.')

    # Happiest and Unhappiest Countries
    st.subheader('Happiest and Unhappiest Countries')
    avg_life_ladder = happiness_df_copy.groupby('Country name')['Life Ladder'].mean().reset_index()
    happiest_countries = avg_life_ladder.sort_values(by='Life Ladder', ascending=False).head(5)
    unhappiest_countries = avg_life_ladder.sort_values(by='Life Ladder', ascending=False).tail(5)
    combined_df = pd.concat([happiest_countries, unhappiest_countries])

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Life Ladder', y='Country name', data=combined_df, palette='plasma', ax=ax)
    ax.set_title('Happiest and Unhappiest Countries', fontsize=16)
    st.pyplot(fig)

    st.subheader('Happiness trend for Unhappiest Countries Over Years ')
    # Happiness trend for Unhappiest country
    unhappiest_country = avg_life_ladder.sort_values(by='Life Ladder', ascending=True).iloc[0]['Country name']
    unhappiest_country_data = happiness_df_copy[happiness_df_copy['Country name'] == unhappiest_country]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='year', y='Life Ladder', data=unhappiest_country_data, ax=ax, color='red')
    ax.set_title(f'Life Ladder Trend for {unhappiest_country}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Life Ladder Score')
    st.pyplot(fig)

    # Heatmap
    happiness_df_copy_drop_country_name = happiness_df_copy.drop(columns=['Country name'])
    corr_matrix = happiness_df_copy_drop_country_name.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True, ax=ax)
    ax.set_title('Correlation Heatmap of Features')
    st.pyplot(fig)
