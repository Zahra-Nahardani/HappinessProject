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

# The number of missing values
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

    # shape of the DataFrame
    st.write('<b>Shape of Dataset:</b>', happiness_df_copy.shape, unsafe_allow_html=True)

    # the first and last 5 rows of the DataFrame
    option = st.selectbox('Select an option', ['First 5 rows of the dataset', 'Last 5 rows of the dataset'])
    if option == 'First 5 rows of the dataset':
        happiness_df_copy['year'] = happiness_df_copy['year'].astype(str)
        st.write(happiness_df_copy.head(5))
    elif option == 'Last 5 rows of the dataset':
        happiness_df_copy['year'] = happiness_df_copy['year'].astype(str)
        st.write(happiness_df_copy.tail(5))

    # descriptive statistics of the numerical columns
    st.subheader('Dataset Describtion')
    st.write(happiness_df_copy.describe(), unsafe_allow_html=True)
    # information about the DataFrame
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
    st.write('Healthy life expectancy at birth is the average number of years a newborn infant would live in good '
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

    avg_life_expectancy = happiness_df_copy.groupby('Country name')['Healthy life expectancy at birth'].mean().reset_index()
    avg_life_expectancy_sorted = avg_life_expectancy.sort_values(by='Healthy life expectancy at birth', ascending=False)
    top_5_countries = avg_life_expectancy_sorted.head(5)
    bottom_5_countries = avg_life_expectancy_sorted.tail(5)
    combined_countries = pd.concat([top_5_countries, bottom_5_countries])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Country name', y='Healthy life expectancy at birth', data=combined_countries, palette='viridis')
    ax.set_title('Top and Bottom 5 Countries by Healthy Life Expectancy at Birth', fontsize=16)
    ax.set_xlabel('Country')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel('Healthy Life Expectancy at Birth')
    st.pyplot(fig)

    # Distribution of Healthy life expectancy
    st.subheader('Box Plot')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(y=happiness_df_copy['Healthy life expectancy at birth'], ax=ax)
    ax.set_title('Box Plot of Healthy Life Expectancy at Birth', fontsize=16)
    ax.set_xlabel('Healthy Life Expectancy at Birth')
    st.pyplot(fig)

    st.write('In this plot, there are several outliers below the lower whisker. So, as the number of outliers (22) is relatively'
             ' low compared to the total volume of the dataset (2239), it can be reasonable to remove them. As outliers can skew'
             ' statistical measures and effect on analysis of data.')

# Remove outliers
Q1 = happiness_df_copy['Healthy life expectancy at birth'].quantile(0.25)
Q3 = happiness_df_copy['Healthy life expectancy at birth'].quantile(0.75)
IQR = Q3 - Q1
# lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = happiness_df_copy[(happiness_df_copy['Healthy life expectancy at birth'] < lower_bound) |
                             (happiness_df_copy['Healthy life expectancy at birth'] > upper_bound)]
num_outliers = outliers.shape[0]
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
    fig.suptitle('Distribution of Ladder Score and Logged GDP per Capita', fontsize=20)
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
    combined_df_happiness = pd.concat([happiest_countries, unhappiest_countries])

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Life Ladder', y='Country name', data=combined_df_happiness, palette='plasma', ax=ax)
    ax.set_title('Happiest and Unhappiest Countries', fontsize=16)
    st.pyplot(fig)

    # Happiness trend for Unhappiest country
    st.subheader('Happiness trend for Unhappiest Country Over Years ')
    unhappiest_country = avg_life_ladder.sort_values(by='Life Ladder', ascending=True).iloc[0]['Country name']
    unhappiest_country_data = happiness_df_copy[happiness_df_copy['Country name'] == unhappiest_country]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(x='year', y='Life Ladder', data=unhappiest_country_data, ax=ax, color='red')
    ax.set_title(f'Life Ladder Trend for {unhappiest_country}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Life Ladder Score')
    st.pyplot(fig)

    # Generosity
    st.subheader('Most Generous and Most Ungenerous Countries')
    average_generosity = happiness_df_copy.groupby('Country name')['Generosity'].mean().reset_index()
    sorted_generosity = average_generosity.sort_values(by='Generosity', ascending=False)
    top_5_countries = sorted_generosity.head(5)
    bottom_5_countries = sorted_generosity.tail(5)
    top_bottom_generous_countries = pd.concat([top_5_countries, bottom_5_countries])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=top_bottom_generous_countries, x='Country name', y='Generosity', palette='spring')
    ax.set_title('Top 5 and Bottom 5 Countries by Average Generosity', fontsize=16)
    ax.set_xlabel('Country')
    ax.set_ylabel('Average Generosity')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Heatmap
    st.subheader('Heatmap')
    happiness_df_copy_drop_country_name = happiness_df_copy.drop(columns=['Country name'])
    corr_matrix = happiness_df_copy_drop_country_name.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True, ax=ax)
    ax.set_title('Correlation Heatmap of Features', fontsize=16)
    st.pyplot(fig)

########################################## Machine Learning

####### Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

if selected == 'Machine Learning':
    st.subheader('Model 1: Linear Regression')
    features = happiness_df_copy[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                                  'Freedom to make life choices', 'Generosity', 'Positive affect']]
    target = happiness_df_copy['Life Ladder']

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=60)

    model_linear_Reg = LinearRegression()
    model_linear_Reg.fit(x_train, y_train)

    y_pred_train = model_linear_Reg.predict(x_train)
    y_pred_test = model_linear_Reg.predict(x_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    st.write(f"**Root Mean Squared Error for Test (RMSE):** {test_rmse:.2f}")
    st.write(f"**R-Squared (R²) Score for Test:** {test_r2:.2f}")

    st.write(f"**Root Mean Squared Error for Train (RMSE):** {train_rmse:.2f}")
    st.write(f"**R-Squared (R²) Score for Train:** {train_r2:.2f}")

    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred_test, ax=ax, color='orange', label='Predicted vs Actual')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal fit')
    ax.set_xlabel('Actual Life Ladder')
    ax.set_ylabel('Predicted Life Ladder')
    ax.set_title('Actual vs Predicted Life Ladder (Linear Regression)')
    ax.legend()
    st.pyplot(fig)

    ####### Random Forest Regression
    st.subheader('Model 2: Random Forest Regression')

    from sklearn.ensemble import RandomForestRegressor

    model_random_forest = RandomForestRegressor(random_state=1)
    model_random_forest.fit(x_train, y_train)

    y_predict_rf = model_random_forest.predict(x_test)

    mean_sqe_rf = mean_squared_error(y_test, y_predict_rf)
    r_mean_sqe_rf = np.sqrt(mean_squared_error(y_test, y_predict_rf))
    r_squared_rf = r2_score(y_test, y_predict_rf)

    st.markdown(f"**Mean Squared Error (MSE):** {mean_sqe_rf:.2f}")
    st.markdown(f"**Root Mean Squared Error (RMSE):** {r_mean_sqe_rf:.2f}")
    st.markdown(f"**R-squared (R²) Score:** {r_squared_rf:.2f}")

    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_predict_rf, ax=ax, color='red', label='Predicted vs Actual')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal fit')
    ax.set_xlabel('Actual Life Ladder')
    ax.set_ylabel('Predicted Life Ladder')
    ax.set_title('Actual vs Predicted Life Ladder (Random Forest)')
    ax.legend()
    st.pyplot(fig)

    st.write("The comparison of models indicates that the Random Forest Regression significantly outperforms Linear Regression."
             " The Random Forest model achieved a lower Root Mean Squared Error (RMSE) of 0.41 and a higher R-squared (R²) score "
             "of 0.86 on the test set, compared to the Linear Regression model's RMSE of 0.56 and R² score of 0.74. These results"
             " suggest that the Random Forest model provides more accurate predictions and explains a greater proportion of variance"
             " in the data, making it the preferred choice for our analysis.")


