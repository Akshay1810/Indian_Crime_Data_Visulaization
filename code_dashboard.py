import dash
import numpy as np
from dash import dcc, html, Input, Output, State, ClientsideFunction, no_update
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from rapidfuzz import process, fuzz
import os
from sklearn.metrics import silhouette_samples, silhouette_score
import warnings
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans

os.environ["LOKY_MAX_CPU_COUNT"] = "2"

# Enabled dynamic callback exceptions.
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=['assets/drawer_styles.css'], 
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    title='Watchtower' 
)
server = app.server


crime_options = {
    'ipc': [
        "MURDER", "ATTEMPT_TO_MURDER", "CULPABLE_HOMICIDE_NOT_AMOUNTING_TO_MURDER", "RAPE",
        "CUSTODIAL_RAPE", "OTHER_RAPE", "KIDNAPPING_AND_ABDUCTION", 
        "KIDNAPPING_AND_ABDUCTION_OF_WOMEN_AND_GIRLS", "KIDNAPPING_AND_ABDUCTION_OF_OTHERS",
        "DACOITY", "PREPARATION_AND_ASSEMBLY_FOR_DACOITY", "ROBBERY", "BURGLARY",
        "THEFT", "AUTO_THEFT", "OTHER_THEFT", "RIOTS", "CRIMINAL_BREACH_OF_TRUST",
        "CHEATING", "COUNTERFIETING", "ARSON", "HURT/GREVIOUS_HURT", "DOWRY_DEATHS",
        "ASSAULT_ON_WOMEN_WITH_INTENT_TO_OUTRAGE_HER_MODESTY", "INSULT_TO_MODESTY_OF_WOMEN",
        "CRUELTY_BY_HUSBAND_OR_HIS_RELATIVES", "IMPORTATION_OF_GIRLS_FROM_FOREIGN_COUNTRIES",
        "CAUSING_DEATH_BY_NEGLIGENCE", "OTHER_IPC_CRIMES"
    ],
    'sc': [
        "MURDER", "RAPE", "KIDNAPPING_AND_ABDUCTION", "DACOITY", "ROBBERY",
        "ARSON", "HURT", "PREVENTION_OF_ATROCITIES_(POA)_ACT",
        "PROTECTION_OF_CIVIL_RIGHTS_(PCR)_ACT", "OTHER_CRIMES_AGAINST_SCS"
    ],
    'st': [
        "MURDER", "RAPE", "KIDNAPPING_AND_ABDUCTION", 
        "DACOITY", "ROBBERY", "ARSON",
        "HURT", "PROTECTION_OF_CIVIL_RIGHTS_(PCR)_ACT",
        "PREVENTION_OF_ATROCITIES_(POA)_ACT", "OTHER_CRIMES_AGAINST_STS"
    ],
    'children': [
        "RAPE", "KIDNAPPING_AND_ABDUCTION", 
        "FOETICIDE", "ABETMENT_OF_SUICIDE",
        "EXPOSURE_AND_ABANDONMENT", "PROCURATION_OF_MINOR_GIRLS",
        "BUYING_OF_GIRLS_FOR_PROSTITUTION", "SELLING_OF_GIRLS_FOR_PROSTITUTION",
        "PROHIBITION_OF_CHILD_MARRIAGE_ACT", "OTHER_CRIMES"
    ],
    'women' : [
        "RAPE","KIDNAPPING_AND_ABDUCTION", "DOWRY_DEATHS",
        "ASSAULT_ON_WOMEN_WITH_INTENT_TO_OUTRAGE_HER_MODESTY","INSULT_TO_MODESTY_OF_WOMEN",
        "CRUELTY_BY_HUSBAND_OR_HIS_RELATIVES","IMPORTATION_OF_GIRLS"
    ]
}
crime_features = [ 
    "MURDER", "ATTEMPT_TO_MURDER", "CULPABLE_HOMICIDE_NOT_AMOUNTING_TO_MURDER", "RAPE",
        "CUSTODIAL_RAPE", "OTHER_RAPE", "KIDNAPPING_AND_ABDUCTION", 
        "KIDNAPPING_AND_ABDUCTION_OF_WOMEN_AND_GIRLS", "KIDNAPPING_AND_ABDUCTION_OF_OTHERS",
        "DACOITY", "PREPARATION_AND_ASSEMBLY_FOR_DACOITY", "ROBBERY", "BURGLARY",
        "THEFT", "AUTO_THEFT", "OTHER_THEFT", "RIOTS", "CRIMINAL_BREACH_OF_TRUST",
        "CHEATING", "COUNTERFIETING", "ARSON", "HURT/GREVIOUS_HURT", "DOWRY_DEATHS",
        "ASSAULT_ON_WOMEN_WITH_INTENT_TO_OUTRAGE_HER_MODESTY", "INSULT_TO_MODESTY_OF_WOMEN",
        "CRUELTY_BY_HUSBAND_OR_HIS_RELATIVES", "IMPORTATION_OF_GIRLS_FROM_FOREIGN_COUNTRIES",
        "CAUSING_DEATH_BY_NEGLIGENCE", "OTHER_IPC_CRIMES"
]

# -------------------------
# Standardization and Cleaning for CSV Data
# -------------------------
def standardize_columns(df):
    df.columns = [col.upper().strip().replace(" ", "_") for col in df.columns]
    if "STATE/UT" in df.columns:
        df = df.rename(columns={"STATE/UT": "STATE"})
    if "YEAR" in df.columns and df["YEAR"].dtype == object: 
        df = df.rename(columns={"YEAR": "YEAR"})
    elif "Year" in df.columns:
         df = df.rename(columns={"Year": "YEAR"}) # Standardize 'Year' to 'YEAR'

    # Rename specific kidnapping columns for ST and Children to match others
    if "KIDNAPPING_ABDUCTION" in df.columns: # ST
        df = df.rename(columns={"KIDNAPPING_ABDUCTION": "KIDNAPPING_AND_ABDUCTION"})
    if "KIDNAPPING_&_ABDUCTION" in df.columns: # IPC
        df = df.rename(columns={"KIDNAPPING_&_ABDUCTION": "KIDNAPPING_AND_ABDUCTION"})

    for col in df.columns:
        if col not in ['STATE', 'DISTRICT', 'YEAR']: 
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def calculate_total_crimes(df, category):
    """Calculates a TOTAL_CRIMES column based on category."""

    total_col_name = 'TOTAL_CRIMES'
    if category == 'ipc':
        if 'TOTAL_IPC_CRIMES' in df.columns:
            df = df.rename(columns={'TOTAL_IPC_CRIMES': total_col_name})
        else:
             df[total_col_name] = df[crime_options['ipc']].sum(axis=1, skipna=True)
    elif category == 'sc':
        cols_to_sum = crime_options['sc']
        valid_cols = [col for col in cols_to_sum if col in df.columns]
        df[total_col_name] = df[valid_cols].sum(axis=1, skipna=True)
    elif category == 'st':
        cols_to_sum = crime_options['st']
        if "KIDNAPPING_ABDUCTION" in cols_to_sum:
            cols_to_sum = [c if c != "KIDNAPPING_ABDUCTION" else "KIDNAPPING_AND_ABDUCTION" for c in cols_to_sum]
        valid_cols = [col for col in cols_to_sum if col in df.columns]
        df[total_col_name] = df[valid_cols].sum(axis=1, skipna=True)
    elif category == 'children':
        if 'TOTAL' in df.columns:
            df = df.rename(columns={'TOTAL': total_col_name})
        else:
            cols_to_sum = crime_options['children']
            valid_cols = [col for col in cols_to_sum if col in df.columns]
            df[total_col_name] = df[valid_cols].sum(axis=1, skipna=True)
    elif category == 'women':
        cols_to_sum = crime_options['women']
        valid_cols = [col for col in cols_to_sum if col in df.columns]
        df[total_col_name] = df[valid_cols].sum(axis=1, skipna=True)

    if 'YEAR' in df.columns:
        df['YEAR'] = df['YEAR'].astype(str)

    return df


def load_and_clean_csv1(file_path):
    df = pd.read_csv(file_path)
    df = standardize_columns(df)
    print(df.shape)
    if "AREA_NAME" in df.columns:
        df = df[df["AREA_NAME"]   != "TOTAL"]
    df["YEAR"] = df["YEAR"].astype(str)
    return df


def load_and_clean_csv(file_path, category):
    df = pd.read_csv(file_path)
    df = standardize_columns(df)
    df = df[df["DISTRICT"] != "TOTAL"]
    df = calculate_total_crimes(df, category) 
    df['YEAR'] = df['YEAR'].astype(str)
    return df



# Load CSV files (update file paths if necessary)
df_ipc = load_and_clean_csv('final_data/01_District_wise_crimes_committed_IPC_final.csv', 'ipc')
df_sc = load_and_clean_csv('final_data/02_01_District_wise_crimes_committed_against_SC_final.csv', 'sc')
df_st = load_and_clean_csv('final_data/02_District_wise_crimes_committed_against_ST_final.csv', 'st')
df_children = load_and_clean_csv('final_data/03_District_wise_crimes_committed_against_children_final.csv', 'children')
df_women = load_and_clean_csv('final_data/42_District_wise_crimes_committed_against_women_2001_2013.csv', 'women')
df_juv_edu      = load_and_clean_csv1('final_data/18_01_Juveniles_arrested_Education.csv')
df_juv_econ     = load_and_clean_csv1('final_data/18_02_Juveniles_arrested_Economic_setup.csv')
df_juv_family   = load_and_clean_csv1('final_data/18_03_Juveniles_arrested_Family_background.csv')
df_juv_recidiv  = load_and_clean_csv1('final_data/18_04_Juveniles_arrested_Recidivism.csv')
df_murder = pd.read_csv('final_data/32_Murder_victim_age_sex.csv')
df_cust = pd.read_csv("./final_data/40_05_Custodial_death_others.csv")
df_bar = pd.read_csv('final_data/serious_fraud.csv')
df_stolen = pd.read_csv('final_data/property_stolen.csv')
df_stolen['Year'] = pd.to_numeric(df_stolen['Year'], errors='coerce').astype('Int64')
df_stolen = df_stolen.dropna(subset=['Year'])
df_stolen['Year'] = df_stolen['Year'].astype(int)
avail_states = sorted(df_stolen['Area_Name'].unique())
yr_min, yr_max = df_stolen['Year'].min(), df_stolen['Year'].max()
default_range = [yr_min, yr_max]
CSV_PATH = "final_data/38_Unidentified_dead_bodies_recovered_and_inquest_conducted.csv"

dropdown_style = {
    'backgroundColor': '#ffffff',
    'borderRadius': '4px',
    'border': '1px solid #e0e0e0',
    'marginBottom': '15px'
}
card_style = { 
    'backgroundColor': '#ffffff', 'borderRadius': '8px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)',
    'padding': '15px', 'marginBottom': '20px'
}


CLUSTER_COLORS = px.colors.qualitative.Plotly
UNSELECTED_CLUSTER_COLOR = 'rgb(255, 255, 255)' 
MISSING_DATA_COLOR = 'rgb(200, 200, 200)' #Grey

df_heat = pd.read_csv(CSV_PATH)
heatmap_data = df_heat.pivot(
    index="Area_Name", columns="Year",
    values="Unidentified_Dead_bodies_Recovered_Inquest_Conducted"
)
fig_heat = px.imshow(
    heatmap_data,
    labels=dict(x="Year", y="State/UT", color="No. of Cases"),
    x=heatmap_data.columns, y=heatmap_data.index,
    color_continuous_scale="YlOrRd", aspect="auto",
    title="Unidentified Dead Bodies Recovered & Inquests Conducted (2001–2010)"
)
fig_heat.update_yaxes(categoryorder="total ascending")

loss_cols = [
    'Loss_of_Property_1_25_Crores',
    'Loss_of_Property_25_100_Crores',
    'Loss_of_Property_Above_100_Crores'
]
df_long = (
    df_bar.melt(
        id_vars=['Area_Name','Year','Group_Name'],
        value_vars=loss_cols, var_name='Loss_Band',
        value_name='Loss'
    )
    .query('Loss > 0')
)
band_map = {
    'Loss_of_Property_1_25_Crores':'1–25 Cr',
    'Loss_of_Property_25_100_Crores':'25–100 Cr',
    'Loss_of_Property_Above_100_Crores':'>100 Cr'
}
df_long['Loss_Band'] = df_long['Loss_Band'].map(band_map)

years  = ['All'] + sorted(df_long['Year'].unique())
groups = ['All'] + sorted(df_long['Group_Name'].unique())
states = ['All'] + sorted(df_long['Area_Name'].unique())
default_states = ['Andhra Pradesh','Madhya Pradesh','Kerala']


# Combine DataFrames for easier access
dataframes = {
    'ipc': df_ipc,
    'sc': df_sc,
    'st': df_st,
    'children': df_children,
    'women': df_women
}

df_cust.replace("NULL", np.nan, inplace=True)
cols = [
    'CD_Accidents',
    'CD_By_Mob_AttackRiots',
    'CD_By_other_Criminals',
    'CD_By_Suicide',
    'CD_IllnessNatural_Death',
    'CD_While_Escaping_from_Custody'
]
for c in cols:
    df_cust[c] = df_cust[c].astype(float).fillna(0)

melted = df_cust.melt(
    id_vars=['Area_Name', 'Year'],
    value_vars=cols,
    var_name='Cause',
    value_name='Count'
)

# 3. National aggregate
nat = (
    df_cust
    .groupby('Year')[cols]
    .sum()
    .reset_index()
)
nat_melt = nat.melt(id_vars=['Year'], var_name='Cause', value_name='Count')
cust_states = sorted(df_cust['Area_Name'].unique())

# Define global options for the dropdown
STATE_OPTIONS = [{'label': s, 'value': s} for s in cust_states]


csv_file_path = 'final_data/34_Use_of_fire_arms_in_murder_cases.csv'
df_place = pd.read_csv('final_data/17_Crime_by_place_of_occurrence_2001_2012.csv')
df_place = df_place.rename(columns={'STATE/UT':'STATE'})
df_relative = pd.read_csv('final_data/21_Offenders_known_to_the_victim.csv')
df_place['YEAR'] = df_place['YEAR'].astype(str)
# Prepare long‐form for offender relationships
rel_cols = [
    col for col in df_relative.columns
    if col.startswith('No_of_Cases_in_which_offenders_were_')
]
df_rel_long = df_relative.melt(
    id_vars=['Area_Name'],
    value_vars=rel_cols,
    var_name='Relationship',
    value_name='Count'
)
df_rel_long['Relationship'] = (
    df_rel_long['Relationship']
       .str.replace('No_of_Cases_in_which_offenders_were_', '')
       .str.replace('_', ' ')
)

# Identify all "PLACE - CRIME" columns
pom_cols = [c for c in df_place.columns if ' - ' in c]

# Melt into long form: STATE, YEAR, PLACE_CRIME, COUNT
df_place_long = df_place.melt(
    id_vars=['STATE','YEAR'],
    value_vars=pom_cols,
    var_name='PLACE_CRIME',
    value_name='COUNT'
)

# Split PLACE_CRIME into separate PLACE and CRIME fields
df_place_long[['PLACE','CRIME']] = df_place_long['PLACE_CRIME'].str.split(' - ', expand=True)

# Define column names based on your description
area_col = 'Area_Name'
year_col = 'Year' # Though we aggregate over it
comparison_cols = ['Victims', 'By Registered Arms', 'By Unregistered Arms']

# Initialize empty DataFrame and areas list in case the file is not found
df_aggregated = pd.DataFrame(columns=[area_col] + comparison_cols)
areas = []


def load_and_clean_kidnapping_purpose_csv(file_path):
    try:
        # Read CSV, explicitly handle 'NULL' as NaN
        df = pd.read_csv(file_path, na_values=['NULL', 'Null', 'null', ''])
        print(f"Initial columns in kidnapping CSV: {df.columns.tolist()}") 

        # Standardize columns
        df = standardize_columns(df)
        print(f"Standardized columns in kidnapping CSV: {df.columns.tolist()}") 

        # Rename key columns for clarity and consistency
        rename_map = {
            'AREA_NAME': 'STATE',
            'YEAR': 'YEAR',
            'SUB_GROUP_NAME': 'PURPOSE',
            'K_A_GRAND_TOTAL': 'COUNT' 
        }
        
        valid_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=valid_rename_map)
        print(f"Renamed columns in kidnapping CSV: {df.columns.tolist()}") 

        required_cols = ['STATE', 'YEAR', 'PURPOSE', 'COUNT']
        missing_req = [col for col in required_cols if col not in df.columns]
        if missing_req:
            raise ValueError(f"Essential columns missing after standardization/rename: {missing_req}")

        # Convert COUNT and YEAR to numeric, coercing errors
        df['COUNT'] = pd.to_numeric(df['COUNT'], errors='coerce')
        df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce') # numeric for range slider

        # Fill NaNs in COUNT with 0
        df['COUNT'] = df['COUNT'].fillna(0).astype(int) 

        # Drop rows where YEAR is NaN
        df = df.dropna(subset=['YEAR'])
        df['YEAR'] = df['YEAR'].astype(int) 
        df['PURPOSE'] = df['PURPOSE'].fillna('').astype(str)
        df['PURPOSE_CLEAN'] = df['PURPOSE'].str.replace(r'^\d+\.\s*', '', regex=True).str.strip()

        print(f"Unique Cleaned Purposes: {df['PURPOSE_CLEAN'].unique()}")
        # Filter out any potential total rows if they exist
        df = df[~df['STATE'].str.contains("TOTAL", na=False, case=False)]
        # print(f"Kidnapping data loaded successfully. Shape: {df.shape}")
        return df

    except FileNotFoundError:
        print(f"Error: Kidnapping CSV file not found at '{file_path}'")
        return pd.DataFrame() # Return empty DataFrame on error
    except ValueError as ve:
        print(f"Data Error during kidnapping CSV processing: {ve}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred loading/cleaning kidnapping CSV: {e}")
        # traceback.print_exc() # Uncomment for more detailed traceback during debugging
        return pd.DataFrame()

df_kidnap = load_and_clean_kidnapping_purpose_csv('final_data/39_Specific_purpose_of_kidnapping_and_abduction.csv')

def standardize_columns1(df):
    """
    Standardizes column names (uppercase, replaces spaces with underscores)
    and attempts numeric conversion for columns not explicitly identified as
    non-numeric (e.g., STATE, DISTRICT, text categories).

    Handles specific renames for STATE/UT and common variations of
    Kidnapping column names.
    """
    # Standardize column names
    df.columns = [col.upper().strip().replace(" ", "_") for col in df.columns]

    # Rename STATE/UT if present
    if "STATE/UT" in df.columns:
        df = df.rename(columns={"STATE/UT": "STATE"})
    # Rename AREA_NAME if present (used in some files like murder, relatives, kidnap)
    if "AREA_NAME" in df.columns:
        df = df.rename(columns={"AREA_NAME": "STATE"}) #AREA_NAME maps to STATE

    # Handle variations of Year column
    if "YEAR" in df.columns and df["YEAR"].dtype == object:
        pass # Already standardized or correct type
    elif "YEAR" in df.columns and df["YEAR"].dtype != object:
         df['YEAR'] = df['YEAR'].astype(str) 
    elif "Year" in df.columns:
         df = df.rename(columns={"Year": "YEAR"}) 
         df['YEAR'] = df['YEAR'].astype(str) 

    if "KIDNAPPING_ABDUCTION" in df.columns: # Found in ST data
        df = df.rename(columns={"KIDNAPPING_ABDUCTION": "KIDNAPPING_AND_ABDUCTION"})
    if "KIDNAPPING_&_ABDUCTION" in df.columns: # Found in IPC data
        df = df.rename(columns={"KIDNAPPING_&_ABDUCTION": "KIDNAPPING_AND_ABDUCTION"})

    non_numeric_cols = [
        'STATE', 'DISTRICT', 'YEAR',          
        'GROUP_NAME', 'SUB_GROUP_NAME',      
        'PURPOSE', 'PURPOSE_CLEAN',          
        'RELATIONSHIP',                      
        'PLACE_CRIME', 'PLACE', 'CRIME',      
        'CRIME_TYPE', 'CRIME_LABEL',         
        'CLUSTERLABEL',                      
        'TYPE'                               
    ]
    non_numeric_cols_upper = [col.upper() for col in non_numeric_cols]

    # Attempt numeric conversion only for columns not in the non_numeric_cols list
    for col in df.columns:
        if col not in non_numeric_cols_upper:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"Warning: Could not process column '{col}' for numeric conversion: {e}")
        elif col == 'YEAR' and 'YEAR' in df.columns:
             if df['YEAR'].dtype != object:
                 df['YEAR'] = df['YEAR'].astype(str)


    return df


def load_and_clean_kidnapping_purpose_csv(file_path):
    try:
        # Read CSV, explicitly handle 'NULL' as NaN
        df = pd.read_csv(file_path, na_values=['NULL', 'Null', 'null', ''])
        # print(f"Initial columns in kidnapping CSV: {df.columns.tolist()}") 

        df = standardize_columns1(df)
        # print(f"Standardized columns in kidnapping CSV: {df.columns.tolist()}") 

        # Rename key columns for clarity and consistency
        rename_map = {
            'AREA_NAME': 'STATE',
            'YEAR': 'YEAR',
            'SUB_GROUP_NAME': 'PURPOSE',
            'K_A_GRAND_TOTAL': 'COUNT'
        }
        
        valid_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=valid_rename_map)
        # print(f"Renamed columns in kidnapping CSV: {df.columns.tolist()}") 

        required_cols = ['STATE', 'YEAR', 'PURPOSE', 'COUNT']
        missing_req = [col for col in required_cols if col not in df.columns]
        if missing_req:
            raise ValueError(f"Essential columns missing after standardization/rename: {missing_req}")

        # Convert COUNT and YEAR to numeric, coercing errors
        df['COUNT'] = pd.to_numeric(df['COUNT'], errors='coerce')
        df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce') # numeric for range slider

        # Fill NaNs in COUNT with 0
        df['COUNT'] = df['COUNT'].fillna(0).astype(int) # Convert to int after filling NaN
        df = df.dropna(subset=['YEAR'])
        df['YEAR'] = df['YEAR'].astype(int) # Convert YEAR to int
        df['PURPOSE'] = df['PURPOSE'].fillna('').astype(str)
        df['PURPOSE_CLEAN'] = df['PURPOSE'].str.replace(r'^\d+\.\s*', '', regex=True).str.strip()
        df = df[~df['STATE'].str.contains("TOTAL", na=False, case=False)]
        return df

    except FileNotFoundError:
        print(f"Error: Kidnapping CSV file not found at '{file_path}'")
        return pd.DataFrame() 
    except ValueError as ve:
        print(f"Data Error during kidnapping CSV processing: {ve}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred loading/cleaning kidnapping CSV: {e}")
        return pd.DataFrame()


if os.path.exists(csv_file_path):
    try:
        df_raw = pd.read_csv(csv_file_path)

        # --- Data Preprocessing and Aggregation ---
        numeric_cols_to_aggregate = []
        standardized_comparison_cols = [] 
        for col in comparison_cols:
            if col in df_raw.columns:
                standardized_col_name = col.upper().strip().replace(" ", "_")
                df_raw = df_raw.rename(columns={col: standardized_col_name}) 
                df_raw[standardized_col_name] = pd.to_numeric(df_raw[standardized_col_name], errors='coerce')
                numeric_cols_to_aggregate.append(standardized_col_name) 
                standardized_comparison_cols.append(standardized_col_name)
            else:
                print(f"Warning: Column '{col}' not found in CSV. Skipping.")

        df_raw = df_raw.fillna(0) 
        if area_col in df_raw.columns:
             standardized_area_col = area_col.upper().strip().replace(" ", "_")
             df_raw = df_raw.rename(columns={area_col: standardized_area_col})
             area_col = standardized_area_col 
        elif area_col.upper().strip().replace(" ", "_") in df_raw.columns:
             area_col = area_col.upper().strip().replace(" ", "_")
        else:
             raise ValueError(f"Area column '{area_col}' not found in the CSV.")

        # Aggregating Group by Area_Name and sum the numeric comparison columns
        if numeric_cols_to_aggregate: 
             df_aggregated = df_raw.groupby(area_col)[numeric_cols_to_aggregate].sum().reset_index()
             comparison_cols = standardized_comparison_cols
        else:
             print("Error: No valid numeric columns found for aggregation.")

        #unique area names for dropdowns
        if not df_aggregated.empty:
             areas = sorted(df_aggregated[area_col].unique())

    except FileNotFoundError:
         print(f"Error: CSV file not found at '{csv_file_path}'")
    except ValueError as ve:
         print(f"Data Error: {ve}")
    except Exception as e:
        print(f"Error loading or processing CSV '{csv_file_path}': {e}")
else:
    print(f"Error: CSV file not found at '{csv_file_path}'")


df_rape = pd.read_csv('final_data/20_Victims_of_rape.csv')
df_rape.rename(columns={'Area_Name':'State'}, inplace=True)
df_rape['Year'] = df_rape['Year'].astype(int)

all_states = sorted(df_rape['State'].unique())
all_subs   = sorted(df_rape['Subgroup'].unique())

max_year   = df_rape['Year'].max()
next_year  = max_year + 1

# predictions
preds = []
for state in all_states + ['All States']:
    for sub in all_subs + ['All Subgroups']:
        if state=='All States' and sub=='All Subgroups':
            train = df_rape.groupby('Year')['Victims_of_Rape_Total'].sum().reset_index()
        elif state=='All States':
            train = (df_rape[df_rape['Subgroup']==sub]
                     .groupby('Year')['Victims_of_Rape_Total']
                     .sum().reset_index())
        elif sub=='All Subgroups':
            train = (df_rape[df_rape['State']==state]
                     .groupby('Year')['Victims_of_Rape_Total']
                     .sum().reset_index())
        else:
            train = df_rape[(df_rape['State']==state)&(df_rape['Subgroup']==sub)][['Year','Victims_of_Rape_Total']]
        if train.empty:
            continue
        lr = LinearRegression().fit(train[['Year']], train['Victims_of_Rape_Total'])
        val = max(0, lr.predict([[next_year]])[0])
        preds.append({'State':state,'Subgroup':sub,'Year':next_year,'Predicted':val})

pred_df = pd.DataFrame(preds)

state_opts = [{'label': s,'value': s} for s in all_states]
state_opts.insert(0,{'label':'All States','value':'All States'})

sub_opts = [{'label': s,'value': s} for s in all_subs]
sub_opts.insert(0,{'label':'All Subgroups','value':'All Subgroups'})

years = list(range(df_rape['Year'].min(), next_year+1))
marks = {y:(str(y) if y<=max_year else f"{y} (pred)") for y in years}

# -------------------------
# Load and Normalize GeoJSON Files
# -------------------------
def normalize_geojson(file_path, prop_key, new_key):
    try:
        with open(file_path, "r") as f:
            geo = json.load(f)
    except FileNotFoundError:
        print(f"Error: GeoJSON file not found at {file_path}")
        return None 
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None 

    features = geo.get("features", [])
    if not features:
         print(f"Warning: No features found in {file_path}")
         return geo 

    for feature in features:
        if "properties" not in feature or not isinstance(feature["properties"], dict):
            feature["properties"] = {} 
        val = str(feature["properties"].get(prop_key, "")).upper().strip()
        feature["properties"][new_key] = val
    return geo


india_state_geo = normalize_geojson("india_state.geojson", prop_key="NAME_1", new_key="STATE_UPPER")
india_district_geo = normalize_geojson("india_district.geojson", prop_key="NAME_2", new_key="DISTRICT_UPPER")


# -------------------------
# Fuzzy Matching with RapidFuzz to Fix Missing Names
# -------------------------
def best_match_fuzzy(candidate, candidates, threshold=70):
    """Return the best match for candidate from candidates using RapidFuzz;
    if best score >= threshold, return the match, else return None."""
    result = process.extractOne(candidate, candidates, scorer=fuzz.WRatio)
    # result is a tuple: (best_match, score, index)
    if result and result[1] >= threshold:
        return result[0]
    return None

def fix_geojson_names(geo, csv_names, key):
    if geo is None: # Check if geojson loaded correctly
        print(f"Skipping name fixing for {key} as GeoJSON is missing.")
        return None
    if not csv_names: # Check if csv names are available
        print(f"Skipping name fixing for {key} as CSV names are missing.")
        return geo 

    candidates = list(csv_names)
    mismatched_count = 0
    updated_count = 0
    for feature in geo["features"]:
        if "properties" not in feature or key not in feature["properties"]:
            continue 

        current = feature["properties"].get(key, "")
        if current not in csv_names:
            mismatched_count += 1
            best = best_match_fuzzy(current, candidates, threshold=75) 
            if best:
                feature["properties"][key] = best
                updated_count +=1

    print(f"GeoJSON Name Fixing for '{key}': {mismatched_count} initial mismatches, {updated_count} updated.")
    return geo

all_states = set()
all_districts = set()
for df in dataframes.values():
    if 'STATE' in df.columns:
        all_states.update(df['STATE'].unique())
    if 'DISTRICT' in df.columns:
        all_districts.update(df['DISTRICT'].unique())

# fixing names using the comprehensive list
india_state_geo = fix_geojson_names(india_state_geo, all_states, "STATE_UPPER")
india_district_geo = fix_geojson_names(india_district_geo, all_districts, "DISTRICT_UPPER")


# After fixing, calculating missing entries.
missing_states_info = "GeoJSON for states not loaded or has issues."
missing_districts_info = "GeoJSON for districts not loaded or has issues."

if india_state_geo and 'features' in india_state_geo:
    geo_states = {feature["properties"].get("STATE_UPPER", "") for feature in india_state_geo["features"] if "properties" in feature}
    missing_states = geo_states - all_states
    missing_states_info = f"Missing States (GeoJSON states not found in any CSV): {sorted(list(missing_states))}" if missing_states else "All GeoJSON states found in CSV data."

if india_district_geo and 'features' in india_district_geo:
    geo_districts = {feature["properties"].get("DISTRICT_UPPER", "") for feature in india_district_geo["features"] if "properties" in feature}
    missing_districts = geo_districts - all_districts
    missing_districts_info = f"Missing Districts (GeoJSON districts not found in any CSV): {sorted(list(missing_districts))}" if missing_districts else "All GeoJSON districts found in CSV data."


# print(missing_states_info)
# print(missing_districts_info)

# -------------------------
# Helper Functions to Build Maps 
# -------------------------

def build_offender_treemap(top_n=15):
    df_rel_long['Count'] = pd.to_numeric(df_rel_long['Count'], errors='coerce').fillna(0)

    totals = (
        df_rel_long
        .groupby('Area_Name')['Count']
        .sum()
        .nlargest(top_n)
        .index
    )
    df_top = df_rel_long[df_rel_long['Area_Name'].isin(totals)]

    fig = px.treemap(
        df_top,
        path=['Area_Name', 'Relationship'],
        values='Count',
        color='Relationship', # Color by relationship type
        color_discrete_sequence=px.colors.qualitative.Pastel, 
        title=f"Offender Relationship Breakdown (Top {top_n} States by Total Cases)"
    )
    fig.update_traces(
        textinfo='label+percent entry',
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Pct of Parent: %{percentParent:.1%}<br>Pct of Total: %{percentRoot:.1%}<extra></extra>'
    )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25),
                       title_x=0.5) 
    return fig


###################
# Layout Snippets 
###################
area_comparison_layout = html.Div([
    html.H3("Compare Firearm Victims Between Areas (Aggregated Across Years)"),
    html.P("Select two areas to compare their total victims, victims by registered arms, and victims by unregistered arms."),

    html.Div([
        # Dropdown for Area 1
        html.Div([
            html.Label("Select Area 1:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='compare-area-dropdown-1',
                options=[{'label': area, 'value': area} for area in areas], 
                value=areas[0] if areas else None, 
                clearable=False
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),

        # Dropdown for Area 2
        html.Div([
            html.Label("Select Area 2:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='compare-area-dropdown-2',
                options=[{'label': area, 'value': area} for area in areas], 
                value=areas[1] if len(areas) > 1 else None, 
                clearable=False
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),

    ], style={'marginBottom': '20px'}),

    # Graph for Radar Plot
    dcc.Loading(
        id="loading-radar-chart",
        type="circle", 
        children=[dcc.Graph(id='area-comparison-radar-plot')]
    )

], style={'padding': '25px'})

dropdown_style = {
    'backgroundColor': '#ffffff',
    'borderRadius': '4px',
    'border': '1px solid #e0e0e0',
    'marginBottom': '15px'
}

radio_style = {
    'display': 'flex',
    'flexDirection': 'row', 
    'flexWrap': 'wrap', 
    'justifyContent': 'center', 
    'marginBottom': '15px',
    'padding': '10px',
    'backgroundColor': '#f8f9fa', 
    'borderRadius': '4px'
}

radio_label_style = {
    'display': 'inline-block',
    'margin': '5px 15px 5px 0', 
    'cursor': 'pointer'
}

#===========================================================================================================================
#=======================================================dash app layout==============================================================================================
app.layout = html.Div([
    # --- Header Section ---
    html.Div([
        # Hamburger Button
        html.Button(
            [html.Span(className="bar1"), html.Span(className="bar2"), html.Span(className="bar3")],
            id="btn-open-drawer", className="hamburger-button", n_clicks=0
        ),
        html.H1("Indian Crime Data Visualization Portal", className="main-title", style={'color':'white'})
    ], className="header-container"),

    # --- Slide-out Drawer (Navigation) ---
    html.Nav(id="slide-out-drawer", className="drawer", children=[
        html.Button("×", id="btn-close-drawer", className="close-button", n_clicks=0),
        html.H4("Navigation", className="drawer-header"),
        # List of links corresponding to our tabs
        html.Ul([
            html.Li(html.A("State Wise Crimes", href="#", id="link-tab-statewise", className="drawer-link")),
            html.Li(html.A("District Wise Crimes", href="#", id="link-tab-districtwise", className="drawer-link")),
            html.Li(html.A("Year Wise Crimes", href="#", id="link-tab-yearwise", className="drawer-link")),
            html.Li(html.A("Firearms Victims Statewise", href="#", id="link-tab-areacomparison", className="drawer-link")),
            html.Li(html.A("Place of Occurrence", href="#", id="link-tab-placeoccurrence", className="drawer-link")),
            html.Li(html.A("Murder Victims by Age and Sex", href="#", id="link-tab-murderflow", className="drawer-link")),
            html.Li(html.A("Relationship with Offender", href="#", id="link-tab-offenderrel", className="drawer-link")),
            html.Li(html.A("Clustering of Districts on IPC Crimes", href="#", id="link-tab-clusters", className="drawer-link")), 
            html.Li(html.A("Custodial Deaths Plots", href="#", id="link-tab-custodial", className="drawer-link")), 
            html.Li(html.A("Juvenile Background Analysis", href="#", id="link-tab-juvenile", className="drawer-link")), 
            html.Li(html.A("Unidentified Bodies Recovered", href="#", id="link-tab-heatmap", className="drawer-link")), 
            html.Li(html.A("Serious Fraud Losses", href="#", id="link-tab-fraud", className="drawer-link")), 
            html.Li(html.A("Property Stolen", href="#", id="link-tab-stolen", className="drawer-link")), 
            html.Li(html.A("Rape Victims Trends", href="#", id="link-tab-rape", className="drawer-link")), 
            html.Li(html.A("Kidnappings and Abduction", href="#", id="link-tab-kidnapping", className="drawer-link")), 

        ], className="drawer-links-list")
    ]),

    # Overlay (to dim content when drawer is open)
    html.Div(id="drawer-overlay", className="drawer-overlay", n_clicks=0),

    # --- Main Content Area ---
    
    html.Div([ 
        dcc.Tabs(
            id="main-tabs",
            value='tab-statewise', # Default tab
            children=[
                # === State Wise Tab ===
                dcc.Tab(label="State Wise", value='tab-statewise', children=[
                    html.Div([
                        html.H3("State Level Crime Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
                        html.Div([
                             html.H4("Select Crime Category:", style={'marginTop': '10px', 'textAlign': 'center'}),
                             dcc.RadioItems(
                                id="state-category-radio", 
                                options=[{"label": k.upper(), "value": k} for k in dataframes.keys()], 
                                value="ipc", # Default value
                                labelStyle=radio_label_style,
                                style=radio_style,
                                inputStyle={"marginRight": "5px"} 
                             )
                        ], style={'marginBottom': '20px'}),
                        dcc.Loading(
                            id="loading-state-map",
                            type="circle",
                            children=dcc.Graph(id="state-map", style={'width': '100%', 'height': '65vh'}, responsive=True) 
                        ),
                        dcc.Store(id="selected-state-store"),
                        html.Div(id="selected-state-display", style={'fontWeight': 'bold', 'marginTop': '15px', 'textAlign':'center', 'fontSize':'1.1em'}),
                        html.Div(id="state-analysis-options", style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
                        dcc.Loading( 
                            id="loading-state-visualizations",
                            type="circle", 
                            children=html.Div(id="state-visualizations-container", style={'marginTop': '20px'})
                        )
                    ], style={'padding': '25px'})
                ]), # End State Wise Tab

                # === District Wise Tab ===
                dcc.Tab(label="District Wise", value='tab-districtwise', children=[
                    html.Div([
                        html.H3("District Level Crime Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
                        html.Div([
                            html.Div([
                                html.Label("Select Crime Category:", style={'fontWeight': 'bold'}),
                                dcc.RadioItems(
                                    id="district-category-radio", 
                                    options=[{"label": k.upper(), "value": k} for k in dataframes.keys()],
                                    value="ipc", 
                                    labelStyle={'display': 'inline-block', 'marginRight': '15px', 'cursor': 'pointer'},
                                    style={'textAlign': 'center', 'marginBottom': '10px'}
                                )
                            ], style={'width': '100%', 'marginBottom': '15px'}),
                            html.Div([
                                html.Div([
                                    html.Label("Select Year(s):", style={'fontWeight': 'bold'}),
                                    dcc.RangeSlider(
                                        id='district-year-slider',
                                        step=1,
                                        marks=None, 
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        allowCross=False
                                    )
                                ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),

                                html.Div([
                                    html.Label("Select Specific Crime (Optional):", style={'fontWeight': 'bold'}),
                                    dcc.Dropdown(
                                        id='district-crime-dropdown',
                                        options=[], 
                                        placeholder="Select specific crime type (defaults to Total Crimes)",
                                        clearable=True,
                                        style=dropdown_style
                                    )
                                ], style={'width': '48%', 'display': 'inline-block'}),
                            ], style={'marginBottom': '20px'}),
                        ], style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'marginBottom': '25px'}),
                        html.Div([
                            html.H4("Crime Distribution by District", style={'textAlign': 'center'}),
                             dcc.Loading(
                                id="loading-district-map",
                                type="circle",
                                children=dcc.Graph(id="district-map", style={'height': '70vh'}, responsive=True) 
                            )
                        ], style={'marginBottom': '25px'}),
                        html.Div([
                             html.H4("Detailed District Analysis", style={'textAlign': 'center', 'borderBottom': '1px solid #ddd', 'paddingBottom': '10px', 'marginBottom': '20px'}),
                             dcc.Store(id="selected-district-store"), 
                             html.Div(id="selected-district-display", style={'fontWeight': 'bold', 'textAlign': 'center', 'fontSize': '1.2em', 'marginBottom': '15px'}),
                             dcc.Loading(
                                id="loading-district-details",
                                type="circle",
                                children=html.Div(id="district-detail-graphs")
                            )
                        ], style={'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px', 'marginBottom': '25px', 'minHeight': '200px'}), 
                        html.Div([
                            html.H4("Compare Districts", style={'textAlign': 'center', 'borderBottom': '1px solid #ddd', 'paddingBottom': '10px', 'marginBottom': '20px'}),
                            html.Div([
                                html.Div([
                                    html.Label("Select Districts to Compare (2 or more):", style={'fontWeight': 'bold'}),
                                    dcc.Dropdown(
                                        id='compare-districts-multi', 
                                        options=[], 
                                        placeholder="Select Districts",
                                        multi=True, 
                                        style=dropdown_style
                                    )
                                ], style={'width': '98%', 'display': 'inline-block'}), 
                            ], style={'marginBottom': '20px'}),
                            dcc.Loading(
                                id="loading-district-comparison",
                                type="circle",
                                children=html.Div(id="district-comparison-graphs") 
                            )
                        ], style={'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px', 'marginBottom': '25px', 'minHeight': '200px'}), # Added minHeight
                         html.Div([
                             html.H4("Crime Hotspot Analysis", style={'textAlign': 'center', 'borderBottom': '1px solid #ddd', 'paddingBottom': '10px', 'marginBottom': '20px'}),
                             html.Div([
                                 html.Div([
                                     html.Label("Select Crime for Hotspot Analysis:", style={'fontWeight': 'bold'}),
                                     dcc.Dropdown(
                                         id='crime-hotspot-dropdown',
                                         options=[], 
                                         placeholder="Select Crime Type",
                                         style=dropdown_style
                                     )
                                 ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),
                                 html.Div([
                                     html.Label("Number of Top Districts:", style={'fontWeight': 'bold'}),
                                     dcc.Slider(
                                         id='crime-hotspot-top-n-slider',
                                         min=5, max=30, step=5, value=10,
                                         marks={i: str(i) for i in range(5, 31, 5)},
                                         tooltip={"placement": "bottom", "always_visible": True}
                                     )
                                 ], style={'width': '48%', 'display': 'inline-block'}),
                             ], style={'marginBottom': '20px'}),
                             dcc.Loading(
                                 id="loading-crime-comparison",
                                 type="circle",
                                 children=html.Div(id="crime-comparison-graphs")
                             )
                         ], style={'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px', 'minHeight': '200px'}), 
                    ], style={'padding': '25px'})
                ]), # End District Wise Tab

                # === Year Wise Tab ===
                dcc.Tab(label="Year Wise", value='tab-yearwise', children=[
                     html.Div([
                        html.H3("Year-wise Crime Data Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
                        html.Div([
                            html.H4("Select Year Range:"),
                            dcc.RangeSlider(
                                id="year-slider",
                                min=min([int(y) for df in dataframes.values() for y in df["YEAR"].unique()]),
                                max=max([int(y) for df in dataframes.values() for y in df["YEAR"].unique()]),
                                value=[min([int(y) for df in dataframes.values() for y in df["YEAR"].unique()]),
                                      max([int(y) for df in dataframes.values() for y in df["YEAR"].unique()])],
                                marks={int(year): {'label': str(year), 'style': {'transform': 'rotate(45deg)', 'color': '#1f77b4', 'whiteSpace': 'nowrap'}}
                                      for year in sorted([int(y) for df in dataframes.values() for y in df["YEAR"].unique()])}, # Combine years
                                step=1, 
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),

                            html.H4("Select Crime Category:", style={'marginTop': '20px'}),
                            dcc.RadioItems(
                                id="year-category-radio",
                                options=[{"label": k.upper(), "value": k} for k in dataframes.keys()],
                                value="ipc",
                                labelStyle=radio_label_style,
                                style=radio_style,
                                inputStyle={"marginRight": "5px"}
                            ),

                            html.H4("Select Crime Type:", style={'marginTop': '20px'}),
                            dcc.Dropdown(
                                id="year-crime-type-dropdown",
                                style=dropdown_style 
                            ),
                            html.H4("Select Visualization Type:", style={'marginTop': '20px'}),
                            dcc.RadioItems(
                                id="year-viz-type-radio",
                                options=[
                                    {"label": "Trend Analysis", "value": "trend"},
                                    {"label": "State Comparison", "value": "state_comparison"},
                                    {"label": "Crime Type Breakdown", "value": "crime_breakdown"}
                                ],
                                value="trend",
                                labelStyle=radio_label_style, 
                                style=radio_style, 
                                inputStyle={"marginRight": "5px"}
                            ),
                        ]),
                        dcc.Loading( 
                             id="loading-year-visualizations",
                             type="circle",
                             children=html.Div(id="year-visualizations-container", style={'marginTop': '20px'})
                        )
                        
                    ], style={'padding': '25px'})
                ]), # End Year Wise Tab

                # === Area Comparison Tab ===
                dcc.Tab(label="Area Comparison", value='tab-areacomparison', children=[
                    area_comparison_layout 
                ]),

                # === Place Occurrence Tab ===
                dcc.Tab(label="Place Occurrence", value='tab-placeoccurrence', children=[
                    html.Div([
                        html.H3("Crime by Place of Occurrence (2001–2012)", style={'textAlign': 'center', 'marginBottom': '20px'}),
                        html.Label("Select Year Range:", style={'fontWeight':'bold'}),
                        dcc.RangeSlider(
                            id='place-year-slider',
                            min=int(df_place_long['YEAR'].min()),
                            max=int(df_place_long['YEAR'].max()),
                            value=[int(df_place_long['YEAR'].min()), int(df_place_long['YEAR'].max())],
                            marks={int(y): str(y) for y in sorted(df_place_long['YEAR'].unique())},
                            step=None,
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Br(),
                        dcc.Loading(id='loading-place-sunburst', type='circle',children=[dcc.Graph(id='place-sunburst')]),
                        dcc.Loading(id='loading-place-multiples', type='circle',children=[dcc.Graph(id='place-small-multiples')]),
                    ], style={'padding':'1rem'})
                ]), # End Place Occurrence Tab

                # === Murder Victims Tab ===
                dcc.Tab(label="Murder Victims Flow", value='tab-murderflow', children=[
                    html.Div([
                        html.H3("Murder Victims Flow Analysis (State → Gender → Age)", style={'textAlign': 'center', 'marginBottom': '20px'}),
                        html.Div([
                            html.Label("Select Year:", style={'fontWeight':'bold'}),
                            dcc.Dropdown(
                                id='year-dropdown-new', 
                                options=[{'label': str(year), 'value': year} for year in sorted(df_murder['Year'].unique())],
                                value=df_murder['Year'].max(),
                                clearable=False
                            ),
                        ], style={'width': '30%', 'marginBottom': '20px'}),
                        html.Div([
                            html.Label("Select Number of Top States (by total victims):", style={'fontWeight':'bold'}),
                            dcc.Slider(
                                id='states-slider', 
                                min=5,
                                max=len(df_murder['Area_Name'].unique()),
                                step=1,
                                value=10,
                                marks={i: str(i) for i in range(5, min(36, len(df_murder['Area_Name'].unique())+1), 5)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ], style={'width': '70%', 'marginTop': '20px', 'marginBottom': '20px'}),
                        dcc.Loading(id='loading-sankey', type = 'circle',children=[dcc.Graph(id='sankey-diagram', style={'height': '700px'})])
                    ], style={'padding': '1rem'})
                ]), # End Murder Victims Tab

                # === Offender Relationships Tab ===
                dcc.Tab(label="Offender Relationships", value='tab-offenderrel', children=[
                    html.Div([
                        html.H3("Offenders Known to the Victim", style={'textAlign': 'center', 'marginBottom': '20px'}),
                        html.P("Treemap showing breakdown of known offender categories across states.", style={'textAlign': 'center', 'marginBottom': '15px'}),
                        html.Label("Select Number of Top States (by total cases):", style={'fontWeight':'bold'}),
                        dcc.Slider(
                            id='rel-top-n', 
                            min=5, max=30, step=5, value=15,
                            marks={i: str(i) for i in [5,10,15,20,25,30]},
                            tooltip={'placement':'bottom', 'always_visible': True}
                        ),
                         dcc.Loading(id='loading-relative-treemap', type='circle',children=[dcc.Graph(id='relative-treemap', style={'marginTop': '20px'})])
                    ], style={'padding':'1rem'})
                ]), # End Offender Relationships Tab

                # === Crime Clusters Tab ===
                dcc.Tab(label="Crime Clusters (IPC)", value='tab-clusters', children=[
                    html.Div([
                        html.H3("K‑Means Clustering of Districts based on IPC Crimes", style={'textAlign': 'center', 'marginBottom': '20px'}),
                        html.P("Select crime types (features will be scaled) and number of clusters. Clustering uses mean values across all years.", style={'textAlign': 'center', 'marginBottom': '15px'}),
                        html.Div([
                            html.Div([
                                html.Label("Select IPC Crime Features:", style={'fontWeight':'bold'}),
                                dcc.Dropdown(
                                    id='cluster-features',
                                    options=[{'label': f.replace('_',' ').title(), 'value': f} for f in crime_features], 
                                    value=crime_features[:5], # Default to first 5 features
                                    multi=True, style={'marginBottom': '15px'}
                                ),
                                html.Label("Select Number of Clusters (K):", style={'fontWeight':'bold'}),
                                dcc.Slider(
                                    id='cluster-count', min=2, max=10, step=1, value=4,
                                    marks={i:str(i) for i in range(2,11)},
                                    tooltip={'placement':'bottom', 'always_visible': True}, updatemode='drag'
                                )
                            ], style={'width':'65%', 'display':'inline-block', 'paddingRight':'20px', 'verticalAlign':'top'}),
                            html.Div([
                                html.Label("Show Clusters on Map:", style={'fontWeight':'bold'}),
                                dcc.Checklist(
                                    id='cluster-visibility-checklist',
                                    options=[], value=[],
                                    labelStyle={'display': 'block', 'marginBottom': '5px'},
                                    style={'maxHeight': '150px', 'overflowY': 'auto', 'border': '1px solid #ccc', 'padding': '10px', 'borderRadius': '5px'}
                                )
                            ], style={'width':'30%', 'display':'inline-block', 'verticalAlign':'top'}),
                        ], style={'width':'90%', 'margin': '0 auto', 'marginBottom':'20px'}, className='control-panel'),
                        dcc.Loading(id='loading-cluster-map', type='circle',children=[dcc.Graph(id='cluster-map', style={'height':'55vh'})]), # Map
                        html.Div([
                            html.Div(
                                dcc.Loading(id='loading-cluster-bar',type='circle', children=[dcc.Graph(id='cluster-centroid-bar')])
                            , style={'width':'50%','display':'inline-block', 'paddingRight':'10px', 'boxSizing': 'border-box'}),
                            html.Div(
                                dcc.Loading(id='loading-cluster-sil',type='circle', children=[dcc.Graph(id='cluster-centroid-silhouette')])
                            , style={'width':'50%','display':'inline-block', 'paddingLeft':'10px', 'boxSizing': 'border-box'})
                        ], style={'marginTop': '15px'}),
                        html.Br(),
                        html.Div([
                            html.H4("Cluster Crime Trends (Avg Selected, incl. 1‑year ARIMA forecast)", style={'textAlign': 'center', 'marginTop': '20px'}),
                            dcc.Loading(id='loading-cluster-trends-avg',type='circle', children=[dcc.Graph(id='cluster-trends-avg', style={'height':'50vh'})]) # Trends Average
                        ], style={'marginBottom': '30px'}),
                        html.Div([
                            html.H4("Cluster Crime Trends (IPC Total, incl. 1‑year ARIMA forecast)", style={'textAlign': 'center', 'marginTop': '20px'}),
                            dcc.Loading(id='loading-cluster-trends-total',type='circle', children=[dcc.Graph(id='cluster-trends-total', style={'height':'50vh'})]) # Trends Total
                        ]),
                    ], style={'padding':'1rem'})
                ]), # End Crime Clusters Tab

                # === Custodial Deaths Tab ===
                dcc.Tab(label="Custodial Deaths Plots", value='tab-custodial', children=[
                     html.Div([
                        html.H3("Custodial Death Causes: Small Multiples vs National Total", style={'textAlign': 'center', 'marginBottom': '20px', 'marginTop': '80px'}),
                        html.Div([
                            html.Label("Chart type:"),
                            dcc.RadioItems( id='chart-type', options=[ {'label': 'Stacked Area', 'value': 'area'}, {'label': 'Stacked Bar',  'value': 'bar'} ], value='area', inline=True )
                        ], style={'margin-bottom': '1em'}),
                        html.Div([
                            html.Label("Mode:"),
                            dcc.RadioItems( id='mode', options=[ {'label': 'Few States (small multiples)', 'value': 'states'}, {'label': 'National aggregate', 'value': 'nat'} ], value='states', inline=True )
                        ], style={'margin-bottom': '1em'}),
                        html.Div([
                            html.Label("Select up to 6 states:"),
                            dcc.Dropdown( id='state-select', options=STATE_OPTIONS, value=['Maharashtra','Uttar Pradesh','Gujarat','West Bengal','Tamil Nadu'], multi=True, placeholder="Choose states" )
                        ], style={'margin-bottom': '2em'}),
                        dcc.Graph(id='timeline-graph')
                    ])
                ]),

                # === Juvenile Plots Tab ===
                dcc.Tab(label="Juvenile Plots", value='tab-juvenile', children=[
                     html.Div([
                        html.H3("Juvenile Arrests Breakdown", style={'textAlign': 'center', 'marginBottom': '20px'}),
                        dcc.Dropdown(
                            id="juv-category-dropdown",
                            options=[ {'label': 'Education', 'value': 'education'}, {'label': 'Economic Setup', 'value': 'economic'}, {'label': 'Family Background','value': 'family'}, {'label': 'Recidivism', 'value': 'recidivism'} ],
                            placeholder="Select one or more categories", multi=True
                        ),
                        html.Div(id="juv-plots-container", style={'marginTop': '20px'})
                    ], style={'padding': '25px'})
                ]),

                # === Unidentified Bodies Heatmap Tab ===
                dcc.Tab(label="Unidentified Bodies Heatmap", value='tab-heatmap', children=[
                      html.Div([
                          dcc.Graph(id='heatmap', figure=fig_heat)
                          
                      ], style={'width':'90%','margin':'auto'})
                ]),

                # === Serious Fraud Losses Tab ===
                dcc.Tab(label="Serious Fraud Losses", value='tab-fraud', children=[
                     html.Div([
                        html.H3("Serious Fraud Losses", style={'textAlign': 'center', 'marginBottom': '20px', 'marginTop':'60px'}),
                        dcc.Dropdown( id='year-dropdown', options=[{'label': y,'value': y} for y in years], value='All', clearable=False, style={'width':'80%','margin':'10px auto'}, placeholder='Select Year' ),
                        dcc.Dropdown( id='group-dropdown', options=[{'label': g,'value': g} for g in groups], value='All', clearable=False, style={'width':'80%','margin':'10px auto'} ),
                        dcc.Dropdown( id='state-dropdown', options=[{'label': s,'value': s} for s in states], value=default_states, multi=True, placeholder='Select states', style={'width':'80%','margin':'10px auto'} ),
                        dcc.Graph(id='bar-chart', style={'height':'70vh'})
                    ])
                ]),

                # === Property Stolen Analysis Tab ===
                dcc.Tab(label="Property Stolen Analysis", value='tab-stolen', children=[
                     html.Div([
                        html.H3("Property Stolen", style={'textAlign': 'center', 'marginBottom': '20px', 'marginTop':'60px'}),
                        html.Div([ html.Label("Year Range:"), dcc.RangeSlider( id='stolen-year-slider', min=yr_min, max=yr_max, step=1, marks={y: str(y) for y in range(yr_min, yr_max+1)}, value=default_range, tooltip={"always_visible": True} ) ], style={'width':'80%','margin':'auto','padding':'20px'}),
                        html.Div([ html.Label("Mode:"), dcc.RadioItems( id='stolen-mode', options=[ {'label':'Single State','value':'single'}, {'label':'Compare States','value':'multi'} ], value='single', labelStyle={'display':'inline-block','marginRight':'20px'} ) ], style={'textAlign':'center','padding':'10px'}),
                        html.Div(id='stolen-state-select', style={'width':'60%','margin':'auto'}, children=[
                            html.Div(id='single-container', children=[ html.Label("Select State:"), dcc.Dropdown( id='stolen-single-dd', options=[{'label': s,'value': s} for s in avail_states], placeholder="State..." ) ]),
                            html.Div(id='multi-container', style={'display':'none'}, children=[ html.Label("Select 2–7 States:"), dcc.Dropdown( id='stolen-multi-dd', options=[{'label': s,'value': s} for s in avail_states], multi=True, placeholder="States..." ) ])
                        ]),
                        html.Div(id='stolen-plots-div', style={'padding':'20px'})
                    ])
                ]),

                # === Rape Victims Trend & Prediction Tab ===
                dcc.Tab(label="Rape Victims Trend & Prediction", value='tab-rape', children=[
                     html.Div([
                        html.H3("Rape Victim Trend and Prediction for next year", style={'textAlign': 'center', 'marginBottom': '20px', 'marginTop':'60px'}),
                        html.Div([ html.Label('Select States'), dcc.Dropdown(id='test-state-dd', options=state_opts, value=['All States'], multi=True) ], style={'width':'48%','display':'inline-block'}),
                        html.Div([ html.Label('Select Subgroups'), dcc.Dropdown(id='test-sub-dd', options=sub_opts, value=['All Subgroups'], multi=True) ], style={'width':'48%','display':'inline-block'}),
                        html.Br(),
                        html.Label('Year Range'),
                        dcc.RangeSlider(id='test-year-slider', min=years[0], max=years[-1], step=1, value=[years[0], years[-1]], marks=marks),
                        dcc.Graph(id='test-time-series')
                    ], style={'padding':'20px'})
                ]),

                 # === Kidnappings & Abductions Tab ===
                dcc.Tab(label="Kidnappings & Abductions", value='tab-kidnapping', children=[
                      html.Div([
                        html.H3("Kidnapping & Abduction Analysis", style={'textAlign': 'center', 'marginBottom': '20px', 'marginTop': '60px'}),
                        html.Div([
                            html.Label("Select Year Range:", style={'fontWeight':'bold'}),
                            html.Div([ dcc.RangeSlider( id="kidnap-year-slider", min=df_kidnap['YEAR'].min() if not df_kidnap.empty else 2001, max=df_kidnap['YEAR'].max() if not df_kidnap.empty else 2010, value=[df_kidnap['YEAR'].min(), df_kidnap['YEAR'].max()] if not df_kidnap.empty else [2001, 2010], marks={int(year): {'label': str(year), 'style': {'transform': 'rotate(45deg)', 'color': '#1f77b4', 'whiteSpace': 'nowrap'}} for year in sorted(df_kidnap['YEAR'].unique())} if not df_kidnap.empty else {2001: '2001', 2010: '2010'}, step=1, tooltip={"placement": "bottom", "always_visible": True}, ) ], style={'marginBottom': '25px'}),
                            html.Label("Select Purpose (for Trend/State/Demographics):", style={'fontWeight':'bold'}),
                            dcc.Dropdown( id="kidnap-purpose-dropdown", value='TOTAL_KIDNAPPINGS', clearable=False, style={**dropdown_style, 'marginBottom': '15px'} ),
                            html.Label("Select States (for Profile Comparison):", style={'fontWeight':'bold'}),
                            dcc.Dropdown( id="kidnap-state-multiselect", options=[{'label': state, 'value': state} for state in sorted(df_kidnap['STATE'].unique())] if not df_kidnap.empty else [], value=sorted(df_kidnap['STATE'].unique())[:5] if not df_kidnap.empty else [], multi=True, style={**dropdown_style, 'marginBottom': '15px'} ),
                            html.Label("Select State (for Demographics):", style={'fontWeight':'bold'}),
                            dcc.Dropdown( id="kidnap-state-demographics-dropdown", options=[{'label': 'All India', 'value': 'All India'}] + [{'label': state, 'value': state} for state in sorted(df_kidnap['STATE'].unique())] if not df_kidnap.empty else [], value='All India', clearable=False, style={**dropdown_style, 'marginBottom': '25px'} ),
                            html.Label("Select Visualization Type:", style={'fontWeight':'bold'}),
                            dcc.RadioItems( id="kidnap-viz-type-radio", options=[ {"label": "Trend Analysis", "value": "trend"}, {"label": "State Comparison (Counts)", "value": "state_comparison"}, {"label": "Purpose Profile Comparison (%)", "value": "profile_comparison"}, {"label": "Victim Demographics Breakdown", "value": "victim_demographics"}, ], value="trend", labelStyle=radio_label_style, style=radio_style, inputStyle={"marginRight": "5px"} ),
                        ], style={'width': '80%', 'margin': '0 auto', 'marginBottom':'20px'}),
                        dcc.Loading( id="loading-kidnap-visualizations", type="circle", children=html.Div(id="kidnap-visualizations-container", style={'marginTop': '20px'}) )
                        
                    ], style={'padding': '25px'})
                ]),

            ],
        ), # End dcc.Tabs
    ], style={'padding': '15px', 'maxWidth': '1300px', 'margin': '0 auto', 'backgroundColor': '#ffffff',
              'borderRadius': '8px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'}),

    # --- Footer Section --- Footer ---
    html.Footer([
        html.Span("WatchTower", className='footer-left'),
        # html.Span("Made with ❤️", className='footer-center'),
        html.Div([
            html.Span("Created by: ", style={'marginRight': '5px'}),
            html.A("Roshan Kumar", href="https://www.linkedin.com/in/roshan03/", target="_blank", className='footer-link'),
            html.Span(" | ", style={'margin': '0 5px'}), # Optional label
            html.A("Akshay Toshniwal", href="linkedin.com/in/akshay-toshniwal-a3aa77206", target="_blank", className='footer-link'),
            html.Span(" | ", style={'margin': '0 5px'}), # Separator
            html.A("Ansh Makwe", href="https://www.linkedin.com/in/ansh-makwe-1908bb227/", target="_blank", className='footer-link'),
            html.Span(" | ", style={'margin': '0 5px'}), # Separator
            html.A("Devang Agrawal", href="https://www.linkedin.com/in/devang-agrawal-18239b172/", target="_blank", className='footer-link'),
            html.Span(" | ", style={'margin': '0 5px'}), # Separator
            html.A("Divyansh Chaurasia", href="https://www.linkedin.com/in/divyanshchaurasia22/", target="_blank", className='footer-link'),
            html.Span(" | ", style={'margin': '0 5px'}), # Separator
            html.A("Kunal Anand", href="https://www.linkedin.com/in/kunal-anand-752169214/", target="_blank", className='footer-link'),
            html.Span(" | ", style={'margin': '0 5px'}), # Separator
            html.A("Parjanya Aditya Shukla", href="https://www.linkedin.com/in/parjanya-aditya-shukla/", target="_blank", className='footer-link'),
            html.Span(" | ", style={'margin': '0 5px'}), # Separator
            html.A("Prakhar Mandloi", href="https://www.linkedin.com/in/prakhar-mandloi-016851207/", target="_blank", className='footer-link'),
        ], className='footer-right') # Assign class for styling
    ], className='app-footer')
]) # End app.layout

#===============================Callbacks for tabs and functionalities =======================================================

# Clientside callback to toggle drawer visibility 
app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='toggleDrawer'),
    Output("slide-out-drawer", "className"),
    Output("drawer-overlay", "className"),
    Input("btn-open-drawer", "n_clicks"),
    Input("btn-close-drawer", "n_clicks"),
    Input("drawer-overlay", "n_clicks"),
    State("slide-out-drawer", "className"),
    prevent_initial_call=True
)

# Dynamically generating Inputs/Outputs/States for link callbacks
tab_values = []
try:
    tabs_component = app.layout.children[1].children[0] 

    if isinstance(tabs_component, dcc.Tabs) and hasattr(tabs_component, 'children'):
        for tab in tabs_component.children:
             if isinstance(tab, dcc.Tab) and hasattr(tab, 'value') and tab.value:
                tab_values.append(tab.value)
        if not tab_values: 
            raise ValueError("No tab values extracted from layout.")
        # print(f"Successfully extracted tab values: {tab_values}")
    else:
        raise TypeError("Could not find dcc.Tabs component or its children as expected.")

except (AttributeError, IndexError, TypeError, ValueError) as e:
    #  print(f"Error introspecting layout for tab values: {e}. Falling back to manual list.")
     tab_values = [
         'tab-statewise', 'tab-districtwise', 'tab-yearwise', 'tab-areacomparison',
         'tab-placeoccurrence', 'tab-murderflow', 'tab-offenderrel', 'tab-clusters',
         'tab-custodial', 'tab-juvenile', 'tab-heatmap', 'tab-fraud', 'tab-stolen',
         'tab-rape', 'tab-kidnapping' 
     ]

# Create Inputs for the link click callback
link_inputs = [Input(f"link-{val}", "n_clicks") for val in tab_values if val] # Filter Nones

# --- Clientside callback to handle clicks on drawer links ---
if link_inputs: # Only register if inputs were created
    app.clientside_callback(
        ClientsideFunction(namespace='clientside', function_name='handleDrawerLinkClick'),
        Output("main-tabs", "value"), # Update the active tab
        Output("slide-out-drawer", "className", allow_duplicate=True), # Close drawer
        Output("drawer-overlay", "className", allow_duplicate=True), # Hide overlay
        *link_inputs,
        prevent_initial_call=True
    )
else:
    print("Warning: No link inputs generated for drawer link callback.")


# --- Callback to Update Active Link Style ---
link_outputs = [Output(f"link-{val}", "className", allow_duplicate=True) for val in tab_values if val]
link_id_states = [State(f"link-{val}", "id") for val in tab_values if val]

if link_outputs: # Only register if outputs exist
    app.clientside_callback(
        ClientsideFunction(namespace='clientside', function_name='updateActiveLink'),
        link_outputs,
        Input("main-tabs", "value"),
        *link_id_states,
        prevent_initial_call='initial_duplicate' # Allows initial call with allow_duplicate
    )
else:
     print("Warning: No link outputs generated for active link style callback.")


# -------------------------
# Callbacks for Year Wise Tab
# -------------------------

@app.callback(
    Output("year-crime-type-dropdown", "options"),
    [Input("year-category-radio", "value")]
)
def update_year_crime_types(category):
    """Dynamically update crime type dropdown options based on selected category."""
    if category not in crime_options:
        return []
    # Add 'TOTAL_CRIMES' as an option for every category
    opts = [{"label": "TOTAL CRIMES", "value": "TOTAL_CRIMES"}]
    opts.extend([{"label": x.replace("_", " ").title(), "value": x} for x in crime_options.get(category, [])])
    return opts

# Set default value for year-crime-type-dropdown (optional, but good UX)
@app.callback(
    Output("year-crime-type-dropdown", "value"),
    Input("year-crime-type-dropdown", "options"),
)
def set_default_year_crime_type(options):
     if options:
         # Default to 'TOTAL_CRIMES' if available, else the first option
         default_value = next((opt['value'] for opt in options if opt['value'] == 'TOTAL_CRIMES'), options[0]['value'])
         return default_value
     return None


@app.callback(
    Output("year-visualizations-container", "children"),
    [Input("year-slider", "value"),
     Input("year-category-radio", "value"),
     Input("year-crime-type-dropdown", "value"),
     Input("year-viz-type-radio", "value")]
)
def update_year_visualizations(year_range, category, crime_type, viz_type):
    """Update visualizations in the Year Wise tab based on selections."""
    if not year_range or not category or not crime_type or not viz_type:
        return html.Div("Please select year range, category, crime type, and visualization type.",
                       style={'padding': '20px', 'color': '#777777', 'fontStyle': 'italic', 'textAlign':'center'})

    # Get the correct dataframe
    df = dataframes.get(category)
    if df is None:
        return html.Div(f"Error: Data for category '{category}' not found.", style={'color': 'red', 'textAlign':'center'})

    # Ensure the selected crime_type exists in the dataframe (or is TOTAL_CRIMES)
    if crime_type != 'TOTAL_CRIMES' and crime_type not in df.columns:
         # Check if it's a known option for the category but just missing from this specific DF (edge case)
         if crime_type in crime_options.get(category, []):
             return html.Div(f"Warning: Crime type '{crime_type.replace('_',' ').title()}' selected but not found in the '{category.upper()}' dataset for the chosen years. It might have only zero values or be missing.",
                             style={'padding': '20px', 'color': 'orange', 'textAlign':'center'})
         else:
            # This case should ideally not happen if dropdowns are populated correctly
            return html.Div(f"Error: Selected crime type '{crime_type}' is invalid for category '{category.upper()}'.",
                           style={'padding': '20px', 'color': 'red', 'textAlign':'center'})


    # Convert year values to strings for filtering
    # Handle single year selection vs range
    min_year, max_year = year_range
    years = [str(year) for year in range(min_year, max_year + 1)]


    # Filter dataframe for the selected years
    df_years = df[df["YEAR"].isin(years)].copy() # Use .copy() to avoid SettingWithCopyWarning

    if df_years.empty:
        return html.Div(f"No data available for {category.upper()} crimes between {min_year} and {max_year}.",
                       style={'padding': '20px', 'color': '#777777', 'fontStyle': 'italic', 'textAlign':'center'})

    # Use the specific crime type or the calculated TOTAL_CRIMES
    target_crime_col = crime_type # This will be 'TOTAL_CRIMES' if selected, or a specific crime type


    # --- Visualization Logic ---
    graphs = []
    crime_label = target_crime_col.replace('_', ' ').title() # For titles and labels

    # General Card Style
    card_style = {
        'backgroundColor': '#ffffff',
        'borderRadius': '8px',
        'boxShadow': '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)',
        'padding': '15px',
        'marginBottom': '20px'
    }

    # === Trend Analysis ===
    if viz_type == "trend":
        # Aggregate nationally by year
        trend_df = df_years.groupby("YEAR")[target_crime_col].sum().reset_index()
        trend_df = trend_df.sort_values("YEAR")
        trend_df['YEAR'] = pd.to_numeric(trend_df['YEAR']) # Ensure year is numeric for plotting/sorting

        line_fig = px.line(trend_df, x="YEAR", y=target_crime_col,
                         title=f"National Trend of {crime_label} ({min_year}–{max_year})")

        line_fig.update_layout(
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
            title={'font': {'size': 18, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
            margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
            xaxis={'gridcolor': '#f0f0f0', 'title': 'Year'},
            yaxis={'gridcolor': '#f0f0f0', 'title': crime_label}
        )
        line_fig.update_traces(
            line=dict(color='#1f77b4', width=3), mode='lines+markers', marker=dict(size=8),
            hovertemplate='Year: %{x}<br>' + crime_label + ': %{y:,}<extra></extra>' # Added comma formatting
        )
        graphs.append(html.Div(dcc.Graph(figure=line_fig), className="card-container", style=card_style))

        # Year-over-year percentage change (only if more than one year)
        if len(trend_df) > 1:
            trend_df['YOY_CHANGE'] = trend_df[target_crime_col].pct_change() * 100
            # Handle potential division by zero or infinite values if previous year was 0
            trend_df['YOY_CHANGE'] = trend_df['YOY_CHANGE'].replace([float('inf'), -float('inf')], None) # Replace infinities with None/NaN
            trend_df_yoy = trend_df.dropna(subset=['YOY_CHANGE']) # Drop rows where change couldn't be calculated

            if not trend_df_yoy.empty:
                # Determine bar colors based on change
                bar_colors = ['#2ca02c' if x >= 0 else '#d62728' for x in trend_df_yoy['YOY_CHANGE']] # Green for increase, Red for decrease

                yoy_fig = px.bar(trend_df_yoy, x="YEAR", y="YOY_CHANGE",
                               title=f"Year-over-Year % Change in {crime_label} ({min_year+1}–{max_year})") # Adjust title year range

                yoy_fig.update_layout(
                    plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
                    title={'font': {'size': 18, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
                    margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
                    xaxis={'gridcolor': '#f0f0f0', 'title': 'Year', 'tickmode': 'linear'}, # Ensure all years show if space allows
                    yaxis={'gridcolor': '#f0f0f0', 'title': '% Change'}
                )
                yoy_fig.update_traces(
                    marker_color=bar_colors,
                    hovertemplate='Year: %{x}<br>% Change: %{y:.2f}%<extra></extra>'
                )
                graphs.append(html.Div(dcc.Graph(figure=yoy_fig), className="card-container", style=card_style))
            else:
                 graphs.append(html.Div("Not enough data points for Year-over-Year change calculation.", style={**card_style, 'fontStyle':'italic', 'color':'grey'}))

    # === State Comparison ===
    elif viz_type == "state_comparison":
        # Aggregate by state over the selected years
        state_agg = df_years.groupby("STATE")[target_crime_col].sum().reset_index()
        top_states = state_agg.sort_values(target_crime_col, ascending=False).head(15) # Show top 15 states

        # Horizontal bar chart for top states
        state_fig = px.bar(top_states, y="STATE", x=target_crime_col, orientation='h',
                        title=f"Top 15 States by Total {crime_label} ({min_year}–{max_year})")
        state_fig.update_layout(
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
            title={'font': {'size': 18, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
            margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
            xaxis={'gridcolor': '#f0f0f0', 'title': f"Total {crime_label}"},
            yaxis={'gridcolor': '#f0f0f0', 'title': '', 'categoryorder': 'total ascending'} # Keep states sorted by value
        )
        state_fig.update_traces(marker_color='#1f77b4', hovertemplate='State: %{y}<br>Total: %{x:,}<extra></extra>') # Added comma formatting
        graphs.append(html.Div(dcc.Graph(figure=state_fig), className="card-container", style=card_style))

        # Pie chart for contribution of top 5 states
        top5_states_df = state_agg.sort_values(target_crime_col, ascending=False).head(5)
        # Calculate 'Other' category for the rest
        total_top5 = top5_states_df[target_crime_col].sum()
        total_all = state_agg[target_crime_col].sum()
        other_val = total_all - total_top5
        if other_val > 0 and len(state_agg) > 5: # Only add 'Other' if there are more than 5 states and 'Other' has value
             other_df = pd.DataFrame([{'STATE': 'Other States', target_crime_col: other_val}])
             pie_data = pd.concat([top5_states_df, other_df], ignore_index=True)
        else:
             pie_data = top5_states_df

        pie_fig = px.pie(pie_data, values=target_crime_col, names='STATE',
                        title=f"Contribution of Top 5 States to Total {crime_label} ({min_year}–{max_year})",
                        hole=0.3) # Add a donut hole

        pie_fig.update_traces(textposition='inside', textinfo='percent+label',
                             hovertemplate='<b>%{label}</b><br>Total: %{value:,}<br>Percentage: %{percent:.1%}<extra></extra>') # Improved hover
        pie_fig.update_layout(
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
            title={'font': {'size': 18, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
            margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
            legend={'orientation': 'v', 'yanchor':'top', 'y':0.7, 'xanchor':'left', 'x':-0.1} # Vertical legend to the side
        )
        graphs.append(html.Div(dcc.Graph(figure=pie_fig), className="card-container", style=card_style))


    # === Crime Type Breakdown ===
    elif viz_type == "crime_breakdown":
        # Use all crime types defined for the selected category
        all_crime_types = crime_options[category]
        # Ensure these columns actually exist in the filtered df_years
        valid_crime_types = [ct for ct in all_crime_types if ct in df_years.columns]

        if not valid_crime_types:
             graphs.append(html.Div(f"No specific crime type data found for {category.upper()} category in the selected years.", style={**card_style, 'fontStyle':'italic', 'color':'grey'}))
        else:
            # Aggregate sum for each valid crime type over the selected years
            breakdown_data = df_years[valid_crime_types].sum().reset_index()
            breakdown_data.columns = ['CRIME_TYPE', 'TOTAL_COUNT']
            breakdown_data = breakdown_data[breakdown_data['TOTAL_COUNT'] > 0] # Only show crimes with counts > 0
            breakdown_data = breakdown_data.sort_values('TOTAL_COUNT', ascending=False)
            breakdown_data['CRIME_LABEL'] = breakdown_data['CRIME_TYPE'].str.replace('_', ' ').str.title() # Create clean labels

            # Pie chart for distribution of crime types
            pie_fig = px.pie(breakdown_data, values='TOTAL_COUNT', names='CRIME_LABEL',
                            title=f"Breakdown of {category.upper()} Crime Types ({min_year}–{max_year})",
                            hole=0.3)
            pie_fig.update_traces(textposition='inside', textinfo='percent+label',
                                 hovertemplate='<b>%{label}</b><br>Total Count: %{value:,}<br>Percentage: %{percent:.1%}<extra></extra>')
            pie_fig.update_layout(
                plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
                title={'font': {'size': 18, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
                margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
                legend_title_text='Crime Type',
                legend={'orientation': 'v', 'yanchor':'top', 'y':0.7, 'xanchor':'left', 'x':-0.1}
            )
            graphs.append(html.Div(dcc.Graph(figure=pie_fig), className="card-container", style=card_style))

            # Horizontal bar chart for top N crime types (e.g., top 10)
            top_n_crimes = breakdown_data.head(10)
            bar_fig = px.bar(top_n_crimes, y='CRIME_LABEL', x='TOTAL_COUNT', orientation='h',
                            title=f"Top 10 {category.upper()} Crime Types by Count ({min_year}–{max_year})")
            bar_fig.update_layout(
                plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
                title={'font': {'size': 18, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
                margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
                xaxis={'gridcolor': '#f0f0f0', 'title': 'Total Count'},
                yaxis={'gridcolor': '#f0f0f0', 'title': '', 'categoryorder': 'total ascending'}
            )
            bar_fig.update_traces(marker_color='#1f77b4', hovertemplate='%{y}: %{x:,}<extra></extra>')
            graphs.append(html.Div(dcc.Graph(figure=bar_fig), className="card-container", style=card_style))

    return html.Div(graphs)


# -------------------------
# Callbacks for State Wise Tab
# -------------------------

# Callback to update the State Map based on selected category
@app.callback(
    Output("state-map", "figure"),
    [Input("state-category-radio", "value")]
)
def update_state_map(selected_category):
    """Updates the main state choropleth map based on the selected crime category."""
    if selected_category not in dataframes:
        # Return a default empty map or a map indicating an error
        fig = go.Figure()
        fig.update_layout(title=f"Error: Data for '{selected_category}' not found.", title_x=0.5)
        fig.update_geos(visible=False) # Hide geo outlines
        return fig

    df = dataframes[selected_category]

    # Aggregate data by state for the selected category's total crimes
    # Use all years available in the dataframe for the map aggregation
    state_summary = df.groupby("STATE")["TOTAL_CRIMES"].sum().reset_index()

    # Check if GeoJSON is loaded
    if not india_state_geo or 'features' not in india_state_geo:
         fig = go.Figure()
         fig.update_layout(title="Error: State GeoJSON data could not be loaded.", title_x=0.5)
         fig.update_geos(visible=False)
         return fig


    # Determine color scale range dynamically (optional, but can improve contrast)
    # You might want to cap the max value or use quantiles for better color distribution
    min_z = 0
    max_z = state_summary["TOTAL_CRIMES"].quantile(0.95) # Use 95th percentile as max to handle outliers
    if max_z == 0: max_z = state_summary["TOTAL_CRIMES"].max() # Handle case where 95th percentile is 0


    fig = go.Figure(go.Choropleth(
        geojson=india_state_geo,
        locations=state_summary["STATE"],
        z=state_summary["TOTAL_CRIMES"],
        featureidkey="properties.STATE_UPPER", # Key in GeoJSON properties
        colorscale="Viridis",
        marker_line_color="#d4d4d4",
        marker_line_width=0.5,
        zmin=min_z,
        zmax=max_z, # Use calculated max
        zauto=False, # Disable auto-scaling of z
        colorbar_title=f"Total {selected_category.upper()} Crimes<br>(All Years)",
        customdata=state_summary["STATE"], # Pass state name for hover
        hovertemplate = '<b>State:</b> %{customdata}<br>' +
                        f'<b>Total {selected_category.upper()} Crimes:</b> %{{z:,}}<br>' + # Add category name and formatting
                        '<extra></extra>' # Remove trace info
    ))

    fig.update_geos(
        visible=False, # Start with base map invisible, fitbounds will make it visible
        scope="asia",
        projection_type="mercator",
        # center={"lat": 22, "lon": 80}, # Centering might not be needed with fitbounds
        lataxis_range=[5, 38], # Approximate latitude range for India
        lonaxis_range=[67, 99], # Approximate longitude range for India
        bgcolor='rgba(0,0,0,0)', # Transparent background for geos
        fitbounds="locations" # Fit map to the locations with data
    )

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff", # Ensure plot background is white
        margin={"r":10, "t":40, "l":10, "b":10},
        title={
            "text": f"Aggregate {selected_category.upper()} Crimes by State (All Years)",
            "font": {"size": 18, "color": "#333333"},
            "x": 0.5,
            "xanchor": "center"
        },
        geo=dict(bgcolor= 'rgba(0,0,0,0)') # Ensure geo background remains transparent
    )

    return fig

# Callback to handle state selection from the map click
@app.callback(
    [Output("selected-state-store", "data"),
     Output("selected-state-display", "children"),
     Output("state-analysis-options", "children")], # Populate the options container
    [Input("state-map", "clickData"),
     Input("state-category-radio", "value")], # Also depends on category now
)
def state_selected(clickData, selected_category):
    """Handles clicks on the state map to select a state and show further analysis options."""
    # Use dash.callback_context to check which input triggered the callback
    # ctx = dash.callback_context
    # triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # We only want to update options when a map click happens
    # If the category changes, we don't necessarily reset the selection, but the analysis below will update
    if not clickData:
        # If no click, return default states (no state selected, empty options)
        # Keep previously selected state if category changes? Or reset? Resetting seems clearer.
        return None, "Click on a state in the map to see details.", ""

    state_val = clickData["points"][0]["location"]
    display_text = f"Selected State: {state_val} (Category: {selected_category.upper()})"

    # Get the relevant dataframe
    df = dataframes.get(selected_category)
    if df is None or state_val not in df['STATE'].unique():
        # Handle case where state exists in GeoJSON but not in the selected category's data
        options_content = html.Div(f"No data available for {state_val} in the {selected_category.upper()} dataset.", style={'color':'orange'})
        return state_val, display_text, options_content # Store state but show message

    # Get available years for the selected state and category
    years_available = sorted(df[df["STATE"] == state_val]["YEAR"].unique())
    if not years_available:
         options_content = html.Div(f"No yearly data found for {state_val} in the {selected_category.upper()} dataset.", style={'color':'orange'})
         return state_val, display_text, options_content

    min_yr, max_yr = int(min(years_available)), int(max(years_available))

    # Create marks for the RangeSlider, handle single year case
    if min_yr == max_yr:
         marks = {min_yr: str(min_yr)}
         slider_value = [min_yr, max_yr]
    else:
         marks = {int(year): {'label': str(year), 'style': {'transform': 'rotate(45deg)', 'color': '#1f77b4', 'whiteSpace': 'nowrap'}}
                 for year in years_available}
         slider_value = [min_yr, max_yr] # Default to full range for the state


    # Create analysis options: Year Slider and Crime Type Dropdown
    year_slider = dcc.RangeSlider(
        id="state-year-slider", # New ID
        min=min_yr,
        max=max_yr,
        value=slider_value,
        marks=marks,
        step=1 if min_yr != max_yr else None, # Allow step=1 if range exists
        allowCross=False,
        tooltip={"placement": "bottom", "always_visible": True}
    )

    # Crime type dropdown options for the selected category
    state_crime_options = [{"label": "TOTAL CRIMES", "value": "TOTAL_CRIMES"}]
    state_crime_options.extend([{"label": x.replace("_", " ").title(), "value": x} for x in crime_options.get(selected_category, [])])

    crime_type_dd = dcc.Dropdown(
        id="state-crime-type-dropdown", # New ID
        options=state_crime_options,
        value="TOTAL_CRIMES", # Default to total
        clearable=False,
        style=dropdown_style # Use common style
    )

    # Assemble the options div
    options_content = html.Div([
        html.H4(f"Analyze {state_val} ({selected_category.upper()})", style={'textAlign':'center', 'marginBottom':'15px'}),
        html.Label("Select Year Range:", style={'fontWeight':'bold'}), year_slider,
        html.Label("Select Crime Type:", style={'fontWeight':'bold', 'marginTop':'15px'}), crime_type_dd
    ])

    return state_val, display_text, options_content


# Callback to update state-specific visualizations based on selections in the options div
@app.callback(
    Output("state-visualizations-container", "children"),
    [Input("state-year-slider", "value"),         # From the dynamically created slider
     Input("state-crime-type-dropdown", "value")], # From the dynamically created dropdown
    [State("selected-state-store", "data"),      # The currently selected state
     State("state-category-radio", "value")],    # The overall category selected
)
def update_state_visualizations(year_range, crime_type, selected_state, selected_category):
    """Generates visualizations (bar/line charts) for the selected state, category, years, and crime type."""
    # print(f"State Viz Update: State={selected_state}, Cat={selected_category}, Year={year_range}, Crime={crime_type}") # Debug print

    if not selected_state or not year_range or not crime_type or not selected_category:
        return "" # Return empty if any input is missing

    df = dataframes.get(selected_category)
    if df is None:
        return html.Div(f"Error: Data for category '{selected_category}' not found.", style={'color': 'red', 'textAlign':'center'})

    # Filter data for the selected state and years
    min_year, max_year = year_range
    years = [str(year) for year in range(min_year, max_year + 1)]
    df_state_years = df[(df["STATE"] == selected_state) & (df["YEAR"].isin(years))].copy()

    if df_state_years.empty:
        return html.Div(f"No {selected_category.upper()} data available for {selected_state} between {min_year} and {max_year}.",
                       style={'padding': '10px', 'color': '#777777', 'fontStyle': 'italic', 'textAlign':'center'})

    # Check if the selected crime type exists
    target_crime_col = crime_type
    if target_crime_col not in df_state_years.columns:
         # TOTAL_CRIMES should always exist due to preprocessing
         return html.Div(f"Error: Crime type '{target_crime_col}' not found in data for {selected_state}, {selected_category.upper()}. This shouldn't happen.",
                        style={'color': 'red', 'textAlign':'center'})

    graphs = []
    crime_label = target_crime_col.replace('_', ' ').title()
    year_label = f"{min_year}–{max_year}" if min_year != max_year else str(min_year)

    card_style = { # Consistent card styling
        'backgroundColor': '#ffffff', 'borderRadius': '8px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)',
        'padding': '15px', 'marginBottom': '20px'
    }

    # --- Visualization 1: Crime Count by District (Bar Chart) ---
    # Aggregate by district for the selected state/years/crime
    district_agg = df_state_years.groupby("DISTRICT")[target_crime_col].sum().reset_index()
    # Filter out districts with 0 crime for this selection to avoid clutter
    district_agg = district_agg[district_agg[target_crime_col] > 0].sort_values(target_crime_col, ascending=False)

    if not district_agg.empty:
        bar_title = f"{crime_label} by District in {selected_state} ({year_label})"
        bar_fig = px.bar(district_agg.head(20), x="DISTRICT", y=target_crime_col, # Show top 20 districts
                        title=bar_title)
        bar_fig.update_layout(
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
            title={'font': {'size': 16, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
            margin={'l': 40, 'r': 40, 't': 50, 'b': 40},
            xaxis={'gridcolor': '#f0f0f0', 'title': 'District'},
            yaxis={'gridcolor': '#f0f0f0', 'title': f"Total {crime_label}"}
        )
        bar_fig.update_traces(marker_color='#1f77b4', hovertemplate='District: %{x}<br>Total: %{y:,}<extra></extra>')
        graphs.append(html.Div(dcc.Graph(figure=bar_fig), style=card_style))
    else:
         graphs.append(html.Div(f"No districts in {selected_state} reported {crime_label} between {year_label}.", style={**card_style, 'fontStyle':'italic', 'color':'grey'}))


    # --- Visualization 2: Trend Over Selected Years (Line Chart) ---
    # Only show trend line if a range of years is selected
    if min_year != max_year:
        # Aggregate by year for the selected state/crime
        trend_df = df_state_years.groupby("YEAR")[target_crime_col].sum().reset_index()
        trend_df = trend_df.sort_values("YEAR")
        trend_df['YEAR'] = pd.to_numeric(trend_df['YEAR']) # Ensure numeric for plotting

        if len(trend_df) > 1: # Need at least 2 points for a line
            line_title = f"Trend of {crime_label} in {selected_state} ({year_label})"
            line_fig = px.line(trend_df, x="YEAR", y=target_crime_col, title=line_title, markers=True)
            line_fig.update_layout(
                plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
                title={'font': {'size': 16, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
                margin={'l': 40, 'r': 40, 't': 50, 'b': 40},
                xaxis={'gridcolor': '#f0f0f0', 'title': 'Year'},
                yaxis={'gridcolor': '#f0f0f0', 'title': crime_label}
            )
            line_fig.update_traces(line=dict(color='#1f77b4', width=2), hovertemplate='Year: %{x}<br>Total: %{y:,}<extra></extra>')
            graphs.append(html.Div(dcc.Graph(figure=line_fig), style=card_style))
        elif len(graphs) == 0: # If bar chart was also empty, show a message
            graphs.append(html.Div(f"Only one year selected ({min_year}). No trend line to display.", style={**card_style, 'fontStyle':'italic', 'color':'grey'}))


    return html.Div(graphs)

# -------------------------
# Callbacks for District Wise Tab
# -------------------------
@app.callback(
    [Output('district-year-slider', 'min'),
     Output('district-year-slider', 'max'),
     Output('district-year-slider', 'value'),
     Output('district-year-slider', 'marks'),
     Output('district-crime-dropdown', 'options'),
     Output('district-crime-dropdown', 'value'),
     Output('compare-districts-multi', 'options'), # UPDATED: Output for multi-select dropdown
     Output('compare-districts-multi', 'value'),   # UPDATED: Reset multi-select value
     Output('crime-hotspot-dropdown', 'options'),
     Output('crime-hotspot-dropdown', 'value')],
    [Input('district-category-radio', 'value')]
)
def update_district_controls(category):
    """
    Updates year slider, crime dropdown, and district selection dropdowns
    based on the selected category.
    """
    if category not in dataframes:
        # Return default/empty values if category data is missing
        default_marks = {2001: '2001', 2013: '2013'}
        return 2001, 2013, [2001, 2013], default_marks, [], 'TOTAL_CRIMES', [], None, [], None

    df = dataframes[category]

    # Get unique years, convert to int, find min/max
    try:
        years = sorted([int(y) for y in df['YEAR'].unique() if str(y).isdigit()])
        if not years: # Handle case where no valid years found
             raise ValueError("No valid integer years found in data")
        min_yr, max_yr = min(years), max(years)
    except Exception as e:
        print(f"Error processing years for category {category}: {e}")
        # Fallback to default years if processing fails
        min_yr, max_yr = 2001, 2013
        years = list(range(min_yr, max_yr + 1))

    # Create marks for slider
    marks = {yr: {'label': str(yr), 'style': {'transform': 'rotate(45deg)', 'color': '#1f77b4', 'whiteSpace': 'nowrap'}}
             for yr in years if yr % 2 == 0 or yr == min_yr or yr == max_yr} # Mark every other year + ends

    # Crime options for the main dropdown (including TOTAL_CRIMES)
    crime_opts_list = crime_options.get(category, [])
    # Ensure TOTAL_CRIMES column exists before adding it as an option
    crime_opts = []
    if 'TOTAL_CRIMES' in df.columns:
         crime_opts.append({"label": "TOTAL CRIMES", "value": "TOTAL_CRIMES"})
    # Add other specific crimes if they exist in the dataframe
    crime_opts.extend([{"label": x.replace("_", " ").title(), "value": x}
                       for x in crime_opts_list if x in df.columns])

    # Crime options for hotspot analysis (excluding TOTAL_CRIMES)
    hotspot_crime_opts = [{"label": x.replace("_", " ").title(), "value": x}
                          for x in crime_opts_list if x in df.columns] # Only include if column exists

    # District options for the multi-select comparison dropdown
    district_opts = []
    if 'DISTRICT' in df.columns:
        district_opts = [{'label': d, 'value': d} for d in sorted(df['DISTRICT'].unique())]

    # Default crime value: TOTAL_CRIMES if available, else the first specific crime, or None
    default_crime_value = 'TOTAL_CRIMES' if 'TOTAL_CRIMES' in df.columns else (crime_opts[0]['value'] if crime_opts else None)


    return min_yr, max_yr, [min_yr, max_yr], marks, crime_opts, default_crime_value, district_opts, [], hotspot_crime_opts, None # Return empty list [] for multi-select value


# Callback to update the District Map (MODIFIED to handle missing districts)
@app.callback(
    Output("district-map", "figure"),
    [Input("district-category-radio", "value"),
     Input("district-year-slider", "value"),
     Input("district-crime-dropdown", "value")]
)
def update_district_map_detailed(selected_category, year_range, selected_crime):
    """
    Updates the district choropleth map based on category, selected years,
    and specific crime type. Ensures all districts from GeoJSON are shown.
    """
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Prevent update if GeoJSON is missing
    if not india_district_geo or 'features' not in india_district_geo:
         return go.Figure().update_layout(title="Error: District GeoJSON data could not be loaded.", title_x=0.5, xaxis_visible=False, yaxis_visible=False)

    # Get all district names from GeoJSON properties
    # Ensure the key 'DISTRICT_UPPER' matches your normalize_geojson function output
    try:
        all_geojson_districts = [
            feat['properties']['DISTRICT_UPPER']
            for feat in india_district_geo['features']
            if 'properties' in feat and 'DISTRICT_UPPER' in feat['properties']
        ]
        if not all_geojson_districts:
             raise ValueError("No districts found in GeoJSON features properties.")
        # Create a DataFrame from GeoJSON districts for merging
        geojson_districts_df = pd.DataFrame({'DISTRICT': all_geojson_districts})
    except Exception as e:
        print(f"Error extracting districts from GeoJSON: {e}")
        return go.Figure().update_layout(title="Error processing GeoJSON district names.", title_x=0.5, xaxis_visible=False, yaxis_visible=False)


    # Basic validation for inputs
    if not selected_category or not year_range or not selected_crime:
        # If controls haven't fully initialized, show a blank map with all districts (value 0)
        print("Map Update: Controls not fully initialized, showing blank map.")
        fig = go.Figure(go.Choropleth(
            geojson=india_district_geo,
            locations=geojson_districts_df["DISTRICT"],
            z=pd.Series([0] * len(geojson_districts_df)), # Assign 0 to all
            featureidkey="properties.DISTRICT_UPPER",
            colorscale=[[0, 'rgb(240,240,240)'], [1, 'rgb(240,240,240)']], # Use a single light grey color
            colorbar_title="No Data Selected",
            showscale=False, # Hide scale for blank map
            customdata=geojson_districts_df["DISTRICT"],
            hovertemplate = '<b>District:</b> %{customdata}<br>No Data Selected<extra></extra>'
        ))
        fig.update_geos(visible=False, scope="asia", fitbounds="locations", bgcolor='rgba(0,0,0,0)')
        fig.update_layout(title="Select Category, Year, and Crime", title_x=0.5, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", margin={"r":10, "t":40, "l":10, "b":10}, geo=dict(bgcolor= 'rgba(0,0,0,0)'))
        return fig


    if selected_category not in dataframes:
        return go.Figure().update_layout(title=f"Error: Data for '{selected_category}' not found.", title_x=0.5, xaxis_visible=False, yaxis_visible=False)

    df = dataframes[selected_category]
    min_year, max_year = year_range
    years = [str(year) for year in range(min_year, max_year + 1)]

    # Filter by year first
    df_filtered = df[df["YEAR"].isin(years)].copy()

    # Determine the crime column to aggregate
    crime_col = selected_crime # Already defaults to TOTAL_CRIMES from previous callback

    if crime_col not in df_filtered.columns:
        # This case should be less likely now due to checks in update_district_controls
        print(f"Warning: Column '{crime_col}' not found for map. Check data consistency.")
        # Show blank map if column is unexpectedly missing
        fig = go.Figure(go.Choropleth(
            geojson=india_district_geo, locations=geojson_districts_df["DISTRICT"], z=pd.Series([0] * len(geojson_districts_df)),
            featureidkey="properties.DISTRICT_UPPER", colorscale=[[0, 'rgb(240,240,240)'], [1, 'rgb(240,240,240)']],
            showscale=False, customdata=geojson_districts_df["DISTRICT"], hovertemplate = '<b>District:</b> %{customdata}<br>Data Error<extra></extra>'
        ))
        fig.update_geos(visible=False, scope="asia", fitbounds="locations", bgcolor='rgba(0,0,0,0)')
        fig.update_layout(title=f"Error: Column '{crime_col}' missing", title_x=0.5, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", margin={"r":10, "t":40, "l":10, "b":10}, geo=dict(bgcolor= 'rgba(0,0,0,0)'))
        return fig

    # Aggregate data by district for the selected crime column and years
    district_summary = df_filtered.groupby("DISTRICT")[crime_col].sum().reset_index()

    # --- Merge with GeoJSON districts ---
    # Perform a left merge: keep all districts from geojson_districts_df
    # Use the 'DISTRICT' column name assumed to be consistent after cleaning/fuzzy matching
    merged_data = pd.merge(geojson_districts_df, district_summary, on='DISTRICT', how='left')

    # Fill NaN values in the crime column with 0 for districts present in GeoJSON but not in data
    merged_data[crime_col] = merged_data[crime_col].fillna(0).astype(int) # Ensure integer type

    # --- Create Map using merged_data ---
    if merged_data.empty:
         # This case is unlikely if geojson_districts_df is populated
         return go.Figure().update_layout(title="Error creating merged map data.", title_x=0.5, xaxis_visible=False, yaxis_visible=False)

    # Dynamic Z range based on merged data (excluding potential massive outliers if needed)
    # Use quantiles on the non-zero values if there are many zeros, otherwise use max
    non_zero_values = merged_data[merged_data[crime_col] > 0][crime_col]
    if not non_zero_values.empty:
        max_z = non_zero_values.quantile(0.98) if len(non_zero_values) > 10 else non_zero_values.max()
        min_z = 0 # Always start from 0
        if max_z == 0: max_z = 1 # Avoid max_z being 0 if only zeros exist after filtering
    else:
        min_z, max_z = 0, 1 # Default range if all values are 0

    crime_label_disp = crime_col.replace("_", " ").title()
    map_title = f"{crime_label_disp} ({selected_category.upper()}) by District ({min_year}–{max_year})"
    colorbar_title_text = f"{crime_label_disp}<br>({min_year}–{max_year})"

    fig = go.Figure(go.Choropleth(
        geojson=india_district_geo,
        locations=merged_data["DISTRICT"], # Use the DISTRICT column from merged data
        z=merged_data[crime_col],          # Use the crime column (with zeros filled)
        featureidkey="properties.DISTRICT_UPPER", # Key in GeoJSON properties
        colorscale="Viridis",              # Or choose another scale like "Blues" or "Reds"
        reversescale=False,                # Adjust as needed
        marker_line_color="#d4d4d4",
        marker_line_width=0.2,
        zmin=min_z,
        zmax=max_z,
        zauto=False,
        colorbar_title=colorbar_title_text,
        customdata=merged_data[["DISTRICT", crime_col]], # Pass district and value
        hovertemplate = '<b>District:</b> %{customdata[0]}<br>' +
                        f'<b>{crime_label_disp}:</b> %{{customdata[1]:,}}<br>' + # Use customdata for value
                        '<extra></extra>'
    ))

    fig.update_geos(
        visible=False, scope="asia", projection_type="mercator",
        lataxis_range=[5, 38], lonaxis_range=[67, 99], # Adjust ranges if needed
        bgcolor='rgba(0,0,0,0)', fitbounds="locations"
    )

    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin={"r":10, "t":40, "l":10, "b":10},
        title={"text": map_title, "font": {"size": 18, "color": "#333333"}, "x": 0.5, "xanchor": "center"},
        geo=dict(bgcolor= 'rgba(0,0,0,0)')
    )

    return fig


# Callback to handle district selection from the map click (Populates details section trigger)
# (Keep this callback as it was - it just stores the clicked district)
@app.callback(
    [Output("selected-district-store", "data"),
     Output("selected-district-display", "children")],
    [Input("district-map", "clickData")],
    [State("district-category-radio", "value")], # Need category to provide context
    prevent_initial_call=True
)
def district_map_click_handler(clickData, selected_category):
    """Handles clicks on the district map ONLY to store the selected district and update display."""
    if not clickData:
        return no_update, no_update # Keep existing selection

    # Extract district name from the clicked feature's location property
    # Check if customdata is available (it should be from the updated map callback)
    if 'customdata' in clickData['points'][0]:
         district_val = clickData['points'][0]['customdata'][0] # Get district name from customdata
    else:
         district_val = clickData["points"][0]["location"] # Fallback to location if customdata missing

    # Find the state(s) this district belongs to in the data for context
    df = dataframes.get(selected_category)
    state_info = ""
    if df is not None and 'DISTRICT' in df.columns and 'STATE' in df.columns:
        possible_states = df[df['DISTRICT'] == district_val]['STATE'].unique()
        if len(possible_states) == 1:
            state_info = f" (State: {possible_states[0]})"
        elif len(possible_states) > 1:
            # This might happen if district names are reused across states, show them all
            state_info = f" (States: {', '.join(possible_states)})"


    display_text = f"Selected for Details: {district_val}{state_info}"
    # Store a dictionary with district and category for the detail callback
    selected_data = {'district': district_val, 'category': selected_category}

    return selected_data, display_text


# Callback to update district-specific detail visualizations based on stored selection
# (Keep this callback as it was)
@app.callback(
    Output("district-detail-graphs", "children"),
    [Input("selected-district-store", "data")], # Triggered when stored data changes
    prevent_initial_call=True
)
def update_district_detail_visualizations(selected_data):
    """Generates detailed visualizations (time series, crime breakdown) for the selected district."""
    if not selected_data or 'district' not in selected_data or 'category' not in selected_data:
        return html.Div("Click on a district in the map to view detailed analysis.", style={'padding': '20px', 'textAlign': 'center', 'color': 'grey'})

    selected_district = selected_data['district']
    selected_category = selected_data['category']

    df = dataframes.get(selected_category)
    if df is None:
        return html.Div(f"Error: Data for category '{selected_category}' not found.", style={'color': 'red', 'textAlign':'center'})

    # Filter data for the selected district (across all available years in the dataset)
    df_district = df[df["DISTRICT"] == selected_district].copy()

    if df_district.empty:
        return html.Div(f"No data available for {selected_district} in the {selected_category.upper()} dataset.",
                       style={'padding': '10px', 'color': '#777777', 'fontStyle': 'italic', 'textAlign':'center'})

    # Find state for context
    possible_states = df_district['STATE'].unique()
    state_info = f" (State: {possible_states[0]})" if len(possible_states) == 1 else f" (Aggregated across States: {', '.join(possible_states)})" if len(possible_states) > 1 else ""

    graphs = []
    # Using the globally defined card_style

    # --- Visualization 1: Time Series of TOTAL CRIMES ---
    if 'TOTAL_CRIMES' in df_district.columns:
        trend_df = df_district.groupby("YEAR")["TOTAL_CRIMES"].sum().reset_index()
        try: # Add error handling for year conversion
            trend_df['YEAR'] = pd.to_numeric(trend_df['YEAR']) # Ensure numeric for sorting/plotting
            trend_df = trend_df.sort_values("YEAR")
        except ValueError:
             print(f"Warning: Could not convert YEAR to numeric for district {selected_district} trend.")
             trend_df = pd.DataFrame() # Set to empty to skip plot

        if not trend_df.empty and trend_df["TOTAL_CRIMES"].sum() > 0:
            line_title = f"Total {selected_category.upper()} Crimes Trend in {selected_district}{state_info}"
            fig_trend = px.line(trend_df, x="YEAR", y="TOTAL_CRIMES", title=line_title, markers=True)
            fig_trend.update_layout(
                plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
                title={'font': {'size': 16, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
                margin={'l': 40, 'r': 40, 't': 50, 'b': 40},
                xaxis={'gridcolor': '#f0f0f0', 'title': 'Year'},
                yaxis={'gridcolor': '#f0f0f0', 'title': 'Total Crimes'}
            )
            fig_trend.update_traces(line=dict(color='#1f77b4', width=2), hovertemplate='Year: %{x}<br>Total: %{y:,}<extra></extra>')
            graphs.append(html.Div(dcc.Graph(figure=fig_trend), style=card_style))
        else:
            graphs.append(html.Div(f"No 'Total Crimes' data to display for {selected_district}.", style={**card_style, 'fontStyle':'italic', 'color':'grey'}))
    else:
         graphs.append(html.Div(f"'Total Crimes' column not found for {selected_district}.", style={**card_style, 'fontStyle':'italic', 'color':'orange'}))


    # --- Visualization 2: Breakdown by Specific Crime Type (Pie Chart for latest year or aggregated) ---
    specific_crimes = crime_options.get(selected_category, [])
    valid_crimes = [c for c in specific_crimes if c in df_district.columns]

    if valid_crimes:
        # Aggregate across all years for the selected district
        breakdown_data = df_district[valid_crimes].sum().reset_index()
        breakdown_data.columns = ['CRIME_TYPE', 'TOTAL_COUNT']
        breakdown_data = breakdown_data[breakdown_data['TOTAL_COUNT'] > 0] # Only show crimes with counts > 0
        breakdown_data = breakdown_data.sort_values('TOTAL_COUNT', ascending=False)
        breakdown_data['CRIME_LABEL'] = breakdown_data['CRIME_TYPE'].str.replace('_', ' ').str.title()

        if not breakdown_data.empty:
            pie_title = f"Breakdown of {selected_category.upper()} Crime Types in {selected_district}{state_info} (All Years)"
            fig_pie = px.pie(breakdown_data.head(10), values='TOTAL_COUNT', names='CRIME_LABEL', # Show top 10 types
                            title=pie_title, hole=0.3)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                                 hovertemplate='<b>%{label}</b><br>Total Count: %{value:,}<br>Percentage: %{percent:.1%}<extra></extra>')
            fig_pie.update_layout(
                plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
                title={'font': {'size': 16, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
                margin={'l': 40, 'r': 40, 't': 50, 'b': 40},
                legend_title_text='Crime Type (Top 10)',
                legend={'orientation': 'v', 'yanchor':'top', 'y':0.7, 'xanchor':'left', 'x':-0.1}
            )
            graphs.append(html.Div(dcc.Graph(figure=fig_pie), style=card_style))
        else:
             graphs.append(html.Div(f"No specific crime breakdown data to display for {selected_district}.", style={**card_style, 'fontStyle':'italic', 'color':'grey'}))


    return html.Div(graphs)


# Callback for District Comparison (MODIFIED for multi-select and specific crime)
@app.callback(
    Output('district-comparison-graphs', 'children'),
    [Input('compare-districts-multi', 'value')], # UPDATED: Input from multi-select dropdown
    [State('district-category-radio', 'value'),
     State('district-year-slider', 'value'),
     State('district-crime-dropdown', 'value')], # Use the main crime dropdown value
    prevent_initial_call=True
)
def update_district_comparison(selected_districts, category, year_range, crime_type):
    """
    Compares multiple selected districts based on the main controls
    (category, year range, and selected crime type).
    """
    # Validate selections
    if not selected_districts:
        return html.Div("Select two or more districts to compare.", style={'padding': '20px', 'textAlign': 'center', 'color': 'grey'})
    if len(selected_districts) < 2:
        return html.Div("Please select at least two districts for comparison.", style={'padding': '20px', 'textAlign': 'center', 'color': 'orange'})

    if not category or not year_range or not crime_type:
         return html.Div("Please select category, year range, and crime type.", style={'padding': '20px', 'textAlign': 'center', 'color': 'orange'})

    df = dataframes.get(category)
    if df is None:
        return html.Div(f"Error loading data for category {category}.", style={'color': 'red', 'textAlign':'center'})

    min_year, max_year = year_range
    years = [str(year) for year in range(min_year, max_year + 1)]

    # Use the crime column selected in the main dropdown
    crime_col = crime_type # Already defaults to TOTAL_CRIMES if nothing else selected
    if crime_col not in df.columns:
         return html.Div(f"Error: Cannot find comparison column '{crime_col}'.", style={'color': 'red', 'textAlign':'center'})

    # Filter data for the selected districts and years
    df_comp = df[df['DISTRICT'].isin(selected_districts) & df['YEAR'].isin(years)].copy()

    if df_comp.empty:
        return html.Div(f"No data found for the selected districts in the specified category/years.", style={'padding': '20px', 'textAlign': 'center', 'color': 'grey'})

    # Aggregate by District and Year
    comp_agg = df_comp.groupby(['DISTRICT', 'YEAR'])[crime_col].sum().reset_index()
    try: # Add error handling for year conversion
        comp_agg['YEAR'] = pd.to_numeric(comp_agg['YEAR']) # Ensure numeric year
        comp_agg = comp_agg.sort_values(['DISTRICT', 'YEAR'])
    except ValueError:
        print(f"Warning: Could not convert YEAR to numeric for district comparison.")
        # Decide how to handle - maybe return error or try to proceed without sorting?
        # For now, let's return an error message.
        return html.Div("Error converting Year data for comparison.", style={'color': 'red', 'textAlign':'center'})


    graphs = []
    crime_label = crime_col.replace("_", " ").title()
    year_label = f"{min_year}–{max_year}"
    # Using the globally defined card_style

    # --- Visualization 1: Trend Lines for all selected districts ---
    if len(years) > 1 and not comp_agg.empty: # Only show trend if more than one year selected and data exists
        title_districts = ", ".join(selected_districts)
        fig_trend_comp = px.line(comp_agg, x='YEAR', y=crime_col, color='DISTRICT',
                                 title=f"{crime_label} Trend: {title_districts} ({year_label})",
                                 markers=True,
                                 color_discrete_sequence=px.colors.qualitative.Plotly) # Use a qualitative color scale
        fig_trend_comp.update_layout(
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
            title={'font': {'size': 16, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
            margin={'l': 40, 'r': 40, 't': 50, 'b': 40},
            xaxis={'gridcolor': '#f0f0f0', 'title': 'Year'},
            yaxis={'gridcolor': '#f0f0f0', 'title': crime_label},
            legend_title_text='District'
        )
        fig_trend_comp.update_traces(hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Count: %{y:,}<extra></extra>')
        graphs.append(html.Div(dcc.Graph(figure=fig_trend_comp), style=card_style))
    elif len(years) == 1:
         graphs.append(html.Div("Trend line requires selecting more than one year.", style={**card_style, 'fontStyle':'italic', 'color':'grey'}))


    # --- Visualization 2: Bar Chart comparing total over the period ---
    total_comp = comp_agg.groupby('DISTRICT')[crime_col].sum().reset_index()
    # Ensure the order matches the selection order if possible, or sort alphabetically/by value
    total_comp = total_comp.sort_values(crime_col, ascending=False) # Sort by value descending

    if not total_comp.empty:
        fig_bar_comp = px.bar(total_comp, x='DISTRICT', y=crime_col, color='DISTRICT',
                              title=f"Total {crime_label} Comparison ({year_label})",
                              text=crime_col,
                              color_discrete_sequence=px.colors.qualitative.Plotly) # Match line colors
        fig_bar_comp.update_layout(
            plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
            title={'font': {'size': 16, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
            margin={'l': 40, 'r': 40, 't': 50, 'b': 40},
            xaxis={'gridcolor': '#f0f0f0', 'title': 'District', 'categoryorder':'total descending'}, # Order bars by value
            yaxis={'gridcolor': '#f0f0f0', 'title': f"Total {crime_label}"},
            showlegend=False # Colors match bars, legend redundant
        )
        fig_bar_comp.update_traces(texttemplate='%{text:,}', textposition='outside',
                                  hovertemplate='<b>%{x}</b><br>Total: %{y:,}<extra></extra>')
        graphs.append(html.Div(dcc.Graph(figure=fig_bar_comp), style=card_style))

    if not graphs: # If neither plot could be generated
         return html.Div("Could not generate comparison plots with the selected data.", style={'padding': '20px', 'textAlign': 'center', 'color': 'orange'})


    return html.Div(graphs)


# Callback for Crime Hotspot Analysis
# (Keep this callback as it was - it uses its own crime dropdown)
@app.callback(
    Output('crime-comparison-graphs', 'children'),
    [Input('crime-hotspot-dropdown', 'value'),
     Input('crime-hotspot-top-n-slider', 'value')],
    [State('district-category-radio', 'value'),
     State('district-year-slider', 'value')],
    prevent_initial_call=True
)
def update_crime_hotspots(selected_crime, top_n, category, year_range):
    """Shows the top N districts for a specific crime in the selected category/years."""
    if not selected_crime:
        return html.Div("Select a specific crime type for hotspot analysis.", style={'padding': '20px', 'textAlign': 'center', 'color': 'grey'})

    if not category or not year_range:
         return html.Div("Please select category and year range.", style={'padding': '20px', 'textAlign': 'center', 'color': 'orange'})

    df = dataframes.get(category)
    if df is None:
        return html.Div(f"Error loading data for category {category}.", style={'color': 'red', 'textAlign':'center'})

    min_year, max_year = year_range
    years = [str(year) for year in range(min_year, max_year + 1)]

    # Check if selected crime column exists
    if selected_crime not in df.columns:
         return html.Div(f"Error: Crime column '{selected_crime}' not found for category {category}.", style={'color': 'red', 'textAlign':'center'})

    # Filter data
    df_filtered = df[df['YEAR'].isin(years)].copy()
    if df_filtered.empty:
         return html.Div(f"No data found for {category.upper()} between {min_year}-{max_year}", style={'padding': '20px', 'textAlign': 'center', 'color': 'grey'})

    # Aggregate by district for the specific crime
    hotspot_agg = df_filtered.groupby('DISTRICT')[selected_crime].sum().reset_index()
    hotspot_agg = hotspot_agg[hotspot_agg[selected_crime] > 0] # Filter out zero counts
    hotspot_agg = hotspot_agg.sort_values(selected_crime, ascending=False).head(top_n)

    if hotspot_agg.empty:
        crime_label = selected_crime.replace("_", " ").title()
        return html.Div(f"No districts reported {crime_label} between {min_year}-{max_year}.", style={'padding': '20px', 'textAlign': 'center', 'color': 'grey'})

    graphs = []
    crime_label = selected_crime.replace("_", " ").title()
    year_label = f"{min_year}–{max_year}"
    # Using global card_style

    # --- Visualization: Bar Chart of Top N Districts ---
    fig_hotspot = px.bar(hotspot_agg, x='DISTRICT', y=selected_crime,
                         title=f"Top {top_n} Districts for {crime_label} ({year_label}) - (one is TOTAL)",
                         text=selected_crime)
    fig_hotspot.update_layout(
        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
        title={'font': {'size': 16, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
        margin={'l': 40, 'r': 40, 't': 50, 'b': 40},
        xaxis={'gridcolor': '#f0f0f0', 'title': 'District', 'categoryorder':'total descending'}, # Keep sorted by value
        yaxis={'gridcolor': '#f0f0f0', 'title': f"Total {crime_label}"}
    )
    fig_hotspot.update_traces(marker_color='#d62728', # Use a distinct color for hotspots
                             texttemplate='%{text:,}', textposition='outside',
                             hovertemplate='<b>%{x}</b><br>Count: %{y:,}<extra></extra>')
    graphs.append(html.Div(dcc.Graph(figure=fig_hotspot), style=card_style))

    return html.Div(graphs)


# -------------------------
# Other Callbacks (Area Comparison, Place Occurrence, Murder Flow, Relatives, Clusters)
# Keep these as they were, ensuring they use standardized column names if necessary.
# Note: Clustering tab specifically uses IPC data (df_ipc) and its features.
# -------------------------

# --- Callback for Area Comparison Radar Plot ---
@app.callback(
    Output('area-comparison-radar-plot', 'figure'),
    [Input('compare-area-dropdown-1', 'value'),
     Input('compare-area-dropdown-2', 'value')]
)
def update_area_comparison_radar(selected_area1, selected_area2):
    # Use the potentially standardized area_col and comparison_cols names
    # (standardization happens during data loading now)
    if not selected_area1 or not selected_area2 or df_aggregated.empty or not comparison_cols:
        fig = go.Figure()
        fig.update_layout(title="Please select two areas (ensure firearm data loaded)", title_x=0.5, xaxis_visible=False, yaxis_visible=False)
        return fig

    if selected_area1 == selected_area2:
        fig = go.Figure()
        fig.update_layout(title="Please select two different areas", title_x=0.5, xaxis_visible=False, yaxis_visible=False)
        return fig

    # Filter aggregated data using standardized area_col
    data_area1 = df_aggregated[df_aggregated[area_col] == selected_area1]
    data_area2 = df_aggregated[df_aggregated[area_col] == selected_area2]

    if data_area1.empty or data_area2.empty:
        missing = [area for area, d in zip([selected_area1, selected_area2], [data_area1, data_area2]) if d.empty]
        fig = go.Figure()
        fig.update_layout(title=f"Aggregated firearm data not found for: {', '.join(missing)}", title_x=0.5, xaxis_visible=False, yaxis_visible=False)
        return fig

    # Extract values using standardized comparison_cols
    values1 = data_area1[comparison_cols].iloc[0].values.tolist()
    values2 = data_area2[comparison_cols].iloc[0].values.tolist()

    # Format categories from standardized column names
    categories = [col.replace('_', ' ').replace('BY ', '').replace('VICTIMS ', '').title() for col in comparison_cols] # Cleaner labels


    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values1, theta=categories, fill='toself', name=selected_area1,
                                   hovertemplate=f'<b>{selected_area1}</b><br>%{{theta}}: %{{r:,}}<extra></extra>'))
    fig.add_trace(go.Scatterpolar(r=values2, theta=categories, fill='toself', name=selected_area2,
                                   hovertemplate=f'<b>{selected_area2}</b><br>%{{theta}}: %{{r:,}}<extra></extra>'))

    title_cols_str = ', '.join(categories) # Use cleaned categories for title
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, tickformat=',')),
        title=f"Aggregated Firearm Use ({title_cols_str}):<br>{selected_area1} vs {selected_area2}",
        title_x=0.5, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=100, b=100)
    )
    return fig

# --- Callback for Place Occurrence ---
@app.callback(
    [Output('place-sunburst','figure'),
     Output('place-small-multiples','figure')],
    [Input('place-year-slider','value')]
)
def update_place_viz(year_range):
    yrs = [str(y) for y in range(year_range[0], year_range[1]+1)]
    # Ensure COUNT is numeric
    df_place_long['COUNT'] = pd.to_numeric(df_place_long['COUNT'], errors='coerce').fillna(0)
    dfp = df_place_long[df_place_long['YEAR'].isin(yrs)]
    dfp = dfp[dfp['COUNT'] > 0] # Filter out zero counts for cleaner viz

    if dfp.empty:
         empty_fig = go.Figure().update_layout(title=f"No Place of Occurrence data found for {year_range[0]}-{year_range[1]}", xaxis_visible=False, yaxis_visible=False, title_x=0.5)
         return empty_fig, empty_fig

    # Sunburst
    sb_fig = px.sunburst(
        dfp, path=['YEAR','PLACE','CRIME'], values='COUNT',
        title='Crime by Place Occurrence (Year → Place → Crime)',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        maxdepth=2 # Limit initial depth? Or -1 for full depth
    )
    sb_fig.update_layout(margin=dict(t=50, l=0, r=0, b=0), title_x=0.5)
    sb_fig.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value:,}<extra></extra>')


    # Small multiples (Grouped Bar)
    # Aggregate first to avoid overly dense plots if many years selected
    dfp_agg = dfp.groupby(['PLACE', 'CRIME'])['COUNT'].sum().reset_index()
    # Consider only top N places or crimes if needed
    # top_places = dfp.groupby('PLACE')['COUNT'].sum().nlargest(9).index
    # dfp_filtered = dfp[dfp['PLACE'].isin(top_places)]

    sm_fig = px.bar(
        dfp, # Use original dfp to show year trends within facets
        x='YEAR', y='COUNT', color='CRIME',
        facet_col='PLACE', facet_col_wrap=3, # Max 3 columns
        facet_col_spacing=0.06, facet_row_spacing=0.1,
        title=f'Yearly Crime Counts by Place of Occurrence ({year_range[0]}–{year_range[1]})',
        labels={'COUNT': 'Count', 'CRIME': 'Crime Type'},
        category_orders={"YEAR": sorted(yrs)} # Ensure years are sequential
    )
    sm_fig.update_layout(
        # barmode='group', # Group bars by crime type within each year/place
        margin=dict(t=60, l=20, r=20, b=20), showlegend=True,
        legend_title_text='Crime Type', title_x=0.5,
        height=max(600, 200 * ((len(dfp['PLACE'].unique())-1) // 3 + 1)) # Dynamic height
    )
    sm_fig.for_each_yaxis(lambda ax: ax.update(matches=None, showticklabels=True, title='')) # Unlink y-axes, remove title
    sm_fig.for_each_xaxis(lambda ax: ax.update(matches=None, title='Year')) # Unlink x-axes
    sm_fig.update_traces(hovertemplate='Year: %{x}<br>Count: %{y:,}<extra></extra>')


    return sb_fig, sm_fig


# --- Callback for Murder Victims Flow (Sankey) ---
@app.callback(
    Output('sankey-diagram', 'figure'),
    [Input('year-dropdown-new', 'value'),
     Input('states-slider', 'value')],
     prevent_initial_call=True
)
def update_sankey(selected_year, top_n_states):
    # Ensure columns are standardized and numeric
    df_murder_std = df_murder.rename(columns={'Area_Name': 'STATE', 'Year':'YEAR',
                                            'Sub_Group_Name':'SUBGROUP',
                                            'Victims_Above_50_Yrs':'VICTIMS_ADULT', # Grouping example
                                            'Victims_Upto_10_15_18_30_50_Yrs': 'VICTIMS_NON_ADULT'}) # Simplified

    # Combine age groups (Example: Adult vs Non-Adult - adjust logic as needed)
    # This requires careful definition based on available columns
    # For this example, let's assume 'victims adult' and 'victims non adult' columns exist (as in original code)
    # If not, they need to be created by summing appropriate age columns.
    # Assuming 'victims adult' and 'victims non adult' columns are correctly available or created:
    numeric_cols = ['victims adult', 'victims non adult'] # Make sure these columns exist in df_murder
    for col in numeric_cols:
         if col not in df_murder.columns:
             # Handle missing columns - maybe create them or return error figure
             print(f"Warning: Sankey diagram requires column '{col}' which is missing.")
             # As a fallback, let's try summing all victim columns if specific ones are missing
             victim_cols = [c for c in df_murder.columns if c.startswith('Victims_')]
             if len(victim_cols) >= 2:
                 df_murder['victims adult'] = df_murder[[c for c in victim_cols if 'Above' in c or '30_50' in c]].sum(axis=1, skipna=True)
                 df_murder['victims non adult'] = df_murder[[c for c in victim_cols if 'Upto' in c]].sum(axis=1, skipna=True)
                 numeric_cols = ['victims adult', 'victims non adult']
             else:
                 fig = go.Figure().update_layout(title=f"Error: Required victim columns for Sankey missing.", title_x=0.5)
                 return fig
         else:
             df_murder[col] = pd.to_numeric(df_murder[col], errors='coerce').fillna(0)


    year_df = df_murder[df_murder['Year'] == selected_year]
    #print(year_df.head)
    # Get top N states by total victims
    total_victims = year_df.groupby('Area_Name')[numeric_cols].sum()
    total_victims['total'] = total_victims.sum(axis=1)
    # Ensure top_n_states is not greater than available states
    top_n_states = min(top_n_states, len(total_victims))
    top_states = total_victims.nlargest(top_n_states, 'total').index.tolist()


    # Filter for top states and relevant subgroups (assuming Male/Female subgroups exist)
    gender_subgroups = ['1. Male Victims', '2. Female Victims'] # Check actual subgroup names
    # Find actual subgroup names if they differ
    actual_subgroups = [sg for sg in gender_subgroups if sg in year_df['Sub_Group_Name'].unique()]
    if not actual_subgroups:
         # Handle case where expected subgroups are missing
         # Try finding any gender-like subgroups
         potential_genders = [sg for sg in year_df['Sub_Group_Name'].unique() if 'Male' in sg or 'Female' in sg]
         if potential_genders:
             actual_subgroups = potential_genders
             print(f"Warning: Using subgroups {actual_subgroups} for Sankey gender nodes.")
         else:
             fig = go.Figure().update_layout(title=f"Error: Gender subgroups not found for Sankey.", title_x=0.5)
             return fig


    sankey_df = year_df[
        (year_df['Area_Name'].isin(top_states)) &
        (year_df['Sub_Group_Name'].isin(actual_subgroups))
    ]

    if sankey_df.empty:
        fig = go.Figure().update_layout(title=f"No data found for Sankey diagram based on selections.", title_x=0.5)
        return fig

    # --- Sankey Logic ---
    sources, targets, values = [], [], []
    node_labels = []
    node_map = {} # Map label to index

    def get_node_index(label):
        if label not in node_map:
            node_map[label] = len(node_labels)
            node_labels.append(label)
        return node_map[label]

    # Define node types and their colors
    state_nodes = top_states
    gender_nodes = [sg.split('. ')[-1] for sg in actual_subgroups] # Clean labels 'Male Victims' -> 'Male Victims'
    age_nodes = ['Adult', 'Non-Adult']

    node_colors_map = {
        'state': '#a6cee3', # Light Blue
        'gender': '#fdbf6f', # Light Orange (adjust if more than 2 genders)
        'age': '#b2df8a'     # Light Green
    }
    node_colors_list = []

    # 1. State to Gender Links
    for state in state_nodes:
        state_idx = get_node_index(state)
        node_colors_list.append(node_colors_map['state']) # Add color for state node
        for sg_raw, gender_label in zip(actual_subgroups, gender_nodes):
            gender_idx = get_node_index(gender_label)
            if gender_label not in [node_labels[i] for i in range(len(node_labels)-1)]: # Add gender color only once
                 node_colors_list.append(node_colors_map['gender'])

            val = sankey_df[
                (sankey_df['Area_Name'] == state) &
                (sankey_df['Sub_Group_Name'] == sg_raw)
            ][numeric_cols].sum().sum() # Sum adult and non-adult for this state/gender

            if val > 0:
                sources.append(state_idx)
                targets.append(gender_idx)
                values.append(val)

    # 2. Gender to Age Links
    for gender_label in gender_nodes:
        gender_idx = get_node_index(gender_label) # Should exist now
        sg_raw = next(sg for sg, gl in zip(actual_subgroups, gender_nodes) if gl == gender_label) # Find original subgroup name

        # Adult Link
        adult_idx = get_node_index('Adult')
        if 'Adult' not in [node_labels[i] for i in range(len(node_labels)-1)]: node_colors_list.append(node_colors_map['age'])
        adult_val = sankey_df[sankey_df['Sub_Group_Name'] == sg_raw]['victims adult'].sum()
        if adult_val > 0:
            sources.append(gender_idx)
            targets.append(adult_idx)
            values.append(adult_val)

        # Non-Adult Link
        non_adult_idx = get_node_index('Non-Adult')
        if 'Non-Adult' not in [node_labels[i] for i in range(len(node_labels)-1)]: node_colors_list.append(node_colors_map['age'])
        non_adult_val = sankey_df[sankey_df['Sub_Group_Name'] == sg_raw]['victims non adult'].sum()
        if non_adult_val > 0:
            sources.append(gender_idx)
            targets.append(non_adult_idx)
            values.append(non_adult_val)


    # Create Link Colors (optional, can make complex)
    link_colors = ['rgba(166,206,227,0.6)' if node_labels[s] in state_nodes else # State links: semi-transparent blue
                   'rgba(253,191,111,0.6)' if node_labels[s] == 'Male Victims' else # Male links: semi-transparent orange
                   'rgba(251,154,153,0.6)' # Female links: semi-transparent pink (adjust if more genders)
                   for s in sources]


    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors_list # Apply node colors
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=link_colors, # Optional link coloring
            hovertemplate='Flow from %{source.label} to %{target.label}: %{value:,}<extra></extra>' # Add formatting
        )
    )])

    fig.update_layout(
        title_text=f"Murder Victims Flow ({top_n_states} States): State → Gender → Age ({selected_year})",
        font_size=11, height=700, title_x=0.5
    )

    return fig

# --- Callback for Offender Relationships Treemap ---
@app.callback(
    Output('relative-treemap', 'figure'),
    Input('rel-top-n', 'value')
)
def update_relative_treemap(top_n):
    return build_offender_treemap(top_n) # Call the helper function


# --- Updated Callback for Clusters Tab ---
@app.callback(
    [
        Output('cluster-map', 'figure'),
        Output('cluster-centroid-bar', 'figure'),
        Output('cluster-centroid-silhouette', 'figure'),
        Output('cluster-trends-total', 'figure'), # Updated ID
        Output('cluster-trends-avg', 'figure'),   # New Output for avg trend
        Output('cluster-visibility-checklist', 'options'), # Update checklist options
        Output('cluster-visibility-checklist', 'value')    # Update checklist values
    ],
    [
        Input('cluster-features', 'value'),
        Input('cluster-count', 'value'),
        Input('cluster-visibility-checklist', 'value'), # Add checklist value as input
    ],
    # *** CORRECTED ***: Removed State for checklist value as it's an Input
    prevent_initial_call=True
)
def update_clusters(selected_feats, n_clusters, visible_clusters): # visible_clusters is now directly from Input
    """
    Updates all cluster-related plots based on selected crime features,
    the number of clusters, and selected visible clusters for the map.
    Includes feature scaling and ARIMA forecasts for total and average crimes.

    Args:
        selected_feats (list): List of crime features selected by the user.
        n_clusters (int): The number of clusters specified by the user.
        visible_clusters (list): List of cluster indices (int) to show on map (from Input).

    Returns:
        tuple: Updated figures and checklist options/values.
    """
    # --- Identify Trigger ---
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # --- Create Empty/Error Figures ---
    empty_fig = go.Figure().update_layout(
        title='Please select features/clusters',
        xaxis_visible=False, yaxis_visible=False, title_x=0.5,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    error_fig = lambda msg: go.Figure().update_layout(
        title=msg, xaxis_visible=False, yaxis_visible=False, title_x=0.5,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    default_checklist_opts = []
    default_checklist_vals = []

    # --- Input Validation ---
    if not selected_feats:
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, default_checklist_opts, default_checklist_vals
    if n_clusters < 2:
        err_msg = 'Number of clusters must be at least 2.'
        return error_fig(err_msg), error_fig(''), error_fig(''), error_fig(''), error_fig(''), default_checklist_opts, default_checklist_vals

    # Use IPC data (ensure df_ipc is loaded and preprocessed)
    if 'df_ipc' not in globals() or df_ipc.empty:
        err_msg = "IPC Crime data not loaded."
        return error_fig(err_msg), error_fig(''), error_fig(''), error_fig(''), error_fig(''), default_checklist_opts, default_checklist_vals
    df_clust_orig = df_ipc.copy()

    # --- Data Preprocessing ---
    # Ensure DISTRICT column exists
    if 'DISTRICT' not in df_clust_orig.columns:
        err_msg = "DISTRICT column missing in IPC data."
        return error_fig(err_msg), error_fig(''), error_fig(''), error_fig(''), error_fig(''), default_checklist_opts, default_checklist_vals

    # Filter out "TOTAL" districts before aggregation
    df_clust = df_clust_orig[~df_clust_orig['DISTRICT'].str.contains("TOTAL", case=False, na=False)].copy()
    if df_clust.empty:
        err_msg = "No district data remaining after removing 'TOTAL' entries."
        return error_fig(err_msg), error_fig(''), error_fig(''), error_fig(''), error_fig(''), default_checklist_opts, default_checklist_vals


    # Validate selected features
    valid_feats = [f for f in selected_feats if f in df_clust.columns]
    if not valid_feats:
        err_msg = 'Selected features not found in IPC data (after filtering).'
        return error_fig(err_msg), error_fig(''), error_fig(''), error_fig(''), error_fig(''), default_checklist_opts, default_checklist_vals

    # Convert selected features to numeric and aggregate by district (mean)
    for f in valid_feats:
        df_clust[f] = pd.to_numeric(df_clust[f], errors='coerce')

    X_agg = df_clust.groupby('DISTRICT')[valid_feats].mean().fillna(0)

    # Remove districts with zero variance across selected features
    X_agg = X_agg.loc[X_agg.var(axis=1) > 1e-6]

    # Check for sufficient data points
    if X_agg.empty or len(X_agg) < n_clusters:
        err_msg = 'Not enough valid district data for clustering.'
        return error_fig(err_msg), error_fig(''), error_fig(''), error_fig(''), error_fig(''), default_checklist_opts, default_checklist_vals

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_agg)
    X_scaled_df = pd.DataFrame(X_scaled, index=X_agg.index, columns=X_agg.columns)

    # --- K-Means Clustering ---
    try:
        # Use init='k-means++' and n_init=10 (or 'auto' in newer sklearn)
        km = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10).fit(X_scaled_df)
        labels = dict(zip(X_scaled_df.index, km.labels_))  # District to cluster mapping
        cluster_labels = km.labels_  # Cluster assignments for silhouette
    except Exception as e:
        print(f"Error during KMeans: {e}")
        err_msg = f'Error during clustering: {e}'
        return error_fig(err_msg), error_fig(''), error_fig(''), error_fig(''), error_fig(''), default_checklist_opts, default_checklist_vals

    # --- Update Checklist Options and Values ---
    unique_clusters_found = sorted(list(np.unique(cluster_labels)))
    checklist_opts = [{'label': f'Cluster {i}', 'value': i} for i in unique_clusters_found]

    # *** CORRECTED ***: Use the 'visible_clusters' argument directly
    # If only the checklist triggered, keep the current selection. Otherwise, reset.
    if triggered_id != 'cluster-visibility-checklist' or not visible_clusters:
         # Reset to all if K changed, features changed, or checklist was empty
         checklist_vals = unique_clusters_found
    else:
         # Keep the user's selection from the Input, filtering for valid clusters
         checklist_vals = [val for val in visible_clusters if val in unique_clusters_found]
         # If the previous selection is now empty (e.g., K decreased drastically), select all
         if not checklist_vals and unique_clusters_found:
             checklist_vals = unique_clusters_found


    # --- Generate Plots ---

    # 1. Choropleth Map (Always update map)
    map_fig = create_choropleth_map(X_agg, labels, valid_feats, n_clusters, checklist_vals)

    # 2. Centroid Bar Chart
    # 3. Silhouette Plot
    # 4. Trend Plots (Total and Average)

    # Use no_update if only the checklist changed
    if triggered_id == 'cluster-visibility-checklist':
        bar_fig = no_update
        sil_fig = no_update
        trend_fig_total = no_update
        trend_fig_avg = no_update
    else:
        # Regenerate plots if features or K changed
        try:
            centroids_original_scale = scaler.inverse_transform(km.cluster_centers_)
            bar_fig = create_centroid_bar_chart(centroids_original_scale, valid_feats, n_clusters)
        except Exception as e:
            print(f"Error inverse transforming centroids: {e}")
            bar_fig = error_fig('Error generating centroid plot')

        sil_fig = create_silhouette_plot(X_scaled_df, cluster_labels, n_clusters)

        # Pass the *original* dataframe (with TOTAL districts filtered out) for trends
        trend_fig_total = create_trend_forecast_plot(df_clust, labels, n_clusters, crime_col='TOTAL_CRIMES', title_suffix='(Total IPC)')
        trend_fig_avg = create_trend_forecast_plot(df_clust, labels, n_clusters, crime_col=valid_feats, title_suffix='(Avg Selected)')

    return map_fig, bar_fig, sil_fig, trend_fig_total, trend_fig_avg, checklist_opts, checklist_vals


# --- Helper Functions for Plot Creation (with modifications) ---

def create_choropleth_map(X_agg, labels, valid_feats, n_clusters, visible_clusters):
    """
    Creates the choropleth map visualizing district clusters.
    Uses original aggregated data (X_agg) for hover info.
    Colors unselected clusters white. Correctly handles hover info for AVG_CRIME.

    Args:
        X_agg (pd.DataFrame): Aggregated data (original scale) with districts as index.
        labels (dict): Mapping of district name to cluster index.
        valid_feats (list): List of features used for clustering.
        n_clusters (int): Total number of clusters generated.
        visible_clusters (list): List of cluster indices (int) to display.

    Returns:
        go.Figure: Plotly choropleth map figure.
    """
    # Check if GeoJSON is loaded
    if 'india_district_geo' not in globals() or not india_district_geo or 'features' not in india_district_geo:
        return go.Figure().update_layout(
            title='Cluster Map (Error: GeoJSON missing)',
            xaxis_visible=False, yaxis_visible=False, title_x=0.5
        )

    # --- Prepare Data for Map ---
    # Districts that were actually used in clustering (present in X_agg)
    clustered_districts = X_agg.index.tolist()

    # Create DataFrame mapping clustered districts to labels
    cluster_map_df = pd.DataFrame({
        'DISTRICT': clustered_districts,
        'CLUSTER': [labels[d] for d in clustered_districts]
    })

    # Calculate average crime value using X_agg (original scale means)
    avg_crime_series = X_agg[valid_feats].mean(axis=1)
    avg_crime_series.name = 'AVG_CRIME'

    # Merge cluster labels and average crime onto the cluster_map_df
    map_data = pd.merge(cluster_map_df, avg_crime_series, left_on='DISTRICT', right_index=True, how='left')


    # --- Merge with GeoJSON ---
    try:
        all_geojson_districts = [
            feat['properties']['DISTRICT_UPPER']
            for feat in india_district_geo['features']
            if 'properties' in feat and 'DISTRICT_UPPER' in feat['properties']
        ]
        if not all_geojson_districts: raise ValueError("No districts in GeoJSON.")
        geojson_districts_df = pd.DataFrame({'DISTRICT': all_geojson_districts})
    except Exception as e:
        print(f"Error extracting districts from GeoJSON for cluster map: {e}")
        return go.Figure().update_layout(title="Cluster Map (Error: GeoJSON processing issue)", title_x=0.5)

    # Merge cluster data (map_data) with all GeoJSON districts
    merged_map_data = pd.merge(geojson_districts_df, map_data, on='DISTRICT', how='left')

    # --- Cluster Visibility & Color Logic ---
    merged_map_data['CLUSTER_DISPLAY'] = merged_map_data['CLUSTER'].fillna(-1).astype(int)
    merged_map_data.loc[~merged_map_data['CLUSTER_DISPLAY'].isin(visible_clusters + [-1]), 'CLUSTER_DISPLAY'] = -2

    # --- Define Color Mapping ---
    color_map = {i: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(n_clusters)}
    color_map[-1] = MISSING_DATA_COLOR
    color_map[-2] = UNSELECTED_CLUSTER_COLOR

    # Create discrete colorscale
    unique_display_vals = sorted(merged_map_data['CLUSTER_DISPLAY'].unique())
    colorscale = []
    if len(unique_display_vals) > 1:
        val_range = max(unique_display_vals) - min(unique_display_vals)
        if val_range == 0: val_range = 1
        for i, val in enumerate(unique_display_vals):
            norm_val = (val - min(unique_display_vals)) / val_range
            colorscale.append([norm_val, color_map[val]])
            if i < len(unique_display_vals) - 1:
                 next_norm_val = (unique_display_vals[i+1] - min(unique_display_vals)) / val_range
                 mid_point = (norm_val + next_norm_val) / 2
                 colorscale.append([mid_point - 0.00001, color_map[val]])
                 colorscale.append([mid_point + 0.00001, color_map[unique_display_vals[i+1]]])
            else:
                 colorscale.append([1.0, color_map[val]])
    elif len(unique_display_vals) == 1:
         val = unique_display_vals[0]
         colorscale = [[0.0, color_map[val]], [1.0, color_map[val]]]
    if colorscale and colorscale[0][0] > 0: colorscale.insert(0, [0.0, colorscale[0][1]])
    if colorscale and colorscale[-1][0] < 1: colorscale.append([1.0, colorscale[-1][1]])


    # --- Create Figure ---
    map_fig = go.Figure(
        go.Choropleth(
            geojson=india_district_geo,
            locations=merged_map_data["DISTRICT"],
            z=merged_map_data["CLUSTER_DISPLAY"],
            featureidkey='properties.DISTRICT_UPPER',
            colorscale=colorscale,
            zmin=min(unique_display_vals),
            zmax=max(unique_display_vals),
            marker_line_width=0.1,
            marker_line_color='#666',
            colorbar=None,
            customdata=merged_map_data[['CLUSTER', 'AVG_CRIME', 'CLUSTER_DISPLAY']],
            hovertemplate="<b>District:</b> %{location}<br>" + "<extra></extra>"
        )
    )

    # --- Conditional Hovertemplate ---
    hover_texts = []
    for _, row in merged_map_data.iterrows():
        cluster_display = row['CLUSTER_DISPLAY']
        avg_crime = row['AVG_CRIME']

        if cluster_display == -1:
            hover_texts.append("<b>District:</b> %{location}<br><i>No Clustering Data</i><extra></extra>")
        elif cluster_display == -2:
            hover_texts.append("<b>District:</b> %{location}<br><i>Cluster Hidden</i><extra></extra>")
        else:
            cluster_num = int(row['CLUSTER'])
            avg_crime_text = f"{avg_crime:.1f}" if pd.notna(avg_crime) else "N/A"
            hover_texts.append(
                f"<b>District:</b> %{{location}}<br>"
                f"<b>Cluster:</b> {cluster_num}<br>"
                f"<i>Avg (Selected Feats):</i> {avg_crime_text}<extra></extra>"
            )
    map_fig.update_traces(hovertemplate=hover_texts)
    # --------------------------------

    map_fig.update_geos(fitbounds='locations', visible=False, bgcolor='rgba(0,0,0,0)')
    map_fig.update_layout(
        title=f'District Clusters (K={n_clusters}) based on Mean Scaled IPC Crimes',
        title_font_size=14,
        margin=dict(t=40, l=0, r=0, b=0),
        title_x=0.5,
        geo=dict(bgcolor='rgba(0,0,0,0)')
    )

    return map_fig


def create_centroid_bar_chart(centroids_original_scale, valid_feats, n_clusters):
    """
    Creates a bar chart visualizing cluster centroids in their original scale.
    Uses the consistent color palette.

    Args:
        centroids_original_scale (np.array): Cluster centers in original data scale.
        valid_feats (list): List of features used.
        n_clusters (int): Number of clusters.

    Returns:
        go.Figure: Plotly bar chart figure.
    """
    cent = pd.DataFrame(centroids_original_scale, columns=valid_feats)
    cent['Cluster'] = cent.index

    bar_data = cent.melt(
        id_vars='Cluster',
        value_vars=valid_feats,
        var_name='Crime Feature',
        value_name='Average Count',
    )
    bar_data['Crime Feature'] = (
        bar_data['Crime Feature'].str.replace('_', ' ').str.title()
    )
    bar_data['ClusterLabel'] = 'Cluster ' + bar_data['Cluster'].astype(str)

    cluster_color_map = {f'Cluster {i}': CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(n_clusters)}

    bar_fig = px.bar(
        bar_data,
        x='Crime Feature',
        y='Average Count',
        color='ClusterLabel',
        barmode='group',
        title='Cluster Centroid Profiles (Original Scale)',
        color_discrete_map=cluster_color_map,
        category_orders={'ClusterLabel': sorted(bar_data['ClusterLabel'].unique())}
    )

    bar_fig.update_layout(
        xaxis_title=None,
        yaxis_title='Avg Count per District (Original Scale)',
        title_font_size=14,
        font_size=10,
        margin=dict(t=40, l=20, r=20, b=20),
        title_x=0.5,
        legend_title_text='Cluster',
        xaxis_tickangle=-45
    )
    bar_fig.update_traces(hovertemplate='<b>%{fullData.name}</b><br>%{x}<br>Avg Count: %{y:.2f}<extra></extra>')

    return bar_fig


def create_silhouette_plot(X_scaled_df, cluster_labels, n_clusters):
    """
    Creates a silhouette plot using scaled data, adds average score,
    and uses consistent discrete coloring with full opacity.

    Args:
        X_scaled_df (pd.DataFrame): Scaled data used for clustering.
        cluster_labels (np.array): Cluster assignments for each sample.
        n_clusters (int): Number of clusters.

    Returns:
        go.Figure: Plotly silhouette plot figure.
    """
    n_unique_labels = len(np.unique(cluster_labels))
    if n_unique_labels < 2 or n_unique_labels >= len(X_scaled_df):
         print(f"Cannot calculate silhouette score with {n_unique_labels} clusters for {len(X_scaled_df)} samples.")
         return go.Figure().update_layout(
                title=f'Silhouette Plot (Error: Invalid cluster/sample combination - Need 2 <= k < n_samples)',
                xaxis_visible=False, yaxis_visible=False, title_x=0.5,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )

    try:
        sil_vals = silhouette_samples(X_scaled_df, cluster_labels)
        avg_sil_score = silhouette_score(X_scaled_df, cluster_labels)

        sil_df = pd.DataFrame({
            'district': X_scaled_df.index,
            'cluster': cluster_labels,
            'silhouette': sil_vals,
        })
        sil_df = sil_df.sort_values(['cluster', 'silhouette'], ascending=[True, False]).reset_index(drop=True)
        sil_df['ClusterLabel'] = 'Cluster ' + sil_df['cluster'].astype(str)
        sil_df['y_pos'] = sil_df.index

        cluster_color_map = {f'Cluster {i}': CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(n_clusters)}

        # *** CORRECTED ***: Added opacity=1 directly to px.bar
        sil_fig = px.bar(
            sil_df,
            x='silhouette',
            y='y_pos',
            orientation='h',
            color='ClusterLabel',
            title=f'Silhouette Plot (Avg Score: {avg_sil_score:.3f})',
            labels={'silhouette':'Silhouette Score', 'y_pos':'', 'ClusterLabel':'Cluster'},
            color_discrete_map=cluster_color_map,
            category_orders={'ClusterLabel': sorted(sil_df['ClusterLabel'].unique())},
            custom_data=['district','ClusterLabel'],
            opacity=1 # Ensure bars are opaque from the start
        )

        # Keep update_traces for hover and potentially other marker properties if needed
        sil_fig.update_traces(
            marker_opacity=1, # Reinforce opacity setting
            selected_marker_opacity=1,
            unselected_marker_opacity=1,
            hovertemplate=(
                'District: %{customdata[0]}<br>'
                'Cluster: %{customdata[1]}<br>'
                'Silhouette: %{x:.3f}<extra></extra>'
            )
        )

        sil_fig.add_vline(x=avg_sil_score, line_dash='dash', line_color='red',
                        annotation_text='Average', annotation_position='top right')
        sil_fig.update_layout(
            yaxis=dict(visible=False, showticklabels=False),
            xaxis_title='Silhouette Score',
            margin=dict(t=50, l=20, r=20, b=20),
            title_x=0.5,
            legend_title_text='Cluster'
        )

        return sil_fig

    except ValueError as e:
        print(f"Silhouette score error: {e}")
        error_message = f'Silhouette Plot Error: {e}'
        if "must have less than" in str(e) or "must have more than 1" in str(e):
             error_message = 'Silhouette Plot Error: Need at least 2 samples per cluster.'
        return go.Figure().update_layout(
            title=error_message,
            xaxis_visible=False, yaxis_visible=False, title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
    except Exception as e:
        print(f"Silhouette plot generation error: {e}")
        return go.Figure().update_layout(
            title='Silhouette Plot (Unexpected Error)',
            xaxis_visible=False, yaxis_visible=False, title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )

def create_trend_forecast_plot(df, labels, n_clusters, crime_col, title_suffix):
    """
    Creates time series plot with ARIMA forecasts for each cluster.
    Handles potential errors during ARIMA fitting more gracefully.
    Can calculate trend for a single column ('TOTAL_CRIMES') or average of multiple columns.

    Args:
        df (pd.DataFrame): Original DataFrame (e.g., df_ipc, filtered for non-TOTAL districts).
        labels (dict): Mapping of district name to cluster index.
        n_clusters (int): Number of clusters.
        crime_col (str or list): Column name (e.g., 'TOTAL_CRIMES') or list of
                                 column names (e.g., valid_feats) to average.
        title_suffix (str): String to append to the plot title (e.g., '(Total IPC)').

    Returns:
        go.Figure: Plotly line chart figure with forecasts.
    """
    required_cols = ['DISTRICT', 'YEAR']
    target_col_input = crime_col
    if isinstance(crime_col, str):
        required_cols.append(crime_col)
    elif isinstance(crime_col, list):
        required_cols.extend(crime_col)
    else:
        return go.Figure().update_layout(title=f"Trend Plot Error {title_suffix}: Invalid crime_col type", title_x=0.5)

    if not all(col in df.columns for col in required_cols):
         missing = [col for col in required_cols if col not in df.columns]
         return go.Figure().update_layout(title=f"Trend Plot Error {title_suffix}: Missing required columns ({', '.join(missing)})", title_x=0.5)

    if not labels or df.empty:
        return go.Figure().update_layout(title=f"Trend Plot Error {title_suffix}: No cluster labels or data available.", title_x=0.5)

    df_trend = df.copy()
    df_trend['cluster'] = df_trend['DISTRICT'].map(labels)
    df_clustered = df_trend.dropna(subset=['cluster']).copy()
    df_clustered['cluster'] = df_clustered['cluster'].astype(int)

    if df_clustered.empty:
        return go.Figure().update_layout(title=f"Trend Plot Error {title_suffix}: No districts mapped to clusters.", title_x=0.5)

    df_clustered['YEAR'] = pd.to_numeric(df_clustered['YEAR'], errors='coerce')
    df_clustered = df_clustered.dropna(subset=['YEAR'])
    if df_clustered.empty:
         return go.Figure().update_layout(title=f"Trend Plot Error {title_suffix}: No valid YEAR data after conversion.", title_x=0.5)

    # --- Calculate the value to plot ---
    if isinstance(target_col_input, str):
        df_clustered[target_col_input] = pd.to_numeric(df_clustered[target_col_input], errors='coerce')
        df_clustered = df_clustered.dropna(subset=[target_col_input])
        target_col_name = target_col_input
        if df_clustered.empty:
             return go.Figure().update_layout(title=f"Trend Plot Error {title_suffix}: No valid data for {target_col_name}.", title_x=0.5)
        ts = df_clustered.groupby(['cluster', 'YEAR'])[target_col_name].mean().reset_index()
        yaxis_label = f"Avg {target_col_name.replace('_',' ').title()}"
    elif isinstance(target_col_input, list):
        valid_numeric_cols = []
        for col in target_col_input:
             try:
                 df_clustered[col] = pd.to_numeric(df_clustered[col], errors='coerce')
                 valid_numeric_cols.append(col)
             except Exception as e:
                 print(f"Warning: Could not convert column {col} to numeric for averaging. Skipping. Error: {e}")
        if not valid_numeric_cols:
            return go.Figure().update_layout(title=f"Trend Plot Error {title_suffix}: No valid numeric columns to average.", title_x=0.5)

        df_clustered['AVG_SELECTED_CRIME'] = df_clustered[valid_numeric_cols].mean(axis=1, skipna=True)
        df_clustered = df_clustered.dropna(subset=['AVG_SELECTED_CRIME'])
        target_col_name = 'AVG_SELECTED_CRIME'
        if df_clustered.empty:
             return go.Figure().update_layout(title=f"Trend Plot Error {title_suffix}: No valid data after averaging.", title_x=0.5)
        ts = df_clustered.groupby(['cluster', 'YEAR'])[target_col_name].mean().reset_index()
        yaxis_label = "Avg of Selected Crimes"
    # ------------------------------------

    if ts.empty:
         return go.Figure().update_layout(title=f"Trend Plot Error {title_suffix}: No data after aggregation.", title_x=0.5)

    trend_frames = []
    all_clusters = sorted(ts['cluster'].unique())
    cluster_color_map = {i: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(n_clusters)}

    for cl in all_clusters:
        grp = ts[ts['cluster'] == cl].sort_values('YEAR').set_index('YEAR')
        h = grp.reset_index().copy()
        h['type'] = 'History'
        h['cluster'] = cl
        forecast_df = pd.DataFrame()

        if len(grp) >= 5:
            series = grp[target_col_name]
            if series.nunique() <= 1:
                 print(f"Skipping ARIMA for cluster {cl} ({title_suffix}) due to constant series.")
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(series, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
                        fit = model.fit()
                        fcst = fit.forecast(steps=1)

                    forecast_value = max(0, fcst.iloc[0])
                    forecast_df = pd.DataFrame({
                        'cluster': [cl],
                        'YEAR': [series.index[-1] + 1],
                        target_col_name: [forecast_value],
                        'type': ['Forecast']
                    })
                except Exception as e:
                    print(f"ARIMA fitting/forecast error for cluster {cl} ({title_suffix}): {e}")
                    pass
        else:
             print(f"Skipping ARIMA for cluster {cl} ({title_suffix}) due to insufficient data points ({len(grp)})")

        if not forecast_df.empty:
            trend_frames.append(pd.concat([h, forecast_df], ignore_index=True))
        else:
            trend_frames.append(h)

    if not trend_frames:
        return go.Figure().update_layout(
            title=f'Cluster Trends {title_suffix} (Error: No trend data generated)',
            xaxis_visible=False, yaxis_visible=False, title_x=0.5
        )

    df_all_trends = pd.concat(trend_frames, ignore_index=True)
    df_all_trends['ClusterLabel'] = 'Cluster ' + df_all_trends['cluster'].astype(str)

    num_clusters_present = len(all_clusters)
    facet_col_wrap = min(num_clusters_present, 4)

    try:
        trend_fig = px.line(
            df_all_trends,
            x='YEAR',
            y=target_col_name,
            color='ClusterLabel',
            line_dash='type',
            facet_col='ClusterLabel',
            facet_col_wrap=facet_col_wrap,
            facet_col_spacing=0.04,
            facet_row_spacing=0.1 if num_clusters_present > facet_col_wrap else 0,
            title=f'Mean Crime Trends by Cluster {title_suffix}',
            labels={target_col_name: yaxis_label, 'ClusterLabel': 'Cluster', 'type': 'Data Type'},
            markers=True,
            color_discrete_map={f'Cluster {i}': cluster_color_map[i] for i in all_clusters},
            category_orders={'ClusterLabel': sorted(df_all_trends['ClusterLabel'].unique())}
        )

        trend_fig.update_layout(
            title_font_size=14,
            font_size=10,
            margin=dict(t=50, l=20, r=20, b=20),
            title_x=0.5,
            legend_title_text='Data Type',
            height=max(400, 200 * ((num_clusters_present - 1) // facet_col_wrap + 1))
        )
        trend_fig.for_each_xaxis(lambda ax: ax.update(title='') )
        trend_fig.for_each_yaxis(lambda ax: ax.update(title='') )
        trend_fig.update_layout(
            xaxis_title="Year" if num_clusters_present <= facet_col_wrap else "",
            yaxis_title=yaxis_label
        )

        trend_fig.update_traces(hovertemplate='Year: %{x}<br>Avg Crimes: %{y:.1f}<extra></extra>')
        trend_fig.update_yaxes(matches=None, showticklabels=True)
        trend_fig.update_xaxes(matches=None, showticklabels=True, dtick=1 if df_all_trends['YEAR'].max() - df_all_trends['YEAR'].min() < 15 else None)

        return trend_fig
    except Exception as e:
         print(f"Error creating trend figure {title_suffix}: {e}")
         return go.Figure().update_layout(title=f"Trend Plot Error {title_suffix}: {e}", title_x=0.5)

@app.callback(
  Output("juv-plots-container", "children"),
  Input("juv-category-dropdown", "value")
)
def update_juv_plots(selected_categories):
    if not selected_categories:
        return html.Div("Please select at least one category.", style={'color':'#777'})

    figs = []
    # map dropdown value → (dataframe, column‐prefix, title)
    mapping = {
        'education':    (df_juv_edu,    "EDUCATION_",           "By Education Level"),
        'economic':     (df_juv_econ,   "ECONOMIC_SET_UP_",     "By Annual Income"),
        'family':       (df_juv_family, "FAMILY_BACK_GROUND_",  "By Family Background"),
        'recidivism':   (df_juv_recidiv,"RECIDIVISM_",          "By Recidivism Status")
    }    
  
    for cat in selected_categories:
        df, prefix, subtitle = mapping[cat]
        # collect only the numeric columns with that prefix
        val_cols = [c for c in df.columns if c.startswith(prefix) and not c.endswith('_TOTAL')]
        print(f"Selected columns for {cat}: {val_cols}")
        
        # sum across all rows
        totals = df[val_cols].sum().reset_index()
        totals.columns = ['label','value']
        # clean up label text
        totals['label'] = totals['label'].str.replace(prefix,'').str.replace('_',' ').str.title()

        # Use go.Figure instead of px.pie for more control
        fig = go.Figure(data=[go.Pie(
            labels=totals['label'],
            values=totals['value'],
            hole=0.3,
            textinfo='percent+label',
            textfont_size=12,  # Set smaller font size for labels
            insidetextfont=dict(size=12),
            outsidetextfont=dict(size=12),
            hoverinfo='label+percent+value'
        )])

        fig.update_layout(
            title=f"Juvenile Arrests {subtitle}",
            title_x=0.5,  # Center the title
            height=700,  # Fixed height for all charts
            width=550,   # Fixed width for all charts
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0,           # move legend below the chart (adjust as needed)
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(t=80, b=20, l=40, r=40),
            # margin=dict(t=60, b=40, l=40, r=120),
            paper_bgcolor='white',
            plot_bgcolor='white',
            uniformtext_minsize=10,  # Minimum font size
            uniformtext_mode='hide'  # Hide labels that are too small
        )

        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent:.1%}<extra></extra>',
            textposition='outside'  # Position all text outside for clarity
        )

        figs.append(
            html.Div(
                dcc.Graph(
                    figure=fig, 
                    config={'displayModeBar': False, 'responsive': True}
                ), 
                style={
                    'width':'48%',
                    'display':'inline-block',
                    'verticalAlign':'top',
                    'padding':'5px',
                    'height':'600px'  # Fixed container height
                }
            )
        )

    return html.Div(figs)


## Custodial Deaths 


# 5. Callback to disable options once 6 states are chosen
@app.callback(
    Output('state-select', 'options'),
    Input('state-select', 'value')
)
def disable_extra_states(selected_states):
    opts = STATE_OPTIONS.copy()
    if selected_states and len(selected_states) >= 6:
        for o in opts:
            if o['value'] not in selected_states:
                o['disabled'] = True
    return opts

# 6. Callback to update chart based on controls
@app.callback(
    Output('timeline-graph', 'figure'),
    Input('chart-type',    'value'),
    Input('mode',          'value'),
    Input('state-select',  'value')
)
def update_chart(chart_type, mode, selected_states):
    if mode == 'nat':
        df_plot = nat_melt
        facet_args = {}
        title = "National Aggregate Stacked " + ("Area" if chart_type == 'area' else "Bar") + " Chart"
    else:
        df_plot = melted[melted['Area_Name'].isin(selected_states)]
        facet_args = dict(
            facet_col='Area_Name',
            facet_col_wrap=3,
            category_orders={'Area_Name': selected_states}
        )
        title = "Small-Multiple Stacked " + ("Areas" if chart_type == 'area' else "Bars")
    
    if chart_type == 'area':
        fig = px.area(
            df_plot,
            x='Year', y='Count', color='Cause',
            title=title,
            **facet_args
        )
    else:
        fig = px.bar(
            df_plot,
            x='Year', y='Count', color='Cause',
            title=title,
            **facet_args
        )
    
    fig.update_layout(
        margin=dict(t=60, r=20, l=50, b=50),
        legend_title_text='Cause of Death'
    )
    return fig

@app.callback(
    Output('bar-chart','figure'),
    Input('year-dropdown','value'),
    Input('group-dropdown','value'),
    Input('state-dropdown','value')
)
def update_bar_chart(selected_year, selected_group, selected_states):
    dff = df_long.copy()
    if selected_year != 'All':
        dff = dff[dff['Year']==selected_year]
    if selected_group!='All':
        dff = dff[dff['Group_Name']==selected_group]
    if 'All' not in selected_states:
        dff = dff[dff['Area_Name'].isin(selected_states)]

    agg = (
        dff.groupby(['Area_Name','Loss_Band'])['Loss']
        .sum().reset_index()
    )
    order = agg.groupby('Area_Name')['Loss'].sum().sort_values(ascending=False).index

    fig = px.bar(
        agg, y='Area_Name', x='Loss', color='Loss_Band',
        orientation='h',
        category_orders={'Area_Name': order.tolist()},
        labels={'Loss':'Count of Cases','Area_Name':'State','Loss_Band':'Loss Band'},
        title=(
            f"Year: {selected_year} | Category: {selected_group} | "
            f"States: {', '.join(selected_states)}"
        )
    )
    fig.update_layout(
        barmode='stack',
        yaxis={'categoryorder':'array','categoryarray':order.tolist()},
        margin={'l':120,'r':20,'t':50,'b':50},
        legend_title='Loss Band'
    )
    return fig

@app.callback(
    Output('single-container','style'),
    Output('multi-container','style'),
    Input('stolen-mode','value')
)
def toggle_stolen(mode):
    if mode == 'multi':
        return {'display':'none'}, {'display':'block'}
    return {'display':'block'}, {'display':'none'}

# Generate the plots
@app.callback(
    Output('stolen-plots-div','children'),
    Input('stolen-year-slider','value'),
    Input('stolen-mode','value'),
    Input('stolen-single-dd','value'),
    Input('stolen-multi-dd','value'),
    prevent_initial_call=True
)
def update_stolen(year_range, mode, single, multi):
    df = df_stolen[
        (df_stolen['Year'] >= year_range[0]) & 
        (df_stolen['Year'] <= year_range[1])
    ]
    if df.empty:
        return html.P("No data for selected years.", style={'textAlign':'center'})

    graphs = []
    if mode == 'single':
        if not single:
            return html.P("Select a state.", style={'textAlign':'center'})
        sdf = df[df['Area_Name']==single].groupby('Year').sum().reset_index()

        # Value plot
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=sdf['Year'], y=sdf['Value_of_Property_Stolen'], mode='lines+markers', name='Value Stolen'))
        fig_v.add_trace(go.Scatter(x=sdf['Year'], y=sdf['Value_of_Property_Recovered'], mode='lines+markers', name='Value Recovered'))
        fig_v.update_layout(title=f"Value Stolen vs Recovered in {single}", xaxis_title='Year', yaxis_title='Value')
        graphs.append(dcc.Graph(figure=fig_v))

        # Cases plot
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=sdf['Year'], y=sdf['Cases_Property_Stolen'], mode='lines+markers', name='Cases Stolen'))
        fig_c.add_trace(go.Scatter(x=sdf['Year'], y=sdf['Cases_Property_Recovered'], mode='lines+markers', name='Cases Recovered'))
        fig_c.update_layout(title=f"Cases Stolen vs Recovered in {single}", xaxis_title='Year', yaxis_title='Cases')
        graphs.append(dcc.Graph(figure=fig_c))

    else:
        if not multi or len(multi) < 2 or len(multi) > 7:
            return html.P("Select 2–7 states.", style={'textAlign':'center'})
        mdf = df[df['Area_Name'].isin(multi)].groupby('Area_Name').sum().reset_index()

        # Cases comparison
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=mdf['Area_Name'], y=mdf['Cases_Property_Stolen'], name='Cases Stolen'))
        fig1.add_trace(go.Bar(x=mdf['Area_Name'], y=mdf['Cases_Property_Recovered'], name='Cases Recovered'))
        fig1.update_layout(barmode='group', title="Cases Comparison")
        graphs.append(dcc.Graph(figure=fig1))

        # Value comparison
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=mdf['Area_Name'], y=mdf['Value_of_Property_Stolen'], name='Value Stolen'))
        fig2.add_trace(go.Bar(x=mdf['Area_Name'], y=mdf['Value_of_Property_Recovered'], name='Value Recovered'))
        fig2.update_layout(barmode='group', title="Value Comparison")
        graphs.append(dcc.Graph(figure=fig2))

    return graphs

@app.callback(
    Output('test-time-series','figure'),
    Input('test-state-dd','value'),
    Input('test-sub-dd','value'),
    Input('test-year-slider','value')
)
def update_test(states_sel, subs_sel, yr_range):
    # Resolve "All"
    # ensure order is preserved (sort if you like)
    state_list = list(all_states)
    sub_list   = list(all_subs)

    states = state_list + ['All States'] if not states_sel or 'All States' in states_sel else states_sel
    subs   = sub_list   + ['All Subgroups'] if not subs_sel   or 'All Subgroups' in subs_sel else subs_sel

    low, hi = yr_range
    hist_hi = min(hi, max_year)
    hist_df = df_rape[df_rape['Year'].between(low, hist_hi)]

    series = [(st, sb) for st in states for sb in subs]
    palette = px.colors.qualitative.Plotly
    color_map = {key: palette[i % len(palette)] for i, key in enumerate(series)}

    fig = go.Figure()

    # Historical lines
    for st, sb in series:
        if st=='All States' and sb=='All Subgroups':
            tmp = hist_df.groupby('Year')['Victims_of_Rape_Total'].sum().reset_index()
        elif st=='All States':
            tmp = (hist_df[hist_df['Subgroup']==sb]
                   .groupby('Year')['Victims_of_Rape_Total']
                   .sum().reset_index())
        elif sb=='All Subgroups':
            tmp = (hist_df[hist_df['State']==st]
                   .groupby('Year')['Victims_of_Rape_Total']
                   .sum().reset_index())
        else:
            tmp = hist_df[(hist_df['State']==st)&(hist_df['Subgroup']==sb)]
        if tmp.empty:
            continue
        col = color_map[(st,sb)]
        fig.add_trace(go.Scatter(
            x=tmp['Year'], y=tmp['Victims_of_Rape_Total'],
            mode='lines+markers', name=f"{st} — {sb}",
            line=dict(color=col), marker=dict(color=col)
        ))

    # Predicted point
    if hi >= next_year:
        dfp = pred_df[
            pred_df['State'].isin(states) &
            pred_df['Subgroup'].isin(subs)
        ]
        for _, r in dfp.iterrows():
            col = color_map.get((r['State'],r['Subgroup']), 'black')
            fig.add_trace(go.Scatter(
                x=[r['Year']], y=[r['Predicted']],
                mode='markers', marker=dict(symbol='diamond', size=12, color=col),
                name=f"{r['State']}—{r['Subgroup']} (pred)", showlegend=False
            ))

    fig.update_layout(
        template='plotly_white',
        title='Rape Victims: Historical & Next-Year Prediction',
        xaxis=dict(title='Year', range=[years[0]-0.5, years[-1]+0.5]),
        yaxis_title='Total Victims',
        hovermode='x unified'
    )

    return fig

# ---------------------------------------------------
# Callbacks for Kidnappings & Abductions Tab (NEW)
# ---------------------------------------------------

# Callback to populate the Purpose dropdown
@app.callback(
    Output("kidnap-purpose-dropdown", "options"),
    Input("main-tabs", "value") # Trigger when the tab becomes active or on load
)
def update_kidnap_purpose_options(tab_value):
    # This callback runs to populate options.
    # It might run multiple times, but the logic remains the same.
    if df_kidnap.empty:
        return [{'label': 'Data Not Loaded', 'value': 'error'}]

    # Get unique original purpose names and their cleaned versions
    purposes = df_kidnap[['PURPOSE', 'PURPOSE_CLEAN']].drop_duplicates().sort_values('PURPOSE')
    
    # --- Filter out the specific 'Total' purpose ---
    total_purpose_label = 'Total (Sum of 1-13 Above)'
    purposes_filtered = purposes[purposes['PURPOSE_CLEAN'] != total_purpose_label]
    # ---------------------------------------------

    options = [{'label': 'TOTAL KIDNAPPINGS', 'value': 'TOTAL_KIDNAPPINGS'}]
    options.extend([
        {'label': row['PURPOSE_CLEAN'], 'value': row['PURPOSE']}
        for index, row in purposes_filtered.iterrows() if pd.notna(row['PURPOSE']) # Ensure purpose is not NaN
    ])
    return options

# Callback to update kidnapping with specific purpose visualizations
@app.callback(
    Output("kidnap-visualizations-container", "children"),
    [Input("kidnap-year-slider", "value"),
     Input("kidnap-purpose-dropdown", "value"),
     Input("kidnap-viz-type-radio", "value"),
     Input("kidnap-state-multiselect", "value"),
     Input("kidnap-state-demographics-dropdown", "value")] # <-- ADDED DEMOGRAPHICS STATE DROPDOWN
)
def update_kidnap_visualizations(year_range, selected_purpose, viz_type, selected_states, selected_demo_state):
    """Update visualizations in the Kidnapping tab based on selections."""
    
    if df_kidnap.empty:
        return html.Div("Kidnapping data could not be loaded.", style={'color': 'red', 'textAlign':'center', 'padding':'20px'})

        # Input validation differs slightly depending on viz_type
    if viz_type == "trend":
        if not year_range or not selected_purpose:
            return html.Div("Please select year range and purpose for Trend Analysis.",
                           style={'padding': '20px', 'color': '#777777', 'fontStyle': 'italic', 'textAlign':'center'})
    elif viz_type == "state_comparison":
         if not year_range or not selected_purpose:
            return html.Div("Please select year range and purpose for State Comparison.",
                           style={'padding': '20px', 'color': '#777777', 'fontStyle': 'italic', 'textAlign':'center'})
    elif viz_type == "profile_comparison":
        # Profile comparison needs states selected, but not necessarily a specific purpose (though it uses all purposes implicitly)
        if not year_range or not selected_states:
             return html.Div("Please select year range and at least one state for Profile Comparison.",
                           style={'padding': '20px', 'color': '#777777', 'fontStyle': 'italic', 'textAlign':'center'})
    elif viz_type == "victim_demographics": # <-- ADDED THIS CHECK
        # Victim demographics needs a purpose (even TOTAL) and a state (even All India)
        if not year_range or not selected_purpose or not selected_demo_state:
             return html.Div("Please select year range, purpose, and state (or 'All India') for Demographics Breakdown.",
                           style={'padding': '20px', 'color': '#777777', 'fontStyle': 'italic', 'textAlign':'center'})
    elif not viz_type: # Handle case where viz_type might be None initially or unexpectedly
        return html.Div("Please select a visualization type.",
                        style={'padding': '20px', 'color': '#777777', 'fontStyle': 'italic', 'textAlign':'center'})
    else: # Fallback for truly unexpected viz_type values
        print(f"ERROR: Unexpected viz_type received: {viz_type}") # Log unexpected value
        return html.Div(f"Invalid visualization type selected: {viz_type}", style={'color': 'red', 'textAlign':'center'})

    # --- Common Filtering by Year (moved after validation) ---
    min_year, max_year = year_range
    df_filtered_years = df_kidnap[(df_kidnap["YEAR"] >= min_year) & (df_kidnap["YEAR"] <= max_year)].copy()

    if df_filtered_years.empty:
        return html.Div(f"No kidnapping data available between {min_year} and {max_year}.",
                       style={'padding': '20px', 'color': '#777777', 'fontStyle': 'italic', 'textAlign':'center'})

    # --- Visualization Logic ---
    graphs = []
    year_label_str = f"{min_year}–{max_year}"
    card_style = {
        'backgroundColor': '#ffffff', 'borderRadius': '8px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)',
        'padding': '15px', 'marginBottom': '20px'
    }

    # === Trend Analysis ===
    if viz_type == "trend":
        # --- Data Aggregation based on Purpose --- (Same as before)
        if selected_purpose == 'TOTAL_KIDNAPPINGS':
            df_agg = df_filtered_years.groupby(['STATE', 'YEAR'])['COUNT'].sum().reset_index()
            target_col = 'COUNT'
            purpose_label = 'Total Kidnappings'
        else:
            df_purpose = df_filtered_years[df_filtered_years['PURPOSE'] == selected_purpose]
            if df_purpose.empty:
                 return html.Div(f"No data found for the selected purpose '{selected_purpose}' between {min_year} and {max_year}.",
                           style={'padding': '20px', 'color': '#777777', 'fontStyle': 'italic', 'textAlign':'center'})
            df_agg = df_purpose.copy()
            target_col = 'COUNT'
            purpose_label = df_agg['PURPOSE_CLEAN'].iloc[0] if not df_agg.empty else selected_purpose

        # Aggregate nationally by year
        trend_df = df_agg.groupby("YEAR")[target_col].sum().reset_index()
        trend_df = trend_df.sort_values("YEAR")

        if trend_df.empty or trend_df[target_col].sum() == 0:
             graphs.append(html.Div(f"No trend data to display for {purpose_label} ({year_label_str}).", style={**card_style, 'fontStyle':'italic', 'color':'grey'}))
        else:
            # Ensure YEAR is numeric before plotting
            trend_df['YEAR'] = pd.to_numeric(trend_df['YEAR'])

            # --- Create figure using graph_objects ---
            line_fig = go.Figure()

            # Add the line/marker trace explicitly
            line_fig.add_trace(go.Scatter(
                x=trend_df['YEAR'],
                y=trend_df[target_col],
                mode='lines+markers',  # Specify mode directly here
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8, color='#1f77b4'), # Apply marker style directly
                name=purpose_label, # Optional: for legend if ever needed
                hovertemplate='Year: %{x}<br>Cases: %{y:,}<extra></extra>'
            ))

            # Apply layout settings
            line_fig.update_layout(
                title=f"National Trend of {purpose_label} ({year_label_str})",
                plot_bgcolor='#ffffff',
                paper_bgcolor='#ffffff',
                font={'color': '#333333'},
                title_font=dict(size=18, color='#1f77b4'), # Use explicit title font dict
                title_x=0.5,
                title_xanchor='center',
                margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
                xaxis=dict(gridcolor='#f0f0f0', title='Year', dtick=1),
                yaxis=dict(gridcolor='#f0f0f0', title='Number of Cases'),
                showlegend=False # Hide legend if only one trace
            )
            graphs.append(html.Div(dcc.Graph(figure=line_fig), className="card-container", style=card_style))

            # --- YOY Bar Chart --- (Same as before)
            if len(trend_df) > 1:
                trend_df['YOY_CHANGE'] = trend_df[target_col].pct_change() * 100
                trend_df['YOY_CHANGE'] = trend_df['YOY_CHANGE'].replace([float('inf'), -float('inf')], None)
                trend_df_yoy = trend_df.dropna(subset=['YOY_CHANGE'])

                if not trend_df_yoy.empty:
                    bar_colors = ['#2ca02c' if x >= 0 else '#d62728' for x in trend_df_yoy['YOY_CHANGE']]
                    yoy_fig = px.bar(trend_df_yoy, x="YEAR", y="YOY_CHANGE",
                                   title=f"Year-over-Year % Change in {purpose_label} ({min_year+1 if min_year != max_year else min_year}–{max_year})")
                    yoy_fig.update_layout(
                        plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
                        title={'font': {'size': 18, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
                        margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
                        xaxis={'gridcolor': '#f0f0f0', 'title': 'Year', 'tickmode': 'linear', 'dtick': 1},
                        yaxis={'gridcolor': '#f0f0f0', 'title': '% Change'}
                    )
                    yoy_fig.update_traces(
                        marker_color=bar_colors,
                        hovertemplate='Year: %{x}<br>% Change: %{y:.2f}%<extra></extra>'
                    )
                    graphs.append(html.Div(dcc.Graph(figure=yoy_fig), className="card-container", style=card_style))
                else:
                     graphs.append(html.Div("Not enough data points for Year-over-Year change calculation.", style={**card_style, 'fontStyle':'italic', 'color':'grey'}))

    # === State Comparison (Counts) ===
    elif viz_type == "state_comparison":
        # --- Data Aggregation based on Purpose --- (Same as before)
        if selected_purpose == 'TOTAL_KIDNAPPINGS':
            df_agg = df_filtered_years.groupby(['STATE', 'YEAR'])['COUNT'].sum().reset_index()
            target_col = 'COUNT'
            purpose_label = 'Total Kidnappings'
        else:
            df_purpose = df_filtered_years[df_filtered_years['PURPOSE'] == selected_purpose]
            if df_purpose.empty:
                 return html.Div(f"No data found for the selected purpose '{selected_purpose}' between {min_year} and {max_year}.",
                           style={'padding': '20px', 'color': '#777777', 'fontStyle': 'italic', 'textAlign':'center'})
            df_agg = df_purpose.copy()
            target_col = 'COUNT'
            purpose_label = df_agg['PURPOSE_CLEAN'].iloc[0] if not df_agg.empty else selected_purpose

        # Aggregate by state over the selected years
        state_agg_k = df_agg.groupby("STATE")[target_col].sum().reset_index()
        state_agg_k = state_agg_k[state_agg_k[target_col] > 0] # Filter out zeros

        if state_agg_k.empty:
             graphs.append(html.Div(f"No states reported cases for {purpose_label} between {year_label_str}.", style={**card_style, 'fontStyle':'italic', 'color':'grey'}))
        else:
            # --- Top States Bar Chart --- (Same as before)
            top_states_k = state_agg_k.sort_values(target_col, ascending=False).head(15)
            state_fig_k = px.bar(top_states_k, y="STATE", x=target_col, orientation='h',
                            title=f"Top 15 States by {purpose_label} Cases ({year_label_str})")
            state_fig_k.update_layout(
                plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
                title={'font': {'size': 18, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
                margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
                xaxis={'gridcolor': '#f0f0f0', 'title': f"Number of Cases"},
                yaxis={'gridcolor': '#f0f0f0', 'title': '', 'categoryorder': 'total ascending'}
            )
            state_fig_k.update_traces(marker_color='#1f77b4', hovertemplate='State: %{y}<br>Cases: %{x:,}<extra></extra>')
            graphs.append(html.Div(dcc.Graph(figure=state_fig_k), className="card-container", style=card_style))

            # --- Top 5 States Pie Chart --- (Same as before)
            top5_states_k_df = state_agg_k.sort_values(target_col, ascending=False).head(5)
            total_top5_k = top5_states_k_df[target_col].sum()
            total_all_k = state_agg_k[target_col].sum()
            other_val_k = total_all_k - total_top5_k
            if other_val_k > 0 and len(state_agg_k) > 5:
                 other_df_k = pd.DataFrame([{'STATE': 'Other States', target_col: other_val_k}])
                 pie_data_k = pd.concat([top5_states_k_df, other_df_k], ignore_index=True)
            else:
                 pie_data_k = top5_states_k_df

            pie_fig_k = px.pie(pie_data_k, values=target_col, names='STATE',
                            title=f"Contribution of Top 5 States to {purpose_label} Cases ({year_label_str})",
                            hole=0.3)
            pie_fig_k.update_traces(textposition='inside', textinfo='percent+label',
                                 hovertemplate='<b>%{label}</b><br>Cases: %{value:,}<br>Percentage: %{percent:.1%}<extra></extra>')
            pie_fig_k.update_layout(
                plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
                title={'font': {'size': 18, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
                margin={'l': 40, 'r': 40, 't': 60, 'b': 40},
                legend={'orientation': 'v', 'yanchor':'top', 'y':0.7, 'xanchor':'left', 'x':-0.1}
            )
            graphs.append(html.Div(dcc.Graph(figure=pie_fig_k), className="card-container", style=card_style))

        # === Purpose Profile Comparison (%) ===
    elif viz_type == "profile_comparison":
        # Filter by selected states *in addition* to years
        df_filtered = df_filtered_years[df_filtered_years['STATE'].isin(selected_states)].copy()

        # --- IMPORTANT: Exclude the 'Total' purpose row BEFORE calculating percentages ---
        total_purpose_label = 'Total (Sum of 1-13 Above)' # Define the exact label to filter out
        df_filtered = df_filtered[df_filtered['PURPOSE_CLEAN'] != total_purpose_label]
        # -----------------------------------------------------------------------------

        # --- Robustness Check: Ensure COUNT is numeric ---
        df_filtered['COUNT'] = pd.to_numeric(df_filtered['COUNT'], errors='coerce').fillna(0).astype(int)

        if df_filtered.empty or df_filtered['COUNT'].sum() == 0: # Check if any actual cases exist
            graphs.append(html.Div(f"No non-'Total' kidnapping cases found for the selected states/years.", style={**card_style, 'fontStyle':'italic', 'color':'grey'}))
        else:
            # Aggregate count per state and purpose (only for non-zero counts initially)
            purpose_agg = df_filtered[df_filtered['COUNT'] > 0].groupby(['STATE', 'PURPOSE_CLEAN'])['COUNT'].sum().reset_index()

            # Calculate total count per state (using the filtered data *before* purpose aggregation)
            state_totals = df_filtered.groupby('STATE')['COUNT'].sum().reset_index().rename(columns={'COUNT':'STATE_TOTAL'})
            # Keep only states with a positive total count
            state_totals = state_totals[state_totals['STATE_TOTAL'] > 0]

            # Merge totals back with purpose aggregates. Use left merge to keep all purposes for states that HAVE totals.
            purpose_agg = pd.merge(state_totals, purpose_agg, on='STATE', how='left')
            # Fill NaN counts/purposes resulting from the merge (states with totals but 0 for specific purpose)
            purpose_agg['COUNT'] = purpose_agg['COUNT'].fillna(0).astype(int)
            purpose_agg['PURPOSE_CLEAN'] = purpose_agg['PURPOSE_CLEAN'].fillna('Unknown/None') # Or handle differently if needed
            # Calculate percentage
            purpose_agg['PERCENTAGE'] = (purpose_agg['COUNT'] / purpose_agg['STATE_TOTAL']) * 100
            purpose_agg = purpose_agg.sort_values(by=['STATE_TOTAL','STATE', 'PURPOSE_CLEAN'], ascending=[False, True, True]) # Sort states by total count desc

            # Get unique purposes and ordered states based on the processed data
            unique_purposes = sorted(purpose_agg['PURPOSE_CLEAN'].unique())
            ordered_states = purpose_agg['STATE'].unique().tolist() # Now ordered by total cases

            # --- Add Total Cases Display ---
            # Use the state_totals dataframe calculated earlier
            total_cases_text_lines = [f"Total Cases ({year_label_str}) for Selected States:"]
            total_cases_text_lines.extend([f"- {row['STATE']}: {row['STATE_TOTAL']:,}" for index, row in state_totals.iterrows() if row['STATE'] in ordered_states]) # Filter totals for included states
            total_cases_display = html.Div([
                 html.H5(total_cases_text_lines[0], style={'color': '#1f77b4', 'marginBottom':'5px'}),
                 html.Ul([html.Li(line) for line in total_cases_text_lines[1:]], style={'listStyleType':'none', 'paddingLeft':'10px', 'fontSize':'0.9em'})
            ], style={**card_style, 'marginTop': '-10px'}) # Style as a card, adjust margin

            graphs.append(total_cases_display)


            # --- Create Heatmap ---
            try:
                # Pivot the percentage data
                purpose_pivot_perc = purpose_agg.pivot_table(index='STATE', columns='PURPOSE_CLEAN', values='PERCENTAGE', fill_value=0)
                # Pivot the count data (for customdata)
                purpose_pivot_count = purpose_agg.pivot_table(index='STATE', columns='PURPOSE_CLEAN', values='COUNT', fill_value=0)

                # Ensure row order matches the bar chart
                purpose_pivot_perc = purpose_pivot_perc.reindex(ordered_states)
                purpose_pivot_count = purpose_pivot_count.reindex(ordered_states)

                # Ensure column order is consistent (alphabetical)
                purpose_pivot_perc = purpose_pivot_perc[sorted(purpose_pivot_perc.columns)]
                purpose_pivot_count = purpose_pivot_count[sorted(purpose_pivot_count.columns)]

                heatmap_fig = px.imshow(
                    purpose_pivot_perc, # Using percentage data for color
                    text_auto='.1f', # Show percentage overlay
                    aspect="auto",
                    color_continuous_scale='Viridis',
                    labels=dict(x="Purpose of Kidnapping", y="State", color="Percentage (%)"),
                    title=f"Heatmap of Kidnapping Purpose Profile (%) ({year_label_str})"
                )
                heatmap_fig.update_xaxes(side="bottom", tickangle=-30) # Keep x-axis settings
                heatmap_fig.update_traces(
                    customdata=purpose_pivot_count.values.tolist(), 
                    hovertemplate='<b>State:</b> %{y}<br><b>Purpose:</b> %{x}<br><b>Cases:</b> %{customdata:,.0f}<br><b>Percentage:</b> %{z:.1f}%<extra></extra>' 
                )
                heatmap_fig.update_layout(
                    plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
                    title={'font': {'size': 18, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
                    margin={'l': 40, 'r': 40, 't': 60, 'b': 120}, # Increased bottom margin
                    coloraxis_colorbar=dict(title="%")
                )
                graphs.append(html.Div(dcc.Graph(figure=heatmap_fig), className="card-container", style=card_style))

            except Exception as e:
                print(f"Error creating heatmap: {e}")
                graphs.append(html.Div(f"Could not generate heatmap. Error: {e}", style={**card_style, 'color':'red'}))

        # ... (existing code for trend, state_comparison, profile_comparison) ...

    # === Victim Demographics Breakdown ===
    elif viz_type == "victim_demographics":
        # Input Validation for this specific viz
        if not year_range or not selected_purpose or not selected_demo_state:
             return html.Div("Please select year range, purpose, and state (or 'All India') for Demographics Breakdown.",
                           style={'padding': '20px', 'color': '#777777', 'fontStyle': 'italic', 'textAlign':'center'})

        # Start with year-filtered data
        df_demographics = df_filtered_years.copy()

        # Filter by Purpose (handle TOTAL case by aggregating demographics)
        if selected_purpose == 'TOTAL_KIDNAPPINGS':
            purpose_label_demo = "Total Kidnappings"
            # We need to sum the demographic columns *before* grouping by state/year if needed later
            # Let's do the state/all india filtering first
        else:
            # Get the clean purpose label
            purpose_label_demo = df_demographics[df_demographics['PURPOSE'] == selected_purpose]['PURPOSE_CLEAN'].iloc[0] if selected_purpose in df_demographics['PURPOSE'].values else selected_purpose
            # Filter for the specific purpose
            df_demographics = df_demographics[df_demographics['PURPOSE'] == selected_purpose]

        if df_demographics.empty and selected_purpose != 'TOTAL_KIDNAPPINGS':
             return html.Div(f"No data found for purpose '{purpose_label_demo}' between {year_label_str}.",
                       style={**card_style, 'fontStyle':'italic', 'color':'grey'})


        # Filter by State (if not "All India")
        if selected_demo_state != "All India":
            df_demographics = df_demographics[df_demographics['STATE'] == selected_demo_state]
            location_label = selected_demo_state
        else:
            location_label = "All India"

        if df_demographics.empty:
             return html.Div(f"No data found for '{purpose_label_demo}' in '{location_label}' between {year_label_str}.",
                       style={**card_style, 'fontStyle':'italic', 'color':'grey'})

        # --- Aggregate Demographic Data ---
        demo_cols = [
            'K_A_FEMALE_UPTO_10_YEARS', 'K_A_FEMALE_10_18_YEARS', 'K_A_FEMALE_18_30_YEARS',
            'K_A_FEMALE_30_50_YEARS', 'K_A_FEMALE_ABOVE_50_YEARS',
            'K_A_MALE_UPTO_10_YEARS', 'K_A_MALE_10_18_YEARS', 'K_A_MALE_18_30_YEARS',
            'K_A_MALE_30_50_YEARS', 'K_A_MALE_ABOVE_50_YEARS'
        ]
        # Check if columns exist (should be guaranteed by standardize_columns)
        valid_demo_cols = [col for col in demo_cols if col in df_demographics.columns]
        if not valid_demo_cols:
             return html.Div("Required demographic columns not found in the data.", style={**card_style, 'color':'red'})

        # Sum across the selected scope (years, purpose, state/all)
        demographic_counts = df_demographics[valid_demo_cols].sum()

        # --- Prepare Data for Plotting ---
        demo_df = demographic_counts.reset_index()
        demo_df.columns = ['Demographic_Group_Raw', 'COUNT']
        demo_df = demo_df[demo_df['COUNT'] > 0] # Only plot non-zero groups

        if demo_df.empty:
            graphs.append(html.Div(f"No victims recorded in specific demographic groups for '{purpose_label_demo}' in '{location_label}' ({year_label_str}).", style={**card_style, 'fontStyle':'italic', 'color':'grey'}))
        else:
            # Clean up labels for display
            def clean_demo_label(raw_label):
                label = raw_label.replace('K_A_', '').replace('_YEARS', '').replace('_', ' ')
                # Simple replacements - could be more sophisticated
                label = label.replace('FEMALE UPTO 10', 'Female 0-10')
                label = label.replace('FEMALE 10 18', 'Female 10-18')
                label = label.replace('FEMALE 18 30', 'Female 18-30')
                label = label.replace('FEMALE 30 50', 'Female 30-50')
                label = label.replace('FEMALE ABOVE 50', 'Female 50+')
                label = label.replace('MALE UPTO 10', 'Male 0-10')
                label = label.replace('MALE 10 18', 'Male 10-18')
                label = label.replace('MALE 18 30', 'Male 18-30')
                label = label.replace('MALE 30 50', 'Male 30-50')
                label = label.replace('MALE ABOVE 50', 'Male 50+')
                return label.strip()

            demo_df['Demographic_Group'] = demo_df['Demographic_Group_Raw'].apply(clean_demo_label)
            # Sort for better visualization (optional)
            demo_df = demo_df.sort_values('COUNT', ascending=False)

            # --- Create Grouped Bar Chart ---
            demo_fig = px.bar(
                demo_df,
                x='Demographic_Group',
                y='COUNT',
                color='Demographic_Group', # Color by group for distinction
                title=f"Victim Demographics for '{purpose_label_demo}'<br>in {location_label} ({year_label_str})",
                labels={'COUNT': 'Number of Cases', 'Demographic_Group': 'Victim Age & Gender Group'}
            )

            demo_fig.update_layout(
                plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', font={'color': '#333333'},
                title={'font': {'size': 16, 'color': '#1f77b4'}, 'x': 0.5, 'xanchor': 'center'},
                margin={'l': 40, 'r': 40, 't': 80, 'b': 40}, # Increased top margin for longer title
                xaxis={'gridcolor': '#f0f0f0', 'title': 'Demographic Group', 'categoryorder': 'total descending'}, # Order bars by count
                yaxis={'gridcolor': '#f0f0f0', 'title': 'Number of Cases'},
                showlegend=False # Legend is redundant with colored bars and x-axis labels
            )
            demo_fig.update_traces(
                hovertemplate='<b>Group:</b> %{x}<br><b>Cases:</b> %{y:,}<extra></extra>'
            )

            graphs.append(html.Div(dcc.Graph(figure=demo_fig), className="card-container", style=card_style))


    # --- Return all generated graphs ---
    return html.Div(graphs)


if __name__ == '__main__':
    # Set host='0.0.0.0' to make it accessible on your network if needed
    app.run(debug=True) # debug=True enables hot reloading and error pages
