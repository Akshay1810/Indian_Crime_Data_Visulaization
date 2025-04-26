# Indian Crime Data Visualization Portal (Watchtower)

This project is a Dash-based web application for visualizing various Indian crime datasets from 2001 onwards. It provides interactive maps, charts, and analysis across different crime categories, geographical levels (state and district), and specific crime types.

## Prerequisites

Before running the application, ensure you have the following files and folders set up correctly in the same directory as the `code_dashboard.py` script:

1.  **Data Files (`final_data/` folder):** All the necessary CSV data files must be placed inside a folder named `final_data`. The script expects the following files within this folder:

    - `01_District_wise_crimes_committed_IPC_final.csv`
    - `02_01_District_wise_crimes_committed_against_SC_final.csv`
    - `02_District_wise_crimes_committed_against_ST_final.csv`
    - `03_District_wise_crimes_committed_against_children_final.csv`
    - `17_Crime_by_place_of_occurrence_2001_2012.csv`
    - `18_01_Juveniles_arrested_Education.csv`
    - `18_02_Juveniles_arrested_Economic_setup.csv`
    - `18_03_Juveniles_arrested_Family_background.csv`
    - `18_04_Juveniles_arrested_Recidivism.csv`
    - `20_Victims_of_rape.csv`
    - `21_Offenders_known_to_the_victim.csv`
    - `31_Serious_fraud.csv`
    - `32_Murder_victim_age_sex.csv`
    - `34_Use_of_fire_arms_in_murder_cases.csv`
    - `38_Unidentified_dead_bodies_recovered_and_inquest_conducted.csv`
    - `39_Specific_purpose_of_kidnapping_and_abduction.csv`
    - `40_05_Custodial_death_others.csv`
    - `42_District_wise_crimes_committed_against_women_2001_2013.csv`
    - `property_stolen.csv`

2.  **GeoJSON Files (Root folder):** The GeoJSON files used for map visualizations must be present in the root directory (same level as the script):

    - `india_state.geojson`
    - `india_district.geojson`

3.  **Assets Folder (`assets/` folder):** A folder named `assets` containing the CSS file for the navigation drawer:
    - `drawer_styles.css`

## Installation

You need Python installed on your system (Python 3.7+ recommended). You can install the required Python libraries using pip. Open your terminal or command prompt and run:

```bash
pip install dash numpy dash_daq plotly pandas scikit-learn rapidfuzz statsmodels


Note: Depending on your system setup, you might need to use pip3 instead of pip.
Running the Dashboard
Once the prerequisites are met and the libraries are installed, you can run the dashboard application.
Navigate to the directory containing the code_dashboard.py script, the final_data folder, the assets folder, and the GeoJSON files in your terminal.
Run the script using one of the following commands:
Using python:
python code_dashboard.py

Using python3:
python3 code_dashboard.py


After running the command, you should see output similar to this in your terminal:
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'code_dashboard'
 * Debug mode: on


Open your web browser and navigate to the URL provided (usually http://127.0.0.1:8050/). The dashboard should load.
```
