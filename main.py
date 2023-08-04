import random
import streamlit as st
import pandas as pd
import numpy as np
import wfdb
import ast
import time
import os.path
import altair as alt
from streamlit_javascript import st_javascript as st_js
from streamlit.components.v1 import html
from csscolor import parse
import subprocess
import matplotlib.pyplot as plt

# Define constants
path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
sampling_rate = 100

# Download data from kaggle
if not os.path.isfile(path + 'ptbxl_database.csv'):
    subprocess.run(['pip', 'uninstall', '-y', 'kaggle'])
    subprocess.run(['pip', 'install', '--user', 'kaggle'])
    try:
        # Streamlit cloud
        subprocess.run(['/home/appuser/.local/bin/kaggle', 'datasets', 'download',
                        'khyeh0719/ptb-xl-dataset', '--unzip'])
    except:
        # Hugging Face
        subprocess.run(['/home/user/.local/bin/kaggle', 'datasets', 'download',
                        'khyeh0719/ptb-xl-dataset', '--unzip'])

# Configure libraries
st.set_page_config(
    page_title="ECG Database",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)
pd.set_option('display.max_columns', None)

# Initialize session state
if "expander_state" not in st.session_state:
    st.session_state["expander_state"] = True
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
if "history" not in st.session_state:
    st.session_state["history"] = []


def query_to_filters():
    filters = {}
    query_params = st.experimental_get_query_params()
    if "id" in query_params:
        filters["record_index"] = int(query_params["id"][0]) - 1
    if "validated" in query_params:
        filters["validated_by_human"] = query_params["validated"][0].lower() == "true"
    if "second_opinion" in query_params:
        filters["second_opinion"] = query_params["second_opinion"][0].lower() == "true"
    if "axis" in query_params:
        filters["heart_axis"] = query_params["axis"][0].lower() == "true"
    if "clean" in query_params:
        filters["no_artifacts"] = query_params["clean"][0].lower() == "true"
    if "condition" in query_params:
        filters["scp_code"] = query_params["condition"]
    if "d_class" in query_params:
        filters["diagnostic_class"] = query_params["d_class"]
    return filters


def filters_to_query():
    query_params = {}
    if "record_index" in filters:
        query_params["id"] = filters["record_index"] + 1
    if "validated_by_human" in filters:
        query_params["validated"] = filters["validated_by_human"]
    if "second_opinion" in filters:
        query_params["second_opinion"] = filters["second_opinion"]
    if "heart_axis" in filters:
        query_params["axis"] = filters["heart_axis"]
    if "no_artifacts" in filters:
        query_params["clean"] = filters["no_artifacts"]
    if "scp_code" in filters:
        query_params["condition"] = filters["scp_code"]
    if "diagnostic_class" in filters:
        query_params["d_class"] = filters["diagnostic_class"]
    st.experimental_set_query_params(**query_params)


filters = query_to_filters()


st.write("""
# ECG Database

Filter and view the ECG, VCG and diagnosis data from the PTB-XL ECG Database.
""")

site_link = 'https://huggingface.co/spaces/lysine/ecg-db'
st_cloud = os.path.isdir('/home/appuser')
if st_cloud:
    st.markdown(f"""
**ecg-db has a new home with increased stability. Redirecting you to the new site in 3 seconds...**

Link to the new site: [{site_link}]({site_link})
""")
    html(f"""
<script>
console.log("Redirect script loaded");
setTimeout(() => {{
    console.log("Redirecting to {site_link}");
    const queryString = window.top.location.search;
    window.top.location.href = "{site_link}" + queryString;
}}, 3000);
</script>
    """)
    st.stop()


@st.cache_data(ttl=60 * 60)
def load_records():
    """
    Load and convert the ECG records to a DataFrame.
    One record for each ECG taken.
    """
    def optional_int(x): return pd.NA if x == '' else int(float(x))
    def optional_float(x): return pd.NA if x == '' else float(x)
    def optional_string(x): return pd.NA if x == '' else x

    record_df = pd.read_csv(
        path+'ptbxl_database.csv',
        index_col='ecg_id',
        converters={
            'patient_id': optional_int,
            'age': optional_int,
            'sex': lambda x: 'M' if x == '0' else 'F',
            'height': optional_float,
            'weight': optional_float,
            'nurse': optional_int,
            'site': optional_int,
            'scp_codes': lambda x: ast.literal_eval(x),
            'heart_axis': optional_string,
            'infarction_stadium1': optional_string,
            'infarction_stadium2': optional_string,
            'validated_by': optional_int,
            'baseline_drift': optional_string,
            'static_noise': optional_string,
            'burst_noise': optional_string,
            'electrodes_problems': optional_string,
            'extra_beats': optional_string,
            'pacemaker': optional_string,
        }
    )
    return record_df.reset_index()


record_df = load_records()


@st.cache_data(ttl=60 * 60)
def load_annotations():
    """
    Load and convert the ECG annotations to a DataFrame.
    One row for each condition in SCP code.
    """
    def int_bool(x): return False if x == '' else True
    def optional_int(x): return pd.NA if x == '' else int(float(x))
    def optional_string(x): return pd.NA if x == '' else x
    def mandatory_string(x): return 'OTHER' if x == '' else x
    annotation_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    annotation_df = pd.read_csv(
        path+'scp_statements.csv',
        index_col=0,
        converters={
            'diagnostic': int_bool,
            'form': int_bool,
            'rhythm': int_bool,
            'diagnostic_class': mandatory_string,
            'diagnostic_subclass': mandatory_string,
            'AHA code': optional_int,
            'aECG REFID': optional_string,
            'CDISC Code': optional_string,
            'DICOM Code': optional_string,
        }
    )
    annotation_df.index.name = 'scp_code'
    annotation_df.sort_values('description', inplace=True)
    return annotation_df


annotation_df = load_annotations()

# ===============================
# Browsing history
# ===============================

with st.sidebar:
    st.write("**Browsing history:**")
    if len(st.session_state['history']) == 0:
        st.write('No ECGs viewed yet.')
    else:
        for history in st.session_state['history']:
            st.write(
                f"""<a href="?id={history + 1}">{history + 1} - {', '.join(record_df.iloc[history].scp_codes.keys())}</a>""", unsafe_allow_html=True)


# ===============================
# ECG Filters
# ===============================

with st.form("filter_form"):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        filters['validated_by_human'] = st.checkbox("Human-validated", key='new_ecg1', value=filters['validated_by_human'] if 'validated_by_human' in filters else True,
                                                    help='Filter ECGs with results validated by a human')
    with col2:
        filters['second_opinion'] = st.checkbox("Double-validated", key='new_ecg2', value=filters['second_opinion'] if 'second_opinion' in filters else False,
                                                help='Filter ECGs with results validated twice')
    with col3:
        filters['heart_axis'] = st.checkbox("Heart axis", key='new_ecg3', value=filters['heart_axis'] if 'heart_axis' in filters else False,
                                            help='Filter ECGs with heart axis data')
    with col4:
        filters['no_artifacts'] = st.checkbox("No artifacts", key='new_ecg4', value=filters['no_artifacts'] if 'no_artifacts' in filters else True,
                                              help='Filter ECGs with no artifacts (e.g. baseline drift and noises)')

    with st.expander("Filter by class"):
        if 'diagnostic_class' in filters:
            del filters['diagnostic_class']
        cols = st.columns(2)
        class_df = annotation_df.groupby(['diagnostic_class'])[
            'Statement Category'].apply(set).reset_index()
        for i in range(len(class_df)):
            key = class_df.iloc[i]['diagnostic_class']
            description = 'Other conditions' if key == 'OTHER' else ', '.join(
                class_df.iloc[i]['Statement Category'])
            selected_class = cols[i % 2].checkbox(
                description, key=f'filter_class_{i}', value=key in filters['diagnostic_class'] if 'diagnostic_class' in filters else False)
            if selected_class:
                if 'diagnostic_class' not in filters:
                    filters['diagnostic_class'] = [key]
                elif key not in filters['diagnostic_class']:
                    filters['diagnostic_class'].append(key)

    with st.expander("Filter by condition"):
        if 'scp_code' in filters:
            del filters['scp_code']
        cols = st.columns(4)
        for i in range(len(annotation_df)):
            key = annotation_df.iloc[i].name
            description = annotation_df.iloc[i]['description']
            selected_code = cols[i % 4].checkbox(
                description, key=f'filter_condition_{i}', value=key in filters['scp_code'] if 'scp_code' in filters else False)
            if selected_code:
                if 'scp_code' not in filters:
                    filters['scp_code'] = [key]
                elif key not in filters['scp_code']:
                    filters['scp_code'].append(key)

    submitted = st.form_submit_button(
        label='Random ECG', help='Find a new ECG with the selected filters')
    if submitted:
        if 'record_index' in filters:
            del filters['record_index']
        filters_to_query()
        st.session_state["expander_state"] = True


def applyFilter():
    """
    Filter records based on filters in session state.
    """
    global record_df
    filtered_record_df = record_df
    if "validated_by_human" in filters and filters['validated_by_human']:
        filtered_record_df = filtered_record_df[filtered_record_df.validated_by_human]
    if "second_opinion" in filters and filters['second_opinion']:
        filtered_record_df = filtered_record_df[filtered_record_df.second_opinion]
    if "heart_axis" in filters and filters['heart_axis']:
        filtered_record_df = filtered_record_df[pd.isna(
            filtered_record_df.heart_axis) == False]
    if "no_artifacts" in filters and filters['no_artifacts']:
        filtered_record_df = filtered_record_df[pd.isna(filtered_record_df.baseline_drift) & pd.isna(
            filtered_record_df.static_noise) & pd.isna(filtered_record_df.burst_noise) & pd.isna(filtered_record_df.electrodes_problems)]
    if "scp_code" in filters and not "diagnostic_class" in filters:
        filtered_record_df = filtered_record_df[filtered_record_df.scp_codes.apply(
            lambda x: any(code in filters["scp_code"] for code in x))]
    elif not "scp_code" in filters and "diagnostic_class" in filters:
        filtered_codes = annotation_df[annotation_df['diagnostic_class'].isin(
            filters["diagnostic_class"])].reset_index()['scp_code'].values
        filtered_record_df = filtered_record_df[filtered_record_df.scp_codes.apply(
            lambda x: any(code in filtered_codes for code in x))]
    elif "scp_code" in filters and "diagnostic_class" in filters:
        filtered_codes = annotation_df[annotation_df['diagnostic_class'].isin(
            filters["diagnostic_class"])].reset_index()['scp_code'].values
        filtered_record_df = filtered_record_df[filtered_record_df.scp_codes.apply(
            lambda x: any(code in filters["scp_code"] or code in filtered_codes for code in x))]
    return filtered_record_df


filtered_record_df = applyFilter()

if len(filtered_record_df) == 0:
    st.error('No ECGs found with the selected filters.')
    st.stop()

# Select a random ECG record
if "record_index" not in filters or filters["record_index"] == None:
    record = filtered_record_df.iloc[random.randint(
        0, len(filtered_record_df) - 1)]
    filters["record_index"] = record.name
    filters_to_query()
else:
    record = record_df.iloc[filters["record_index"]]

if filters["record_index"] in st.session_state['history']:
    st.session_state['history'].remove(filters["record_index"])
st.session_state['history'].insert(0, filters["record_index"])

st.write(f'*{len(filtered_record_df)} ECGs with the selected filters*')

st.write("----------------------------")

# ===============================
# ECG Verification Status
# ===============================

box = st.warning
if record.validated_by_human:
    box = st.info
if record.second_opinion:
    box = st.success

box(f"""
**Autogenerated report:** {'Yes' if record.initial_autogenerated_report else 'No'}

**Human validated:** {'Yes' if record.validated_by_human else 'No'}

**Second opinion:** {'Yes' if record.second_opinion else 'No'}
""")

# ===============================
# Patient Info
# ===============================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write(f"**Patient ID:** {record.patient_id}")
    st.write(f"**ECG ID:** {record.ecg_id}")

with col2:
    st.write(f"**Age:** {record.age}")
    st.write(f"**Sex:** {record.sex}")

with col3:
    st.write(f"**Height:** {record.height}")
    st.write(f"**Weight:** {record.weight}")

with col4:
    st.write(f"**Date:** {record.recording_date}")
    st.write(f"**ECG Device:** {record.device}")


# ===============================
# ECG Chart
# ===============================


@st.cache_data(max_entries=2)
def load_raw_data(df, sampling_rate, path):
    """
    Load ECG signals from the raw data files.
    """
    if sampling_rate == 100:
        data = wfdb.rdsamp(path + df.filename_lr)
    else:
        data = wfdb.rdsamp(path + df.filename_hr)
    data = pd.DataFrame(data[0], columns=data[1]['sig_name']).reset_index()
    return data


@st.cache_resource(max_entries=2)
def plot_ecg(lead_signals, sampling_rate, chart_mode, theme):
    """
    Draw the ECG chart.
    """
    alt.renderers.set_embed_options(
        padding={"left": 0, "right": 0, "bottom": 0, "top": 0}
    )
    if chart_mode == 'Continuous':
        chart_x_min = 0
        chart_x_max = 10 * sampling_rate
        chart_y_min = -1.5
        chart_y_max = 34.5

        # Prepare DataFrames for the grid lines
        grid_df = pd.DataFrame(columns=['x', 'y', 'x2', 'y2'])
        for i in range(int(chart_y_min * 2), int(chart_y_max * 2), 1):
            grid_df.loc[len(grid_df.index)] = [
                chart_x_min, i / 2, chart_x_max, i / 2]
        for i in range(chart_x_min, chart_x_max, 20):
            grid_df.loc[len(grid_df.index)] = [i, chart_y_min, i, chart_y_max]

        minor_grid_df = pd.DataFrame(columns=['x', 'y', 'x2', 'y2'])
        for i in range(int(chart_y_min * 10), int(chart_y_max * 10), 1):
            minor_grid_df.loc[len(minor_grid_df.index)] = [
                chart_x_min, i / 10, chart_x_max, i / 10]
        for i in range(chart_x_min, chart_x_max, 4):
            minor_grid_df.loc[len(minor_grid_df.index)] = [
                i, chart_y_min, i, chart_y_max]

        # Prepare DataFrames for the text labels and modify the lead signals
        text_df = pd.DataFrame(columns=['x', 'y', 'text'])

        lead_names = lead_signals.columns.values[1:]
        leads_count = len(lead_names)
        for i in range(leads_count):
            lead_signals[lead_names[i]].iloc[int(
                10 * sampling_rate * 48/50):int(10 * sampling_rate * 49/50)] = 1
            lead_signals[lead_names[i]].iloc[int(
                10 * sampling_rate * 49/50):int(10 * sampling_rate * 49.2/50)] = 0
            lead_signals[lead_names[i]].iloc[int(
                10 * sampling_rate * 49.2/50):] = pd.NA
            lead_signals[lead_names[i]] = lead_signals[lead_names[i]
                                                       ] + (leads_count - i - 1) * 3
            text_df.loc[len(text_df.index)] = [
                4, (leads_count - i - 1) * 3 + 1, lead_names[i]]

        # Plot the grid lines
        chart = alt.layer(
            alt.Chart(minor_grid_df).mark_rule(clip=True, stroke=('#252525' if theme == 'dark' else '#dddddd')).encode(
                x=alt.X('x', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_x_min, chart_x_max), padding=0)),
                x2=alt.X2('x2'),
                y=alt.Y('y', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_y_min, chart_y_max), padding=0)),
                y2=alt.Y2('y2'),
                tooltip=alt.value(None),
            ),
            alt.Chart(grid_df).mark_rule(clip=True, stroke=('#555' if theme == 'dark' else '#bbb')).encode(
                x=alt.X('x', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_x_min, chart_x_max), padding=0)),
                x2=alt.X2('x2'),
                y=alt.Y('y', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_y_min, chart_y_max), padding=0)),
                y2=alt.Y2('y2'),
                tooltip=alt.value(None),
            )
        ).properties(
            width=1600,
            height=1600 / 50 * 72 + 20,  # 20px padding
        ).configure_concat(
            spacing=0
        ).configure_facet(
            spacing=0
        ).configure_axis(
            grid=False,
            labels=False,
        )

        # Plot the ECG signals
        for col in lead_signals.columns.values[1:]:
            chart += alt.Chart(lead_signals).mark_line(clip=True, stroke=('#7abaed' if theme == 'dark' else '#05014a')).encode(
                x=alt.X('index', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_x_min, chart_x_max), padding=0)),
                y=alt.Y(col, type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_y_min, chart_y_max), padding=0)),
                tooltip=alt.value(None),
            )

        # Plot the text labels
        chart += alt.Chart(text_df).mark_text(baseline='middle', align='left', size=20, fill=('#fff' if theme == 'dark' else '#020079')).encode(
            text='text',
            x=alt.X('x', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_x_min, chart_x_max), padding=0)),
            y=alt.Y('y', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_y_min, chart_y_max), padding=0)),
            tooltip=alt.value(None),
        )

        return chart
    else:
        # Duplicate lead II into a new column
        lead_signals['II '] = lead_signals['II']
        lead_config = [
            {
                'lead': 'II ',
                'y': 0,
                'start_x': 0,
                'end_x': int(10 * sampling_rate),
            },
            {
                'lead': 'I',
                'y': 3,
                'start_x': 0,
                'end_x': int(10 * sampling_rate * 12 / 50),
            },
            {
                'lead': 'II',
                'y': 2,
                'start_x': 0,
                'end_x': int(10 * sampling_rate * 12 / 50),
            },
            {
                'lead': 'III',
                'y': 1,
                'start_x': 0,
                'end_x': int(10 * sampling_rate * 12 / 50),
            },
            {
                'lead': 'AVR',
                'y': 3,
                'start_x': int(10 * sampling_rate * 12 / 50),
                'end_x': int(10 * sampling_rate * 24 / 50),
            },
            {
                'lead': 'AVL',
                'y': 2,
                'start_x': int(10 * sampling_rate * 12 / 50),
                'end_x': int(10 * sampling_rate * 24 / 50),
            },
            {
                'lead': 'AVF',
                'y': 1,
                'start_x': int(10 * sampling_rate * 12 / 50),
                'end_x': int(10 * sampling_rate * 24 / 50),
            },
            {
                'lead': 'V1',
                'y': 3,
                'start_x': int(10 * sampling_rate * 24 / 50),
                'end_x': int(10 * sampling_rate * 36 / 50),
            },
            {
                'lead': 'V2',
                'y': 2,
                'start_x': int(10 * sampling_rate * 24 / 50),
                'end_x': int(10 * sampling_rate * 36 / 50),
            },
            {
                'lead': 'V3',
                'y': 1,
                'start_x': int(10 * sampling_rate * 24 / 50),
                'end_x': int(10 * sampling_rate * 36 / 50),
            },
            {
                'lead': 'V4',
                'y': 3,
                'start_x': int(10 * sampling_rate * 36 / 50),
                'end_x': int(10 * sampling_rate),
            },
            {
                'lead': 'V5',
                'y': 2,
                'start_x': int(10 * sampling_rate * 36 / 50),
                'end_x': int(10 * sampling_rate),
            },
            {
                'lead': 'V6',
                'y': 1,
                'start_x': int(10 * sampling_rate * 36 / 50),
                'end_x': int(10 * sampling_rate),
            },
        ]

        chart_x_min = 0
        chart_x_max = 10 * sampling_rate
        chart_y_min = -1.5
        chart_y_max = 10.5

        # Prepare DataFrames for the grid lines
        grid_df = pd.DataFrame(columns=['x', 'y', 'x2', 'y2'])
        for i in range(int(chart_y_min * 2), int(chart_y_max * 2), 1):
            grid_df.loc[len(grid_df.index)] = [
                chart_x_min, i / 2, chart_x_max, i / 2]
        for i in range(chart_x_min, chart_x_max, 20):
            grid_df.loc[len(grid_df.index)] = [i, chart_y_min, i, chart_y_max]

        minor_grid_df = pd.DataFrame(columns=['x', 'y', 'x2', 'y2'])
        for i in range(int(chart_y_min * 10), int(chart_y_max * 10), 1):
            minor_grid_df.loc[len(minor_grid_df.index)] = [
                chart_x_min, i / 10, chart_x_max, i / 10]
        for i in range(chart_x_min, chart_x_max, 4):
            minor_grid_df.loc[len(minor_grid_df.index)] = [
                i, chart_y_min, i, chart_y_max]

        # Prepare DataFrames for the text labels and lead separators
        # Also modify the lead signals
        text_df = pd.DataFrame(columns=['x', 'y', 'text'])
        separator_df = pd.DataFrame(columns=['x', 'y', 'x2', 'y2'])

        for config in lead_config:
            if config['start_x'] > 0:
                lead_signals[config['lead']].iloc[:config['start_x']] = pd.NA
                separator_df.loc[len(separator_df.index)] = [
                    config['start_x'], config['y'] * 3 - 0.5, config['start_x'], config['y'] * 3 + 0.5]
            if config['end_x'] < 10 * sampling_rate:
                lead_signals[config['lead']].iloc[config['end_x']:] = pd.NA
            else:
                lead_signals[config['lead']].iloc[int(
                    10 * sampling_rate * 48/50):int(10 * sampling_rate * 49/50)] = 1
                lead_signals[config['lead']].iloc[int(
                    10 * sampling_rate * 49/50):int(10 * sampling_rate * 49.2/50)] = 0
                lead_signals[config['lead']].iloc[int(
                    10 * sampling_rate * 49.2/50):] = pd.NA

            lead_signals[config['lead']
                         ] = lead_signals[config['lead']] + config['y'] * 3
            text_df.loc[len(text_df.index)] = [
                config['start_x'] + 4, config['y'] * 3 + 1, config['lead']]

        # Plot the grid lines
        chart = alt.layer(
            alt.Chart(minor_grid_df).mark_rule(clip=True, stroke=('#252525' if theme == 'dark' else '#dddddd')).encode(
                x=alt.X('x', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_x_min, chart_x_max), padding=0)),
                x2=alt.X2('x2'),
                y=alt.Y('y', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_y_min, chart_y_max), padding=0)),
                y2=alt.Y2('y2'),
                tooltip=alt.value(None),
            ),
            alt.Chart(grid_df).mark_rule(clip=True, stroke=('#555' if theme == 'dark' else '#bbb')).encode(
                x=alt.X('x', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_x_min, chart_x_max), padding=0)),
                x2=alt.X2('x2'),
                y=alt.Y('y', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_y_min, chart_y_max), padding=0)),
                y2=alt.Y2('y2'),
                tooltip=alt.value(None),
            )
        ).properties(
            width=1600,
            height=1600 / 50 * 24 + 20,  # 20px padding
        ).configure_concat(
            spacing=0
        ).configure_facet(
            spacing=0
        ).configure_axis(
            grid=False,
            labels=False,
        )

        # Plot the ECG signals
        for config in lead_config:
            chart += alt.Chart(lead_signals).mark_line(clip=True, stroke=('#7abaed' if theme == 'dark' else '#05014a')).encode(
                x=alt.X('index', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_x_min, chart_x_max), padding=0)),
                y=alt.Y(config['lead'], type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_y_min, chart_y_max), padding=0)),
                tooltip=alt.value(None),
            )

        # Plot the lead separators
        chart += alt.Chart(separator_df).mark_rule(clip=True, stroke=('#7abaed' if theme == 'dark' else '#05014a')).encode(
            x=alt.X('x', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_x_min, chart_x_max), padding=0)),
            x2=alt.X2('x2'),
            y=alt.Y('y', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_y_min, chart_y_max), padding=0)),
            y2=alt.Y2('y2'),
            tooltip=alt.value(None),
        )

        # Plot the text labels
        chart += alt.Chart(text_df).mark_text(baseline='middle', align='left', size=20, fill=('#fff' if theme == 'dark' else '#020079')).encode(
            text='text',
            x=alt.X('x', type='quantitative', title=None, scale=alt.Scale(
                domain=(chart_x_min, chart_x_max), padding=0)),
            y=alt.Y('y', type='quantitative', title=None, scale=alt.Scale(
                domain=(chart_y_min, chart_y_max), padding=0)),
            tooltip=alt.value(None),
        )

        return chart


if st.session_state["expander_state"] == False:
    chart_mode = st.selectbox(
        'ECG Chart Mode',
        options=('Report', 'Continuous'),
    )

    with st.spinner('Loading ECG...'):
        lead_signals = load_raw_data(record, sampling_rate, path)
        fig = plot_ecg(lead_signals, sampling_rate,
                       chart_mode, st.session_state["theme"])
        st.altair_chart(fig, use_container_width=False)
else:
    st.info('**Loading ECG...**', icon='🔃')

# ===============================
# ECG Analysis
# ===============================

# Only render the expander when this is the final re-render
if st.session_state["expander_state"] == False:
    with st.expander("ECG Analysis", expanded=st.session_state["expander_state"]):
        for code, prob in record.scp_codes.items():
            annotation = annotation_df.loc[code]
            st.write(f"""
> `{f"{annotation.diagnostic_class} > {annotation.diagnostic_subclass} > {annotation.name}" if not pd.isna(annotation.diagnostic_class) and not pd.isna(annotation.diagnostic_subclass) else
    f"{annotation.diagnostic_class} > {annotation.name}" if not pd.isna(annotation.diagnostic_class) else annotation.name}` - {"unknown likelihood" if prob == 0 else f"**{prob}%**"}
>
> {annotation['Statement Category']}
>
> **{annotation['SCP-ECG Statement Description']}**


    """)

        st.write("---------------------")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write(f"**Heart Axis:** {record.heart_axis}")
            st.write(f"**Pacemaker:** {record.pacemaker}")
            st.write(f"**Extra Beats:** {record.extra_beats}")

        with col2:
            st.write(f"**Infarction Stadium 1:** {record.infarction_stadium1}")
            st.write(f"**Infarction Stadium 2:** {record.infarction_stadium2}")

        with col3:
            st.write(f"**Baseline Drift:** {record.baseline_drift}")
            st.write(f"**Electrode Problems:** {record.electrodes_problems}")

        with col4:
            st.write(f"**Static Noise:** {record.static_noise}")
            st.write(f"**Burst Noise:** {record.burst_noise}")
else:
    st.write('**Loading...**')


# ===============================
# Vectorcardiogram
# ===============================

kors_transform = [
    {
        'axis': 'X',
        'leads': [
            {'lead': 'V1', 'weight': -0.130},
            {'lead': 'V2', 'weight': 0.050},
            {'lead': 'V3', 'weight': -0.010},
            {'lead': 'V4', 'weight': 0.140},
            {'lead': 'V5', 'weight': 0.060},
            {'lead': 'V6', 'weight': 0.540},
            {'lead': 'I', 'weight': 0.380},
            {'lead': 'II', 'weight': -0.070},
        ]
    },
    {
        'axis': 'Y',
        'leads': [
            {'lead': 'V1', 'weight': 0.060},
            {'lead': 'V2', 'weight': -0.020},
            {'lead': 'V3', 'weight': -0.050},
            {'lead': 'V4', 'weight': 0.060},
            {'lead': 'V5', 'weight': -0.170},
            {'lead': 'V6', 'weight': 0.130},
            {'lead': 'I', 'weight': -0.070},
            {'lead': 'II', 'weight': 0.930},
        ]
    },
    {
        'axis': 'Z',
        'leads': [
            {'lead': 'V1', 'weight': -0.430},
            {'lead': 'V2', 'weight': -0.060},
            {'lead': 'V3', 'weight': -0.140},
            {'lead': 'V4', 'weight': -0.200},
            {'lead': 'V5', 'weight': -0.110},
            {'lead': 'V6', 'weight': 0.310},
            {'lead': 'I', 'weight': 0.110},
            {'lead': 'II', 'weight': -0.230},
        ]
    }
]


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([rho, phi])


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y])


@st.cache_data(max_entries=2)
def calculate_kors_transform(lead_signals):
    """
    Calculate VCG data from the ECG data using the Kors regression transformation.
    https://doi.org/10.3390/s19143072
    """
    vector_signals = lead_signals.copy()

    for axis in kors_transform:
        vector_signals[axis['axis']] = vector_signals.apply(
            lambda r: sum([r[lead['lead']] * lead['weight'] for lead in axis['leads']]), axis=1)

    vector_signals['frontal'] = vector_signals.apply(
        lambda r: cart2pol(r['X'], r['Y']), axis=1)
    vector_signals['frontal_rho'] = vector_signals['frontal'].apply(
        lambda x: x[0])
    vector_signals['frontal_phi'] = vector_signals['frontal'].apply(
        lambda x: x[1])
    vector_signals['transverse'] = vector_signals.apply(
        lambda r: cart2pol(r['X'], -r['Z']), axis=1)
    vector_signals['transverse_rho'] = vector_signals['transverse'].apply(
        lambda x: x[0])
    vector_signals['transverse_phi'] = vector_signals['transverse'].apply(
        lambda x: x[1])
    vector_signals['sagittal'] = vector_signals.apply(
        lambda r: cart2pol(r['Z'], r['Y']), axis=1)
    vector_signals['sagittal_rho'] = vector_signals['sagittal'].apply(
        lambda x: x[0])
    vector_signals['sagittal_phi'] = vector_signals['sagittal'].apply(
        lambda x: x[1])

    return vector_signals


@st.cache_resource(max_entries=2)
def plot_vcg(lead_signals, theme):
    """
    Draw the vectorcardiogram.
    """
    if theme == 'dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    plt.rcParams.update({'font.size': 8, 'axes.titlepad': 40})

    fig, ax = plt.subplots(
        1, 3, subplot_kw={'projection': 'polar'}, figsize=(15, 15))
    fig.patch.set_alpha(0.0)
    fig.tight_layout(pad=2.0)

    ax[0].set_theta_direction(-1)
    ax[0].title.set_text("Frontal Vectorcardiogram")
    fig.text(0.172, 0.662, "0º at +X, positive towards +Y",
             horizontalalignment="center")
    ax[0].set_facecolor("none")
    ax[0].plot(lead_signals['frontal_phi'],
               lead_signals['frontal_rho'], linewidth=0.5, color=('#7abaed' if theme == 'dark' else '#05014a'))

    ax[1].set_theta_direction(-1)
    ax[1].title.set_text("Transverse Vectorcardiogram")
    fig.text(0.5, 0.662, "0º at +X, positive towards -Z",
             horizontalalignment="center")
    ax[1].set_facecolor("none")
    ax[1].plot(lead_signals['transverse_phi'],
               lead_signals['transverse_rho'], linewidth=0.5, color=('#7abaed' if theme == 'dark' else '#05014a'))

    ax[2].set_theta_direction(-1)
    ax[2].title.set_text("Sagittal Vectorcardiogram")
    fig.text(0.829, 0.662, "0º at +Z, positive towards +Y",
             horizontalalignment="center")
    ax[2].set_facecolor("none")
    ax[2].plot(lead_signals['sagittal_phi'],
               lead_signals['sagittal_rho'], linewidth=0.5, color=('#7abaed' if theme == 'dark' else '#05014a'))

    return fig


@st.cache_resource(max_entries=2)
def plot_vcg_3d(lead_signals, h_angle, v_angle, theme):
    """
    Draw the vectorcardiogram in 3D.
    """
    if theme == 'dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    plt.rcParams.update({'font.size': 8})

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(10, 8))
    fig.patch.set_alpha(0.0)

    ax.set_facecolor("none")
    ax.plot(lead_signals['X'],
            lead_signals['Y'], lead_signals['Z'], linewidth=0.5, color="blue")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.invert_yaxis()
    ax.title.set_text("Spatial Vectorcardiogram")
    ax.view_init(v_angle, h_angle, None, vertical_axis='y')

    return fig


# @st.cache_resource(max_entries=2)
# def plot_vcg_interactive(lead_signals, theme):
#     """
#     Draw the vectorcardiogram in 3D.
#     """
#     # pv.global_theme.show_scalar_bar = False

#     p = pv.Plotter()

#     points = np.array(
#         [lead_signals['X'].values, lead_signals['Y'].values, lead_signals['Z'].values])
#     points = points.transpose()

#     spline = pv.Spline(points)

#     p.add_mesh(mesh=spline, color='blue')

#     p.show_grid()
#     if theme == 'dark':
#         p.set_background(color='#0e1117')
#     else:
#         p.set_background(color='#ffffff')

#     return p


if st.session_state["expander_state"] == False:
    with st.expander("Vectorcardiogram (Approximation)", expanded=st.session_state["expander_state"]):
        hd_lead_signals = load_raw_data(record, 500, path)
        vector_signals = calculate_kors_transform(hd_lead_signals)
        fig = plot_vcg(vector_signals, st.session_state["theme"])
        st.pyplot(fig, use_container_width=False)
        col1, col2 = st.columns(spec=[0.2, 0.8])
        with st.spinner('Loading 3D plot...'):
            with col1:
                h_angle = st.slider("Horizontal view angle", min_value=-180,
                                    max_value=180, value=-60, step=5)
                v_angle = st.slider("Vertical view angle", min_value=-180,
                                    max_value=180, value=30, step=5)
            with col2:
                fig3d = plot_vcg_3d(vector_signals, h_angle,
                                    v_angle, st.session_state["theme"])
                st.pyplot(fig3d, use_container_width=False)
        # fig3d = plot_vcg_interactive(
        #     vector_signals, st.session_state["theme"])
        # stpyvista(fig3d)
        # fig_html = mpld3.fig_to_html(fig3d)
        # components.html(fig_html, height=600)
else:
    st.info('**Loading VCG...**', icon='🔃')

# Detect browser theme
if st.session_state["expander_state"] == True:
    theme = st_js(
        """window.getComputedStyle( document.body ,null).getPropertyValue('background-color')""")
    if theme != 0:
        color = parse.color(theme)
        if color.as_hsl_percent_triple()[2] > 50:
            st.session_state["theme"] = "light"
        else:
            st.session_state["theme"] = "dark"


# To forcibly collapse the expanders, the whole page is rendered twice.
# In the first rerender, the expander is replaced by a placeholder markdown text.
# In the second rerender, the expander is rendered and it defaults to collapsed
# because it did not exist in the previous render.
if st.session_state["expander_state"] == True and theme != 0:
    st.session_state["expander_state"] = False
    # Wait for the client to sync up
    time.sleep(0.05)
    # Start the second re-render
    st.experimental_rerun()
