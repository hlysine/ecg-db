import random
import streamlit as st
import pandas as pd
import numpy as np
import wfdb
import ast
import time
from zipfile import ZipFile
import os.path
import altair as alt
from streamlit_javascript import st_javascript as st_js
import streamlit.components.v1 as components
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
    subprocess.run(['/home/appuser/.local/bin/kaggle', 'datasets', 'download',
                    'khyeh0719/ptb-xl-dataset', '--unzip'])

# Configure libraries
st.set_page_config(layout="wide")
pd.set_option('display.max_columns', None)

# Initialize session state
if "expander_state" not in st.session_state:
    st.session_state["expander_state"] = True
if "record_index" not in st.session_state:
    st.session_state["record_index"] = None
if "validated_by_human" not in st.session_state:
    st.session_state["validated_by_human"] = True
if "second_opinion" not in st.session_state:
    st.session_state["second_opinion"] = False
if "heart_axis" not in st.session_state:
    st.session_state["heart_axis"] = False
if "scp_code" not in st.session_state:
    st.session_state["scp_code"] = None
if "diagnostic_class" not in st.session_state:
    st.session_state["diagnostic_class"] = None
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
query_params = st.experimental_get_query_params()
if "ecg" in query_params:
    st.session_state["record_index"] = int(query_params["ecg"][0]) - 1
    st.session_state["validated_by_human"] = False
    st.session_state["second_opinion"] = False
    st.session_state["heart_axis"] = False
    st.session_state["scp_code"] = None
    st.session_state["diagnostic_class"] = None

st.write("""
# ECG Quiz

Click to see a random ECG and try to guess the condition.
""")


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

    return record_df


total_record_df = load_records()
record_df = total_record_df


@st.cache_data(ttl=60 * 60)
def load_annotations():
    """
    Load and convert the ECG annotations to a DataFrame.
    One row for each condition in SCP code.
    """
    def int_bool(x): return False if x == '' else True
    def optional_int(x): return pd.NA if x == '' else int(float(x))
    def optional_string(x): return pd.NA if x == '' else x
    annotation_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    annotation_df = pd.read_csv(
        path+'scp_statements.csv',
        index_col=0,
        converters={
            'diagnostic': int_bool,
            'form': int_bool,
            'rhythm': int_bool,
            'diagnostic_class': optional_string,
            'diagnostic_subclass': optional_string,
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


def applyFilter():
    """
    Filter records based on filters in session state.
    """
    global total_record_df
    global record_df
    record_df = total_record_df
    if st.session_state["validated_by_human"]:
        record_df = record_df[record_df.validated_by_human]
    if st.session_state["second_opinion"]:
        record_df = record_df[record_df.second_opinion]
    if st.session_state["heart_axis"]:
        record_df = record_df[pd.isna(record_df.heart_axis) == False]
    if st.session_state["scp_code"] is not None:
        record_df = record_df[record_df.scp_codes.apply(
            lambda x: st.session_state["scp_code"] in x)]
    if st.session_state["diagnostic_class"] is not None:
        class_list = annotation_df.reset_index().groupby(
            ['diagnostic_class'])['scp_code'].apply(list)[st.session_state["diagnostic_class"]]

        record_df = record_df[record_df.scp_codes.apply(
            lambda x: any(item in class_list for item in x))]


applyFilter()

# Select a random ECG record
if st.session_state["record_index"] is None:
    st.session_state["record_index"] = random.randint(0, len(record_df) - 1)

record = record_df.iloc[st.session_state["record_index"]]

# Display the selected ECG id in the URL
# Only do so if this is the final re-render
if st.session_state["expander_state"] == False:
    st.experimental_set_query_params(ecg=record.name)


def random_record(validated_by_human, second_opinion, heart_axis, scp_code=None, diagnostic_class=None):
    """
    Set session states based on filters.
    The page will be re-rendered automatically because this is used as an event handler.
    """
    global record
    st.session_state["validated_by_human"] = validated_by_human
    st.session_state["second_opinion"] = second_opinion
    st.session_state["heart_axis"] = heart_axis
    st.session_state["scp_code"] = scp_code
    st.session_state["diagnostic_class"] = diagnostic_class
    applyFilter()
    st.experimental_set_query_params(ecg='')
    st.session_state["record_index"] = None
    st.session_state["expander_state"] = True

# ===============================
# ECG Filters
# ===============================


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.button("New ECG", key='new_ecg1',
              help='Click to see a new ECG', on_click=lambda: random_record(False, False, False))
with col2:
    st.button("New human-validated ECG", key='new_ecg2',
              help='Click to see a new ECG with results validated by a human', on_click=lambda: random_record(True, False, False))
with col3:
    st.button("New double-validated ECG", key='new_ecg3',
              help='Click to see a new ECG with results validated twice', on_click=lambda: random_record(True, True, False))
with col4:
    st.button("New ECG with heart axis", key='new_ecg4',
              help='Click to see a new ECG with heart axis data', on_click=lambda: random_record(False, False, True))

if st.session_state["expander_state"] == False:
    with st.expander("Filter by class", expanded=st.session_state["expander_state"]):
        cols = st.columns(2)
        class_df = annotation_df.groupby(['diagnostic_class'])[
            'Statement Category'].apply(set)
        for i in range(len(class_df)):
            description = ', '.join(class_df.iloc[i])
            cols[i % 2].button(description, key=f'filter_class_{i}', help=f"Find a new ECG with description", on_click=lambda i=i: random_record(
                False, False, False, None, class_df.index[i]))
else:
    st.write('**Loading...**')

if st.session_state["expander_state"] == False:
    with st.expander("Filter by condition", expanded=st.session_state["expander_state"]):
        cols = st.columns(4)
        for i in range(len(annotation_df)):
            cols[i % 4].button(annotation_df.iloc[i]['description'], key=f'filter_condition_{i}', help=f"Find a new ECG with {annotation_df.iloc[i]['description']}", on_click=lambda i=i: random_record(
                False, False, False, annotation_df.iloc[i].name))
else:
    st.write('**Loading...**')

st.write(f'*{len(record_df)} ECGs with the selected filters*')

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
    st.write(f"**ECG ID:** {record.name}")

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


@st.cache_data(max_entries=10)
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


lead_signals = load_raw_data(record, sampling_rate, path)


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

hd_lead_signals = load_raw_data(record, 500, path)

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


@st.cache_data(max_entries=10)
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

    plt.rcParams.update({'font.size': 8})

    fig, ax = plt.subplots(
        1, 3, subplot_kw={'projection': 'polar'}, figsize=(15, 15))
    fig.patch.set_alpha(0.0)
    fig.tight_layout(pad=2.0)

    ax[0].set_theta_direction(-1)
    ax[0].title.set_text("Frontal Vectorcardiogram")
    ax[0].set_facecolor("none")
    ax[0].plot(lead_signals['frontal_phi'],
               lead_signals['frontal_rho'], linewidth=0.5, color="blue")

    ax[1].set_theta_direction(-1)
    ax[1].title.set_text("Transverse Vectorcardiogram")
    ax[1].set_facecolor("none")
    ax[1].plot(lead_signals['transverse_phi'],
               lead_signals['transverse_rho'], linewidth=0.5, color="blue")

    ax[2].set_theta_direction(-1)
    ax[2].title.set_text("Sagittal Vectorcardiogram")
    ax[2].set_facecolor("none")
    ax[2].plot(lead_signals['sagittal_phi'],
               lead_signals['sagittal_rho'], linewidth=0.5, color="blue")

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

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(8, 8))
    fig.patch.set_alpha(0.0)

    ax.set_facecolor("none")
    ax.plot(lead_signals['X'],
            lead_signals['Y'], lead_signals['Z'], linewidth=0.5, color="blue")
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
        vector_signals = calculate_kors_transform(hd_lead_signals)
        fig = plot_vcg(vector_signals, st.session_state["theme"])
        st.pyplot(fig, use_container_width=False)
        col1, col2 = st.columns(spec=[0.2, 0.8])
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
