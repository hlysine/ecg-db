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

# Define constants
path = 'ptb-xl/'
sampling_rate = 100

if not os.path.isfile(path + 'ptbxl_database.csv'):
    with ZipFile("ptb-xl.zip", 'r') as zObject:
        zObject.extractall(path=path)

st.set_page_config(layout="wide")
pd.set_option('display.max_columns', None)
if "expander_state" not in st.session_state:
    st.session_state["expander_state"] = False
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

st.write("""
# ECG Quiz

Click to see a random ECG and try to guess the diagnosis.
""")


@st.cache_data(ttl=60 * 60)
def load_records():
    def optional_int(x): return pd.NA if x == '' else int(float(x))
    def optional_float(x): return pd.NA if x == '' else float(x)
    def optional_string(x): return pd.NA if x == '' else x
    # load and convert annotation data
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


def applyFilter():
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


applyFilter()


@st.cache_data(ttl=60 * 60)
def load_annotations():
    # Load scp_statements.csv for diagnostic aggregation
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
    return annotation_df


annotation_df = load_annotations()

if st.session_state["record_index"] is None:
    st.session_state["record_index"] = random.randint(0, len(record_df) - 1)

record = record_df.iloc[st.session_state["record_index"]]


def random_record(validated_by_human, second_opinion, heart_axis, scp_code=None):
    global record
    st.session_state["validated_by_human"] = validated_by_human
    st.session_state["second_opinion"] = second_opinion
    st.session_state["heart_axis"] = heart_axis
    st.session_state["scp_code"] = scp_code
    applyFilter()
    st.session_state["record_index"] = random.randint(0, len(record_df) - 1)
    st.session_state["expander_state"] = True


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

with st.expander("Filter by condition", expanded=st.session_state["expander_state"]):
    cols = st.columns(4)
    for i in range(len(annotation_df)):
        cols[i % 4].button(annotation_df.iloc[i]['description'], key=f'filter_{i}', help=f"Find a new ECG with {annotation_df.iloc[i]['description']}", on_click=lambda i=i: random_record(
            False, False, False, annotation_df.iloc[i].name))

st.write("----------------------------")

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


@st.cache_data(ttl=60 * 60)
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = wfdb.rdsamp(path + df.filename_lr)
    else:
        data = wfdb.rdsamp(path + df.filename_hr)
    data = pd.DataFrame(data[0], columns=data[1]['sig_name']).reset_index()
    return data


lead_signals = load_raw_data(record, sampling_rate, path)


@ st.cache_resource(max_entries=2)
def plot_ecg(lead_signals, sampling_rate, chart_mode):
    alt.renderers.set_embed_options(
        padding={"left": 0, "right": 0, "bottom": 0, "top": 0}
    )
    if chart_mode == 'Continuous':
        chart_x_min = 0
        chart_x_max = 10 * sampling_rate
        chart_y_min = -1.5
        chart_y_max = 34.5

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

        chart = alt.layer(
            alt.Chart(minor_grid_df).mark_rule(clip=True, stroke='#252525').encode(
                x=alt.X('x', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_x_min, chart_x_max), padding=0)),
                x2=alt.X2('x2'),
                y=alt.Y('y', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_y_min, chart_y_max), padding=0)),
                y2=alt.Y2('y2'),
                tooltip=alt.value(None),
            ),
            alt.Chart(grid_df).mark_rule(clip=True, stroke='#555').encode(
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

        for col in lead_signals.columns.values[1:]:
            chart += alt.Chart(lead_signals).mark_line(clip=True).encode(
                x=alt.X('index', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_x_min, chart_x_max), padding=0)),
                y=alt.Y(col, type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_y_min, chart_y_max), padding=0)),
                tooltip=alt.value(None),
            )
        chart += alt.Chart(text_df).mark_text(baseline='middle', align='left', size=20, fill='#fff').encode(
            text='text',
            x=alt.X('x', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_x_min, chart_x_max), padding=0)),
            y=alt.Y('y', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_y_min, chart_y_max), padding=0)),
            tooltip=alt.value(None),
        )
        return chart
    else:
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

        chart = alt.layer(
            alt.Chart(minor_grid_df).mark_rule(clip=True, stroke='#252525').encode(
                x=alt.X('x', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_x_min, chart_x_max), padding=0)),
                x2=alt.X2('x2'),
                y=alt.Y('y', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_y_min, chart_y_max), padding=0)),
                y2=alt.Y2('y2'),
                tooltip=alt.value(None),
            ),
            alt.Chart(grid_df).mark_rule(clip=True, stroke='#555').encode(
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

        for config in lead_config:
            chart += alt.Chart(lead_signals).mark_line(clip=True, stroke="#7abaed").encode(
                x=alt.X('index', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_x_min, chart_x_max), padding=0)),
                y=alt.Y(config['lead'], type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_y_min, chart_y_max), padding=0)),
                tooltip=alt.value(None),
            )
        chart += alt.Chart(separator_df).mark_rule(clip=True, stroke="#7abaed").encode(
            x=alt.X('x', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_x_min, chart_x_max), padding=0)),
            x2=alt.X2('x2'),
            y=alt.Y('y', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_y_min, chart_y_max), padding=0)),
            y2=alt.Y2('y2'),
            tooltip=alt.value(None),
        )
        chart += alt.Chart(text_df).mark_text(baseline='middle', align='left', size=20, fill='#fff').encode(
            text='text',
            x=alt.X('x', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_x_min, chart_x_max), padding=0)),
            y=alt.Y('y', type='quantitative', title=None, scale=alt.Scale(
                    domain=(chart_y_min, chart_y_max), padding=0)),
            tooltip=alt.value(None),
        )
        return chart


chart_mode = st.selectbox(
    'ECG Chart Mode',
    options=('Report', 'Continuous'),
)

fig = plot_ecg(lead_signals, sampling_rate, chart_mode)
st.altair_chart(fig, use_container_width=False)

with st.expander("ECG Analysis", expanded=st.session_state["expander_state"]):
    if st.session_state["expander_state"] == False:
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

if st.session_state["expander_state"] == True:
    st.session_state["expander_state"] = False
    # For some reason this fixes the problem!? 0.05 was as short as I could push it. When I went down to 0.01 sometimes the inconsistent button behavior would show up again.
    time.sleep(0.05)
    st.experimental_rerun()
