import random
import streamlit as st
import pandas as pd
import numpy as np
import wfdb
import ast
import time
from zipfile import ZipFile
import os.path

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

st.write("""
# ECG Quiz

Click to see a random ECG and try to guess the diagnosis.
""")


@st.cache_data
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


applyFilter()


@st.cache_data
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


def random_record(validated_by_human, second_opinion, heart_axis):
    global record
    st.session_state["validated_by_human"] = validated_by_human
    st.session_state["second_opinion"] = second_opinion
    st.session_state["heart_axis"] = heart_axis
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


@st.cache_data
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = wfdb.rdsamp(path + df.filename_lr)
    else:
        data = wfdb.rdsamp(path + df.filename_hr)
    data = (data[0].transpose(), data[1])
    data = list(zip(data[0], data[1]['units'], data[1]['sig_name']))
    data = [{'signals': np.array(signals), 'unit': unit, 'lead': lead,
             'sampling_rate': sampling_rate} for signals, unit, lead in data]
    return data


lead_signals = load_raw_data(record, sampling_rate, path)


@st.cache_resource
def plot_ecg(lead_signals, sampling_rate):
    fig, axes = wfdb.plot_items(
        [lead['signals'] for lead in lead_signals],
        sig_name=[lead['lead'] for lead in lead_signals],
        fs=sampling_rate,
        sig_units=[lead['unit'] for lead in lead_signals],
        ylabel=[lead['lead'] for lead in lead_signals],
        time_units='seconds',
        figsize=(30, 37),
        sharex=True,
        sharey=False,
        return_fig_axes=True,
    )
    fig.subplots_adjust(hspace=.0, wspace=.0)
    fig.set_clip_on(False)
    major_ticks_x = np.arange(0, 10, 0.2)
    minor_ticks_x = np.arange(0, 10, 0.04)
    major_ticks_y = np.arange(-1.5, 1.5, 0.5)
    minor_ticks_y = np.arange(-1.5, 1.5, 0.1)
    for axis in axes:
        axis.set_ylim(-1.5, 1.5)
        axis.set_xlim(0, 10)
        axis.margins(y=0)
        axis.set_clip_on(False)
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.set_xticks(major_ticks_x)
        axis.set_xticks(minor_ticks_x, minor=True)
        axis.set_yticks(major_ticks_y)
        axis.set_yticks(minor_ticks_y, minor=True)
        axis.grid(which='minor', alpha=0.3)
        axis.grid(which='major', alpha=0.6, linewidth=1.5)
    return fig


fig = plot_ecg(lead_signals, sampling_rate)
st.write(fig)

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

if st.session_state["expander_state"] == True:
    st.session_state["expander_state"] = False
    # For some reason this fixes the problem!? 0.05 was as short as I could push it. When I went down to 0.01 sometimes the inconsistent button behavior would show up again.
    time.sleep(0.05)
    st.experimental_rerun()
