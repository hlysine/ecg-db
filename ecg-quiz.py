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

st.write("""
# ECG Quiz

Click to see a random ECG and try to guess the diagnosis.
""")


@st.cache_data
def load_records():
    # load and convert annotation data
    record_df = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    record_df.scp_codes = record_df.scp_codes.apply(
        lambda x: ast.literal_eval(x))
    return record_df


record_df = load_records()


@st.cache_data
def load_annotations():
    # Load scp_statements.csv for diagnostic aggregation
    annotation_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    return annotation_df


annotation_df = load_annotations()

if st.session_state["record_index"] is None:
    st.session_state["record_index"] = random.randint(0, len(record_df) - 1)

record = record_df.iloc[st.session_state["record_index"]]


def random_record():
    global record
    st.session_state["record_index"] = random.randint(0, len(record_df) - 1)
    st.session_state["expander_state"] = True


st.button("New ECG", key='new_ecg',
          help='Click to see a new ECG', on_click=random_record)

col1, col2, *_ = st.columns(4)

with col1:
    st.write(f"**Patient ID:** {record.patient_id}")
    st.write(f"**Age:** {record.age}")
    st.write(f"**Age:** {'M' if record.sex == 0 else 'F'}")
    st.write(f"**Date:** {record.recording_date}")

with col2:
    st.write(f"**Height:** {record.height}")
    st.write(f"**Weight:** {record.weight}")
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


@st.cache_data
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
> `{annotation.diagnostic_class} > {annotation.diagnostic_subclass}`
>
> {annotation['Statement Category']}
>
> {annotation['SCP-ECG Statement Description']}


""")

    st.write("---------------------")

    col1, col2, *_ = st.columns(4)

    with col1:
        st.write(f"**Heart Axis:** {record.heart_axis}")
        st.write(f"**Infarction Stadium 1:** {record.infarction_stadium1}")
        st.write(f"**Infarction Stadium 2:** {record.infarction_stadium2}")
        st.write(f"**Pacemaker:** {record.pacemaker}")

    with col2:
        st.write(f"**Baseline Drift:** {record.baseline_drift}")
        st.write(f"**Static Noise:** {record.static_noise}")
        st.write(f"**Burst Noise:** {record.burst_noise}")
        st.write(f"**Electrode Problems:** {record.electrodes_problems}")
        st.write(f"**Extra Beats:** {record.extra_beats}")

if st.session_state["expander_state"] == True:
    st.session_state["expander_state"] = False
    # For some reason this fixes the problem!? 0.05 was as short as I could push it. When I went down to 0.01 sometimes the inconsistent button behavior would show up again.
    time.sleep(0.05)
    st.experimental_rerun()
