import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode, GridOptionsBuilder

# basedir = "/tmpssd2/how2-dataset/diffaug"
# basedir = "/tmpssd/data/vatex/diffaug"
basedir = "."
# how2
# src_file = basedir + "/train.en.25.tsv"
# vatex
# src_file = basedir + "/train.en.tsv.sample"
# clipdir = f"/clip-train.en"
# diffdir = f"/img-train.en"

# table hacking
MIN_HEIGHT = 27
MAX_HEIGHT = 200
ROW_HEIGHT = 35

@st.cache
def get_src_lines(src_file):
    with open(src_file, 'r') as f:
        lines = [line.rstrip() for line in f]
    return lines


@st.experimental_memo
def get_dataframe(src_file):

    res_df = pd.read_csv(src_file, sep='\t', header=None)
    res_df.columns = ['Line', 'Src']

    return res_df


if __name__ == '__main__':

    # This needs to be first
    st.set_page_config(layout="wide")

    # horizontal radio buttons
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>',
             unsafe_allow_html=True)

    dataset = st.radio("Dataset", ["how2", "vatex"])

    src_file = f"{basedir}/{dataset}/train.en.tsv"

    df = get_dataframe(src_file)
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_auto_height(False)
    go = gb.build()

    res_table = AgGrid(df,
                       height=min(MIN_HEIGHT + len(df) * ROW_HEIGHT, MAX_HEIGHT),
                       gridOptions=go,
                       update_mode=GridUpdateMode.SELECTION_CHANGED,
                       allow_unsafe_jscode=True)

    sel = res_table['selected_rows']
    if len(sel) != 0:
        sel_idx = sel[0]['Line']

        # sel_no, sel_line = sel[0][:2]
        st.write(f'({sel[0]["Line"]}) {sel[0]["Src"]}')
        # st.write(f'({sel_no}) {sel_line} ')

        # print(f'Selected line {sel_idx}')

        col1, col2 = st.columns([0.5, 0.5])

        with col1:
            st.subheader("Video Still")
            st.image(f'{basedir}/{dataset}/clip-train.en/{sel_idx}.jpg')

        with col2:
            st.subheader("Generated")
            st.image(f'{basedir}/{dataset}/img-train.en/{sel_idx}.png')