import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import networkx as nx
from pyvis.network import Network

import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def main():
    st.set_page_config(layout="wide")
    
    st.title("Bible Cross-Reference")

    # HtmlFile = open("network_cr_nx.html", "r", encoding="utf-8")
    # source_code = HtmlFile.read()
    # components.html(source_code, height=1000)
    
    # HtmlFile = open("network_cr_nx_chapter.html", "r", encoding="utf-8")
    # source_code = HtmlFile.read()
    # components.html(source_code, height=1000)
    
    HtmlFile = open("network_cr_pv.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    components.html(source_code, height=1000, scrolling=True)
    
    # with open("data/figure_cr_sd.json", 'r') as f:
    #     figure_cr_sd_json = f.read()
    
    # fig = go.Figure(json.loads(figure_cr_sd_json))
    
    # with open("data/figure_cr_ng.json", 'r') as f:
    #     figure_cr_ng_json = f.read()
    
    # fig = go.Figure(json.loads(figure_cr_ng_json))
    
    # st.plotly_chart(fig, use_container_width=True)
    
    
if __name__ == "__main__":
    main()