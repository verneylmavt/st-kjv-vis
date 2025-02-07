import os
import random
import json

import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.mention import mention
from streamlit_extras.echo_expander import echo_expander

import spacy

import numpy as np
import pandas as pd

import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from PIL import Image

import pydeck as pdk



# ----------------------
# Loading Function
# ----------------------

@st.cache_data
def load_kjv_clean():
    df = pd.read_csv("data/kjv_clean.csv")
    df = df.astype({col: 'string' for col in df.select_dtypes(include='object').columns})
    return df


@st.cache_data
def load_kjv_locs_all():
    df = pd.read_csv("data/kjv_locs_all.csv")
    return df


@st.cache_data
def load_sentiment_by_book():
    df = pd.read_csv("data/sentiment_by_book.csv")
    return df


@st.cache_data
def load_spacy():
    model_path = os.path.join("data", "en_core_web_sm", "en_core_web_sm-3.8.0")
    if os.path.isdir(model_path):
        return spacy.load(model_path)
    else:
        raise FileNotFoundError(f"SpaCy model not found at {model_path}. Please ensure it is correctly placed.")


@st.cache_data
def load_kjv_books():
    
    old_testament = [
        "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
        "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
        "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra",
        "Nehemiah", "Esther", "Job", "Psalms", "Proverbs",
        "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah", "Lamentations",
        "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
        "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk",
        "Zephaniah", "Haggai", "Zechariah", "Malachi"
    ]
    new_testament = [
        "Matthew", "Mark", "Luke", "John", "Acts",
        "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
        "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy",
        "2 Timothy", "Titus", "Philemon", "Hebrews", "James",
        "1 Peter", "2 Peter", "1 John", "2 John", "3 John",
        "Jude", "Revelation"
    ]
    kjv_books = old_testament + new_testament
    return kjv_books


@st.cache_data
def load_kjv_books_abv():
    kjv_books_abv = {
        "Genesis": "GEN",
        "Exodus": "EXO",
        "Leviticus": "LEV",
        "Numbers": "NUM",
        "Deuteronomy": "DEU",
        "Joshua": "JOS",
        "Judges": "JDG",
        "Ruth": "RUT",
        "1 Samuel": "1SA",
        "2 Samuel": "2SA",
        "1 Kings": "1KI",
        "2 Kings": "2KI",
        "1 Chronicles": "1CH",
        "2 Chronicles": "2CH",
        "Ezra": "EZR",
        "Nehemiah": "NEH",
        "Esther": "EST",
        "Job": "JOB",
        "Psalms": "PSA",
        "Proverbs": "PRO",
        "Ecclesiastes": "ECC",
        "Song of Solomon": "SNG",
        "Isaiah": "ISA",
        "Jeremiah": "JER",
        "Lamentations": "LAM",
        "Ezekiel": "EZK",
        "Daniel": "DAN",
        "Hosea": "HOS",
        "Joel": "JOL",
        "Amos": "AMO",
        "Obadiah": "OBA",
        "Jonah": "JON",
        "Micah": "MIC",
        "Nahum": "NAM",
        "Habakkuk": "HAB",
        "Zephaniah": "ZEP",
        "Haggai": "HAG",
        "Zechariah": "ZEC",
        "Malachi": "MAL",
        "Matthew": "MAT",
        "Mark": "MRK",
        "Luke": "LUK",
        "John": "JHN",
        "Acts": "ACT",
        "Romans": "ROM",
        "1 Corinthians": "1CO",
        "2 Corinthians": "2CO",
        "Galatians": "GAL",
        "Ephesians": "EPH",
        "Philippians": "PHP",
        "Colossians": "COL",
        "1 Thessalonians": "1TH",
        "2 Thessalonians": "2TH",
        "1 Timothy": "1TI",
        "2 Timothy": "2TI",
        "Titus": "TIT",
        "Philemon": "PHM",
        "Hebrews": "HEB",
        "James": "JAS",
        "1 Peter": "1PE",
        "2 Peter": "2PE",
        "1 John": "1JN",
        "2 John": "2JN",
        "3 John": "3JN",
        "Jude": "JUD",
        "Revelation": "REV"
    }
    return kjv_books_abv


@st.cache_data
def load_kjv_countries():
    kjv_countries = [
        'Israel', 'State of Palestine', 'Egypt', 'Jordan', 'Lebanon', 'Syria', 
        'Iraq', 'Iran', 
        # 'Saudi Arabia', 'Kuwait', 'United Arab Emirates', 'Oman', 'Yemen',
        # 'Libya', 'Tunisia',
        # 'Sudan', 'Eritrea', 'Djibouti', 'Somalia', 'Uganda', 'Mozambique',
        'Türkiye', 'Cyprus', 'Greece', 'Italy' 
        # 'North Macedonia', 'Malta', 'Spain', 'Croatia',
        # 'Armenia', 'Azerbaijan', 'Georgia',
        # 'Pakistan', 'India', 'Sri Lanka', 'Bangladesh', 'Indonesia'
    ]
    return kjv_countries


@st.cache_data
def load_pos_map():
    pos_map = {
        "All": None,
        "Adjective": "ADJ",
        "Adposition": "ADP",
        "Adverb": "ADV",
        "Auxiliary": "AUX",
        "Coordinating Conjunction": "CCONJ",
        "Determiner": "DET",
        "Interjection": "INTJ",
        "Noun": "NOUN",
        "Numeral": "NUM",
        "Particle": "PART",
        "Pronoun": "PRON",
        "Proper Noun": "PROPN",
        "Punctuation": "PUNCT",
        "Subordinating Conjunction": "SCONJ",
        "Symbol": "SYM",
        "Verb": "VERB"
    }
    return pos_map

@st.cache_data
def load_fig_cr_cd():
    with open("models/figure_cr_cd.json", 'r') as f:
        figure_cr_cd_json = f.read()
    
    fig = go.Figure(json.loads(figure_cr_cd_json))
    return fig

@st.cache_data
def load_fig_cr_sd():
    with open("models/figure_cr_sd.json", 'r') as f:
        figure_cr_sd_json = f.read()
    
    fig = go.Figure(json.loads(figure_cr_sd_json))
    return fig

@st.cache_data
def load_fig_cr_hm():
    with open("models/figure_cr_hm.json", 'r') as f:
        figure_cr_hm_json = f.read()
    
    fig = go.Figure(json.loads(figure_cr_hm_json))
    return fig

@st.cache_data
def load_fig_cr_ng():
    with open("models/figure_cr_ng.json", 'r') as f:
        figure_cr_ng_json = f.read()
    
    fig = go.Figure(json.loads(figure_cr_ng_json))
    return fig

@st.cache_data
def load_fig_timeline_bc():
    with open("models/figure_timeline_ad.json", 'r') as f:
        figure_timeline_bc_json = f.read()
    
    fig = go.Figure(json.loads(figure_timeline_bc_json))
    return fig

@st.cache_data
def load_fig_timeline_ad():
    with open("models/figure_timeline_ce.json", 'r') as f:
        figure_timeline_ad_json = f.read()
    
    fig = go.Figure(json.loads(figure_timeline_ad_json))
    return fig

@st.cache_data
def load_fig_cr_pv():
    html = open("models/network_cr_pv.html", "r", encoding="utf-8").read()
    return html


# ----------------------
# Page Function (Charts)
# ----------------------


def bible_overview():
    st.header("Bible Overview")
    
    # labels = [
    #     "Bible",
        
    #     "Old Testament", "New Testament",
        
    #     "Torah", "Former Prophets", "Novel", "Poetry", "Latter Prophets",
        
    #     "Major Prophets", "Minor Prophets",
        
    #     "Early Church",
        
    #     "Gospels",
    #     "Acts",
    #     "Epistles",
    #     "Pauline Letters",
    #     "General Letters",
    #     "Prophecy",
    #     # Old Testament
    #     "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    #     "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
    #     "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah",
    #     "Tobit", "Judith", "Esther", "1 Maccabees", "2 Maccabees",
    #     "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Songs", "Wisdom", "Sirach",
    #     "Isaiah", "Jeremiah", "Lamentations", "Baruch", "Ezekiel", "Daniel",
    #     "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk",
    #     "Zephaniah", "Haggai", "Zechariah", "Malachi",
    #     # New Testament
    #     "Matthew", "Mark", "Luke", "John",
    #     "Acts of The Apostles",
    #     "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    #     "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    #     "1 Timothy", "2 Timothy", "Titus", "Philemon",
    #     "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude",
    #     "Revelation",
    #     ""
    # ]

    # parents = [
    #     "",
        
    #     "Bible", "Bible",
        
    #     "Old Testament", "Old Testament", "Old Testament", "Old Testament", "Old Testament",
        
    #     "Latter Prophets", "Latter Prophets",
        
    #     "New Testament", "Early Church", "Early Church", "New Testament",
        
    #     "Epistles", "Epistles", "New Testament",
    #     # Old Testament
    #     "Torah", "Torah", "Torah",
    #     "Torah", "Torah",
    #     "Former Prophets", "Former Prophets", "Former Prophets",
    #     "Former Prophets", "Former Prophets", "Former Prophets",
    #     "Former Prophets", "Former Prophets", "Former Prophets",
    #     "Former Prophets",
    #     "Novel", "Novel", "Novel", "Novel", "Novel",
    #     "Poetry", "Poetry", "Poetry", "Poetry", "Poetry", "Poetry", "Poetry",
    #     "Major Prophets", "Major Prophets", "Major Prophets", "Major Prophets", "Major Prophets", "Major Prophets",
    #     "Minor Prophets", "Minor Prophets", "Minor Prophets", "Minor Prophets",
    #     "Minor Prophets", "Minor Prophets", "Minor Prophets", "Minor Prophets",
    #     "Minor Prophets", "Minor Prophets", "Minor Prophets", "Minor Prophets",
    #     # New Testament
    #     "Gospels", "Gospels", "Gospels",
    #     "Gospels",
    #     "Acts",
    #     "Pauline Letters", "Pauline Letters", "Pauline Letters", "Pauline Letters", "Pauline Letters",
    #     "Pauline Letters", "Pauline Letters", "Pauline Letters", "Pauline Letters",
    #     "Pauline Letters", "Pauline Letters", "Pauline Letters", "Pauline Letters",
    #     "General Letters", "General Letters",
    #     "General Letters", "General Letters",
    #     "General Letters", "General Letters",
    #     "General Letters", "General Letters",
    #     "Prophecy"
    # ]
    
    labels = [
        # 1 = 1
        "Bible",
        # 2 = 2
        "Old Testament", "New Testament",
        # 3 = 5
        "Torah", "Former Prophets", "Novel", "Poetry", "Latter Prophets",
        # 4 = 2
        "Major Prophets", "Minor Prophets",
        # 5 = 3
        "Early Church", "Epistles", "Prophecy",
        # 6 = 2
        "Gospels", "Acts",
        "Synoptic Gospels",
        # 7 = 2
        "Pauline Letters", "General Letters",
        "Pastoral Epistles",
        # 8 = 5
        "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
        # 9 = 11
        "Joshua", "Judges", "Ruth", "1 Samuel", 
        "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", 
        "2 Chronicles", "Ezra", "Nehemiah",
        # 10 = 5
        "Tobit", "Judith", "Esther", "1 Maccabees", "2 Maccabees",
        # 11 = 7
        "Job", "Psalms", "Proverbs", "Ecclesiastes", 
        "Song of Songs", "Wisdom", "Sirach",
        # 12 = 6
        "Isaiah", "Jeremiah", "Lamentations", 
        "Baruch", "Ezekiel", "Daniel",
        # 13 = 12
        "Hosea", "Joel", "Amos", "Obadiah", "Jonah", 
        "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", 
        "Zechariah", "Malachi",
        # 14 = 4
        "Matthew", "Mark", "Luke", "John",
        # 15 = 1
        "Acts of The Apostles",
        # 16 = 13
        "Romans", "1 Corinthians", "2 Corinthians", "Galatians", 
        "Ephesians", "Philippians", "Colossians", "1 Thessalonians", 
        "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon",
        # 17 = 8
        "Hebrews", "James", "1 Peter", "2 Peter", 
        "1 John", "2 John", "3 John", "Jude",
        # 18 = 1
        "Revelation"
    ]

    parents = [
        # 1 =1
        "",
        # 2 = 2
        "Bible", "Bible",
        # 3 = 5
        "Old Testament", "Old Testament", "Old Testament", "Old Testament", "Old Testament",
        # 4 = 2
        "Latter Prophets", "Latter Prophets",
        # 5 = 3
        "New Testament", "New Testament", "New Testament",
        # 6 = 2
        "Early Church", "Early Church",
        "Gospels",
        # 7 = 2
        "Epistles", "Epistles",
        "Pauline Letters",
        # 8 = 5
        "Torah", "Torah", "Torah", "Torah", "Torah",
        # 9 = 11
        "Former Prophets", "Former Prophets", "Former Prophets", "Former Prophets", 
        "Former Prophets", "Former Prophets", "Former Prophets", "Former Prophets", 
        "Former Prophets", "Former Prophets", "Former Prophets",
        # 10 = 5
        "Novel", "Novel", "Novel", "Novel", "Novel",
        # 11 = 7
        "Poetry", "Poetry", "Poetry", "Poetry", 
        "Poetry", "Poetry", "Poetry",
        # 12 = 6
        "Major Prophets", "Major Prophets", "Major Prophets", 
        "Major Prophets", "Major Prophets", "Major Prophets",
        # 13 = 12
        "Minor Prophets", "Minor Prophets", "Minor Prophets", "Minor Prophets", "Minor Prophets", 
        "Minor Prophets", "Minor Prophets", "Minor Prophets", "Minor Prophets", "Minor Prophets", 
        "Minor Prophets", "Minor Prophets",
        # 14 = 4
        "Synoptic Gospels", "Synoptic Gospels", "Synoptic Gospels", "Gospels",
        # 15 = 1
        "Acts",
        # 16 = 13
        "Pauline Letters", "Pauline Letters", "Pauline Letters", "Pauline Letters", 
        "Pauline Letters", "Pauline Letters", "Pauline Letters", "Pauline Letters", 
        "Pauline Letters", "Pastoral Epistles", "Pastoral Epistles", "Pastoral Epistles", "Pauline Letters",
        # 17 = 8
        "General Letters", "General Letters", "General Letters", "General Letters", 
        "General Letters", "General Letters", "General Letters", "General Letters",
        # 18 = 1
        "Prophecy"
    ]

    # st.write(len(labels))
    # st.write(len(parents))
    
    # st.dataframe(
    #     pd.DataFrame(
    #         {
    #             'labels': labels,
    #             'parents': parents
    #         }
    #     )
    # )
    
    values = [
        # 1 = 1
        100,
        # 2 = 2
        85, 75,
        # 3 = 5
        20, 40, 20, 30, 55,
        # 4 = 2
        25, 45,
        # 5 = 3
        20, 60, 10,
        # 6 = 2
        17.5, 10,
        15,
        # 7 = 2 
        45, 25,
        15,
        
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5
    ]
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        marker=dict(
            colors=values,
            colorscale='Reds'
        ),
        branchvalues="total",
    ))
    
    fig.update_traces(insidetextorientation="radial")

    fig.update_layout(
        width=1200,
        height=1200,
        margin=dict(t=0, l=0, r=0, b=0),
    )
    # with st.container(border=True):
    st.plotly_chart(fig, use_container_width=True)
    
    

def verses_book():
    st.header("# of Verses per Book")
    
    df = load_kjv_clean().copy()
    kjv_books = load_kjv_books()
    
    verse_count_per_book = df.groupby("book_name").size().reset_index(name="verse_count")
    verse_count_per_book.rename(columns={"book_name": "Book", "verse_count": "# of Verses"}, inplace=True)
    
    # chart_verse_count = (
    #     alt.Chart(verse_count_per_book)
    #     .mark_bar()
    #     .encode(
    #         x=alt.X("# of Verses:Q", title="# of Verses"),
    #         y=alt.Y("Book:N", sort=kjv_books, title="Book"),
    #         tooltip=["Book", "# of Verses"],
    #     )
    #     .properties(
    #                 # title="# of Verses per Book",
    #                 width='container',
    #                 height=1500)
    #     .interactive()
    # )
    
    chart_verse_count = go.Figure(
        data=[
            go.Bar(
                x=verse_count_per_book["# of Verses"],
                y=verse_count_per_book["Book"],
                orientation='h',
                text=verse_count_per_book["# of Verses"],
                hovertemplate="Book: %{y}<br># of Verses: %{x}<extra></extra>",
                marker=dict(
                    color=verse_count_per_book["# of Verses"],
                    colorscale="Reds",
                    cmin=0,
                    cmax=1500
                )
            )
        ]
    )
    
    chart_verse_count.update_layout(
        xaxis_title="# of Verses",
        yaxis_title="Book",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(kjv_books))),
        height=1500
    )
    
    # st.write("")
    # st.write("")
    # with st.container(border=True):
    # st.altair_chart(chart_verse_count, use_container_width=True, theme="streamlit")
    st.plotly_chart(chart_verse_count, use_container_width=True, selection_mode="points")
        
    # st.feedback("thumbs")
    # mention(
    #         label="GitHub Repo: verneylmavt/st-kjv-vis",
    #         icon="github",
    #         url="https://github.com/verneylmavt/st-kjv-vis"
    #     )
    # mention(
    #         label="Other ML Tasks",
    #         icon="streamlit",
    #         url="https://verneylogyt.streamlit.app/"
    #     )



def chapters_book():
    st.header("# of Chapters per Book")
    
    df = load_kjv_clean().copy()
    kjv_books = load_kjv_books()
    
    chapter_count_per_book = df.groupby("book_name")["chapter_number"].max().reset_index()
    chapter_count_per_book.rename(columns={"book_name": "Book", "chapter_number": "# of Chapters"}, inplace=True)
    
    # chart_chapter_count = (
    #     alt.Chart(chapter_count_per_book)
    #     .mark_bar()
    #     .encode(
    #         x=alt.X("# of Chapters:Q", title="# of Chapters"),
    #         y=alt.Y("Book:N", sort=kjv_books, title="Book"),
    #         tooltip=["Book", "# of Chapters"],
    #     )
    #     .properties(
    #                 # title="# of Chapters per Book",
    #                 width='container',
    #                 height=1500)
    #     .interactive()
    # )
    
    chart_chapter_count = go.Figure(
        data=[
            go.Bar(
                x=chapter_count_per_book["# of Chapters"],
                y=chapter_count_per_book["Book"],
                orientation='h',
                text=chapter_count_per_book["# of Chapters"],
                hovertemplate="Book: %{y}<br># of Chapters: %{x}<extra></extra>",
                marker=dict(
                    color=chapter_count_per_book["# of Chapters"],
                    colorscale="Reds",
                    cmin=0,
                    cmax=75
                )
            )
        ]
    )
    
    chart_chapter_count.update_layout(
        xaxis_title="# of Chapters",
        yaxis_title="Book",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(kjv_books))),
        height=1500
    )
    
    # st.write("")
    # st.write("")
    # with st.container(border=True):
    # st.altair_chart(chart_chapter_count, use_container_width=True, theme="streamlit")
    st.plotly_chart(chart_chapter_count, use_container_width=True, selection_mode="points")
        
    # st.feedback("thumbs")
    # mention(
    #         label="GitHub Repo: verneylmavt/st-kjv-vis",
    #         icon="github",
    #         url="https://github.com/verneylmavt/st-kjv-vis"
    #     )
    # mention(
    #         label="Other ML Tasks",
    #         icon="streamlit",
    #         url="https://verneylogyt.streamlit.app/"
    #     )



def verses_chapter():
    st.header("# of Verses per Chapter")
    
    df = load_kjv_clean().copy()
    kjv_books = load_kjv_books()
    
    chapter_verse_counts = df.groupby(["book_name", "chapter_number"]).size().reset_index(name="verse_count")
    chapter_verse_counts.rename(columns={"book_name": "Book", "chapter_number": "Chapter", "verse_count": "# of Verses"}, inplace=True)
    
    # chart_verse_heatmap = (
    #     alt.Chart(chapter_verse_counts)
    #     .mark_rect()
    #     .encode(
    #         x=alt.X("Chapter:O", title="Chapter"),
    #         y=alt.Y("Book:N", sort=kjv_books, title="Book"),
    #         color=alt.Color("# of Verses:Q", title="# of Verses", scale=alt.Scale(scheme="blues")),
    #         tooltip=["Book", "Chapter", "# of Verses"],
    #     )
    #     .properties(
    #                 # title="# of Verses per Chapter",
    #                 width='container', 
    #                 height=1500)
    #     .interactive()
    # )
    
    chart_verse_heatmap = go.Figure(
        data=[
            go.Heatmap(
                x=chapter_verse_counts["Chapter"],
                y=chapter_verse_counts["Book"],
                z=chapter_verse_counts["# of Verses"],
                colorscale="Reds",
                colorbar=dict(title="# of Verses"),
                zmin=0,
                zmax=100,
                hovertemplate="Book: %{y}<br>Chapter: %{x}<br># of Verses: %{z}<extra></extra>"
            )
        ]
    )
    
    chart_verse_heatmap.update_layout(
        xaxis_title="Chapter",
        yaxis_title="Book",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(kjv_books))),
        height=1500
    )
    
    # st.write("")
    # st.write("")
    # with st.container(border=True):
    # st.altair_chart(chart_verse_heatmap, use_container_width=True, theme="streamlit")
    st.plotly_chart(chart_verse_heatmap, use_container_width=True, selection_mode="points")
        
    # st.feedback("thumbs")
    # mention(
    #         label="GitHub Repo: verneylmavt/st-kjv-vis",
    #         icon="github",
    #         url="https://github.com/verneylmavt/st-kjv-vis"
    #     )
    # mention(
    #         label="Other ML Tasks",
    #         icon="streamlit",
    #         url="https://verneylogyt.streamlit.app/"
    #     )



def lex_rich_book():
    st.header("Lexical Richness (Unique-to-Total Word Ratio) per Book")
    
    df = load_kjv_clean().copy()
    kjv_books = load_kjv_books()

    
    def calculate_lexical_richness(df):
        lexical_data = []
        
        for book, verses in df.groupby("book_name"):
            all_text = " ".join(verses["verse_text"])
            words = all_text.split()
            total_words = len(words)
            unique_words = len(set(words))
            lexical_richness = unique_words / total_words if total_words > 0 else 0
            lexical_data.append({"book_name": book, "lexical_richness": lexical_richness})
            
        return pd.DataFrame(lexical_data)
    
    
    lexical_richness_df = calculate_lexical_richness(df)
    lexical_richness_df.rename(columns={"book_name": "Book", "lexical_richness": "Lexical Richness"}, inplace=True)
    
    # chart_lexical_richness = (
    #     alt.Chart(lexical_richness_df)
    #     .mark_bar()
    #     .encode(
    #         x=alt.X("Lexical Richness:Q", title="Lexical Richness (Unique-to-Total Word Ratio)"),
    #         y=alt.Y("Book:N", title="Book", sort=kjv_books),
    #         tooltip=["Book", "Lexical Richness"]
    #     )
    #     .properties(
    #                 # title="Lexical Richness by Book", 
    #                 width='container', 
    #                 height=1500)
    #     .interactive()
    # )
    
    chart_lexical_richness = go.Figure(
        data=[
            go.Bar(
                x=lexical_richness_df["Lexical Richness"],
                y=lexical_richness_df["Book"],
                orientation='h',
                text=lexical_richness_df["Lexical Richness"],
                hovertemplate="Book: %{y}<br>Lexical Richness: %{x}<extra></extra>",
                marker=dict(
                    color=lexical_richness_df["Lexical Richness"],
                    colorscale="Reds",
                    cmin=0,
                    cmax=0.6
                )
            )
        ]
    )
    
    chart_lexical_richness.update_layout(
        xaxis_title="Lexical Richness (Unique-to-Total Word Ratio)",
        yaxis_title="Book",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(kjv_books))),
        height=1500
    )
    
    # st.write("")
    # st.write("")
    # with st.container(border=True):
    # st.altair_chart(chart_lexical_richness, use_container_width=True, theme="streamlit")
    st.plotly_chart(chart_lexical_richness, use_container_width=True, selection_mode="points")



def snt_book():
    st.header("Sentiment Analysis per Book")
    
    sentiment_by_book = load_sentiment_by_book().copy()
    kjv_books = load_kjv_books()
    
    chart_sentiment_by_book = go.Figure(
        data=[
            go.Bar(
                x=sentiment_by_book["Average Sentiment"],
                y=sentiment_by_book["Book"],
                orientation='h',
                text=sentiment_by_book["Average Sentiment"],
                hovertemplate="Book: %{y}<br>Average Sentiment: %{x}<extra></extra>",
                marker=dict(
                    color=sentiment_by_book["Average Sentiment"],
                    colorscale="RdBu",
                    cmin=-0.2,
                    cmax=0.5
                )
            )
        ]
    )

    chart_sentiment_by_book.update_layout(
        xaxis_title="Average Sentiment",
        yaxis_title="Book",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(kjv_books))),  # Ensure Genesis appears first
        height=1500
    )
    # st.write("")
    # st.write("")
    # with st.container(border=True):
    # st.altair_chart(chart_sentiment_by_book, use_container_width=True, theme="streamlit")
    st.plotly_chart(chart_sentiment_by_book, use_container_width=True, selection_mode="points")
    # st.divider()



def word_cloud():
    st.header("Word Cloud")
    
    nlp = load_spacy()
    df = load_kjv_clean().copy()
    kjv_books = load_kjv_books()
    pos_map = load_pos_map()
    
    
    def generate_word_cloud(book_name, df, pos_tag=None, bg="White"):
        book_verses = df[df["book_name"] == book_name]["verse_text"]
        
        if book_verses.empty:
            print(f"No Verses in Book: {book_name}")
            return
        
        text = " ".join(book_verses)
        
        if pos_tag is None:
            texts = text
        else:
            doc = nlp(text)
            tokens = [token.text for token in doc if token.pos_ == pos_tag]
            texts = " ".join(tokens)
            
        number = random.randint(1, 7)    
        coloring = np.array(Image.open(f"assets/wordcloud/{number}.png"))
        
        stopwords = set(STOPWORDS)
        
        wordcloud = WordCloud(
                            background_color="white",
                            mask = coloring,
                            stopwords=stopwords,
                            repeat = False
                            )
        wordcloud.generate(texts)
        
        image_colors = ImageColorGenerator(coloring)
        wordcloud.recolor(color_func=image_colors)
        
        wordcloud_image = wordcloud.to_image()
        
        def remove_bg(wordcloud_image):
            img = wordcloud_image.convert("RGBA")
            datas = img.getdata()
            
            new_data = []
            for item in datas:
                if item[0] == 255 and item[1] == 255 and item[2] == 255:
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)

            img.putdata(new_data)
            return img
        
        if bg == "White":
            img = wordcloud_image
        elif bg == "Transparent":
            img = remove_bg(wordcloud_image)
            
        return img
    
    
    with st.form(key='word_sims_form'):
        book_name = st.selectbox(
            "Book",
            kjv_books,
        )
        pos_tag = st.selectbox(
            "Part of Speech",
            list(pos_map.keys()),
        )
        bg = st.selectbox(
            "Background",
            ["White", "Transparent"],
        )
        submit_button = st.form_submit_button(label='Generate')
        if submit_button:
            if book_name and pos_tag:
                wordcloud_image = generate_word_cloud(book_name, df, pos_tag=pos_map[pos_tag], bg=bg)
                st.image(wordcloud_image, use_container_width=True)
                
                
    st.feedback("thumbs")
    mention(
            label="GitHub Repo: verneylmavt/st-kjv-vis",
            icon="github",
            url="https://github.com/verneylmavt/st-kjv-vis"
        )
    mention(
            label="Other ML Tasks",
            icon="streamlit",
            url="https://verneylogyt.streamlit.app/"
        )



def bib_sites():
    st.header("Bible Sites")
    
    df = load_kjv_locs_all().copy()
    df['country'] = df['country'].replace('-', 'State of Palestine')
    # st.dataframe(df.head())
    kjv_books = load_kjv_books()
    kjv_books_abv = load_kjv_books_abv()
    kjv_countries = load_kjv_countries()
    
    with st.container(border=True):
        selected_books = st.multiselect("Books", kjv_books, default=kjv_books)

        selected_countries = st.multiselect("Countries", kjv_countries, default=kjv_countries)

        filtered_df = df[(df['book_name'].isin(selected_books)) & (df['country'].isin(selected_countries))].copy()
        
        filtered_df["count"] = filtered_df.groupby(["latitude", "longitude"])["latitude"].transform("count")
        min_count, max_count = filtered_df["count"].min(), filtered_df["count"].max()
        filtered_df["color_intensity"] = np.interp(filtered_df["count"], [min_count, max_count], [50, 255])
        # st.dataframe(filtered_df)

        if filtered_df.empty:
            pass
        else:
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=filtered_df,
                get_position=["longitude", "latitude"],
                get_fill_color="[255, 0, 0, color_intensity]",
                get_radius=500,
                pickable=True,
            )

            mid_lat = filtered_df["latitude"].mean()
            mid_long = filtered_df["longitude"].mean()

            view_state = pdk.ViewState(
                longitude=mid_long,
                latitude=mid_lat,
                zoom=5,
                pitch=0,
            )

            tooltip = {
                "html": (
                    "<b>Verse:</b> {book_name} {chapter_number}:{verse_number}<br/>"
                    "<b>Scripture:</b> \"{verse_text}\"<br/>"
                    "<b>Version:</b> KJV (King James Version)<br/>"
                    "<b>Ancient Place:</b> {name_id_ancient}<br/>"
                    "<b>Modern Place:</b> {name_id_modern}<br/>"
                    "<b>----------</b><br/>"
                    # "<b>Address:</b> {formatted_address}<br/>"
                    # "<b>Latitude:</b> {latitude}<br/>"
                    # "<b>Longitude:</b> {longitude}<br/>"
                    "<b>District / County:</b> {administrative_area_level_2}<br/>"
                    "<b>State / Province:</b> {administrative_area_level_1}<br/>"
                    "<b>Country:</b> {country}<br/>"
                    "<b>----------</b><br/>"
                    "<b>Google Maps:</b> <a href='https://www.google.com/maps/place/{latitude},{longitude}' target='_blank'>{name_id_modern}</a> <br/>"
                    "<b>Bible.com:</b> <a href='https://www.bible.com/bible/1/{kjv_books_abv[book_name]}.{chapter_number}.{verse_number}' target='_blank'>{book_name} {chapter_number}:{verse_number}</a>"
                ),
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white"
                }
            }

            deck = pdk.Deck(
                layers=[layer],
                map_provider="carto",
                initial_view_state=view_state,
                height = 1000,
                tooltip=tooltip,
            )

            st.pydeck_chart(deck)
            
    st.feedback("thumbs")
    mention(
            label="GitHub Repo: verneylmavt/st-kjv-vis",
            icon="github",
            url="https://github.com/verneylmavt/st-kjv-vis"
        )
    mention(
            label="Other ML Tasks",
            icon="streamlit",
            url="https://verneylogyt.streamlit.app/"
        )
    
    st.divider()
    container = st.container(border=True)
    container.markdown("""
    __Source__: OpenBible.info - Bible Geocoding  
    __URL__: https://www.openbible.info/geo/
    """)



def bib_events():
    st.header("Bible Events")

    st.subheader("BC (Before Christ)")
    fig = load_fig_timeline_bc()
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("")
    
    st.subheader("AD (Anno Domini) 'In the Year of Our Lord'")
    fig = load_fig_timeline_ad()
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    container = st.container(border=True)
    container.markdown("""
    __Source__: Viz.Bible - Events by Robert Rouse  
    __URL__: https://viz.bible/
    """)
    

def bib_cr():
    st.header("Bible Cross-Reference")
    
    chart = st.selectbox(
            "Chart",
            ["Heatmap", "Chord Diagram", "Sankey Diagram",  "Network Graph (Plotly)", "Network Graph (Pyvis)"]
        )
    
    if chart:
        if chart == "Heatmap":
            fig = load_fig_cr_hm()
            fig.update_layout(
                width=1200,
                height=1200,
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart == "Chord Diagram":
            fig = load_fig_cr_cd()
            fig.update_layout(
                width=1200,
                height=1200,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart == "Sankey Diagram":
            fig = load_fig_cr_sd()
            fig.update_layout(
                width=1200,
                height=1500,
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart == "Network Graph (Plotly)":
            fig = load_fig_cr_ng()
            fig.update_layout(
                width=1200,
                height=1200,
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart == "Network Graph (Pyvis)":
            html = load_fig_cr_pv()
            components.html(html, height=1100, scrolling=True)
        
        st.divider()
        container = st.container(border=True)
        container.markdown("""
        __Source__: OpenBible.info - Bible Cross References  
        __URL__: https://www.openbible.info/labs/cross-references/
        """)
    
    
    
    
# ----------------------
# Page UI
# ----------------------

def main():
    st.set_page_config(layout="wide")
    
    st.title("KJV Bible Analysis and Visualization")
    st.divider()
    
    with st.sidebar:
        st.title("Pages")
        category = st.radio("Charts", ["Bible Overview", 
                                    "Verses / Book", "Chapters / Book", "Verses / Chapter", 
                                    "Lexical Richness / Book", "Sentiment / Book",
                                    "Word Cloud", 
                                    "Bible Sites", "Bible Events", "Bible Cross-Reference"])
    
    if category == "Bible Overview":
        bible_overview()    
    elif category == "Verses / Book":
        verses_book()
    elif category == "Chapters / Book":
        chapters_book()
    elif category == "Verses / Chapter":
        verses_chapter()
    elif category == "Lexical Richness / Book":
        lex_rich_book()
    elif category == "Sentiment / Book":
        snt_book()
    elif category == "Word Cloud":
        word_cloud()
    elif category == "Bible Sites":
        bib_sites()
    elif category == "Bible Events":
        bib_events()
    elif category == "Bible Cross-Reference":
        bib_cr()

if __name__ == "__main__":
    main()