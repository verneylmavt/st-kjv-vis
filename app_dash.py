import os
import random
import json
import functools
import base64
import io

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

import spacy
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from PIL import Image

import pydeck as pdk

# ----------------------
# Loading Functions
# ----------------------

@functools.lru_cache(maxsize=None)
def load_kjv_clean():
    df = pd.read_csv("data/kjv_clean.csv")
    df = df.astype({col: 'string' for col in df.select_dtypes(include='object').columns})
    return df

@functools.lru_cache(maxsize=None)
def load_kjv_locs_all():
    df = pd.read_csv("data/kjv_locs_all.csv")
    return df

@functools.lru_cache(maxsize=None)
def load_sentiment_by_book():
    df = pd.read_csv("data/sentiment_by_book.csv")
    return df

@functools.lru_cache(maxsize=None)
def load_spacy():
    model_path = os.path.join("data", "en_core_web_sm", "en_core_web_sm-3.8.0")
    if os.path.isdir(model_path):
        return spacy.load(model_path)
    else:
        raise FileNotFoundError(f"SpaCy model not found at {model_path}. Please ensure it is correctly placed.")

@functools.lru_cache(maxsize=None)
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

@functools.lru_cache(maxsize=None)
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

@functools.lru_cache(maxsize=None)
def load_kjv_countries():
    kjv_countries = [
        'Israel', 'State of Palestine', 'Egypt', 'Jordan', 'Lebanon', 'Syria', 
        'Iraq', 'Iran', 
        'Türkiye', 'Cyprus', 'Greece', 'Italy' 
    ]
    return kjv_countries

@functools.lru_cache(maxsize=None)
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

@functools.lru_cache(maxsize=None)
def load_fig_cr_cd():
    with open("models/figure_cr_cd.json", 'r') as f:
        figure_cr_cd_json = f.read()
    fig = go.Figure(json.loads(figure_cr_cd_json))
    return fig

@functools.lru_cache(maxsize=None)
def load_fig_cr_sd():
    with open("models/figure_cr_sd.json", 'r') as f:
        figure_cr_sd_json = f.read()
    fig = go.Figure(json.loads(figure_cr_sd_json))
    return fig

@functools.lru_cache(maxsize=None)
def load_fig_cr_hm():
    with open("models/figure_cr_hm.json", 'r') as f:
        figure_cr_hm_json = f.read()
    fig = go.Figure(json.loads(figure_cr_hm_json))
    return fig

@functools.lru_cache(maxsize=None)
def load_fig_cr_ng():
    with open("models/figure_cr_ng.json", 'r') as f:
        figure_cr_ng_json = f.read()
    fig = go.Figure(json.loads(figure_cr_ng_json))
    return fig

@functools.lru_cache(maxsize=None)
def load_fig_timeline_bc():
    with open("models/figure_timeline_ad.json", 'r') as f:
        figure_timeline_bc_json = f.read()
    fig = go.Figure(json.loads(figure_timeline_bc_json))
    return fig

@functools.lru_cache(maxsize=None)
def load_fig_timeline_ad():
    with open("models/figure_timeline_ce.json", 'r') as f:
        figure_timeline_ad_json = f.read()
    fig = go.Figure(json.loads(figure_timeline_ad_json))
    return fig

@functools.lru_cache(maxsize=None)
def load_fig_cr_pv():
    html_str = open("models/network_cr_pv.html", "r", encoding="utf-8").read()
    return html_str

# ----------------------
# Page Functions (Charts)
# ----------------------

def bible_overview_page():
    header = html.H1("Bible Overview")
    
    labels = [
        "Bible",
        "Old Testament", "New Testament",
        "Torah", "Former Prophets", "Novel", "Poetry", "Latter Prophets",
        "Major Prophets", "Minor Prophets",
        "Early Church", "Epistles", "Prophecy",
        "Gospels", "Acts",
        "Synoptic Gospels",
        "Pauline Letters", "General Letters",
        "Pastoral Epistles",
        "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
        "Joshua", "Judges", "Ruth", "1 Samuel", 
        "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", 
        "2 Chronicles", "Ezra", "Nehemiah",
        "Tobit", "Judith", "Esther", "1 Maccabees", "2 Maccabees",
        "Job", "Psalms", "Proverbs", "Ecclesiastes", 
        "Song of Songs", "Wisdom", "Sirach",
        "Isaiah", "Jeremiah", "Lamentations", 
        "Baruch", "Ezekiel", "Daniel",
        "Hosea", "Joel", "Amos", "Obadiah", "Jonah", 
        "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", 
        "Zechariah", "Malachi",
        "Matthew", "Mark", "Luke", "John",
        "Acts of The Apostles",
        "Romans", "1 Corinthians", "2 Corinthians", "Galatians", 
        "Ephesians", "Philippians", "Colossians", "1 Thessalonians", 
        "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon",
        "Hebrews", "James", "1 Peter", "2 Peter", 
        "1 John", "2 John", "3 John", "Jude",
        "Revelation"
    ]

    parents = [
        "",
        "Bible", "Bible",
        "Old Testament", "Old Testament", "Old Testament", "Old Testament", "Old Testament",
        "Latter Prophets", "Latter Prophets",
        "New Testament", "New Testament", "New Testament",
        "Early Church", "Early Church",
        "Gospels",
        "Epistles", "Epistles",
        "Pauline Letters",
        "Torah", "Torah", "Torah", "Torah", "Torah",
        "Former Prophets", "Former Prophets", "Former Prophets", "Former Prophets", 
        "Former Prophets", "Former Prophets", "Former Prophets", "Former Prophets", 
        "Former Prophets", "Former Prophets", "Former Prophets",
        "Novel", "Novel", "Novel", "Novel", "Novel",
        "Poetry", "Poetry", "Poetry", "Poetry", 
        "Poetry", "Poetry", "Poetry",
        "Major Prophets", "Major Prophets", "Major Prophets", 
        "Major Prophets", "Major Prophets", "Major Prophets",
        "Minor Prophets", "Minor Prophets", "Minor Prophets", "Minor Prophets", "Minor Prophets", 
        "Minor Prophets", "Minor Prophets", "Minor Prophets", "Minor Prophets", "Minor Prophets", 
        "Minor Prophets", "Minor Prophets",
        "Synoptic Gospels", "Synoptic Gospels", "Synoptic Gospels", "Gospels",
        "Acts",
        "Pauline Letters", "Pauline Letters", "Pauline Letters", "Pauline Letters", 
        "Pauline Letters", "Pauline Letters", "Pauline Letters", "Pauline Letters", 
        "Pauline Letters", "Pastoral Epistles", "Pastoral Epistles", "Pastoral Epistles", "Pauline Letters",
        "General Letters", "General Letters", "General Letters", "General Letters", 
        "General Letters", "General Letters", "General Letters", "General Letters",
        "Prophecy"
    ]
    
    values = [
        100,
        85, 75,
        20, 40, 20, 30, 55,
        25, 45,
        20, 60, 10,
        17.5, 10,
        15,
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
    graph = dcc.Graph(figure=fig, config={'responsive': True})
    return html.Div([header, graph])

def verses_book_page():
    header = html.H1("# of Verses per Book")
    
    df = load_kjv_clean().copy()
    kjv_books = load_kjv_books()
    
    verse_count_per_book = df.groupby("book_name").size().reset_index(name="verse_count")
    verse_count_per_book.rename(columns={"book_name": "Book", "verse_count": "# of Verses"}, inplace=True)
    
    chart_verse_count = go.Figure(
        data=[go.Bar(
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
        )]
    )
    
    chart_verse_count.update_layout(
        xaxis_title="# of Verses",
        yaxis_title="Book",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(kjv_books))),
        height=1500
    )
    graph = dcc.Graph(figure=chart_verse_count, config={'responsive': True})
    return html.Div([header, graph])

def chapters_book_page():
    header = html.H1("# of Chapters per Book")
    
    df = load_kjv_clean().copy()
    kjv_books = load_kjv_books()
    
    chapter_count_per_book = df.groupby("book_name")["chapter_number"].max().reset_index()
    chapter_count_per_book.rename(columns={"book_name": "Book", "chapter_number": "# of Chapters"}, inplace=True)
    
    chart_chapter_count = go.Figure(
        data=[go.Bar(
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
        )]
    )
    
    chart_chapter_count.update_layout(
        xaxis_title="# of Chapters",
        yaxis_title="Book",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(kjv_books))),
        height=1500
    )
    graph = dcc.Graph(figure=chart_chapter_count, config={'responsive': True})
    return html.Div([header, graph])

def verses_chapter_page():
    header = html.H1("# of Verses per Chapter")
    
    df = load_kjv_clean().copy()
    kjv_books = load_kjv_books()
    
    chapter_verse_counts = df.groupby(["book_name", "chapter_number"]).size().reset_index(name="verse_count")
    chapter_verse_counts.rename(columns={"book_name": "Book", "chapter_number": "Chapter", "verse_count": "# of Verses"}, inplace=True)
    
    chart_verse_heatmap = go.Figure(
        data=[go.Heatmap(
            x=chapter_verse_counts["Chapter"],
            y=chapter_verse_counts["Book"],
            z=chapter_verse_counts["# of Verses"],
            colorscale="Reds",
            colorbar=dict(title="# of Verses"),
            zmin=0,
            zmax=100,
            hovertemplate="Book: %{y}<br>Chapter: %{x}<br># of Verses: %{z}<extra></extra>"
        )]
    )
    
    chart_verse_heatmap.update_layout(
        xaxis_title="Chapter",
        yaxis_title="Book",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(kjv_books))),
        height=1500
    )
    graph = dcc.Graph(figure=chart_verse_heatmap, config={'responsive': True})
    return html.Div([header, graph])

def lex_rich_book_page():
    header = html.H1("Lexical Richness (Unique-to-Total Word Ratio) per Book")
    
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
    
    chart_lexical_richness = go.Figure(
        data=[go.Bar(
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
        )]
    )
    
    chart_lexical_richness.update_layout(
        xaxis_title="Lexical Richness (Unique-to-Total Word Ratio)",
        yaxis_title="Book",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(kjv_books))),
        height=1500
    )
    graph = dcc.Graph(figure=chart_lexical_richness, config={'responsive': True})
    return html.Div([header, graph])

def snt_book_page():
    header = html.H1("Sentiment Analysis per Book")
    
    sentiment_by_book = load_sentiment_by_book().copy()
    kjv_books = load_kjv_books()
    
    chart_sentiment_by_book = go.Figure(
        data=[go.Bar(
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
        )]
    )
    
    chart_sentiment_by_book.update_layout(
        xaxis_title="Average Sentiment",
        yaxis_title="Book",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(kjv_books))),
        height=1500
    )
    graph = dcc.Graph(figure=chart_sentiment_by_book, config={'responsive': True})
    return html.Div([header, graph])

# ----------------------
# Word Cloud Page and Helper Function
# ----------------------

def generate_word_cloud(book_name, df, pos_tag=None, bg="White"):
    book_verses = df[df["book_name"] == book_name]["verse_text"]
    
    if book_verses.empty:
        print(f"No Verses in Book: {book_name}")
        return None
    
    text = " ".join(book_verses)
    
    if pos_tag is None:
        texts = text
    else:
        nlp = load_spacy()
        doc = nlp(text)
        tokens = [token.text for token in doc if token.pos_ == pos_tag]
        texts = " ".join(tokens)
        
    number = random.randint(1, 7)    
    coloring = np.array(Image.open(f"assets/wordcloud/{number}.png"))
    
    stopwords = set(STOPWORDS)
    
    wordcloud = WordCloud(
        background_color="white",
        mask=coloring,
        stopwords=stopwords,
        repeat=False
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

def word_cloud_page():
    header = html.H1("Word Cloud")
    kjv_books = load_kjv_books()
    pos_map = load_pos_map()
    form = html.Div([
        html.Label("Book"),
        dcc.Dropdown(
            id="wc-book",
            options=[{"label": b, "value": b} for b in kjv_books],
            value=kjv_books[0]
        ),
        html.Label("Part of Speech"),
        dcc.Dropdown(
            id="wc-pos",
            options=[{"label": key, "value": key} for key in list(pos_map.keys())],
            value="All"
        ),
        html.Label("Background"),
        dcc.Dropdown(
            id="wc-bg",
            options=[{"label": "White", "value": "White"}, {"label": "Transparent", "value": "Transparent"}],
            value="White"
        ),
        html.Button("Generate", id="wc-generate", n_clicks=0),
        html.Div(id="wc-output")
    ])
    footer = html.Div([
        html.P("GitHub Repo: verneylmavt/st-kjv-vis - https://github.com/verneylmavt/st-kjv-vis"),
        html.P("Other ML Tasks - https://verneylogyt.streamlit.app/")
    ])
    return html.Div([header, form])

# ----------------------
# Bible Sites Page
# ----------------------

def bib_sites_page():
    header = html.H1("Bible Sites")
    
    df = load_kjv_locs_all().copy()
    df['country'] = df['country'].replace('-', 'State of Palestine')
    kjv_books = load_kjv_books()
    kjv_books_abv = load_kjv_books_abv()
    kjv_countries = load_kjv_countries()
    
    form = html.Div([
        html.Label("Books"),
        dcc.Dropdown(
            id="bs-books",
            options=[{"label": b, "value": b} for b in kjv_books],
            value=kjv_books,
            multi=True
        ),
        html.Label("Countries"),
        dcc.Dropdown(
            id="bs-countries",
            options=[{"label": c, "value": c} for c in kjv_countries],
            value=kjv_countries,
            multi=True
        ),
        html.Div(id="bs-map")
    ])
    footer = html.Div([
        html.P("GitHub Repo: verneylmavt/st-kjv-vis - https://github.com/verneylmavt/st-kjv-vis"),
        html.P("Other ML Tasks - https://verneylogyt.streamlit.app/"),
        html.Hr(),
        html.Div([
            html.P("__Source__: OpenBible.info - Bible Geocoding"),
            html.P("__URL__: https://www.openbible.info/geo/")
        ])
    ])
    return html.Div([header, form])

# ----------------------
# Bible Events Page
# ----------------------

def bib_events_page():
    header = html.H1("Bible Events")
    subheader_bc = html.H2("BC (Before Christ)")
    fig_bc = load_fig_timeline_bc()
    graph_bc = dcc.Graph(figure=fig_bc, config={'responsive': True})
    
    subheader_ad = html.H2("AD (Anno Domini) 'In the Year of Our Lord'")
    fig_ad = load_fig_timeline_ad()
    graph_ad = dcc.Graph(figure=fig_ad, config={'responsive': True})
    
    footer = html.Div([
        html.Hr(),
        html.Div([
            html.P("__Source__: Viz.Bible - Events by Robert Rouse"),
            html.P("__URL__: https://viz.bible/")
        ])
    ])
    return html.Div([header, subheader_bc, graph_bc, subheader_ad, graph_ad])

# ----------------------
# Bible Cross-References Page
# ----------------------

def bib_cr_page():
    header = html.H1("Bible Cross-References")
    dropdown = dcc.Dropdown(
        id="cr-chart",
        options=[
            {"label": "Heatmap", "value": "Heatmap"},
            {"label": "Chord Diagram", "value": "Chord Diagram"},
            {"label": "Sankey Diagram", "value": "Sankey Diagram"},
            {"label": "Network Graph (Plotly)", "value": "Network Graph (Plotly)"},
            {"label": "Network Graph (Pyvis)", "value": "Network Graph (Pyvis)"}
        ],
        value="Heatmap"
    )
    graph_container = html.Div(id="cr-output")
    footer = html.Div([
        html.Hr(),
        html.Div([
            html.P("__Source__: OpenBible.info - Bible Cross References"),
            html.P("__URL__: https://www.openbible.info/labs/cross-references/")
        ])
    ])
    return html.Div([header, dropdown, graph_container])

# ----------------------
# Main App Layout
# ----------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div([
        html.H2("Pages"),
        dcc.RadioItems(
            id="page-selector",
            options=[
                {"label": "Bible Overview", "value": "Bible Overview"},
                {"label": "Verses / Book", "value": "Verses / Book"},
                {"label": "Chapters / Book", "value": "Chapters / Book"},
                {"label": "Verses / Chapter", "value": "Verses / Chapter"},
                {"label": "Lexical Richness / Book", "value": "Lexical Richness / Book"},
                {"label": "Sentiment / Book", "value": "Sentiment / Book"},
                {"label": "Word Cloud", "value": "Word Cloud"},
                {"label": "Bible Sites", "value": "Bible Sites"},
                {"label": "Bible Events", "value": "Bible Events"},
                {"label": "Bible Cross-References", "value": "Bible Cross-References"}
            ],
            value="Bible Overview",
            labelStyle={'display': 'block'}
        )
    ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'borderRight': '1px solid #ccc'}),
    html.Div(id="page-content", style={'width': '75%', 'display': 'inline-block', 'padding': '20px'})
])

# ----------------------
# Callbacks for Page Selection and Interactivity
# ----------------------

@app.callback(Output("page-content", "children"),
              Input("page-selector", "value"))
def display_page(page):
    if page == "Bible Overview":
        return bible_overview_page()
    elif page == "Verses / Book":
        return verses_book_page()
    elif page == "Chapters / Book":
        return chapters_book_page()
    elif page == "Verses / Chapter":
        return verses_chapter_page()
    elif page == "Lexical Richness / Book":
        return lex_rich_book_page()
    elif page == "Sentiment / Book":
        return snt_book_page()
    elif page == "Word Cloud":
        return word_cloud_page()
    elif page == "Bible Sites":
        return bib_sites_page()
    elif page == "Bible Events":
        return bib_events_page()
    elif page == "Bible Cross-References":
        return bib_cr_page()
    else:
        return html.Div("Page not found.")

# Callback for Word Cloud generation
@app.callback(
    Output("wc-output", "children"),
    Input("wc-generate", "n_clicks"),
    State("wc-book", "value"),
    State("wc-pos", "value"),
    State("wc-bg", "value")
)
def update_word_cloud(n_clicks, book_name, pos_label, bg):
    if n_clicks > 0:
        df = load_kjv_clean().copy()
        pos_map = load_pos_map()
        pos_tag = pos_map.get(pos_label, None)
        img = generate_word_cloud(book_name, df, pos_tag=pos_tag, bg=bg)
        if img is None:
            return html.Div("No image generated.")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        src = "data:image/png;base64," + img_str
        return html.Img(src=src, style={'width': '100%'})
    return no_update

# Callback for Bible Sites map generation
@app.callback(
    Output("bs-map", "children"),
    Input("bs-books", "value"),
    Input("bs-countries", "value")
)
def update_bible_sites(selected_books, selected_countries):
    df = load_kjv_locs_all().copy()
    df['country'] = df['country'].replace('-', 'State of Palestine')
    filtered_df = df[(df['book_name'].isin(selected_books)) & (df['country'].isin(selected_countries))].copy()
    if filtered_df.empty:
        return html.Div("No data available.")
    filtered_df["count"] = filtered_df.groupby(["latitude", "longitude"])["latitude"].transform("count")
    min_count, max_count = filtered_df["count"].min(), filtered_df["count"].max()
    filtered_df["color_intensity"] = np.interp(filtered_df["count"], [min_count, max_count], [50, 255])
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
        tooltip=tooltip,
    )
    deck_html = deck.to_html(as_string=True)
    return html.Iframe(srcDoc=deck_html, style={"width": "100%", "height": "1000px", "border": "none"})

# Callback for Bible Cross-References chart selection
@app.callback(
    Output("cr-output", "children"),
    Input("cr-chart", "value")
)
def update_bib_cr(chart):
    if chart == "Heatmap":
        fig = load_fig_cr_hm()
        fig.update_layout(width=1200, height=1200, font=dict(size=12))
        return dcc.Graph(figure=fig, config={'responsive': True})
    elif chart == "Chord Diagram":
        fig = load_fig_cr_cd()
        fig.update_layout(width=1200, height=1200)
        return dcc.Graph(figure=fig, config={'responsive': True})
    elif chart == "Sankey Diagram":
        fig = load_fig_cr_sd()
        fig.update_layout(width=1200, height=1500)
        return dcc.Graph(figure=fig, config={'responsive': True})
    elif chart == "Network Graph (Plotly)":
        fig = load_fig_cr_ng()
        fig.update_layout(width=1200, height=1200)
        return dcc.Graph(figure=fig, config={'responsive': True})
    elif chart == "Network Graph (Pyvis)":
        html_str = load_fig_cr_pv()
        return html.Iframe(srcDoc=html_str, style={"width": "100%", "height": "1100px", "border": "none", "overflow": "auto"})
    else:
        return html.Div("Chart type not recognized.")

if __name__ == "__main__":
    app.run_server(debug=True)