import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

df = pd.read_csv("data/kjv_locs_all.csv")


def main():
    books_kjv = [
        'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy', 'Joshua', 
        'Judges', 'Ruth', '1 Samuel', '2 Samuel', '1 Kings', '2 Kings', 
        '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah', 'Esther', 'Job', 
        'Psalms', 'Proverbs', 'Ecclesiastes', 'Song of Solomon', 'Isaiah', 
        'Jeremiah', 'Lamentations', 'Ezekiel', 'Daniel', 'Hosea', 'Joel', 'Amos', 
        'Obadiah', 'Jonah', 'Micah', 'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai', 
        'Zechariah', 'Malachi', 'Matthew', 'Mark', 'Luke', 'John', 'Acts', 
        'Romans', '1 Corinthians', '2 Corinthians', 'Galatians', 'Ephesians', 
        'Philippians', 'Colossians', '1 Thessalonians', '2 Thessalonians', 
        '1 Timothy', '2 Timothy', 'Titus', 'Philemon', 'Hebrews', 'James', 
        '1 Peter', '2 Peter', '1 John', '2 John', '3 John', 'Jude', 'Revelation'
    ]
    
    books_kjv_abv = {
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


    countries_kjv = [
        'Syria', 'Jordan', 'Israel', 'Lebanon', 'Iraq', 'Egypt', 'Türkiye', 'Greece', 'North Macedonia', 
        'Italy', 'Libya', 'Cyprus', 'Saudi Arabia', 'Sudan', 'Djibouti', 'Iran', 'Yemen', 'Somalia', 'Azerbaijan', 
        'Armenia', 'Spain', 'Uganda', 'Tunisia', 'Croatia', 'Pakistan', 'Mozambique', 'India', 'Sri Lanka', 
        'Eritrea', 'Oman', 'Kuwait', 'Georgia', 'Bangladesh', 'Malta', 'United Arab Emirates', "Indonesia", "-"
    ]

    st.title("Biblical Sites")
    
    with st.container(border=True):
        selected_books = st.multiselect("Books", books_kjv, default=books_kjv)

        selected_countries = st.multiselect("Countries", countries_kjv, default=countries_kjv)

        filtered_df = df[(df['book_name'].isin(selected_books)) & (df['country'].isin(selected_countries))].copy()
        
        filtered_df["count"] = filtered_df.groupby(["latitude", "longitude"])["latitude"].transform("count")
        min_count, max_count = filtered_df["count"].min(), filtered_df["count"].max()
        filtered_df["color_intensity"] = np.interp(filtered_df["count"], [min_count, max_count], [50, 255])
        

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
                    "<b>Bible.com:</b> <a href='https://www.bible.com/bible/1/{books_kjv_abv[book_name]}.{chapter_number}.{verse_number}' target='_blank'>{book_name} {chapter_number}:{verse_number}</a>"
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
    
if __name__ == "__main__":
    main()