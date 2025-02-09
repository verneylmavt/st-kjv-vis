# ✝️ King James Version (KJV) Bible Analysis and Visualization

This repository provides a comprehensive exploration of Bible datasets by delving deep into the multifaceted dimensions of biblical texts. It offers an in‐depth analysis of the scriptures through metrics such as verse counts, chapter distributions, lexical richness, and sentiment across all the books, while also presenting a detailed overview of the Bible’s organizational structure—from the broad divisions of the Old and New Testaments down to the individual books. The visualizations capture the internal structure and thematic nuances of the Bible, revealing patterns and relationships within its extensive textual tradition.

This repository also examines the interconnections between the various books by mapping out the intricate network of cross-references. It highlights how passages in one book relate to and resonate with those in another, using visual tools like heatmaps, chord diagrams, Sankey diagrams, and network graphs to illustrate the flow of ideas and scriptural dialogue. These visual representations bring to light the rich tapestry of intertextual references that underscore the unity and complexity of the biblical narrative.

In addition to the textual and referential analysis, the repository extends its exploration into the historical and geographical realms of the Bible. It presents a timeline of significant biblical events, covering both BC and AD periods, and details the duration and context of these events along with their scriptural connections. Complementing the timeline, a mapping of ancient and modern biblical sites is provided, linking geographical locations with corresponding verses to offer insights into the historical landscape that underpins the biblical world.

For more information about the training process, please check the `kjv.ipynb`, `kjv_cr.ipynb`, and `kjv_timeline.ipynb` file in the `training` folder. [Check here to see my other ML projects and tasks](https://github.com/verneylmavt/ml-model).

## 🎈 Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://verneylogyt-kjv-vis.streamlit.app/)

<!-- ![Demo GIF](https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/demo.gif) -->

If you encounter message `This app has gone to sleep due to inactivity`, click `Yes, get this app back up!` button to wake the app back up.

## ⚙️ Running Locally

If the demo page is not working, you can fork or clone this repository and run the application locally by following these steps:

<!-- ### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- pip (Python Package Installer)

### Installation Steps -->

1. Clone the repository:

   ```bash
   git clone https://github.com/verneylmavt/st-kjv-vis.git
   cd st-kjv-vis
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app_local.py
   ```

## ⚖️ Acknowledgement

I acknowledge the use of the following datasets, which have been instrumental in conducting the research and developing this project:

<!-- ### King James Version Bible

- **Source**: [https://www.kaggle.com/datasets/kk99807/the-king-james-bible](https://www.kaggle.com/datasets/kk99807/the-king-james-bible)
- **License**: Creative Commons 1.0
- **Description**: This dataset contains the full text of King James Version Bible, one of the most widely read and referenced translations of the Bible. -->

### OpenBible.info - Bible Geocoding

- **Source**: [https://www.openbible.info/geo/](https://www.openbible.info/geo/)
- **License**: Creative Commons Attribution 4.0 International
- **Description**: This dataset contains the locations of every identifiable place mentioned in the Bible, compiled from over seventy modern sources.

### Viz.Bible - Events

- **Source**: [https://viz.bible/](https://viz.bible/)
- **License**: Creative Commons Attribution Share Alike 4.0 International
- **Description**: This dataset contains a structured timeline of biblical events, including details such as dates, duration, predecessors, participants, locations, and corresponding verses.

### OpenBible.info - Bible Cross References

- **Source**: [https://www.openbible.info/labs/cross-references/](https://www.openbible.info/labs/cross-references/)
- **License**: Creative Commons Attribution 4.0 International
- **Description**: This dataset contains approximately 340,000 cross references linking different parts of the Bible based on similar themes, words, events, or people.

I deeply appreciate the efforts of the providers in making these datasets available.

## 📊 Chart Gallery

<!-- <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/screenshots/1.png" width="90%"></img>
<img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/screenshots/7.png" width="45%"></img>
<img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/screenshots/8.png" width="45%"></img>
<img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/screenshots/10_1.png" width="45%"></img>
<img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/screenshots/10_2.png" width="45%"></img>
<img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/screenshots/10_3.png" width="45%"></img>
<img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/screenshots/10_4.png" width="45%"></img>
<img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/screenshots/10_5.png" width="45%"></img> -->

<details>
  <summary>Bible Overview</summary>
  <p>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/models/figure_bible_overview.png"/>
  </p>
</details>

<details>
  <summary>Bible Verses, Bible Chapters, Bible Books</summary>
  <p>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/models/figure_verses_book.png"/>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/models/figure_chapters_book.png"/>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/models/figure_verses_chapter.png"/>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/models/figure_lex_rich_book.png"/>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/models/figure_snt_book.png"/>
  </p>
</details>

<details>
  <summary>Bible WordCloud</summary>
  <p>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/wordcloud/1.png" width="45%"></img>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/wordcloud/1_example.jpg" width="45%"></img>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/wordcloud/2.png" width="45%"></img>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/wordcloud/2_example.jpg" width="45%"></img>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/wordcloud/3.png" width="45%"></img>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/wordcloud/3_example.jpg" width="45%"></img>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/wordcloud/4.png" width="45%"></img>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/wordcloud/4_example.jpg" width="45%"></img>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/wordcloud/5.png" width="45%"></img>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/wordcloud/5_example.jpg" width="45%"></img>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/wordcloud/6.png" width="45%"></img>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/wordcloud/6_example.jpg" width="45%"></img>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/wordcloud/7.png" width="45%"></img>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/wordcloud/7_example.jpg" width="45%"></img>
  </p>
</details>

<details>
  <summary>Bible Events</summary>
  <p>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/models/figure_timeline_bc.png"/>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/models/figure_timeline_ad.png"/>
  </p>
</details>

<details>
  <summary>Bible Cross-References</summary>
  <p>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/models/figure_cr_hm.png"/>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/models/figure_cr_cd.png"/>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/models/figure_cr_sd.png"/>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/models/figure_cr_ng.png"/>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/models/figure_cr_ng_spiral.png"/>
    <img src="https://github.com/verneylmavt/st-kjv-vis/blob/main/assets/models/figure_cr_ng_snail.png"/>
  </p>
</details>
