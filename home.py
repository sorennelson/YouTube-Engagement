import streamlit as st
import pandas as pd
import numpy as np
import chromadb, os, cohere
import chromadb.utils.embedding_functions as embedding_functions
from threading import Thread
from helpers import *
from youtube import get_youtube_video_df

import time

DF_PATH = "files/videos.csv"
DF_PATH_FALLBACK = "files/videos-fallback.csv"
DF_CHANNEL_IMPROVEMENT_PATH = "files/channel_improvement.csv"
VIDEO_EMB_COLLECTION = "videos-0926-large-512"
TERMS_EMB_COLLECTION = "terms-large-512"
THUMBNAIL_EMB_COLLECTION = "thumbnails"
EMB_MODEL_NAME = "text-embedding-3-large"
EMB_DIMENSIONS = 512
CHROMA_API_KEY = os.environ['CHROMA_API_KEY']
CHROMA_TENANT = os.environ['CHROMA_TENANT']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
COHERE_API_KEY = os.environ['COHERE_API_KEY']
N_TERMS = 3
FEATURE_SCALE = 0.1
N_BOOST = 3
N_VIDS = 3

# Set up to df / youtube
if "df" not in st.session_state:
    # Use the fallback if the main file hasn't been created yet
    if not os.path.exists(DF_PATH):
      st.session_state.df = pd.read_csv(DF_PATH_FALLBACK)
    else:
      st.session_state.df = pd.read_csv(DF_PATH)
    
    # Load the latest videos on background thread
    Thread(target=get_youtube_video_df, args=(st.session_state, DF_PATH, DF_PATH_FALLBACK), daemon=True).start()

    # Sort by publication
    st.session_state.df['published_at_datetime'] = pd.to_datetime(st.session_state.df['published_at'], utc=True)
    st.session_state.df = st.session_state.df.sort_values(by='published_at_datetime', ascending=False)
    # add a channel_title: title column for selectbox
    st.session_state.df['channel_title_with_title'] = st.session_state.df['channel_title'].str.cat(st.session_state.df['title'], sep=': ')

    st.session_state.client = chromadb.CloudClient(
      api_key=CHROMA_API_KEY,
      tenant=CHROMA_TENANT,
      database='Youtube'
    )

    st.session_state.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
      api_key=OPENAI_API_KEY,
      model_name=EMB_MODEL_NAME,
      dimensions=EMB_DIMENSIONS
    )
    st.session_state.video_collection = st.session_state.client.create_collection(
      name=VIDEO_EMB_COLLECTION,
      embedding_function=st.session_state.openai_ef, 
      get_or_create=True  
    )
    st.session_state.term_collection = st.session_state.client.create_collection(
        name=TERMS_EMB_COLLECTION, 
        embedding_function=st.session_state.openai_ef, 
        get_or_create=True  
    )

    st.session_state.multimodal_cohere_ef = cohere.Client('fYTsfO2Z9yEiOUXNuGblRc8Phcpo6J15S6rgroEF')

    st.session_state.thumbnail_collection = st.session_state.client.create_collection(
        name=THUMBNAIL_EMB_COLLECTION,
        get_or_create=True
    )

    st.session_state.prev_video_id = None


# CSS design tweaks

st.markdown(
    """
    <style>
      /* Main container */
      [data-testid="stMainBlockContainer"] {
          padding-top: 80pxr !important;
          max-width: 900px !important;
      }
      /* Logo */
      [data-testid="stToolbar"] div div div div img {
          height: 40px !important;
      }
      /* Toolbar color */
      [data-testid="stToolbar"] {
          border-bottom: 0.5px solid rgba(60,64,68,.5);
      }

      h1 {
        text-align: center !important;
        padding-top: 0 !important;
        padding-bottom: 32px !important;

        font-size: 2.5rem !important;
      }

      h6, h5 {
          padding-bottom: 8px !important;
      }
      p {
          margin-bottom: 8px !important;
      }

      /* Title tiles */
      .st-key-title {
          padding-top: 8px !important;
          padding-left: 16px !important;
      }

      [data-testid="stCaptionContainer"] {
          margin-bottom: -12px !important;
      }

      button[kind="secondary"], button[kind="primary"], .stButton > button {
          padding-top: 12px !important;
      }

      .stSpinner i {
          margin-top: 8px;
      }

      div[data-testid="stSpinner"] > div:first-child{
          display: flex;
          justify-content: center;
          align-items: center;
          margin-top: 16px !important;
      }

      /* Keep "videos like this section" containers all same height */
      div[class*="st-key-videos-like-this"] [data-testid="stLayoutWrapper"] {
          height: 100% !important;
          min-height: 0 !important;
          flex: 1 1 auto !important;
      }

      /* Floor the captions in videos like this section */
      /* Target the parent of the caption container */
      div[class*="st-key-videos-like-this-caption"] [data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;          
      }

      /* Push the inner caption to the bottom */
      div[class*="st-key-videos-like-this-caption"] [data-testid="stVerticalBlock"] > div:first-child {
        padding-top: 16px !important;
        margin-top: auto !important;      
      }

      /* Keep "thumbnails like this section" containers all same height */
      div[class*="st-key-thumbnails-like-this"] [data-testid="stLayoutWrapper"] {
          height: 100% !important;
          min-height: 0 !important;
          flex: 1 1 auto !important;
      }

      /* Floor the captions in thumbnails like this section */
      /* Target the parent of the caption container */
      div[class*="st-key-thumbnails-like-this-caption"] [data-testid="stVerticalBlock"] {
        display: flex !important;
        flex-direction: column !important;
        height: 100% !important;          
      }

      /* Push the inner caption to the bottom */
      div[class*="st-key-thumbnails-like-this-caption"] [data-testid="stVerticalBlock"] > div:first-child {
        padding-top: 16px !important;
        margin-top: auto !important;      
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# Logo/ title/tab title
st.logo("files/view.png", size="large", icon_image="files/view.png")
st.set_page_config(page_title="View")
st.title("View")

# Select video
title = st.selectbox(
    "Search YouTube Videos",
    st.session_state.df['channel_title_with_title'].to_list(),
    index=0,
    placeholder="Video title",
    accept_new_options=False,
)

# Filter based on selectbox search
filtered_df = st.session_state.df
if title:
  filtered_df = filtered_df.loc[filtered_df['channel_title_with_title'] == title]
  print(len(filtered_df), filtered_df.iloc[:5]['title'])

# Update video to the first video in the selectbox
video = None
if len(filtered_df):
  video = filtered_df.iloc[0]
  # Reset content improvement and title rewrite if we select a new video
  if video['id'] != st.session_state.prev_video_id:
    st.session_state.content_improvement = None
    st.session_state.new_title = None
  st.session_state.prev_video_id = video['id']

# Title and Thumbnail
vid_data = []
if video is not None:
  title_row = st.columns([1,3])
  with title_row[0]:
    st.image(video.loc['thumbnail'] , width='stretch')
  
  with title_row[1]:
    title_tile = st.container(border=False, key="title", height=126)  
    print(video)
    title_tile.markdown(f"##### {video.loc['title']}")
    title_tile.caption(f"{video.loc['channel_title']}")


if video is not None:
  # Capture video info
  channel_df = st.session_state.df[st.session_state.df['channel_id'] == video['channel_id']]
  channel_views = channel_df['views'].sum()
  channel_median_views = channel_df['views'].mean()
  channel_vids_count = channel_df['id'].count()

  vid_data = [
    [
      ('Views', format_number(video.loc['views'])), 
      ('Likes', format_number(video.loc['likes'])), 
      ('Comments', format_number(video.loc['comments'])), 
    ],
    [
      ('Channel views', format_number(channel_views)), 
      ('Avg channel views', format_number(channel_median_views)), 
      ('Subscribers', format_number(video.loc['subscriber_count']))
    ]
  ]

  # Dispaly Video info
  for r, row_data in enumerate(vid_data):
    row1 = st.columns(3)
    for i, col in enumerate(row1):
      with col:
        if len(row_data) > i:
          tile = st.container(border=True, height=60)
          data = row_data[i]
          tile.markdown(f"###### {data[1]} {data[0]}")


# Top/bottom n prediction
if video is not None:
  top_n = predict_top_n_percent(
    st.session_state.df, 
    video, 
    st.session_state.video_collection, 
    st.session_state.openai_ef
  )
  if top_n is not None:
    tile = st.container(border=True)
    top_bottom = "bottom" if top_n <= 0.5 else "top"
    # Want >= 50%/25% not <=
    print(f"Top_n {top_n}")
    if top_n > 0.5:
      top_n = (1-top_n)+0.25
    tile.markdown(f"Expected to be in the {top_bottom} {int(top_n*100)}% in channel views")
    tile.caption(f"Video Prediction")


# Title Rewrite
if video is not None:
  rw_col = None
  if not st.session_state.new_title:
    if st.session_state.get('Rewrite Btn'):
      rw_col, = st.columns(1)
      # Display loader
      with rw_col:
        with st.spinner("Thinking: This may take up to a minute", width="stretch"):
          st.session_state.new_title = rewrite_title(video, OPENAI_API_KEY)

    else:
      title_rewrite = st.button("Title Rewrite With AI", width='stretch', key='Rewrite Btn')

  if st.session_state.new_title:
    col = rw_col if rw_col else st
    tile = col.container(border=True)
    tile.markdown(st.session_state.new_title)
    tile.caption(f"Title Rewrite")


# Content improvement
if video is not None:

  ci_col = None
  if not st.session_state.content_improvement:
    if st.session_state.get('Channel Improvement Btn'):
      ci_col, = st.columns(1)
      # Display loader
      with ci_col:
        with st.spinner("Thinking: This may take a few minutes", width="stretch"):
          st.session_state.content_improvement = get_channel_improvement(
            st.session_state.df, video['channel_id'], DF_CHANNEL_IMPROVEMENT_PATH, OPENAI_API_KEY
          )

    else:
      content_improvement_btn = st.button(
        "Improve My Channel With AI", width='stretch', key='Channel Improvement Btn'
      )

  if st.session_state.content_improvement:
    col = ci_col if ci_col else st
    tile = col.container(border=True)
    tile.subheader("Channel Improvement")
    tile.markdown(st.session_state.content_improvement)
    

st.container(border=False, height=10)
tab1, tab2 = st.tabs(["Video", "Channel"], default=None)

with tab1:
  # Features
  if video is not None:
    # st.container(border=False, height=12)
    st.container(border=False, height=10)
    st.subheader("Video Features")

    # New feature
    new_feature = st.text_input(
      "Feature impact: How much does the model see this feature in a description of the video?", 
      placeholder='Test your own feature (e.g. Viral title, Exciting product, Popular guest)'
    )
    row2 = st.columns(3)

    # Get video embedding
    video_emb_docs = get_from_chroma_with_ids(
      st.session_state.video_collection, 
      [video['id']]
    )

    # Upload embedding if it doesn't exist
    if not len(video_emb_docs['embeddings']):
      print(f"Uploading embedding")

      add_emb_to_chroma(
        st.session_state.video_collection, 
        video['id'], 
        None, 
        video['embedding_text']
      )
      video_emb_docs = get_from_chroma_with_ids(
        st.session_state.video_collection, 
        [video['id']]
      )

    # Query with video embedding
    top_term_docs = st.session_state.term_collection.query(
      query_embeddings=video_emb_docs['embeddings'],
      n_results=N_TERMS,
    )
    print(top_term_docs)

    # If we found any relevant terms, display them
    if top_term_docs.get('documents'):
      top_terms = top_term_docs['documents'][0]
      dist = top_term_docs.get('distances')
      dist = dist[0] if dist else dist

      for i, col in enumerate(row2):
        with col:
          if len(top_terms) > i and format_impact(dist[i]):
            tile = st.container(border=True)
            term = top_terms[i].replace('episode', '').replace('Episode', '')
            tile.markdown(f"###### {term}")
            tile.caption(f"Estimated Impact: {format_impact(dist[i])}")

    # If searched for a feature, display the impact
    if new_feature:
      try:

        # Get video embedding
        video_emb_docs = get_from_chroma_with_ids(
          st.session_state.video_collection, 
          [video['id']]
        )
        video_emb = video_emb_docs.get('embeddings')[0]

        # Get feature embedding
        feature_emb = np.array(st.session_state.openai_ef(new_feature)[0])
        # Compute squared L2 distance between embeddings (default Chroma distance)
        term_dist = np.sum((video_emb - feature_emb) ** 2)

        print(term_dist)

        if term_dist:
          tile = st.container(border=True)
          tile.markdown(f"###### {new_feature}")
          tile.caption(f"Estimated Impact: {format_impact(term_dist)}")

      except Exception as e:
        st.error(f"Error generating embedding: {e}")


  # Boostable features
  if video is not None:

    terms = predict_boostable_features(
          st.session_state.video_collection, 
          st.session_state.term_collection, 
          video['id'], 
          feature_scale=FEATURE_SCALE, 
          n_boostable=N_BOOST
      )
    print(terms)
    if terms:
      st.markdown(
        "<div style='font-weight:400; font-size:14px; margin-bottom:8px;'>Features to improve: Which features are estimated to increase views if enhanced?</div>",
        unsafe_allow_html=True,
        help="Depending on the feature, we recommend strengthening these in the title/description, or adding them to your content."
      )
      for i, col in enumerate(st.columns(3)):
        if len(terms) == i:
          break
        with col:
          tile = st.container(border=True)
          term = terms[i][0].replace('episode', '').replace('Episode', '')
          tile.markdown(f"###### {term}")
          tile.caption(f"Estimated Impact: {int(terms[i][1])}% Increase")


  # Most similar videos
  if video is not None:
    top_videos = get_most_similar_videos(
      st.session_state.df, 
      st.session_state.video_collection,  
      video['id'],  
      n_similar=N_VIDS
    )
    if top_videos is not None and len(top_videos):
      st.container(border=False, height=20)

      st.subheader("Videos like this")
      outer_cont = st.container(border=False, key="videos-like-this")
      cols = outer_cont.columns(3)
      for i in range(len(top_videos[:3])):
        top_video = top_videos.iloc[i]

        col = cols[i]
        container = col.container(border=True)
        container.image(top_video['thumbnail'])
        container.markdown(f"###### {top_video['title']}")
        caption_cont = container.container(border=False, key=f'videos-like-this-caption-{i}')
        left_cap, right_cap = caption_cont.columns([2,1])
        left_cap.caption(f"{top_video['channel_title']}")
        right_cap.caption(f"{format_number(top_video['views'])} Views")

  # Most similar thumbnails
  if video is not None:
    top_videos = get_most_similar_thumbnails(
      st.session_state.df, 
      st.session_state.multimodal_cohere_ef,
      st.session_state.thumbnail_collection,  
      video['id'],  
      n_similar=N_VIDS
    )
    if top_videos is not None and len(top_videos):
      st.container(border=False, height=20)

      st.subheader("Thumbnails like this")
      outer_cont = st.container(border=False, key="thumbnails-like-this")
      cols = outer_cont.columns(3)
      for i in range(len(top_videos[:3])):
        top_video = top_videos.iloc[i]

        col = cols[i]
        container = col.container(border=True)
        container.image(top_video['thumbnail'])
        container.markdown(f"###### {top_video['title']}")
        caption_cont = container.container(border=False, key=f'thumbnails-like-this-caption-{i}')
        left_cap, right_cap = caption_cont.columns([2,1])
        left_cap.caption(f"{top_video['channel_title']}")
        right_cap.caption(f"{format_number(top_video['views'])} Views")

  # Description
  if video is not None:
    st.container(border=False, height=20)

    title_tile = st.container(border=True)  
    title_tile.markdown(f"##### Description \n{video.loc['description']}")


with tab2:
  # Display plots
  st.container(border=False, height=10)
  st.subheader("Channel Views")

  container = st.container()
  chart = plot_channel_over_time(
    st.session_state.df, video['channel_id']
  )
  container.altair_chart(chart, use_container_width=True)

  st.container(border=False, height=10)
  container = st.container()
  chart = plot_channel_duration_over_time(
    st.session_state.df, video['channel_id']
  )
  container.altair_chart(chart)

  st.container(border=False, height=10)
  container = st.container()
  chart = plot_work_per_video_type(
    st.session_state.df, video['channel_id']
  )
  container.altair_chart(chart, use_container_width=True)

  st.container(border=False, height=10)
  container = st.container(border=False, height=500)
  chart = plot_cadence(
    st.session_state.df, video['channel_id']
  )
  container.altair_chart(chart, use_container_width=True)