import pandas as pd
import altair as alt
import numpy as np
import streamlit as st
from datetime import datetime
import pickle, sklearn, openai, tiktoken, requests, base64, os

MAX_TOKENS = 8000  # safe under the 8191 limit
MODEL_PATH = "files/model-uplift-0927.pkl"
EMB_MODEL_NAME = "text-embedding-3-large"


def format_number(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1000:
        return f"{n/1000:.0f}K"
    else:
        return str(n)

def format_impact(distance):
  if distance < 1.2:
    return 'High'
  elif distance < 1.3:
    return 'Medium'
  elif distance < 1.4:
    return 'Low'
  else: 
    return None

# ----- Chroma -----

def truncate_text(text, max_tokens=MAX_TOKENS):
    tokenizer = tiktoken.encoding_for_model(EMB_MODEL_NAME)
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[:max_tokens])
  
def get_all_from_chroma(collection):
    offset = 0
    limit = 300
    chroma_df = None

    while True:
        data = collection.get(include=["documents", "embeddings"], limit=limit, offset=offset)
        if not len(data['documents']):
            break
        if offset % 300 == 0:
            print(f"Documents {offset} - {offset + len(data['documents'])}, {data['ids'][0]}")

        offset += len(data['documents'])
        data['embeddings'] = data['embeddings'].tolist()
        filtered_data = {
            'id': data['ids'],
            'document': data['documents'],
            'embedding': data['embeddings']
        }

        if chroma_df is None:
            chroma_df = pd.DataFrame(filtered_data)
        else:

            chroma_df = pd.concat([chroma_df, pd.DataFrame(filtered_data)], ignore_index=True)
    
        if offset % 300 == 0:
            print(f"Total documents: {len(chroma_df)}")

    print(f"Total documents: {len(chroma_df)}")
    return chroma_df

def get_from_chroma_with_ids(collection, ids):
    data = collection.get(ids=ids, include=["documents", "embeddings"])
    return data

def add_emb_to_chroma(collection, id, emb, doc):
    collection.add(
        documents=[truncate_text(doc)],
        ids=[id],
        embeddings=[emb] if emb is not None else None
    )


def add_images_to_chroma(collection, co_ef, docs, ids=None):
    ids = ids if ids else [str(i) for i in range(len(docs))]

    # Process in batches of 35 (300000 (max tpr) / 8000 (max tpd))
    batch_size = 35
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}")

        # Download images
        images = []
        for doc in batch_docs:
            img_url = doc['image']
            img_data = requests.get(img_url).content
            img_b64 = base64.b64encode(img_data).decode("utf-8")
            images.append(f"data:image/jpeg;base64,{img_b64}")

        resp = co_ef.embed(model="embed-v4.0", images=images)
        embeddings = resp.embeddings

        metadatas = {
            'images': [doc['image'] for doc in batch_docs],
            'text': [doc['text'] for doc in batch_docs],
            'channel': [doc['channel'] for doc in batch_docs]
        }

        collection.add(
            metadatas=batch_docs,
            embeddings=embeddings,
            ids=batch_ids,
        )


  
# ----- Predictions -----


@st.cache_data
def predict_top_n_percent(df_videos, video, _video_collection, _embedder, n_days_ago=90, regr=None):
  ''' Returns whether the video lies in the top n percent of channel videos. 
  The video must be published within the last n days and be in the bottom 50% of views.
  TODO: Verify if this works on videos not in the csv
  '''
  if regr is None:
    with open(MODEL_PATH, 'rb') as f:
      regr = pickle.load(f)

    if regr is None:
        return None

  # Check if more recent than n days ago
  video = video.copy()
  video['published_at_datetime'] = pd.to_datetime(video['published_at'], utc=True)
  time_cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=n_days_ago)
  if video['published_at_datetime'] < time_cutoff:
    return None

  # Sort channel by views
  channel_videos = df_videos[df_videos['channel_id'] == video['channel_id']]
  channel_videos_sorted = channel_videos.sort_values(by='views', ascending=False)
  
  if 'split' in channel_videos.columns:
    # If we trained on channel, use the train median
    channel_median_views = channel_videos[channel_videos['split'] == 'train']['views'].median()
  else:
    # Otherwise grab median of views > time_cutoff
    channel_videos['published_at_datetime'] = pd.to_datetime(channel_videos['published_at'], utc=True)
    channel_median_views = channel_videos[channel_videos['published_at_datetime'] < time_cutoff]['views'].median()

  # Load video embedding from chroma
  video_docs = get_from_chroma_with_ids(_video_collection, [video['id']])

  # If video isn't in chroma then create embedding
  if not video_docs.get('documents'):
    emb = _embedder(truncate_text(video['embedding_text']))
    emb = emb[0]
    # Store in Chroma
    print("Adding to chroma")
    add_emb_to_chroma(_video_collection, video['id'], emb, video['embedding_text'])
    
  else:
    print("Embedding in chroma")
    emb = video_docs['embeddings'][0]

  # Predict
  pred = regr.predict([emb])
  if type(pred) in [np.ndarray, list] and len(pred):
    pred = pred[0]

  print(f"Predicted uplift: {np.exp(pred)}")
  # Denormalize into view space
  if pred == 0:
    pred += 0.00001
  pred = np.exp(pred) * channel_median_views
  print(f"Predicted views: {pred}")

#   Ignoring to keep this exact model predictions and the quantile covers this up some
#   If they outperform the model then good on them
#   pred = min(int(pred), video['views'])

  print(f"Max views for channel: {channel_videos_sorted.iloc[0]['views']}")
  print(f"Min views for channel: {channel_videos_sorted.iloc[-1]['views']}")
  print(f"Median views for channel: {channel_median_views}")

  # "bottom 25% of views" "bottom 50% of views" "top 50% of views" "top 25% of views"
  for n in [0.75, 0.5, 0.25, 0.]:
    top_n_percent_idx = int(len(channel_videos_sorted)*n)
    top_n_percent_views = channel_videos_sorted.iloc[top_n_percent_idx]['views']
    if pred < top_n_percent_views:
        return 1-n
  
  return None


def predict_boostable_features(
        video_collection, 
        term_collection, 
        video_id, 
        term_docs=None,
        regr=None,
        feature_scale=0.2, 
        n_boostable=3
    ):
    ''' Predicts features which if boosted, would improve the prediction. 
    Returns the n_boostable features with the highest boost or None if no features are boostable.
    '''
    if regr is None:
        with open(MODEL_PATH, 'rb') as f:
            regr = pickle.load(f)
    
    if regr is None:
        return None

    # Get video embeddings
    video_docs = get_from_chroma_with_ids(video_collection, [video_id])
    video_embs = video_docs.get('embeddings')
    if video_embs is None or not len(video_embs):
        return None
    video_emb = video_embs[0]

    # Get term embeddings
    if term_docs is None:
        term_docs = get_all_from_chroma(term_collection)
    
    video_pred = regr.predict(video_emb.reshape(1,-1))
    video_pred = np.exp(video_pred)
    if type(video_pred) in [np.ndarray, list] and len(video_pred):
        video_pred = video_pred[0]
    print(f"Predicted uplift: {video_pred}")
    # See which boost the most
    gt_terms = []
    for i in range(len(term_docs)):
        term_emb = np.array(term_docs.iloc[i]['embedding'])
        pred = predict_for_feature(term_emb, video_emb, regr, feature_scale)
        if pred > video_pred:
            # Convert to boost over video_pred
            improvement = ((pred - video_pred) / video_pred)*100
            gt_terms.append((term_docs.iloc[i]['document'], improvement))

    gt_terms.sort(key=lambda x: x[1], reverse=True)
    gt_terms = gt_terms[:n_boostable]

    return gt_terms

def predict_for_feature(feature_embedding, video_embedding, regr, feature_scale=0.2):
    # Predict
    input_embedding = video_embedding + (feature_scale * feature_embedding)
    input_embedding = input_embedding.reshape(1, -1)
    pred = regr.predict(input_embedding)
    if type(pred) in [np.ndarray, list] and len(pred):
        pred = pred[0]
    return np.exp(pred)


def get_most_similar_videos(df_videos, video_collection, video_id, n_similar=3):
    # Get video embedding
    video_docs = get_from_chroma_with_ids(video_collection, [video_id])
    video_emb = video_docs.get('embeddings')
    if video_emb is None or not len(video_emb):
        # Query without embedding
        text = df_videos[df_videos['id'] == video_id]['embedding_text'].values[0]
        add_emb_to_chroma(video_collection, video_id, None, text)
        top_video_docs = video_collection.query(
            query_texts=[text],
            n_results=10,
        )

    else:
        # Query with embedding
        top_video_docs = video_collection.query(
            query_embeddings=video_emb,
            n_results=10,
        )
    ids = top_video_docs.get('ids')
    if ids is None or not len(ids):
        return None
    # 2d array returned - grab list
    if type(ids[0]) in [np.ndarray, list] and len(ids[0]):
        ids = ids[0]

    scores = top_video_docs.get('distances')
    print(f"Scores: {scores}")
    print(f"Ids: {ids}")
    # Remove query video
    ids.remove(video_id)

    top_videos = df_videos[df_videos['id'].isin(ids)]
    top_videos = top_videos.drop_duplicates('title')
    top_videos = top_videos.reset_index(drop=True)
    top_videos = top_videos.iloc[:n_similar]
    top_videos = top_videos.sort_values(by='views', ascending=False)
    print(top_videos[['title', 'views', 'channel_title']].to_string())

    return top_videos

def get_most_similar_thumbnails(df_videos, co_ef, thumbnail_collection, video_id, n_similar=3):
    # Get embedding
    thumbnail_docs = get_from_chroma_with_ids(thumbnail_collection, [video_id])
    thumbnail_emb = thumbnail_docs.get('embeddings')

    if thumbnail_emb is None or not len(thumbnail_emb):
        print("Uploading images to chroma")
        # Upload image
        vid = df_videos[df_videos['id'] == video_id]
        doc = {
            'image': vid['thumbnail'].values[0],
            'text': vid['title'].values[0],
            'channel': vid['channel_id'].values[0]
        }

        add_images_to_chroma(thumbnail_collection, co_ef, [doc], [video_id])
        # Grab embedding
        thumbnail_docs = get_from_chroma_with_ids(thumbnail_collection, [video_id])
        thumbnail_emb = thumbnail_docs.get('embeddings')
        if thumbnail_emb is None or not len(thumbnail_emb):
            return []

    # Query with embedding
    top_thumbnail_docs = thumbnail_collection.query(
        query_embeddings=thumbnail_emb,
        n_results=4,
    )
    ids = top_thumbnail_docs.get('ids')
    if ids is None or not len(ids):
        return None
    # 2d array returned - grab list
    if type(ids[0]) in [np.ndarray, list] and len(ids[0]):
        ids = ids[0]

    scores = top_thumbnail_docs.get('distances')
    print(f"Scores: {scores}")
    print(f"Ids: {ids}")
    # Remove query video
    ids.remove(video_id)

    top_videos = df_videos[df_videos['id'].isin(ids)]
    top_videos = top_videos.drop_duplicates('title')
    top_videos = top_videos.reset_index(drop=True)
    top_videos = top_videos.iloc[:n_similar]
    top_videos = top_videos.sort_values(by='views', ascending=False)
    print(top_videos[['title', 'views', 'channel_title']].to_string())

    return top_videos



# ----- AI -----

def rewrite_title(video, openai_api_key):
    client = openai.OpenAI(api_key=openai_api_key)

    prompt = f"Using the following YouTube title and description, rewrite the title to boost views. Keep the title professional as if you were creating this for @{video['channel_title']}. Remember, the goal is to boost views so tailor yourself to the channel's audience. Do not add information you cannot get from the information provided. Respond with only the title and no other dialog. \n\n {video['embedding_text']}"

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )
    print(prompt)
    if type(response) == str:
        return response
    if not hasattr(response, "output"):
        return None
    output = response.output
    if not output and len(output) < 2:
         return None
    output = output[1].content
    if not output:
         return None
    return output[0].text


def get_recent_cadence_performance(df_videos, channel_id):
    channel_videos = df_videos[df_videos['channel_id'] == channel_id]

    channel_videos['published_at_datetime'] = pd.to_datetime(channel_videos['published_at'], utc=True)
    channel_videos = channel_videos.sort_values(by='published_at_datetime')
    # Grab only videos in past year
    one_year_ago = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
    channel_videos = channel_videos[channel_videos['published_at_datetime'] > one_year_ago]

    oldest = channel_videos['published_at_datetime'].min()

    channel_videos['day'] = (channel_videos['published_at_datetime'].dt.floor('D') - oldest.normalize()).dt.days
    channel_videos['week'] = ((channel_videos['published_at_datetime'] - oldest) / pd.Timedelta(weeks=1)).astype(int)
    channel_videos['month'] = ((channel_videos['published_at_datetime'].dt.year - oldest.year) * 12 + (channel_videos['published_at_datetime'].dt.month - oldest.month))

    weekly = channel_videos.groupby('week')
    monthly = channel_videos.groupby('month')
    
    daily_rate = (weekly.size() >= 4).mean()
    weekly_rate = ((weekly.size() <= 3) & (weekly.size() > 0)).mean()
    monthly_rate = ((monthly.size() <= 2) & (monthly.size() > 0)).mean()

    weekly_views = weekly['views'].sum()
    monthly_views = monthly['views'].sum()
    daily_avg_views = weekly_views[weekly.size() >= 4].mean()
    weekly_avg_views = weekly_views[(weekly.size() <= 3) & (weekly.size() > 0)].mean()
    monthly_avg_views = monthly_views[(monthly.size() <= 2) & (monthly.size() > 0)].mean()

    if not daily_rate:
        daily_avg_views = 0
        daily_rate = 0
    if not weekly_rate:
        weekly_avg_views = 0
        weekly_rate = 0
    if not monthly_rate:
        monthly_avg_views = 0
        monthly_rate = 0
    
    return ("Daily", "Weekly", "Monthly"), \
     (daily_rate, weekly_rate, monthly_rate), \
     (daily_avg_views, weekly_avg_views, monthly_avg_views)


def get_recent_duration_performance(df_videos, channel_id):
    channel_videos = df_videos[df_videos['channel_id'] == channel_id]

    channel_videos['published_at_datetime'] = pd.to_datetime(channel_videos['published_at'], utc=True)
    channel_videos = channel_videos.sort_values(by='published_at_datetime')
    # Grab only videos in past year
    one_year_ago = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
    channel_videos = channel_videos[channel_videos['published_at_datetime'] > one_year_ago]

    channel_videos['video_type'] = channel_videos['duration'].apply(
        lambda x: 'Short' if x < 3*60 else 'Clip' if x < 20*60 else 'Full'
    )

    short_views = channel_videos[channel_videos['video_type'] == 'Short']['views'].mean()
    clip_views = channel_videos[channel_videos['video_type'] == 'Clip']['views'].mean()
    full_views = channel_videos[channel_videos['video_type'] == 'Full']['views'].mean()

    return ("Short (<3M)", "Clip (<20M)", "Long Form (≥20M)"), \
     (short_views, clip_views, full_views)


def get_channel_improvement(df_videos, channel_id, output_path, openai_api_key):
    """
    Suggests improvements for a YouTube channel based on recent and high-performing videos,
    publishing cadence, and video duration analysis.

    Retrieves (and caches) a summary of:
      - The 5 most recent videos (title, metadata, stats, and tags)
      - The 5 most popular videos (title, metadata, stats, and tags)
      - Aggregate channel performance (average likes, views, comments)
      - Publishing cadence breakdown (daily, weekly, monthly rates and view averages)
      - Performance by video duration (Short, Clip, Long Form)

    Uses OpenAI to generate a natural-language improvement summary, based on the above, 
    for the given channel ID.

    Args:
        df_videos: Pandas DataFrame of video metadata.
        channel_id: YouTube channel ID string.
        output_path: Path to where channel improvement summaries may be cached/saved.
        openai_api_key: OpenAI API key for text generation.

    Returns:
        A markdown-formatted string containing the natural-language improvement plan.
    """
    output = read_channel_improvement(channel_id, output_path)
    if output is not None:
      return output

    channel_videos = df_videos[df_videos['channel_id'] == channel_id]
    channel_name = channel_videos.iloc[0]['channel_title']
    name_str = f"Channel name: {channel_name}"

    # Get recent videos - drop extra data (description)
    channel_videos = channel_videos.sort_values(by='published_at', ascending=False)
    recent_channel_videos = channel_videos.iloc[:5]
    recent_str = "5 most recent channel videos:\n\n"
    for i in range(len(recent_channel_videos)):
        video = recent_channel_videos.iloc[i]
        recent_str += video['embedding_text']
        recent_str += f"\n## views: {video['views']} likes: {video['likes']} comments {video['comments']}\n"
        recent_str += f"## Tags: {video['tags']}"
        if i < len(recent_channel_videos) - 1:
            recent_str += "\n\n---\n"

    # Get popular videos - drop extra data (description)
    channel_videos = channel_videos.sort_values(by='views', ascending=False)
    high_view_channel_videos = channel_videos.iloc[:5]
    popular_str = "5 most popular channel videos:\n\n"
    for i in range(len(high_view_channel_videos)):
        video = high_view_channel_videos.iloc[i]
        popular_str += video['embedding_text']
        popular_str += f"\n## views: {video['views']} likes: {video['likes']} comments {video['comments']}\n"
        popular_str += f"## Tags: {video['tags']}"
        if i < len(high_view_channel_videos) - 1:
            popular_str += "\n\n---\n"

    # Performance
    avg_likes = channel_videos['likes'].mean()
    avg_views = channel_videos['views'].mean()
    avg_comments = channel_videos['comments'].mean()
    performance_str = f"Avg likes: {avg_likes:.0f}\nAvg views: {avg_views:.0f}\nAvg comments: {avg_comments:.0f}"

    # Cadence
    cadence, rate, avg_cadence_views = get_recent_cadence_performance(df_videos, channel_id)  
    cadence_str = f"Cadence:"  
    for i in range(len(cadence)):
        cadence_str += f" {cadence[i]} - rate {rate[i]}% of the time - avg views ({avg_cadence_views[i]:.0f})"
        if i < len(cadence) - 1:
            cadence_str += ","
    
    # Duration
    duration_type, avg_duration_views = get_recent_duration_performance(df_videos, channel_id)
    duration_str = f"Duration:"  
    for i in range(len(duration_type)):
        duration_str += f" {duration_type[i]} - avg views ({avg_duration_views[i]:.0f})"
        if i < len(duration_type) - 1:
            duration_str += ","

    header_str = "\n\n".join([name_str, performance_str, cadence_str, duration_str])
    context = f"""{header_str}\n\n---\n\n{recent_str}\n\n\n---\n\n{popular_str}"""

    prompt = f"""Below are statistics, recent videos, and top-performing videos from a YouTube channel. Identify data-grounded ways to increase average views and engagement (likes/comments). Base recommendations only on observed differences in titles, topics, durations, and posting cadence, descriptions, etc. Return results as a bulleted list with bolded headers: Improvement Area, Recommendation, Rationale, and Confidence for each bullet. Compare trends between recent and top-performing videos. At the end, summarize the 3 most impactful changes by estimated effect size. Do not ask if the user would like further assistance.

    {context}
    """

    client = openai.OpenAI(api_key=openai_api_key)
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )
    print("Querying GPT")
    if type(response) == str:
        return response
    if not hasattr(response, "output"):
        return None
    output = response.output
    if not output and len(output) < 2:
         return None
    output = output[1].content
    if not output:
         return None

    save_channel_improvement(channel_id, output[0].text, output_path)
    return output[0].text


def read_channel_improvement(channel_id, path):
    """
    Reads the channel improvement output for a given channel_id from the specified CSV path.
    Returns None if the file does not exist or if the last saved date is 7 days ago or older.
    """
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if channel_id not in df['channel_id'].values:
        return None

    row = df[df['channel_id'] == channel_id].iloc[0]
    saved_date = pd.to_datetime(row['date'])
    one_week_ago = pd.to_datetime(datetime.now().date()) - pd.Timedelta(days=7)
    if saved_date < one_week_ago:
        return None

    return row['output']


def save_channel_improvement(channel_id, output, path):
    """
    Saves the channel improvement output to a CSV file,
    updating the existing row if channel_id exists, or appending otherwise. 
    """
    date = datetime.now().date()
    if not os.path.exists(path):
        # Save a new dataframe
        df = pd.DataFrame([{"channel_id": channel_id, "output": output, "date": date}])
        df.to_csv(path, index=False)
        return

    # Write/overwrite in csv
    df = pd.read_csv(path)
    if channel_id in df['channel_id'].values:
        df.loc[df['channel_id'] == channel_id, 'output'] = output
        df.loc[df['channel_id'] == channel_id, 'date'] = date
    else:
        df = pd.concat([
            df, pd.DataFrame([{"channel_id": channel_id, "output": output, "date": date}])
        ], ignore_index=True)
    df.to_csv(path, index=False)



# ----- Plots -----

@st.cache_data
def plot_channel_over_time(df_videos, channel_id):
    channel_videos = df_videos[df_videos['channel_id'] == channel_id]
    channel_name = channel_videos.iloc[0]['channel_title']
    print(f"Channel name: {channel_name}")
    print(f"Channel ID: {channel_id}")
    print(f"Total videos: {len(channel_videos)}")

    channel_videos['published_at_datetime'] = pd.to_datetime(channel_videos['published_at'], utc=True)
    channel_videos = channel_videos.sort_values(by='published_at_datetime')
    # convert datetime to month and year
    channel_videos['published_at_month'] = channel_videos['published_at_datetime'].dt.to_period('M')

    # # Get video published_at_month
    # video_month = channel_videos[channel_videos['id'] == video_id].iloc[0]['published_at_month']
    # print(f"Video published at month: {video_month}")

    # group by month and compute sum of views
    channel_videos = channel_videos.groupby('published_at_month')['views'].sum()
    channel_videos.index = channel_videos.index.astype(str)
    # convert to a DataFrame
    df = pd.DataFrame({
        'Month': channel_videos.index, 
        'Views': channel_videos.values,
        "Cumulative Views": channel_videos.cumsum()
    })

    # Nearest point selector
    nearest = alt.selection_point(
        nearest=True,  # snap to nearest x-value
        on='mouseover',
        fields=['Month',],
        empty=False
    )

    # Transparent points to enable nearest hover
    selectors = alt.Chart(df).mark_point(size=1,).encode(
        x=alt.X(
            'Month:T', 
            axis=alt.Axis(format='%b %Y', tickCount=6, title="Publication Date")
        ),
        y=alt.Y(
            'Views:Q',
            axis=alt.Axis(tickCount=6, title="Channel Views")
        ),
        opacity=alt.value(0),
        tooltip=[
            alt.Tooltip('Month:T', title='Month', format='%b %Y'),
            alt.Tooltip('Views', title='Views', format=',.0f'),
        ]
    ).add_params(nearest)

    # Plot
    pdf = alt.Chart(df).mark_area(opacity=0.5).encode(
        x=alt.X(
            'Month:T', 
            axis=alt.Axis(format='%b %Y', tickCount=6, title="Publication Date")
        ),
        y=alt.Y(
            'Views:Q',
            axis=alt.Axis(tickCount=6, title="Channel Views")
        )
    )

    # line at top of area
    pdf_line = alt.Chart(df).mark_line(
        opacity=0.8,
        strokeWidth=2
    ).encode(
        x='Month:T',
        y=alt.Y(
            'Views:Q',
            axis=alt.Axis(tickCount=6, title="Channel Views")
        ),
    )

    points = pdf_line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    cdf_time = (
        alt.Chart(df)
        .mark_area(color="steelblue", opacity=0.3)
        .encode(
            x="Month:T",
            y=alt.Y('Cumulative Views:Q',
                axis=alt.Axis(tickCount=6, title="Channel Cumulative Views")
            ),
        )
    )
    cdf_time_line = (
        alt.Chart(df)
        .mark_line(color="steelblue", opacity=0.8)
        .encode(
            x="Month:T",
            y=alt.Y('Cumulative Views:Q',
                axis=alt.Axis(tickCount=6, title="Channel Cumulative Views")
            ),
        )
    )

    chart = alt.layer(selectors + points, pdf + pdf_line)
    chart = alt.layer(chart, cdf_time + cdf_time_line
    ).resolve_scale(
      y='independent'
    ).properties(
        title="Is my channel growing? (Monthly channel views by publication)"
    ).configure_title(
        fontSize=16,
        anchor='middle',
    )
    return chart


@st.cache_data
def plot_channel_duration_over_time(df_videos, channel_id):
    ''' Plot the duration of videos over time for a channel '''

    channel_videos = df_videos[df_videos['channel_id'] == channel_id]
    n_videos_lt_5 = len(channel_videos[channel_videos['duration'] < 5*60])
    n_videos_gt_5_lt_20 = len(channel_videos[(channel_videos['duration'] >= 5*60) & (channel_videos['duration'] < 20*60)])
    n_videos_gt_20 = len(channel_videos[channel_videos['duration'] >= 20*60])
    print(f"Total videos: {len(channel_videos)}")
    print(f"Short form - Number of videos < 5 minutes {n_videos_lt_5}")
    print(f"Medium - Number of videos 5 < x <= 20 minutes {n_videos_gt_5_lt_20}")
    print(f"Long form - Number of videos >= 20 minutes {n_videos_gt_20}")

    channel_videos = channel_videos.sort_values(by='duration').reset_index()

    # convert to a DataFrame
    df = pd.DataFrame({
        'Video': channel_videos['title'], 
        'Views': channel_videos['views'],
        "Duration": channel_videos['duration'] / 60,
        "Order": range(len(channel_videos)),
    })

    # Nearest point selector
    nearest = alt.selection_point(
        nearest=True,  # snap to nearest x-value
        on='mouseover',
        fields=['Order',],
        empty=False
    )

    # Transparent points to enable nearest hover
    selectors = alt.Chart(df).mark_point(size=1,).encode(
        x='Order:Q',
        y=alt.Y(
            'Views:Q',
            axis=alt.Axis(tickCount=6, title="Video Views")
        ),
        opacity=alt.value(0),
        tooltip=[
            alt.Tooltip('Video', title='Video'),
            alt.Tooltip('Views', title='Views', format=',.0f'),
            alt.Tooltip('Duration', title='Duration (min)', format=',.0f')
        ]
    ).add_params(nearest)

    # Plot
    video_views_area_chart = alt.Chart(df).mark_area(opacity=0.5).encode(
        x=alt.X('Order:Q', sort=None, title="Videos (sorted by duration)", axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y(
            'Views:Q',
            axis=alt.Axis(tickCount=6, title="Video Views")
        )
    )

    # line at top of area
    video_views_line_chart = alt.Chart(df).mark_line(
        opacity=0.8,
        strokeWidth=2
    ).encode(
        x=alt.X('Order:Q', sort=None, title="Videos (sorted by duration)", axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y(
            'Views:Q',
            axis=alt.Axis(tickCount=6, title="Video Views")
        )
    )

    points = video_views_line_chart.mark_point().encode(
        y=alt.Y(
            'Views:Q',
            axis=alt.Axis(tickCount=6, title="Video Views")
        ),
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    video_duration_area_chart = alt.Chart(df).mark_area(
        color="steelblue", 
        opacity=0.3
    ).encode(
        x=alt.X('Order:Q', sort=None, title="Videos (sorted by duration)", axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y(
            'Duration:Q',
            axis=alt.Axis(tickCount=6, title="Duration (minutes)")
        )
    )
    video_duration_line_chart = alt.Chart(df).mark_line(
        color="steelblue", 
        opacity=0.8
    ).encode(
        x=alt.X('Order:Q', sort=None, title="Videos (sorted by duration)", axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y(
            'Duration:Q',
            axis=alt.Axis(tickCount=6, title="Duration (minutes)")
        ),
    )

    chart = alt.layer(
      selectors + points,
      video_views_area_chart + video_views_line_chart,
    )
    
    # display
    chart = alt.layer(
        chart,
        video_duration_area_chart + video_duration_line_chart,
    ).resolve_scale(
        y='independent'
    ).interactive(
      bind_y=False
    ).properties(
        title="How many viewers are watching videos this long? (Video views by duration)"
    ).configure_title(
        fontSize=16,
        anchor='middle',
    )
    return chart


@st.cache_data
def plot_work_per_video_type(df_videos, channel_id):
    
    channel_videos = df_videos[df_videos['channel_id'] == channel_id]
    channel_videos['video_type'] = channel_videos['duration'].apply(
        lambda x: 'Short (<3M)' if x < 3*60 else 'Clip (<20M)' if x < 20*60 else 'Long Form (≥20M)'
    )
    video_type_groups = channel_videos.groupby('video_type')
    # Get per-group median
    video_type_views = video_type_groups['views'].median()
    work_per_video = video_type_groups['duration'].median() / 60 * 5

    df = pd.DataFrame({
        'Video Type': video_type_views.index, 
        'Views': video_type_views.values,
        'Work': work_per_video.values,
    })
    # Define the desired order for video types
    video_type_order = ['Short (<3M)', 'Clip (<20M)', 'Long Form (≥20M)']
    # Altair chart that shows a scatter plot where each column is a video_type_view, height is a the total duration * 5, and size is video_type_view
    scatter = alt.Chart(df).mark_circle().encode(
        x=alt.X('Video Type:N', sort=video_type_order,
                axis=alt.Axis(labels=True, ticks=False, labelAngle=0)
        ),
        y=alt.Y(
            'Work:Q',
            scale=alt.Scale(domain=[-100, df['Work'].max() + 200]),
            axis=alt.Axis(tickCount=6, title="Production Work (5 Min/Median Duration Min)")
        ),
        size=alt.Size(
            'Views:Q',
            scale=alt.Scale(range=[1000, 10000]),
            legend=None
        ),
        color=alt.Color(
            'Video Type:N',
            legend=None
        ),
        tooltip=[
            # alt.Tooltip('Video Type:N', title='Type'),
            # alt.Tooltip('Views:Q', title='Median Views', format=',.0f'),
            # alt.Tooltip('Work:Q', title='Estimated Work (min)', format=',.0f')
        ]
    )

    labels = alt.Chart(df).mark_text(
        align='center', 
        baseline='middle',  
        color='white'
    ).encode(
        x=alt.X('Video Type:N', sort=video_type_order),
        y='Work:Q',
        text=alt.Text('Views:Q', format=',.0f'),
        tooltip=[
            alt.Tooltip('Video Type:N', title='Type'),
            alt.Tooltip('Views:Q', title='Median Views', format=',.0f'),
            alt.Tooltip('Work:Q', title='Estimated Work (min)', format=',.0f')
        ]
    )

    # Combine scatter and labels
    chart = alt.layer(
      scatter + labels, 
    ).properties(
        title="Is the effort worth the views? (Median views per production cost)",
        height=400
    ).configure_title(
        fontSize=16,
        anchor='middle',
    )

    return chart


@st.cache_data
def plot_cadence(df_videos, channel_id):

    channel_videos = df_videos[df_videos['channel_id'] == channel_id]

    channel_videos['published_at_datetime'] = pd.to_datetime(channel_videos['published_at'], utc=True)
    channel_videos = channel_videos.sort_values(by='published_at_datetime')

    print(channel_videos['published_at_datetime'].min())

    # Grab total days/weeks/months since first publishing
    oldest = channel_videos['published_at_datetime'].min()

    channel_videos['day'] = (channel_videos['published_at_datetime'].dt.floor('D') - oldest.normalize()).dt.days
    channel_videos['week'] = ((channel_videos['published_at_datetime'] - oldest) / pd.Timedelta(weeks=1)).astype(int)
    channel_videos['month'] = ((channel_videos['published_at_datetime'].dt.year - oldest.year) * 12 + (channel_videos['published_at_datetime'].dt.month - oldest.month))

    daily = channel_videos.groupby('day')
    weekly = channel_videos.groupby('week')
    monthly = channel_videos.groupby('month')
    
    daily_rate = (weekly.size() >= 3).mean()
    weekly_rate = ((weekly.size() <= 2) & (weekly.size() > 0)).mean()
    monthly_rate = ((monthly.size() <= 2) & (monthly.size() > 0)).mean()

    weekly_views = weekly['views'].sum()
    monthly_views = monthly['views'].sum()
    daily_median_views = weekly_views[weekly.size() >= 3].mean()
    weekly_median_views = weekly_views[(weekly.size() <= 2) & (weekly.size() > 0)].mean()
    monthly_median_views = monthly_views[(monthly.size() <= 2) & (monthly.size() > 0)].mean()

    if not daily_rate:
        daily_median_views = 0
        daily_rate = 0
    if not weekly_rate:
        weekly_median_views = 0
        weekly_rate = 0
    if not monthly_rate:
        monthly_median_views = 0
        monthly_rate = 0

    # Define the desired order 
    order = ['Daily', 'Weekly', 'Monthly']

    df = pd.DataFrame({
        'Cadence Type': order,
        'Views': [daily_median_views, weekly_median_views, monthly_median_views],
        'Cadence': [daily_rate, weekly_rate, monthly_rate],
        'Order': range(len(order))
    })


    label_padding = \
        15 if df['Cadence'].min() >= 0.1 \
        else 25 if df.loc[df['Cadence'].idxmin(), 'Views'] <= 200000 \
        else 45

    # Chart
    scatter = alt.Chart(df).mark_circle().encode(
        x=alt.X('Cadence Type:N', sort=order,
                axis=alt.Axis(labels=True, ticks=False, labelAngle=0, labelPadding=label_padding)
        ),
        y=alt.Y(
            'Cadence:Q',
            scale=alt.Scale(domain=[0,1.]),
            axis=alt.Axis(tickCount=6, title="Percentage of posts at cadence", format='.0%')
        ),
        size=alt.Size(
            'Views:Q',
            scale=alt.Scale(range=[1000, 10000]),
            legend=None
        ),
        color=alt.Color(
            'Cadence Type:N',
            legend=None
        ),
        tooltip=[
            # alt.Tooltip('Views:Q', title='Total Views', format=',.0f'),
            # alt.Tooltip('Cadence:Q', title='Cadence', format='.0%')
        ]
    )

    labels = alt.Chart(df).mark_text(
        align='center',
        baseline='middle',
        color='white'
    ).encode(
        x=alt.X('Cadence Type:N', sort=order),
        y='Cadence:Q',
        text=alt.Text('Views:Q', format=',.0f'),
        tooltip=[
            alt.Tooltip('Views:Q', title='Avg Views', format=',.0f'),
            alt.Tooltip('Cadence:Q', title='Cadence', format='.0%')
        ]
    )


    chart = alt.layer(
      scatter + labels, 
    ).properties(
        title="Is this the right cadence? (Avg views per cadence)",
        height=450
    ).configure_title(
        fontSize=16,
        anchor='middle',
    )

    return chart