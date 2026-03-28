"""
CallingMarkets News Engine
Runs daily. Fetches top finance news, writes articles using Claude,
pulls a relevant Unsplash photo, and publishes as a draft to WordPress.
"""

import json
import os
import base64
import requests
from datetime import datetime, timedelta

# ── API KEYS ───────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.environ["ANTHROPIC_API_KEY"]
NEWSAPI_KEY        = os.environ["NEWSAPI_KEY"]
UNSPLASH_ACCESS_KEY= os.environ["UNSPLASH_ACCESS_KEY"]
WP_URL             = os.environ["WP_URL"].rstrip("/")
WP_USERNAME        = os.environ["WP_USERNAME"]
WP_APP_PASSWORD    = os.environ["WP_APP_PASSWORD"]

NEWSAPI_URL        = "https://newsapi.org/v2/top-headlines"
NEWSAPI_EVERYTHING = "https://newsapi.org/v2/everything"
ANTHROPIC_URL      = "https://api.anthropic.com/v1/messages"
UNSPLASH_URL       = "https://api.unsplash.com/search/photos"
WP_POSTS_URL       = f"{WP_URL}/wp-json/wp/v2/posts"
WP_MEDIA_URL       = f"{WP_URL}/wp-json/wp/v2/media"

# ── HELPERS ────────────────────────────────────────────────────────────────────
def wp_auth():
    credentials = f"{WP_USERNAME}:{WP_APP_PASSWORD}"
    token = base64.b64encode(credentials.encode()).decode()
    return {"Authorization": f"Basic {token}"}

def fetch_top_finance_news() -> list:
    """Fetch today's top finance headlines from NewsAPI."""
    from_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        r = requests.get(NEWSAPI_EVERYTHING, params={
            "q":        "finance OR markets OR stocks OR economy OR Federal Reserve OR inflation OR earnings",
            "from":     from_date,
            "sortBy":   "popularity",
            "pageSize": 20,
            "language": "en",
            "apiKey":   NEWSAPI_KEY,
        }, timeout=10)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        # Filter out removed articles
        return [a for a in articles if a.get("title") and "[Removed]" not in a.get("title", "") and a.get("description")]
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []

def load_analysis_context() -> str:
    """Load this week's sector analysis for context if available."""
    try:
        with open("analysis.json", "r") as f:
            data = json.load(f)
        # Build a brief summary of sector biases
        lines = [f"Week of {data.get('week_of', '')} — Sector Momentum:"]
        for s in data.get("sectors", [])[:10]:
            lines.append(f"  {s['sector']}: {s['bias']} ({s['signals']['weekly_buy']}/{s['signals']['total']} weekly BUY)")
        return "\n".join(lines)
    except:
        return ""

def load_signals_context() -> str:
    """Load current signals for key tickers."""
    try:
        with open("signals.json", "r") as f:
            data = json.load(f)
        key_tickers = ["SPY", "QQQ", "GLD", "TLT", "BTC/USD"]
        lines = ["Current Momentum Signals:"]
        for row in data.get("signals", []):
            if row["ticker"] in key_tickers:
                d = row["timeframes"]["daily"]["signal"]
                w = row["timeframes"]["weekly"]["signal"]
                m = row["timeframes"]["monthly"]["signal"]
                lines.append(f"  {row['ticker']}: Daily={d} Weekly={w} Monthly={m}")
        return "\n".join(lines)
    except:
        return ""

def pick_top_story(articles: list) -> dict:
    """Use Claude to pick the most compelling story to write about."""
    headlines = "\n".join([
        f"{i+1}. {a['title']} — {a.get('source', {}).get('name', '')}"
        for i, a in enumerate(articles[:15])
    ])

    prompt = f"""You are an editor at CallingMarkets, a momentum-based financial signals publication.

Here are today's top finance headlines:
{headlines}

Pick the single most compelling story for a 500-800 word finance article that would interest retail traders, crypto traders, long-term investors, and finance professionals.

Respond with ONLY a JSON object in this exact format:
{{"index": <1-based index>, "topic": "<brief topic description>", "search_term": "<3-5 word Unsplash photo search term>"}}"""

    r = requests.post(ANTHROPIC_URL,
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-opus-4-5",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    r.raise_for_status()
    text = r.json()["content"][0]["text"].strip()
    # Parse JSON response
    import re
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        result = json.loads(match.group())
        idx = result["index"] - 1
        return {
            "article":     articles[idx],
            "topic":       result["topic"],
            "search_term": result["search_term"],
        }
    return {"article": articles[0], "topic": articles[0]["title"], "search_term": "stock market finance"}

def write_article(story: dict, analysis_context: str, signals_context: str) -> dict:
    """Use Claude to write the full article."""
    article = story["article"]
    today   = datetime.utcnow().strftime("%B %d, %Y")

    prompt = f"""You are a financial journalist writing for CallingMarkets, a momentum-based signals publication for retail traders, crypto traders, long-term investors, and finance professionals.

Write a 500-800 word finance news article about the following story. The tone should be balanced, clear, and analytical — like a sharp Bloomberg or Reuters piece, not a press release.

TODAY'S DATE: {today}

STORY TO COVER:
Title: {article['title']}
Source: {article.get('source', {}).get('name', '')}
Description: {article.get('description', '')}
Content snippet: {article.get('content', '')[:500] if article.get('content') else ''}

MARKET CONTEXT (use where relevant):
{analysis_context}

{signals_context}

INSTRUCTIONS:
- Write a compelling headline (not the same as the source headline — make it original)
- Write the article in flowing prose — no bullet points, no headers
- Open with a strong lede that gets to the point immediately
- Weave in relevant market context and momentum signals where they add insight
- Include 2-3 paragraphs of analysis beyond just reporting the news
- End with what investors and traders should watch going forward
- Do NOT plagiarize the source — write it entirely in your own words
- Do NOT mention CallingMarkets explicitly in the article body
- Do NOT add disclaimers or "this is not financial advice" language

Respond with ONLY a JSON object:
{{"title": "<article headline>", "content": "<full article in HTML paragraphs using <p> tags>", "excerpt": "<2 sentence summary>", "tags": ["tag1", "tag2", "tag3"]}}"""

    r = requests.post(ANTHROPIC_URL,
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model":      "claude-opus-4-5",
            "max_tokens": 2000,
            "messages":   [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    r.raise_for_status()
    text = r.json()["content"][0]["text"].strip()

    import re
    # Extract JSON — handle potential markdown code blocks
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"Could not parse article JSON: {text[:200]}")

def fetch_unsplash_photo(search_term: str) -> dict | None:
    """Fetch a relevant royalty-free photo from Unsplash."""
    try:
        r = requests.get(UNSPLASH_URL, params={
            "query":       search_term,
            "per_page":    5,
            "orientation": "landscape",
            "content_filter": "high",
        }, headers={
            "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}",
        }, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        photo = results[0]
        return {
            "url":          photo["urls"]["regular"],
            "download_url": photo["links"]["download_location"],
            "photographer": photo["user"]["name"],
            "photographer_url": photo["user"]["links"]["html"],
            "alt":          photo.get("alt_description", search_term),
        }
    except Exception as e:
        print(f"Unsplash error: {e}")
        return None

def trigger_unsplash_download(download_url: str):
    """Trigger Unsplash download endpoint (required by their API terms)."""
    try:
        requests.get(download_url, headers={
            "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
        }, timeout=5)
    except:
        pass

def upload_image_to_wordpress(photo: dict, article_title: str) -> int | None:
    """Download photo and upload to WordPress media library."""
    try:
        # Download the image
        img_r = requests.get(photo["url"], timeout=15)
        img_r.raise_for_status()

        # Upload to WordPress
        filename = f"cm-news-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jpg"
        headers  = {
            **wp_auth(),
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "image/jpeg",
        }
        r = requests.post(WP_MEDIA_URL,
            headers=headers,
            data=img_r.content,
            timeout=30,
        )
        r.raise_for_status()
        media_id = r.json()["id"]

        # Add photo credit as caption via update
        credit = f'Photo by <a href="{photo["photographer_url"]}?utm_source=callingmarkets&utm_medium=referral" target="_blank">{photo["photographer"]}</a> on <a href="https://unsplash.com?utm_source=callingmarkets&utm_medium=referral" target="_blank">Unsplash</a>'
        requests.post(f"{WP_MEDIA_URL}/{media_id}",
            headers={**wp_auth(), "Content-Type": "application/json"},
            json={"caption": credit, "alt_text": photo["alt"]},
            timeout=10,
        )

        print(f"  Image uploaded — media ID: {media_id}")
        return media_id
    except Exception as e:
        print(f"  Image upload error: {e}")
        return None

def publish_to_wordpress(article: dict, media_id: int | None, photo: dict | None) -> str:
    """Create a draft post in WordPress."""

    # Add photo credit to end of content if we have a photo
    content = article["content"]
    if photo:
        credit = (f'<p style="font-size:11px;color:#9a9a9a;margin-top:24px;">'
                  f'Photo: <a href="{photo["photographer_url"]}?utm_source=callingmarkets&utm_medium=referral">'
                  f'{photo["photographer"]}</a> / Unsplash</p>')
        content += credit

    post_data = {
        "title":         article["title"],
        "content":       content,
        "excerpt":       article.get("excerpt", ""),
        "status":        "draft",          # Save as draft for review
        "categories":    [],               # Add category IDs here if needed
        "tags":          [],               # Tags added by name below
        "comment_status": "open",
    }

    if media_id:
        post_data["featured_media"] = media_id

    # Create the post
    r = requests.post(WP_POSTS_URL,
        headers={**wp_auth(), "Content-Type": "application/json"},
        json=post_data,
        timeout=30,
    )
    r.raise_for_status()
    post = r.json()
    post_url = post.get("link", "")
    post_id  = post.get("id")

    # Add tags by name
    if article.get("tags") and post_id:
        try:
            # Create/get tag IDs
            tag_ids = []
            for tag_name in article["tags"][:5]:
                tr = requests.post(f"{WP_URL}/wp-json/wp/v2/tags",
                    headers={**wp_auth(), "Content-Type": "application/json"},
                    json={"name": tag_name},
                    timeout=10,
                )
                if tr.status_code in (200, 201):
                    tag_ids.append(tr.json()["id"])
                elif tr.status_code == 400:
                    # Tag exists — get its ID
                    search = requests.get(f"{WP_URL}/wp-json/wp/v2/tags?search={tag_name}",
                        headers=wp_auth(), timeout=10)
                    results = search.json()
                    if results:
                        tag_ids.append(results[0]["id"])

            if tag_ids:
                requests.post(f"{WP_POSTS_URL}/{post_id}",
                    headers={**wp_auth(), "Content-Type": "application/json"},
                    json={"tags": tag_ids},
                    timeout=10,
                )
        except Exception as e:
            print(f"  Tag error (non-fatal): {e}")

    return post_url

# ── MAIN ───────────────────────────────────────────────────────────────────────
def run():
    today = datetime.utcnow().strftime("%B %d, %Y")
    print(f"CallingMarkets News Engine — {today}\n")

    # 1. Fetch news
    print("Fetching finance news…")
    articles = fetch_top_finance_news()
    if not articles:
        print("No articles found — exiting")
        return
    print(f"  Found {len(articles)} articles")

    # 2. Load context
    print("Loading market context…")
    analysis_context = load_analysis_context()
    signals_context  = load_signals_context()

    # 3. Pick top story
    print("Selecting top story…")
    story = pick_top_story(articles)
    print(f"  Selected: {story['article']['title']}")
    print(f"  Topic: {story['topic']}")
    print(f"  Photo search: {story['search_term']}")

    # 4. Write article
    print("Writing article…")
    article = write_article(story, analysis_context, signals_context)
    print(f"  Title: {article['title']}")

    # 5. Fetch photo
    print("Fetching Unsplash photo…")
    photo = fetch_unsplash_photo(story["search_term"])
    if photo:
        print(f"  Photo by: {photo['photographer']}")
        trigger_unsplash_download(photo["download_url"])
    else:
        print("  No photo found — continuing without")

    # 6. Upload photo to WordPress
    media_id = None
    if photo:
        print("Uploading photo to WordPress…")
        media_id = upload_image_to_wordpress(photo, article["title"])

    # 7. Publish draft to WordPress
    print("Publishing draft to WordPress…")
    post_url = publish_to_wordpress(article, media_id, photo)
    print(f"  Draft saved: {post_url}")

    print(f"\n✓ Done — review your draft at {WP_URL}/wp-admin/edit.php")

if __name__ == "__main__":
    run()
