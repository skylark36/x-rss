import logging
import os
import requests
import xml.etree.ElementTree as ET
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional
from filters import ContentFilter
from main import add_tweet_video, get_state, save_state, PREFIX, TWITTER_USERNAME

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# List of Nitter instances to try as fallback (configured via NITTER_INSTANCES env var, separated by semicolon)
NITTER_INSTANCES = [
    url.strip()
    for url in os.getenv(
        "NITTER_INSTANCES",
        "https://nitter.net;https://xcancel.com;https://nitter.privacyredirect.com;https://nitter.poast.org"
    ).split(";")
    if url.strip()
]

def has_video_attachment(item) -> bool:
    """Checks if an RSS item contains a video attachment marker."""
    desc = item.findtext("description", "")
    if not desc:
        return False
    desc_lower = desc.lower()
    if "<br>video<br>" in desc_lower or "video_thumb" in desc_lower or "<video" in desc_lower:
        return True
    for enclosure in item.findall("enclosure"):
        enc_type = enclosure.attrib.get("type", "")
        if "video" in enc_type:
            return True
    return False

def get_item_pub_date(item) -> Optional[str]:
    """Parses pubDate from RSS item and converts to ISO 8601 string."""
    pub_date_elem = item.find("pubDate")
    if pub_date_elem is not None and pub_date_elem.text:
        try:
            return parsedate_to_datetime(pub_date_elem.text).isoformat()
        except Exception as e:
            logger.warning(f"Error parsing pubDate '{pub_date_elem.text}': {e}")
    return None

def get_item_author(item) -> Optional[str]:
    """Extracts creator or original author username from RSS item."""
    ns = {"dc": "http://purl.org/dc/elements/1.1/"}
    creator_elem = item.find("dc:creator", namespaces=ns)
    if creator_elem is not None and creator_elem.text:
        return creator_elem.text.lstrip("@").strip()
    
    link_elem = item.find("link")
    if link_elem is not None and link_elem.text:
        url_clean = link_elem.text.split("#")[0]
        parts = url_clean.split("/status/")[0].split("/")
        if len(parts) > 0 and parts[-1]:
            return parts[-1].lstrip("@").strip()
    return None

def check_and_sync():
    """Fetches Nitter RSS, detects new tweets with videos, and processes them."""
    if not TWITTER_USERNAME:
        logger.error("TWITTER_USERNAME environment variable is not set.")
        return

    logger.info(f"Checking updates for Twitter user: {TWITTER_USERNAME}")

    # 1. Fetch RSS feed using fallback instances
    feed_content = None
    successful_instance = None
    for instance in NITTER_INSTANCES:
        rss_url = f"{instance.rstrip('/')}/{TWITTER_USERNAME}/rss"
        logger.info(f"Fetching RSS feed from: {rss_url}")
        try:
            r = requests.get(rss_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if r.status_code == 200 and len(r.content) > 0:
                feed_content = r.content
                successful_instance = instance
                logger.info(f"Successfully fetched RSS feed from {instance} ({len(r.content)} bytes)")
                break
            else:
                logger.warning(f"Instance {instance} returned status {r.status_code} or empty content.")
        except Exception as e:
            logger.warning(f"Error fetching from {instance}: {e}")

    if not feed_content:
        logger.error("Could not fetch RSS feed from any Nitter instance.")
        return

    # 2. Parse RSS feed
    try:
        root = ET.fromstring(feed_content)
        items = root.findall(".//item")
    except Exception as e:
        logger.error(f"Error parsing RSS XML: {e}")
        return

    logger.info(f"Found {len(items)} items in RSS feed.")
    if not items:
        return

    # 3. Get current state from R2 to check which tweets are already processed
    state = get_state(PREFIX)
    processed_videos = state.get("videos", {})
    checked_tweets = state.get("checked_tweets", [])

    # 4. Extract tweet URLs and filter for new video tweets
    # Nitter RSS items are sorted newest first. We reverse them to process oldest first.
    new_video_tweets = []
    skipped_state_change = False

    for item in reversed(items):
        link_elem = item.find("link")
        if link_elem is None or not link_elem.text:
            continue
        
        nitter_url = link_elem.text
        # Clean Nitter URL and convert to standard x.com URL
        # e.g., https://nitter.net/CLEC168/status/2078542728661712999#m -> https://x.com/CLEC168/status/2078542728661712999
        url_clean = nitter_url.split("#")[0]
        parts = url_clean.split("/status/")
        if len(parts) < 2:
            continue
            
        tweet_id_str = parts[1].split("?")[0].split("/")[0]
        try:
            tweet_id = int(tweet_id_str)
        except ValueError:
            continue

        tweet_url = f"https://x.com/{TWITTER_USERNAME}/status/{tweet_id}"
        video_key = str(tweet_id)
        
        # Check if already processed
        already_exists = (video_key in processed_videos) or (f"{video_key}_1" in processed_videos) or (video_key in checked_tweets)
        if already_exists:
            continue

        # Check original author (skip reposts from other authors)
        author = get_item_author(item)
        if author and author.lower() != TWITTER_USERNAME.lower():
            logger.info(f"Skipping repost by @{author} for tweet {tweet_id} (target user is @{TWITTER_USERNAME}).")
            checked_tweets.append(video_key)
            skipped_state_change = True
            continue

        # Skip RSS items that do not contain a video attachment
        if not has_video_attachment(item):
            logger.info(f"Skipping tweet {tweet_id} as RSS description indicates no video attachment.")
            checked_tweets.append(video_key)
            skipped_state_change = True
            continue

        pub_date = get_item_pub_date(item)
        new_video_tweets.append((tweet_url, pub_date, item.findtext("title", "")))

    if skipped_state_change:
        state["checked_tweets"] = checked_tweets
        save_state(state, PREFIX)

    logger.info(f"Found {len(new_video_tweets)} new video tweet(s) that are not in the R2 state.")
    if not new_video_tweets:
        logger.info("No new video tweets to process.")
        return

    # 4.5 Content filter: LLM decides what counts as course content
    # (only exact-title duplicates are skipped without calling the model)
    content_filter = ContentFilter()
    content_filter.seed_titles([v.get("title", "") for v in processed_videos.values()])
    candidates = [
        {
            "id": url.rsplit("/", 1)[-1],
            "title": title,
            "url": url,
            "pub_date": pub_date,
        }
        for url, pub_date, title in new_video_tweets
    ]
    kept = content_filter.filter_items(candidates)

    dropped_ids = {c["id"] for c in candidates} - {k["id"] for k in kept}
    if dropped_ids:
        checked_tweets.extend(sorted(dropped_ids))
        state["checked_tweets"] = checked_tweets
        save_state(state, PREFIX)
        logger.info(f"Marked {len(dropped_ids)} filtered tweet(s) as checked.")

    if not kept:
        logger.info("All new tweets were filtered out as non-course content.")
        return

    new_video_tweets = [(k["url"], k["pub_date"]) for k in kept]
    tweets_to_process = new_video_tweets
    logger.info(f"Processing new video tweet(s) in this run: {[t[0] for t in tweets_to_process]}")

    # 5. Process new video tweets
    for i, (tweet_url, pub_date) in enumerate(tweets_to_process):
        logger.info(f"Processing tweet {i+1}/{len(tweets_to_process)}: {tweet_url} (pubDate: {pub_date})")
        try:
            add_tweet_video(tweet_url, pub_date=pub_date)
            logger.info(f"[DRY-RUN] Would process tweet: {tweet_url} with pubDate: {pub_date}")
            # Short sleep between processing to avoid rate limits or overlap
            if i < len(tweets_to_process) - 1:
                sleep_time = int(os.getenv("SLEEP_INTERVAL", "5"))
                logger.info(f"Sleeping for {sleep_time} seconds before the next tweet...")
                time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Error processing tweet {tweet_url}: {e}")

if __name__ == "__main__":
    check_and_sync()
