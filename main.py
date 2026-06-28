import os
import json
import logging
import asyncio
import requests
import time
import random
import urllib.request
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import boto3
from botocore.config import Config
from feedgen.feed import FeedGenerator
from twscrape import API, AccountsPool
from notify import send_bark
from config import load_env

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment
load_env()

# Configuration from environment
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")  # Normalize BASE_URL
MAX_NEW_VIDEOS = int(os.getenv("MAX_NEW_VIDEOS", "5"))
ITUNES_IMAGE = os.getenv("ITUNES_IMAGE", "")
ITUNES_AUTHOR = os.getenv("ITUNES_AUTHOR", "")
ITUNES_TITLE = os.getenv("ITUNES_TITLE", "X Podcast")
SLEEP_INTERVAL = int(os.getenv("SLEEP_INTERVAL", "360"))
RUN_ONCE = os.getenv("RUN_ONCE", "false").lower() == "true"

# Twitter/X Configuration
TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")
PREFIX = os.getenv("PREFIX")
RSS_FILENAME = os.getenv("RSS_FILENAME", "rss.xml")
STATE_FILENAME = os.getenv("STATE_FILENAME", "state.json")
TWITTER_COOKIES = os.getenv("TWITTER_COOKIES")

# S3 Client for R2
s3_client = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    config=Config(signature_version="s3v4"),
)


async def init_twscrape(username: str, cookies: str) -> API:
    """Initializes and returns twscrape API instance with the given cookies."""
    logger.info(f"Initializing twscrape for account: {username}")
    pool = AccountsPool(db_file="accounts.db")
    
    # Try to get the existing account
    acc = await pool.get_account(username)
    if acc is None:
        logger.info(f"Adding new Twitter account {username} using cookies")
        await pool.add_account_cookies(username, cookies)
    else:
        # Update cookies in case they changed in .env
        from twscrape.utils import parse_cookies
        logger.info(f"Updating cookies for existing Twitter account: {username}")
        acc.cookies = parse_cookies(cookies)
        if "ct0" in acc.cookies:
            acc.active = True
        await pool.save(acc)

    # Check session login status
    await pool.login_all()
    return API(pool)


async def fetch_x_tweets(api: API, username: str, limit: int = 20) -> List:
    """Fetches recent tweets for the given Twitter username using twscrape."""
    logger.info(f"Fetching recent tweets for: {username}")
    try:
        user = await api.user_by_login(username)
        if user is None:
            logger.error(f"User {username} not found (unauthenticated or invalid session/cookies).")
            return []
        tweets = []
        async for tweet in api.user_tweets(user.id, limit=limit):
            tweets.append(tweet)
        return tweets
    except Exception as e:
        logger.error(f"Error fetching tweets for {username}: {e}")
        send_bark("X-RSS Sync Error", f"Error fetching tweets for {username}: {e}")
        return []


def get_twitter_video_info(tweet_url: str) -> Optional[Dict]:
    """Fetches video download links and metadata from x-twitter-downloader.com API."""
    logger.info(f"Fetching video info from downloader API for: {tweet_url}")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    payload = {"url": tweet_url}
    try:
        r = requests.post(
            "https://x-twitter-downloader.com/api/parse-video",
            json=payload,
            headers=headers,
            timeout=15
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("success"):
                return data
            else:
                logger.error(f"Downloader API returned success=False: {data}")
        else:
            logger.error(f"Downloader API status code: {r.status_code}, response: {r.text}")
    except Exception as e:
        logger.error(f"Error calling downloader API: {e}")
    return None


def get_state(prefix: str) -> Dict:
    """Fetches sync state from Cloudflare R2."""
    key = f"{prefix}/{STATE_FILENAME}"
    try:
        response = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except s3_client.exceptions.NoSuchKey:
        logger.info(f"State file {key} not found on R2, starting fresh.")
        return {"videos": {}}
    except Exception as e:
        logger.error(f"Error fetching state: {e}")
        send_bark("X-RSS Sync Error", f"Error fetching state: {e}")
        return {"videos": {}}


def save_state(state: Dict, prefix: str):
    """Saves sync state to Cloudflare R2."""
    key = f"{prefix}/{STATE_FILENAME}"
    try:
        s3_client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=key,
            Body=json.dumps(state, indent=2, ensure_ascii=False),
            ContentType="application/json",
        )
        logger.info(f"State saved to {key} on R2.")
    except Exception as e:
        logger.error(f"Error saving state: {e}")
        send_bark("X-RSS Sync Error", f"Error saving state: {e}")


def upload_file(local_path: Path, remote_key: str, content_type: str):
    """Uploads a local file to Cloudflare R2."""
    try:
        s3_client.upload_file(
            str(local_path),
            R2_BUCKET_NAME,
            remote_key,
            ExtraArgs={"ContentType": content_type},
        )
        logger.info(f"Uploaded {local_path} to {remote_key}")
    except Exception as e:
        logger.error(f"Error uploading file {local_path}: {e}")
        send_bark("X-RSS Sync Error", f"Error uploading file {local_path.name}: {e}")
        raise


def randomSleep(min_val: int = 10, max_val: int = 60):
    """Sleeps for a random duration to avoid rate limits."""
    delay = random.randint(min_val, max_val)
    logger.info(f"Waiting for {delay} seconds...")
    time.sleep(delay)


def download_and_extract_audio(url: str, tweet_id: str, index: int, prefix: str) -> Optional[Dict]:
    """Downloads the video file, extracts audio to .m4a using ffmpeg, and returns file info."""
    randomSleep(5, 20)
    tmp_dir = Path("downloads")
    tmp_dir.mkdir(exist_ok=True)

    mp4_path = tmp_dir / f"{tweet_id}_{index}.mp4"
    m4a_path = tmp_dir / f"{tweet_id}_{index}.m4a"

    logger.info(f"Downloading video from {url} to {mp4_path}")
    try:
        # Download video file
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(mp4_path, "wb") as out_file:
            out_file.write(response.read())

        if not mp4_path.exists() or mp4_path.stat().st_size == 0:
            logger.error("Downloaded file is empty or missing.")
            return None

        # Extract audio using ffmpeg without re-encoding
        logger.info(f"Extracting audio to {m4a_path}")
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(mp4_path),
            "-vn",
            "-acodec", "copy",
            str(m4a_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error(f"ffmpeg error (fast copy): {res.stderr}")
            # If fast copy fails, try transcoding to AAC
            logger.info("Attempting ffmpeg transcoding fallback...")
            cmd_fallback = [
                "ffmpeg",
                "-y",
                "-i", str(mp4_path),
                "-vn",
                "-c:a", "aac",
                "-b:a", "128k",
                str(m4a_path)
            ]
            res_fallback = subprocess.run(cmd_fallback, capture_output=True, text=True)
            if res_fallback.returncode != 0:
                logger.error(f"ffmpeg fallback transcoding error: {res_fallback.stderr}")
                return None

        # Clean up mp4 file
        if mp4_path.exists():
            mp4_path.unlink()

        if not m4a_path.exists():
            logger.error(f"M4A file not found at {m4a_path}")
            return None

        return {
            "filename": m4a_path.name,
            "local_path": m4a_path,
            "url": f"{BASE_URL}/{prefix}/{m4a_path.name}",
        }
    except Exception as e:
        logger.error(f"Error downloading/extracting audio: {e}")
        if mp4_path.exists():
            mp4_path.unlink()
        if m4a_path.exists():
            m4a_path.unlink()
        return None


def generate_rss(state: Dict, prefix: str):
    """Generates the podcast RSS feed file and uploads it to R2."""
    fg = FeedGenerator()
    fg.load_extension("podcast")
    
    twitter_user_url = f"https://x.com/{TWITTER_USERNAME}"
    fg.id(twitter_user_url)
    fg.title(f"{ITUNES_TITLE}")
    fg.author({"name": ITUNES_AUTHOR})
    fg.link(href=twitter_user_url, rel="alternate")
    fg.description(f"Generated RSS feed from X posts of {TWITTER_USERNAME}")
    
    if ITUNES_IMAGE:
        fg.podcast.itunes_image(ITUNES_IMAGE)
    if ITUNES_AUTHOR:
        fg.podcast.itunes_author(ITUNES_AUTHOR)

    videos = list(state["videos"].values())
    videos = sorted(videos, key=lambda x: x.get("upload_date", ""), reverse=True)

    for video in videos:
        fe = fg.add_entry()
        fe.id(video["id"])
        fe.title(video["title"])
        fe.description(video["description"])
        fe.link(href=video["url"])
        fe.enclosure(video["url"], 0, "audio/mp4")

        if video.get("upload_date"):
            try:
                fe.pubDate(video["upload_date"])
            except:
                pass

    local_rss = Path(RSS_FILENAME)
    fg.rss_file(str(local_rss), encoding="UTF-8", pretty=True)
    upload_file(
        local_rss,
        f"{prefix}/{RSS_FILENAME}",
        "application/rss+xml; charset=utf-8"
    )
    local_rss.unlink()


async def run_sync():
    """Runs a single synchronization cycle for X (Twitter) videos."""
    if not all([
        R2_ACCESS_KEY_ID,
        R2_SECRET_ACCESS_KEY,
        R2_ENDPOINT_URL,
        R2_BUCKET_NAME,
        TWITTER_USERNAME,
        TWITTER_COOKIES,
        BASE_URL
    ]):
        error_msg = "Missing required environment variables for sync in .env."
        logger.error(error_msg)
        send_bark("X-RSS Config Error", error_msg)
        return

    # 1. Initialize twscrape
    try:
        api = await init_twscrape(TWITTER_USERNAME, TWITTER_COOKIES)
    except Exception as e:
        logger.error(f"Failed to initialize twscrape: {e}")
        send_bark("X-RSS Init Error", f"Failed to initialize twscrape: {e}")
        return

    # 2. Get latest tweets
    tweets = await fetch_x_tweets(api, TWITTER_USERNAME)
    if not tweets:
        logger.info("No tweets found or error fetching tweets.")
        return

    # 3. Load state
    state = get_state(PREFIX)

    new_videos_count = 0
    # Process from oldest to newest to preserve chronological order in RSS
    tweets = list(reversed(tweets))

    for tweet in tweets:
        if not tweet.media or not tweet.media.videos:
            continue

        tweet_id = str(tweet.id)
        tweet_url = f"https://x.com/{TWITTER_USERNAME}/status/{tweet_id}"
        num_videos = len(tweet.media.videos)
        
        video_api_data = None

        for idx, video in enumerate(tweet.media.videos, start=1):
            video_key = tweet_id if num_videos == 1 else f"{tweet_id}_{idx}"

            if video_key in state["videos"]:
                continue

            if new_videos_count >= MAX_NEW_VIDEOS:
                logger.info(f"Reached limit of {MAX_NEW_VIDEOS} new videos per run.")
                break

            # Fetch downloader API metadata lazily
            if video_api_data is None:
                video_api_data = get_twitter_video_info(tweet_url)
                if not video_api_data or not video_api_data.get("videos"):
                    logger.error(f"Could not retrieve video download links for: {tweet_url}")
                    break

            # Find matching index
            variants = [v for v in video_api_data["videos"] if v.get("video_index") == idx]
            if not variants:
                if idx == 1:
                    variants = video_api_data["videos"]
                else:
                    logger.error(f"No variants found for video_index: {idx}")
                    continue

            # Sort/select lowest quality
            lowest_variant = min(variants, key=lambda v: v.get("bitrate", 999999))
            direct_url = lowest_variant.get("direct_download_url")
            if not direct_url:
                logger.error(f"No direct download URL found for variant: {lowest_variant}")
                continue

            title = video_api_data.get("title") or f"X Video from {TWITTER_USERNAME}"
            if num_videos > 1:
                title = f"{title} (Part {idx})"

            description = tweet.rawContent or title

            logger.info(f"Processing new video {video_key} ({title})")
            video_data = download_and_extract_audio(direct_url, tweet_id, idx, PREFIX)
            if video_data:
                logger.info(f"Uploading new video: {video_key}")
                upload_file(
                    video_data["local_path"],
                    f"{PREFIX}/{video_data['filename']}",
                    "audio/mp4"
                )
                video_data["local_path"].unlink()

                # Save upload date
                upload_date_str = tweet.date.isoformat() if tweet.date else datetime.now(timezone.utc).isoformat()

                state["videos"][video_key] = {
                    "id": video_key,
                    "title": title,
                    "description": description,
                    "upload_date": upload_date_str,
                    "url": video_data["url"],
                }
                new_videos_count += 1
                save_state(state, PREFIX)

        if new_videos_count >= MAX_NEW_VIDEOS:
            break

    # 4. Check if RSS exists, update if needed
    rss_key = f"{PREFIX}/{RSS_FILENAME}"
    rss_exists = False
    try:
        s3_client.head_object(Bucket=R2_BUCKET_NAME, Key=rss_key)
        rss_exists = True
    except:
        pass

    if new_videos_count > 0 or not rss_exists:
        logger.info(f"Updating RSS feed in {PREFIX}/ with {new_videos_count} new entries.")
        state = get_state(PREFIX)
        generate_rss(state, PREFIX)
    else:
        logger.info("No new videos found and RSS already exists.")


def main():
    logger.info(f"Starting Sync service mode. Syncing every {SLEEP_INTERVAL} minutes.")
    while True:
        try:
            asyncio.run(run_sync())
        except Exception as e:
            logger.error(f"Unexpected error in sync loop: {e}")
            send_bark("X-RSS Critical Error", f"Unexpected error in sync loop: {e}")

        if RUN_ONCE:
            logger.info("RUN_ONCE is True. Exiting.")
            break

        logger.info(f"Sleeping for {SLEEP_INTERVAL} minutes...")
        time.sleep(SLEEP_INTERVAL * 60)


if __name__ == "__main__":
    main()
