import os
import sys
import json
import logging
import requests
import time
import random
import urllib.request
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import boto3
from botocore.config import Config
from feedgen.feed import FeedGenerator
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
ITUNES_IMAGE = os.getenv("ITUNES_IMAGE", "")
ITUNES_AUTHOR = os.getenv("ITUNES_AUTHOR", "")
ITUNES_TITLE = os.getenv("ITUNES_TITLE", "X Podcast")

# Twitter/X Configuration
TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")
PREFIX = os.getenv("PREFIX")
RSS_FILENAME = os.getenv("RSS_FILENAME", "rss.xml")
STATE_FILENAME = os.getenv("STATE_FILENAME", "state.json")

# S3 Client for R2
s3_client = boto3.client(
    "s3",
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    config=Config(signature_version="s3v4"),
)



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


def extract_tweet_id(url: str) -> Optional[int]:
    """Extracts the numerical tweet ID from a tweet URL."""
    try:
        # e.g., https://x.com/CLEC168/status/2070946389975347267
        parts = url.split("/status/")
        if len(parts) > 1:
            tweet_id_str = parts[1].split("?")[0].split("/")[0]
            return int(tweet_id_str)
    except Exception as e:
        logger.error(f"Error parsing tweet URL: {e}")
    return None


def add_tweet_video(tweet_url: str):
    tweet_id = extract_tweet_id(tweet_url)
    if not tweet_id:
        logger.error(f"Could not extract Tweet ID from URL: {tweet_url}")
        return

    state = get_state(PREFIX)

    # 1. Call downloader API directly
    video_api_data = get_twitter_video_info(tweet_url)
    if not video_api_data or not video_api_data.get("videos"):
        logger.error(f"Could not retrieve video download links for: {tweet_url}")
        return

    # 2. Identify all unique videos by video_index
    # Group variants by video_index
    video_groups = defaultdict(list)
    for variant in video_api_data["videos"]:
        v_idx = variant.get("video_index", 1)
        video_groups[v_idx].append(variant)

    num_videos = len(video_groups)
    if num_videos == 0:
        logger.error("No videos found in downloader API response.")
        return

    new_videos_added = 0

    for idx in sorted(video_groups.keys()):
        variants = video_groups[idx]
        video_key = str(tweet_id) if num_videos == 1 else f"{tweet_id}_{idx}"

        if video_key in state["videos"]:
            logger.info(f"Video {video_key} already exists in state. Skipping.")
            continue

        # Select lowest quality format by bitrate
        lowest_variant = min(variants, key=lambda v: v.get("bitrate", 999999))
        direct_url = lowest_variant.get("direct_download_url")
        if not direct_url:
            logger.error(f"No direct download URL found for variant: {lowest_variant}")
            continue

        title = video_api_data.get("title") or f"X Video from {TWITTER_USERNAME or 'X'}"
        if num_videos > 1:
            title = f"{title} (Part {idx})"

        description = title  # Use the title as the description since we don't scrape raw content

        logger.info(f"Downloading audio for {video_key}...")
        video_data = download_and_extract_audio(direct_url, str(tweet_id), idx, PREFIX)
        if not video_data:
            logger.error("Download and audio extraction failed.")
            continue

        # Upload to R2
        logger.info(f"Uploading {video_data['filename']} to R2...")
        upload_file(
            video_data["local_path"],
            f"{PREFIX}/{video_data['filename']}",
            "audio/mp4"
        )
        video_data["local_path"].unlink()

        # Update state (use current time as upload date)
        upload_date_str = datetime.now(timezone.utc).isoformat()

        state["videos"][video_key] = {
            "id": video_key,
            "title": title,
            "description": description,
            "upload_date": upload_date_str,
            "url": video_data["url"],
        }
        new_videos_added += 1
        save_state(state, PREFIX)
        logger.info(f"Video {video_key} successfully added.")

    if new_videos_added > 0:
        # Regenerate RSS
        logger.info("Regenerating RSS feed...")
        generate_rss(state, PREFIX)
        logger.info("Done!")
    else:
        logger.info("No new videos added.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <tweet_url>")
        sys.exit(1)

    url = sys.argv[1]
    add_tweet_video(url)
