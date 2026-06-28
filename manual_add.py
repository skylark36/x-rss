import sys
import logging
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime, timezone
from collections import defaultdict

from main import (
    load_env,
    get_twitter_video_info,
    download_and_extract_audio,
    upload_file,
    get_state,
    save_state,
    generate_rss,
    PREFIX,
    TWITTER_USERNAME,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    load_env()

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
        print("Usage: python manual_add.py <tweet_url>")
        sys.exit(1)

    url = sys.argv[1]
    # add_tweet_video(url)
    state = get_state(PREFIX)
    generate_rss(state, PREFIX)
