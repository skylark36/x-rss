import logging
from main import get_state, generate_rss, PREFIX, load_env

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Fetches the state from R2, regenerates the podcast RSS feed, and uploads it."""
    load_env()
    logger.info(f"Fetching current state from Cloudflare R2 for prefix: '{PREFIX}'...")
    state = get_state(PREFIX)
    
    videos = state.get("videos", {})
    logger.info(f"Found {len(videos)} podcast episode(s) in state.")

    logger.info("Regenerating and uploading rss.xml to R2...")
    generate_rss(state, PREFIX)
    logger.info("RSS feed successfully regenerated and uploaded to R2!")

if __name__ == "__main__":
    main()
