from datetime import datetime
import requests
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logger = logging.getLogger(__name__)


def send_bark(title: str, content: str):
    """
    Send a notification using Bark API.
    
    :param title: Title of the notification
    :param content: Content of the notification
    """
    bark_key = os.getenv("BARK_KEY")
    if not bark_key:
        logger.warning("BARK_KEY not found in environment variables. Skipping Bark notification.")
        return
    
    url = f"https://api.day.app/{bark_key}"
    
    # Add timestamp to content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_content = f"{content}\n\n[Sent at: {timestamp}]"
    
    payload = {
        "title": title,
        "body": full_content,
        "sound": "chime",  # Optional: add a sound
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info(f"Successfully sent Bark notification: {title}")
        else:
            logger.error(f"Failed to send Bark notification: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error sending Bark notification: {e}")
