import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your YouTube API key from environment variable
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# System Limits
MAX_COMMENTS = 1000  # Increased comment limit
