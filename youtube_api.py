import requests
from config import YOUTUBE_API_KEY, MAX_COMMENTS

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"

def get_video_details(video_id):
    """Fetch video metadata"""
    params = {
        "key": YOUTUBE_API_KEY,
        "part": "snippet",
        "id": video_id
    }
    try:
        response = requests.get(YOUTUBE_VIDEO_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("items"):
            return {
                "title": data["items"][0]["snippet"]["title"],
                "publishedAt": data["items"][0]["snippet"]["publishedAt"],
                "channelTitle": data["items"][0]["snippet"]["channelTitle"]
            }
        return None
    except Exception as e:
        print(f"YouTube API Error: {e}")
        return None

def fetch_comments(video_id):
    """Fetch exactly 200 comments maximum"""
    comments = []
    params = {
        "key": YOUTUBE_API_KEY,
        "part": "snippet",
        "videoId": video_id,
        "maxResults": min(100, MAX_COMMENTS),
        "order": "relevance",
        "textFormat": "plainText"
    }
    
    try:
        while len(comments) < MAX_COMMENTS:
            response = requests.get(YOUTUBE_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": comment.get("textDisplay", ""),
                    "likeCount": comment.get("likeCount", 0),
                    "publishedAt": comment.get("publishedAt", ""),
                    "author": comment.get("authorDisplayName", "")
                })
                if len(comments) >= MAX_COMMENTS:
                    break
            
            if "nextPageToken" not in data or len(comments) >= MAX_COMMENTS:
                break
                
            params["pageToken"] = data["nextPageToken"]
            
        return comments[:MAX_COMMENTS]
    except Exception as e:
        print(f"Comment fetch error: {e}")
        return []
