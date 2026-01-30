from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_api import fetch_comments, get_video_details
from sentiment import analyze_sentiment
from config import MAX_COMMENTS
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "youtube-sentiment-backend"})

@app.route("/api/comments", methods=["GET"])
def get_comments():
    try:
        video_id = request.args.get("videoId")
        if not video_id or len(video_id) != 11:
            return jsonify({"error": "Valid videoId (11 chars) is required"}), 400
        
        # Get video details
        video_details = get_video_details(video_id)
        if not video_details:
            return jsonify({"error": "Video not found"}), 404
        
        # Fetch comments
        comments = fetch_comments(video_id)
        if not comments:
            return jsonify({"error": "No comments found"}), 404
        
        # Analyze sentiments
        processed_comments = []
        stats = defaultdict(int)
        
        for comment in comments:
            analysis = analyze_sentiment(comment["text"])
            processed_comments.append({
                **comment,
                "sentiment": analysis["sentiment"],
                "confidence": analysis["confidence"],
                "source": analysis["source"],
                "scores": analysis["breakdown"]["raw_scores"]
            })
            stats[analysis["sentiment"]] += 1

        return jsonify({
            "videoId": video_id,
            "videoTitle": video_details["title"],
            "channelTitle": video_details["channelTitle"],
            "comments": processed_comments,
            "sentimentStats": dict(stats),
            "count": len(processed_comments)
        })

    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
