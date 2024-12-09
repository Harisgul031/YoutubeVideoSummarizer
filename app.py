# app.py
from flask import Flask, render_template, request, jsonify
import re
from youtube_utils import get_youtube_captions, summarize_large_text_with_bart

app = Flask(__name__)

def extract_video_id(url):
    # Handle different YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and shortened URLs
        r'(?:embed\/)([0-9A-Za-z_-]{11})',   # Embed URLs
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # Shortened URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        video_url = request.form.get('video_url')
        video_id = extract_video_id(video_url)
        
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL'}), 400
        
        # Get captions
        captions = get_youtube_captions(video_id)
        if not captions:
            return jsonify({'error': 'Unable to fetch video captions'}), 400
        
        # Generate summary
        summary = summarize_large_text_with_bart(captions)
        
        return jsonify({
            'summary': summary,
            'video_id': video_id
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)