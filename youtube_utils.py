# youtube_utils.py
import re
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from youtube_transcript_api import YouTubeTranscriptApi
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')

def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    cleaned_text = cleaned_text.replace("'", "")
    return cleaned_text

def get_youtube_captions(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        full_transcript = ""
        
        for transcript in transcript_list:
            try:
                english_transcript = transcript.translate('en').fetch()
                for caption in english_transcript:
                    full_transcript += caption['text'] + " "
                break
            except Exception:
                continue
        
        return clean_text(full_transcript)
    
    except Exception as e:
        print(f"Error fetching captions: {e}")
        return None

def summarize_large_text_with_bart(input_text):
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    input_tokens = tokenizer.encode(input_text, add_special_tokens=False)
    total_input_length = len(input_tokens)
    
    desired_min_length = int(total_input_length * 0.28)
    desired_max_length = int(total_input_length * 0.40)
    
    sentences = sent_tokenize(input_text)
    max_chunk_length = 1024
    overlap = 2
    chunks = []
    
    sentence_tokens = [tokenizer.encode(sentence, add_special_tokens=False) for sentence in sentences]
    sentence_lengths = [len(tokens) for tokens in sentence_tokens]
    
    i = 0
    while i < len(sentences):
        current_chunk = []
        current_length = 0
        start = i
        
        while i < len(sentences) and current_length + sentence_lengths[i] <= max_chunk_length:
            current_chunk.append(sentences[i])
            current_length += sentence_lengths[i]
            i += 1
        
        if i < len(sentences):
            i = i - overlap if i - overlap > start else start
        
        chunks.append(' '.join(current_chunk))
    
    summaries = []
    for chunk in chunks:
        inputs = tokenizer.encode(chunk, return_tensors='pt', max_length=1024, truncation=True).to(device)
        
        with torch.no_grad():
            summary_ids = model.generate(
                inputs,
                max_length=desired_max_length // len(chunks),
                min_length=desired_min_length // len(chunks),
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    
    return ' '.join(summaries)