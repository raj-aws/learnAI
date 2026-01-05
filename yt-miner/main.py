import os  # Handles file operations like paths and existence checks
import yt_dlp  # Downloads the video's audio and subtitle files
import difflib  # Compares two text strings for the similarity report
from faster_whisper import WhisperModel  # Local AI engine for speech-to-text
from pydub import AudioSegment  # Slices the audio into the required 30s chunk

# --- CONFIGURATION ---
VIDEO_URL = "https://www.youtube.com/watch?v=wv779vmyPVY"
VIDEO_ID = VIDEO_URL.split("v=")[-1]
TEMP_AUDIO = "temp_media.wav"
CHOPPED_AUDIO = "chunk_30s.wav"
YT_SUB_FILE = "temp_media.en.vtt" # This is the file downloaded FROM YouTube
AI_TXT_FILE = "whisper_ai_output.txt" # This is the file we will CREATE with AI

def clean_vtt_text(filepath):
    """Removes VTT technical headers and timestamps to leave only clean text."""
    if not os.path.exists(filepath): return ""
    clean_lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Ignore VTT headers, metadata, and timestamp lines (marked by -->)
            if any(x in line for x in ["WEBVTT", "Kind:", "Language:", "-->"]) or not line:
                continue
            if line.isdigit(): continue # Skip line numbers
            clean_lines.append(line)
    return " ".join(clean_lines)

def generate_report(ai_text, yt_text):
    """Creates a side-by-side HTML file and calculates a match percentage."""
    print("Step 5: Generating comparison report...")
    matcher = difflib.SequenceMatcher(None, yt_text, ai_text)
    score = matcher.ratio() * 100
    
    # Create the HTML diff file
    diff_engine = difflib.HtmlDiff()
    html_diff = diff_engine.make_file(
        yt_text.split(), 
        ai_text.split(), 
        fromdesc="YouTube Auto-Captions", 
        todesc="Whisper AI Model"
    )
    with open("comparison_report.html", "w", encoding="utf-8") as f:
        f.write(html_diff)
    return score

def main():
    try:
        # --- 1. DOWNLOAD MEDIA ---
        print("Step 1: Downloading media and captions...")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'temp_media',
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([VIDEO_URL])

        # --- 2. CHOP AUDIO ---
        print("Step 2: Slicing 30s chunk...")
        audio = AudioSegment.from_wav(TEMP_AUDIO)
        audio[:30000].export(CHOPPED_AUDIO, format="wav")

        # --- 3. AI TRANSCRIBE & SAVE ---
        print("Step 3: AI Transcribing and saving to TXT...")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        # vad_filter=True removes silence as per manager's requirement
        segments, _ = model.transcribe(CHOPPED_AUDIO, vad_filter=True)
        ai_text = " ".join([s.text for s in segments]).strip()
        
        # SAVE THE AI TEXT TO A FILE
        with open(AI_TXT_FILE, "w", encoding="utf-8") as f:
            f.write(ai_text)

        # --- 4. PARSE YOUTUBE DATA ---
        print("Step 4: Parsing YouTube captions...")
        yt_text = clean_vtt_text(YT_SUB_FILE)
        # Match word count of YouTube text to AI text for a fair 1:1 comparison
        yt_text = " ".join(yt_text.split()[:len(ai_text.split())])

        # --- 5. FINAL REPORTING ---
        similarity = generate_report(ai_text, yt_text)

        print("\n" + "="*50)
        print("WORKFLOW COMPLETE")
        print("="*50)
        print(f"1. Audio Chunk: {CHOPPED_AUDIO}")
        print(f"2. YouTube Subs: {YT_SUB_FILE}")
        print(f"3. AI Transcript: {AI_TXT_FILE}")
        print(f"4. Comparison:  comparison_report.html")
        print(f"\nSimilarity Score: {similarity:.2f}%")
        print("="*50)

    except Exception as e:
        print(f"Critical Error: {e}")

if __name__ == "__main__":
    main()