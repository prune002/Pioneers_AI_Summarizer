import pyaudio
import wave
import speech_recognition as sr
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from nltk.corpus import stopwords

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 25  # 25 seconds of recording
OUTPUT_FILE = "output.wav"

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Recording... Speak now!")
frames = []
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("Recording Stopped.")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(OUTPUT_FILE, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# Convert Speech to Text
recognizer = sr.Recognizer()

with sr.AudioFile(OUTPUT_FILE) as source:
    print("Processing audio...")
    audio_data = recognizer.record(source)  # Read entire file

    try:
        text = recognizer.recognize_google(audio_data)
        print("\nRecognized Text:", text)

        # Improved Summarization Function
        def summarize_text(text, summary_ratio=0.35):
            """Extracts key sentences using frequency-based scoring."""
            sentences = sent_tokenize(text)
            if len(sentences) <= 2:
                return text  # Return original text if it's too short

            # Tokenize words and count word frequencies (excluding stopwords)
            stop_words = set(stopwords.words("english"))
            words = word_tokenize(text.lower())
            word_frequencies = Counter(w for w in words if w.isalnum() and w not in stop_words)

            # Score sentences based on normalized word importance
            sentence_scores = {}
            for sentence in sentences:
                sentence_words = word_tokenize(sentence.lower())
                sentence_length = len(sentence_words)
                if sentence_length > 0:
                    sentence_scores[sentence] = sum(word_frequencies[word] for word in sentence_words if word in word_frequencies) / sentence_length

            # Select top-ranked sentences
            num_sentences = max(1, int(len(sentences) * summary_ratio))
            best_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

            return " ".join(best_sentences)

        summary = summarize_text(text)

        # Generate Meeting Minutes Format
        def format_meeting_summary(summary):
            points = summary.split(". ")
            points = [f"- {point.strip()}" for point in points if point.strip() and not any(word in point.lower() for word in ["kids", "women", "man", "not related"])]
            return "\n".join(points)

        meeting_minutes = f"""
        *Meeting Summary*
        ------------------------
        *Main Discussion Points:*
        {format_meeting_summary(summary)}

        *Action Items:*
        - [ ] Follow up on key decisions
        - [ ] Address any pending issues
        """

        print("\n" + meeting_minutes)

    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError:
        print("Could not connect to Google Speech Recognition service.")