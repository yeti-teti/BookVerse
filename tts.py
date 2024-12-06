import torch
import torchaudio
from transformers import VitsModel, AutoTokenizer, pipeline
import soundfile as sf
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class AdvancedTextToSpeech:
    def __init__(self, model_name="facebook/mms-tts-eng"):
        """
        Initialize advanced text-to-speech with genre detection
        
        Args:
            model_name (str): Hugging Face TTS model name
        """
        # Download necessary NLTK resources
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        # Device and TTS model setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VitsModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Sentiment and emotion analysis pipeline
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.emotion_analyzer = pipeline("text-classification", 
                                         model="j-hartmann/emotion-english-distilroberta-base")
        
        # Genre classification setup
        self.genre_classifier = self.train_genre_classifier()
        
        # Tone modulation mapping
        self.tone_styles = {
            'academic': {'speed': 0.8, 'pitch': 0.9},  # More measured, slightly higher pitch
            'technical': {'speed': 0.85, 'pitch': 0.95},  # Precise, slightly higher pitch
            'narrative': {'speed': 1.0, 'pitch': 1.0},  # Natural, balanced
            'conversational': {'speed': 1.1, 'pitch': 1.05},  # Slightly faster, slightly higher
            'persuasive': {'speed': 1.05, 'pitch': 1.1},  # Energetic, higher pitch
            'emotional': {'speed': 0.9, 'pitch': 1.15}  # Slightly slower, much higher pitch
        }
        
        # Create output directory
        self.output_dir = "tts_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train_genre_classifier(self):
        """
        Train a simple genre classifier using pre-defined training data
        
        Returns:
            sklearn.naive_bayes.MultinomialNB: Trained genre classifier
        """
        # Sample training data
        genres = {
            'academic': [
                "The research indicates that quantum mechanics provides profound insights into particle behavior.",
                "A comprehensive analysis of economic institutions reveals complex systemic interactions.",
                "Empirical evidence suggests a correlation between educational policies and social mobility."
            ],
            'technical': [
                "The API integrates machine learning algorithms to optimize data processing.",
                "Kubernetes deployments require careful configuration of container networking.",
                "Neural network architectures continue to evolve with advanced deep learning techniques."
            ],
            'narrative': [
                "As the sun set, she remembered the promises of her childhood.",
                "The old house stood silent, keeping its secrets locked away.",
                "He walked through the city, observing the intricate dance of urban life."
            ],
            'conversational': [
                "Hey, how's your day going? Anything exciting happening?",
                "I was thinking we could grab coffee and catch up.",
                "So, what do you think about the new project?"
            ],
            'persuasive': [
                "We must act now to address the critical challenges facing our environment.",
                "Investing in education is the most powerful strategy for social transformation.",
                "Our approach will revolutionize how we understand human potential."
            ],
            'emotional': [
                "The pain of loss overwhelmed her, memories flooding back.",
                "His heart raced with a mixture of hope and fear.",
                "The beauty of human connection transcends all boundaries."
            ]
        }
        
        # Prepare training data
        texts = []
        labels = []
        for genre, examples in genres.items():
            texts.extend(examples)
            labels.extend([genre] * len(examples))
        
        # Vectorize texts
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(texts)
        
        # Train classifier
        classifier = MultinomialNB()
        classifier.fit(X, labels)
        
        return (vectorizer, classifier)
    
    def detect_genre(self, text):
        """
        Detect the genre of the input text
        
        Args:
            text (str): Input text to classify
        
        Returns:
            str: Detected genre
        """
        vectorizer, classifier = self.genre_classifier
        vectorized_text = vectorizer.transform([text])
        genre = classifier.predict(vectorized_text)[0]
        
        # Additional analysis for refinement
        sentiment = self.sentiment_analyzer(text)[0]
        emotion = self.emotion_analyzer(text)[0]
        
        print(f"Detected Genre: {genre}")
        print(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})")
        print(f"Dominant Emotion: {emotion['label']} (Score: {emotion['score']:.2f})")
        
        return genre
    
    def generate_speech(self, text, output_filename="speech.wav", max_length=500):
        """
        Generate speech with genre-aware tone modulation
        
        Args:
            text (str): Input text to convert to speech
            output_filename (str): Name of output audio file
            max_length (int): Maximum token length per generation
        
        Returns:
            numpy.ndarray: Generated audio waveform
        """
        # Detect genre and get tone parameters
        genre = self.detect_genre(text)
        tone_style = self.tone_styles.get(genre, self.tone_styles['narrative'])
        
        # Sanitize and split text
        text = text.replace('\n', ' ').strip()
        
        def split_text(text, max_tokens):
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            chunks = []
            
            for i in range(0, len(tokens), max_tokens):
                chunk = tokens[i:i+max_tokens]
                chunks.append(self.tokenizer.decode(chunk))
            
            return chunks
        
        text_chunks = split_text(text, max_length) if len(text) > max_length else [text]
        
        # Generate speech for each chunk
        waveforms = []
        
        for chunk in text_chunks:
            # Tokenize and generate
            inputs = self.tokenizer(chunk, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                # TODO: Apply tone modulation (Note: Current VITS model doesn't support direct pitch/speed modification)
                outputs = self.model(inputs).waveform
            
            waveforms.append(outputs.cpu().squeeze().numpy())
        
        # Concatenate waveforms
        final_waveform = np.concatenate(waveforms) if len(waveforms) > 1 else waveforms[0]
        
        # Save audio file
        full_path = os.path.join(self.output_dir, output_filename)
        sf.write(full_path, final_waveform, self.model.config.sampling_rate)
        
        print(f"Speech generated and saved to {full_path}")
        print(f"Sample Rate: {self.model.config.sampling_rate}")
        print(f"Tone Style: {genre}")
        
        return final_waveform

# Example usage
if __name__ == "__main__":
    # Initialize TTS engine
    tts = AdvancedTextToSpeech()
    
    # Different genre texts
    texts = {
        "academic": """The empirical research conducted by scholars reveals a profound correlation 
        between institutional frameworks and economic development trajectories across multiple 
        socio-economic contexts.""",
        
        "technical": """The implementation of microservices architecture requires a comprehensive 
        understanding of containerization, service discovery, and distributed system design principles.""",
        
        "narrative": """As the evening light cascaded through the old library windows, Sarah discovered 
        a forgotten manuscript that would change everything she understood about her family's history.""",
        
        "persuasive": """We must take immediate and decisive action to address the critical challenges 
        of climate change, recognizing that our collective future depends on the choices we make today.""",
        
        "emotional": """In the silence of the hospital room, he held her hand, memories of their shared 
        journey flooding his heart with an overwhelming mixture of love and loss."""
    }
    
    # Generate speech for each genre
    for genre, text in texts.items():
        tts.generate_speech(text, f"{genre}_speech.wav")