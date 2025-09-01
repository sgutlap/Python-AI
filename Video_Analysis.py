import os
import cv2
import numpy as np
import tempfile
import json
from datetime import datetime
import google.generativeai as genai
from google.api_core import retry
import speech_recognition as sr
from pydub import AudioSegment
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure API (should use environment variables in production)
# Example: export GEMINI_API_KEY='your_actual_api_key_here'
genai.configure(api_key="AIzaSyDKnu87M0x 5IO7YiIug0p6wl0jWXizBPVc")
model = genai.GenerativeModel("gemma-3-27b-it")

def extract_audio(video_path, output_audio="temp_audio.wav"):
    """
    Extracts audio from video using ffmpeg with proper error handling.
    """
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Use absolute paths to avoid issues
        abs_video_path = os.path.abspath(video_path)
        abs_output_audio = os.path.abspath(output_audio)
        
        # Build the ffmpeg command
        cmd = [
            'ffmpeg', '-y', '-i', abs_video_path, 
            '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '44100', '-ac', '2', 
            abs_output_audio
        ]
        
        # Run the command with error checking
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"Audio extraction failed: {result.stderr}")
        
        if os.path.exists(output_audio):
            logger.info(f"Audio extracted successfully to {output_audio}")
            return output_audio
        else:
            raise RuntimeError("Audio extraction failed - output file not created")
            
    except subprocess.TimeoutExpired:
        logger.error("Audio extraction timed out")
        raise RuntimeError("Audio extraction took too long")
    except Exception as e:
        logger.error(f"Error in audio extraction: {str(e)}")
        raise

def transcribe_audio(audio_path):
    """
    Transcribes audio using Google Speech Recognition.
    """
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Initialize recognizer
        r = sr.Recognizer()
        
        # Load audio file
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
            
        # Use Google Speech Recognition
        transcription = r.recognize_google(audio_data)
        logger.info("Audio transcription completed successfully")
        return transcription
        
    except sr.UnknownValueError:
        logger.warning("Google Speech Recognition could not understand the audio")
        return "Audio could not be transcribed clearly"
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Speech Recognition service; {e}")
        return "Audio transcription service unavailable"
    except Exception as e:
        logger.error(f"Error in audio transcription: {str(e)}")
        return f"Audio transcription error: {str(e)}"

def extract_key_frames(video_path, sample_rate=30, max_frames=20):
    """
    Extracts and analyzes key frames from video with more meaningful information.
    """
    key_frames_info = []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"Video details: {frame_count} frames, {fps:.2f} FPS, {duration:.2f} seconds")
        
        # Adjust sample rate based on video length
        if frame_count > 900:  # Longer than 30 seconds at 30fps
            sample_rate = max(15, sample_rate)  # Sample more frequently for longer videos
        
        # Process frames
        frames_processed = 0
        for i in range(0, frame_count, sample_rate):
            if frames_processed >= max_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if ret:
                # Convert frame to RGB (OpenCV uses BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Calculate basic image properties
                height, width, channels = frame.shape
                avg_color = np.mean(frame_rgb, axis=(0, 1))
                brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                
                # Detect edges to estimate visual complexity
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.mean(edges) / 255.0
                
                # Save a temporary frame for Gemini to analyze if needed
                frame_info = {
                    "frame_number": i,
                    "timestamp": i / fps if fps > 0 else 0,
                    "dimensions": f"{width}x{height}",
                    "avg_color": avg_color.tolist(),
                    "brightness": float(brightness),
                    "edge_density": float(edge_density)
                }
                
                key_frames_info.append(frame_info)
                frames_processed += 1
        
        cap.release()
        logger.info(f"Extracted {len(key_frames_info)} key frames for analysis")
        return key_frames_info, duration
        
    except Exception as e:
        logger.error(f"Error in frame extraction: {str(e)}")
        if 'cap' in locals():
            cap.release()
        raise

def analyze_with_gemini(transcription, frame_analysis, video_duration, description):
    """
    Sends analysis request to Gemini API with proper formatting and error handling.
    """
    try:
        # Format frame analysis into a readable summary
        frame_summary = "Visual Analysis:\n"
        for i, frame in enumerate(frame_analysis):
            frame_summary += (
                f"Frame {i+1} (at {frame['timestamp']:.1f}s): "
                f"{frame['dimensions']}, "
                f"brightness: {frame['brightness']:.1f}, "
                f"edge density: {frame['edge_density']:.2f}\n"
            )
        
        # Prepare a comprehensive prompt
        prompt = f"""
        Video Analysis Request:
        
        Description: {description}
        Video Duration: {video_duration:.2f} seconds
        
        Audio Transcript: {transcription}
        
        {frame_summary}
        
        Please provide a detailed analysis of the video content including:
        1. Main subjects and activities in the video
        2. Overall mood or tone based on visual and audio cues
        3. Key events or transitions
        4. Technical quality assessment (lighting, clarity, etc.)
        5. Potential content categorization
        
        Provide your response in a structured format with clear sections.
        """
        
        # Generate content with retry mechanism
        response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text'):
            logger.info("Successfully received analysis from Gemini")
            return response.text
        else:
            raise ValueError("Empty or invalid response from Gemini API")
            
    except Exception as e:
        logger.error(f"Error in Gemini analysis: {str(e)}")
        return f"Analysis failed: {str(e)}"

def analyze_video_content(video_path, description="Analyze the video content and audio"):
    """
    Main function to analyze video content with enhanced features.
    """
    start_time = datetime.now()
    logger.info(f"Starting analysis of {video_path}")
    
    # Validate input
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        
        try:
            # Step 1: Extract audio
            audio_file = extract_audio(video_path, audio_path)
            
            # Step 2: Transcribe audio
            transcription = transcribe_audio(audio_file)
            
            # Step 3: Analyze video frames
            frame_analysis, video_duration = extract_key_frames(video_path)
            
            # Step 4: Send to Gemini for analysis
            analysis_result = analyze_with_gemini(
                transcription, frame_analysis, video_duration, description
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Analysis completed in {processing_time:.2f} seconds")
            
            # Return comprehensive results
            return {
                "analysis": analysis_result,
                "transcription": transcription,
                "processing_time": processing_time,
                "video_duration": video_duration,
                "frames_analyzed": len(frame_analysis)
            }
            
        except Exception as e:
            logger.error(f"Video analysis failed: {str(e)}")
            raise

def save_analysis_results(results, output_file="video_analysis_report.txt"):
    """
    Saves the analysis results to a file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("VIDEO ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing time: {results.get('processing_time', 0):.2f} seconds\n")
            f.write(f"Video duration: {results.get('video_duration', 0):.2f} seconds\n")
            f.write(f"Frames analyzed: {results.get('frames_analyzed', 0)}\n\n")
            
            f.write("TRANSCRIPTION:\n")
            f.write("-" * 20 + "\n")
            f.write(results.get('transcription', 'No transcription available') + "\n\n")
            
            f.write("ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            f.write(results.get('analysis', 'No analysis available') + "\n")
        
        logger.info(f"Analysis report saved to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        return None

# --- Main execution ---
if __name__ == "__main__":
    try:
        test_video = "Basketball.mp4"  
        
        if not os.path.exists(test_video):
            logger.error(f"Test video not found: {test_video}")
            print("Please ensure the video file exists or update the path.")
        else:
            # Analyze the video
            results = analyze_video_content(
                test_video, 
                description="Analyze this basketball game video"
            )
            
            # Display and save results
            print("\n=== VIDEO ANALYSIS COMPLETED ===")
            print(f"Duration: {results['video_duration']:.2f} seconds")
            print(f"Processing time: {results['processing_time']:.2f} seconds")
            print(f"Frames analyzed: {results['frames_analyzed']}")
            
            print("\n=== TRANSCRIPTION ===")
            print(results['transcription'][:500] + "..." if len(results['transcription']) > 500 else results['transcription'])
            
            print("\n=== AI ANALYSIS ===")
            print(results['analysis'])
            
            # Save full results to file
            report_file = save_analysis_results(results)
            if report_file:
                print(f"\nFull report saved to: {report_file}")
                
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        print(f"An error occurred: {str(e)}")