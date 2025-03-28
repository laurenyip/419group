import React, { useState, useCallback, useRef, useEffect } from 'react';
import Webcam from "react-webcam";
import { 
  Container, 
  TextField, 
  Button, 
  Slider, 
  Typography, 
  Box,
  Paper,
  CircularProgress
} from '@mui/material';
import axios from 'axios';

function App() {

  const webcamRef = useRef(null);   // Webcam component, allows us to interact w/ it
  const canvasRef = useRef(null);  // Reference to the canvas element for frame drawing
  const [capturing, setCapturing] = useState(false);  // Boolean state variable, tracks if we're capturing frames

  const [text, setText] = useState('');
  const [valence, setValence] = useState(0.5);
  const [arousal, setArousal] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Every time capturing changes, useEffect runs
  useEffect(() => {
    let animationFrameId;
    if (capturing) {
      // Starts capturing frames using requestAnimationFrame
      const captureFrame = () => {
        const video = webcamRef.current.video;  // Access the video stream
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');

        // Draw the current video frame to the canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Capture the frame as a base64 image (or any other format you need)
        const imageSrc = canvas.toDataURL('image/jpeg');
        
        // Send to your model here (e.g., feeding imageSrc into your ML model)
        processImageWithModel(imageSrc);

        // Request the next animation frame for continuous capture
        animationFrameId = requestAnimationFrame(captureFrame);
      };

      // Start capturing frames
      captureFrame();
    } else {
      cancelAnimationFrame(animationFrameId);  // Stop capturing frames
    }

    // Cleanup on unmount or when capturing stops
    return () => cancelAnimationFrame(animationFrameId);
  }, [capturing]);

  // Handle model processing logic (feeding image to model)
  const processImageWithModel = (imageSrc) => {
    // Example: Send imageSrc to your model for analysis
    console.log("Sending frame to model", imageSrc);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      // First, get the audio from the API
      const apiResponse = await axios.post('YOUR_API_ENDPOINT', {
        text: text
      }, {
        responseType: 'blob'
      });

      // Create a FormData object for the audio processing
      const formData = new FormData();
      formData.append('audio', new Blob([apiResponse.data], { type: 'audio/wav' }));
      formData.append('valence', valence);
      formData.append('arousal', arousal);

      // Process the audio
      const processedAudio = await axios.post('http://localhost:5000/process-audio', formData, {
        responseType: 'blob'
      });

      // Create and play the audio
      const audioUrl = URL.createObjectURL(processedAudio.data);
      const audio = new Audio(audioUrl);
      audio.play();

    } catch (err) {
      setError('Error processing audio: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm">
      <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Webcam Capture
        </Typography>

        <Webcam 
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          width="100%"
        />

        <Button 
          variant="contained" 
          color="primary" 
          sx={{ mt: 3 }}
          onClick={() => setCapturing(!capturing)}
        >
          {capturing ? "Stop Capture" : "Start Capture"}
        </Button>

        {/* Canvas for capturing the frame, hidden */}
        <canvas ref={canvasRef} style={{ display: 'none' }} width="640" height="480" />

        <Typography variant="h4" gutterBottom>
          Audio Processor
        </Typography>
        
        <form onSubmit={handleSubmit}>
          <TextField
            fullWidth
            label="Enter text to convert to speech"
            value={text}
            onChange={(e) => setText(e.target.value)}
            margin="normal"
            required
          />

          <Box sx={{ mt: 3 }}>
            <Typography gutterBottom>
              Valence (affects pitch)
            </Typography>
            <Slider
              value={valence}
              onChange={(e, newValue) => setValence(newValue)}
              min={0}
              max={1}
              step={0.1}
              marks
            />
            <Typography variant="caption" display="block" textAlign="center">
              {valence.toFixed(1)}
            </Typography>
          </Box>

          <Box sx={{ mt: 3 }}>
            <Typography gutterBottom>
              Arousal (affects speed)
            </Typography>
            <Slider
              value={arousal}
              onChange={(e, newValue) => setArousal(newValue)}
              min={0}
              max={1}
              step={0.1}
              marks
            />
            <Typography variant="caption" display="block" textAlign="center">
              {arousal.toFixed(1)}
            </Typography>
          </Box>

          <Button
            type="submit"
            variant="contained"
            color="primary"
            fullWidth
            sx={{ mt: 3 }}
            disabled={loading}
          >
            {loading ? <CircularProgress size={24} /> : 'Process Audio'}
          </Button>

          {error && (
            <Typography color="error" sx={{ mt: 2 }}>
              {error}
            </Typography>
          )}
        </form>
      </Paper>
    </Container>
  );
}

export default App;