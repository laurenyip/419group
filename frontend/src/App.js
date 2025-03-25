import React, { useState } from 'react';
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
  const [text, setText] = useState('');
  const [valence, setValence] = useState(0.5);
  const [arousal, setArousal] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

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