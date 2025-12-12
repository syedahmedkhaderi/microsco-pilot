# MicroscoPilot

Autonomous microscopy agent using Claude Vision API + DTMicroscope simulator

## Overview

MicroscoPilot is an autonomous agent that explores microscope samples by:
1. Moving around the sample surface
2. Capturing images at different positions
3. Analyzing images using Claude Sonnet 4 Vision API
4. Making intelligent decisions about where to explore next
5. Tracking discoveries and creating visualizations

## Project Structure

```
MicroscoPilot/
├── src/
│   ├── __init__.py
│   ├── agent.py          # Main agent logic
│   ├── vision.py         # Claude vision API calls
│   ├── microscope.py     # DTMicroscope wrapper
│   ├── memory.py         # Store what agent has seen
│   └── visualizer.py     # Display results
├── tests/
│   └── test_basic.py
├── outputs/              # Results saved here
├── data/                 # Sample images
├── main.py               # Run the agent
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Claude API key:**
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```
   
   Or create a `.env` file (see `.env.example`)

3. **Ensure DTMicroscope is installed:**
   The DTMicroscope package should already be installed. If not, install it from the DTMicroscope directory.

## Usage

### Basic Usage

Run the agent:
```bash
python main.py
```

The agent will:
- Initialize the microscope
- Start exploring autonomously
- Save results to the `outputs/` directory

### Configuration

You can modify exploration parameters in `src/agent.py`:
- `step_size`: How far to move in each step
- `max_steps`: Maximum number of exploration steps

### Outputs

All results are saved in the `outputs/` directory:
- `memory.json`: Complete history of discoveries
- `exploration_map.png`: Map showing visited positions
- `summary.png`: Summary visualization
- `microscopilot.log`: Detailed log file
- Individual image files for each captured image

## How It Works

1. **Microscope Wrapper** (`microscope.py`):
   - Provides a simple interface to DTMicroscope's AFM digital twin
   - Handles movement, image capture, and position tracking

2. **Vision Analyzer** (`vision.py`):
   - Sends images to Claude Vision API
   - Gets analysis and exploration suggestions
   - Converts numpy arrays to formats Claude can understand

3. **Memory System** (`memory.py`):
   - Tracks all visited positions
   - Stores discoveries and analysis results
   - Prevents revisiting the same locations

4. **Visualizer** (`visualizer.py`):
   - Creates exploration maps
   - Saves captured images
   - Generates summary visualizations

5. **Agent** (`agent.py`):
   - Orchestrates the exploration loop
   - Makes decisions about where to go next
   - Combines Claude's suggestions with exploration strategies

## Example Workflow

```
1. Agent starts at center position
2. Captures an image
3. Sends image to Claude Vision API
4. Claude analyzes and suggests next position
5. Agent moves to suggested position
6. Repeats until max_steps reached
7. Saves all discoveries and creates visualizations
```

## Troubleshooting

### API Key Issues
- Make sure `ANTHROPIC_API_KEY` is set correctly
- Check that you have API credits available

### Microscope Issues
- Ensure DTMicroscope is properly installed
- Check that dataset files are accessible
- Verify the data path in `main.py`

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check that DTMicroscope is in your Python path

## Learning Resources

This code is designed for beginners! Key features:
- Clear variable names
- Extensive comments explaining each part
- Helpful error messages
- Detailed logging

## License

This project is for the Microscopy Hackathon.

## Contributing

Feel free to modify and extend this code for your hackathon project!

