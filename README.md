# Textbook Summary

## Installation
- Clone this repository
- `pip install -r requirements.txt`

## Usage
### Summarizer
You can easily use the summarizer in any large summarization project to assist with paragraph context batching:
- Import the main `Summarizer` class
- Instantiate class into summarizer object with whatever options you prefer
- Pass text into summarizer object through `Summarizer.summarize` or wrapper `Summarizer.__call__`
- Optionally display statistics for instance with `Summarizer.print_stats`
  
Example usage is shown in [summarizer.py](utils/summarizer.py). View an example on how to import in [main.py](main.py)

### Parser
The only situation in which the parser will work for you is if the author of your book used the EXACT same formatting as the original book I developed this for. The code can be used as a reference for recursive parsing, but is by no means perfect.
