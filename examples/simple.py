"""
Setup:
1. Get your API key from https://cloud.browser-use.com/new-api-key
2. Set environment variable: export BROWSER_USE_API_KEY="your-key"
"""

from pathlib import Path

from dotenv import load_dotenv

from browser_use import Agent, Browser, ChatBrowserUse

load_dotenv()

output_dir = Path('runs')
output_dir.mkdir(parents=True, exist_ok=True)

browser = Browser(
	headless=True,
	record_video_dir=str(output_dir / 'videos'),
	record_video_size={'width': 1280, 'height': 720},
)

agent = Agent(
	task='Find the number of stars of the following repos: browser-use, playwright, stagehand, react, nextjs',
	llm=ChatBrowserUse(),
	browser=browser,
	generate_gif=str(output_dir / 'simple.gif'),
	use_vision=True,
)
history = agent.run_sync()
history.save_to_file(output_dir / 'simple_history.json')

print(f'History JSON: {output_dir / "simple_history.json"}')
print(f'GIF: {output_dir / "simple.gif"}')
print(f'Videos: {output_dir / "videos"}')
