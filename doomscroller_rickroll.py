import webbrowser
from collections import deque
from time import time, sleep
from pynput import mouse

# Configuration
SCROLL_WINDOW = 5          # seconds
SCROLL_THRESHOLD = 20      # number of scrolls within the window
COOLDOWN = 60              # seconds before next Rickroll
RICKROLL_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# State
scroll_times = deque(maxlen=SCROLL_THRESHOLD)
last_trigger = 0

def on_scroll(x, y, dx, dy):
    """Called on every mouse scroll event."""
    global last_trigger
    now = time()

    # Add current scroll time
    scroll_times.append(now)

    # Check if we have enough scrolls within the window
    if len(scroll_times) == SCROLL_THRESHOLD:
        oldest = scroll_times[0]
        if now - oldest <= SCROLL_WINDOW:
            # Cooldown check
            if now - last_trigger > COOLDOWN:
                print("🚨 Doomscrolling detected! Time to get Rickrolled!")
                webbrowser.open(RICKROLL_URL)
                last_trigger = now
                scroll_times.clear()  # optional: reset after trigger
            else:
                print("⏳ Still in cooldown, no Rickroll this time.")

# Start listening
print("Doomscrolling detector running. Press Ctrl+C to stop.")
with mouse.Listener(on_scroll=on_scroll) as listener:
    try:
        listener.join()
    except KeyboardInterrupt:
        print("\nStopped.")