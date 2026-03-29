# PathRacer

Draw paths on a track, race them with real physics, and compare against the mathematically optimal route.

## What it does

You give it a track image and some hand-drawn paths as PNGs. It extracts the centerline from each stroke, runs physics on them (wall penalties, curvature-based speed limits, acceleration caps, jerk smoothing), and then animates the whole thing as a race. Paths get colored by speed so you can see where you're fast and where you're slow.

On top of that, it computes the theoretical best path using the Fast Marching Method. So you can actually see how close (or far) your hand-drawn line is from optimal.

## Setup

```bash
git clone https://github.com/DaviBaum/Racing-Line-Sim-Opt.git
cd Racing-Line-Sim-Opt
pip install -r requirements.txt
```

You'll need FFmpeg for MP4 export. On Mac that's `brew install ffmpeg`, on Linux `apt install ffmpeg`.

## Track format

Track PNGs use the alpha channel: transparent pixels are driveable road, opaque pixels are walls. That's the only requirement.

Path PNGs are colored strokes on a transparent background. The actual stroke color doesn't matter for the physics, it just determines the display color in the animation.

## License

MIT
