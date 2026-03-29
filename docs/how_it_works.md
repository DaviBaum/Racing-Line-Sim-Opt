# How PathRacer works

This covers the math and algorithms behind each stage of the pipeline. If you just want to run it, check the README instead.

## Centerline extraction

The input is a thick, hand-drawn stroke painted onto a transparent PNG. The goal is to turn that into a clean, ordered list of (x, y) coordinates that represent the path.

First, I threshold the alpha channel to get a binary mask of the stroke. If there happen to be stray pixels or disconnected blobs, only the largest connected component is kept. Then morphological skeletonization thins the mask down to a 1-pixel-wide spine --- it works by repeatedly peeling away boundary pixels until only the skeleton remains.

To figure out the direction of the path, I look for degree-1 pixels on the skeleton (pixels with exactly one neighbor). Those are the endpoints. I pick the two that are farthest apart using `cdist`, then use `route_through_array` to trace through the skeleton between them and get a properly ordered sequence of coordinates.

The raw skeleton coordinates are pretty jagged at this point, so I run a Savitzky-Golay filter (window 11, polynomial order 3) on the x and y channels independently. That smooths things out while keeping the overall shape accurate.

## Optimal path with FMM

This is probably the most interesting part. The Fast Marching Method finds the globally optimal path through the track --- not just a locally smooth one, but the actual shortest possible route.

I start by upsampling the driveable mask 8x in each dimension. This gives the FMM sub-pixel resolution to work with, which matters a lot for tight corners. From the upsampled mask, I build a signed distance function (negative inside the road, positive outside) and feed it into scikit-fmm's `travel_time`. That solves the Eikonal equation and gives back a scalar field where every pixel holds the shortest travel time from the start point.

The gradient of that field points "uphill" toward longer travel times. So to extract the actual path, I start at the goal and walk downhill through the gradient. I use RK4 integration for this rather than basic Euler steps, because Euler was cutting through walls on sharp turns. RK4 takes four gradient samples per step:

```
k1 = grad(p)
k2 = grad(p - h/2 * k1)
k3 = grad(p - h/2 * k2)
k4 = grad(p - h * k3)
next = p - h/6 * (k1 + 2*k2 + 2*k3 + k4)
```

The path that comes out of FMM is globally optimal but a bit noisy, so there's one more cleanup step: elastic-band relaxation. I resample the path to 128 evenly spaced "beads" and then iteratively nudge each bead toward the midpoint of its two neighbors. That shortens the path and smooths it out. After each nudge, any bead that ended up outside the driveable region gets projected back onto the nearest road pixel.
