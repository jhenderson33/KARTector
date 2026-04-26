# KARTector ML Backend

A Label Studio ML backend that serves a trained YOLOv8 model with BoTSORT tracking for video object tracking and assisted labeling.

## Quick Start

```bash
# Install dependencies
pip install -r ml_backend/requirements.txt

# Run the backend
cd ml_backend/
export MODEL_PATH=../runs/kartector_v1/weights/best.pt
export LABEL_STUDIO_HOST=http://localhost:8080
export LABEL_STUDIO_API_KEY=<your-token>
python kartector_backend.py --port 9090
```

Then connect to Label Studio: Settings → Model → Add Model → `http://localhost:9090`

## Configuration

### Core Settings

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `runs/kartector_v1/weights/best.pt` | Path to trained YOLOv8 `.pt` weights |
| `LABEL_STUDIO_HOST` | `http://localhost:8080` | Label Studio server URL |
| `LABEL_STUDIO_API_KEY` | _(required)_ | LS API token (Account → Access Token) |

### Detection & Tracking

| Variable | Default | Description |
|---|---|---|
| `CONF_THRESH` | `0.20` | Minimum confidence threshold (0.0-1.0) |
| `IOU_THRESH` | `0.7` | IoU threshold for same-class NMS |
| `TRACKER` | `botsort.yaml` | Tracker config (botsort.yaml or bytetrack.yaml) |
| `FRAME_STRIDE` | `1` | Process every Nth frame (1=all, 2=every other) |

### Track Management

| Variable | Default | Description |
|---|---|---|
| `MAX_DISAPPEAR_FRAMES` | `5` | Close tracks after N frames missing (0=disabled) |
| `AUTO_MERGE_TRACKS` | `true` | Merge nearby tracks of same class |
| `MERGE_TIME_THRESHOLD` | `30` | Max frame gap to merge tracks |
| `MERGE_DISTANCE_THRESHOLD` | `30.0` | Max spatial distance (% of frame) to merge |

### Cross-Class NMS (Duplicate Removal)

| Variable | Default | Description |
|---|---|---|
| `CROSS_CLASS_NMS` | `true` | Suppress overlapping different-class detections |
| `CROSS_CLASS_IOU_THRESH` | `0.5` | IoU threshold for cross-class suppression |

### Performance

| Variable | Default | Description |
|---|---|---|
| `SKIP_ANNOTATED_TASKS` | `true` | Skip tasks with existing annotations |
| `MODEL_DIR` | `./ml_backend_data` | Backend job data directory |

### Label Studio Schema

| Variable | Default | Description |
|---|---|---|
| `VIDEO_KEY` | `video` | Key in task data for video URL |
| `LS_BOX_NAME` | `box` | The `from_name` for videorectangle |
| `LS_VID_NAME` | `video` | The `to_name` for videorectangle |

## Running

### Option A: Direct Run (Recommended)

```bash
cd ml_backend/
export MODEL_PATH=../runs/kartector_v1/weights/best.pt
export LABEL_STUDIO_HOST=http://localhost:8080
export LABEL_STUDIO_API_KEY=<your-token>
python kartector_backend.py --port 9090
```

### Option B: Via label-studio-ml CLI

```bash
cd ml_backend/
export MODEL_PATH=../runs/kartector_v1/weights/best.pt
label-studio-ml start . --port 9090
```

### Debug Mode

```bash
python kartector_backend.py --port 9090 --debug
# Shows detailed logs including cross-class NMS suppressions
```

## Connecting to Label Studio

1. Open your Label Studio project → **Settings → Model**
2. Click **Add Model**
3. Set URL to `http://localhost:9090`
4. Click **Validate and Save**

Once connected, predictions appear automatically when you open unannotated tasks.

## Label Studio Config

Your labeling config must use `VideoRectangle`:

```xml
<View>
  <Video name="video" value="$video"/>
  <VideoRectangle name="box" toName="video">
    <Label value="Boost"/>
    <Label value="Charge"/>
    <Label value="Defense"/>
    <Label value="Glide"/>
    <Label value="HP"/>
    <Label value="Offense"/>
    <Label value="Top Speed"/>
    <Label value="Turn"/>
    <Label value="Weight"/>
  </VideoRectangle>
</View>
```

## How It Works

### Pipeline

1. **Video Access**: Downloads from Label Studio or accesses local files directly
2. **Detection**: Runs YOLOv8 on each frame (or every Nth frame if `FRAME_STRIDE` > 1)
3. **Cross-Class NMS**: Removes duplicate detections of different classes at same location
4. **Tracking**: BoTSORT assigns consistent IDs across frames
5. **Track Closing**: Ends tracks when objects disappear for `MAX_DISAPPEAR_FRAMES` frames
6. **Track Merging**: Combines fragmented tracks of same class (spatial + temporal proximity)
7. **Output**: Returns `videorectangle` annotations with per-frame boxes for each track

### Key Features

#### Track Closing
When objects leave the frame (e.g., item pickups), tracks are automatically closed after N frames of absence:
```bash
export MAX_DISAPPEAR_FRAMES=5  # Close after 5 frames missing
```
This prevents tracks from extending to the end of the video.

#### Track Merging
Combines fragmented tracks when the tracker temporarily loses an object:
```bash
export AUTO_MERGE_TRACKS=true           # Enable merging
export MERGE_TIME_THRESHOLD=30          # Max 30 frame gap
export MERGE_DISTANCE_THRESHOLD=30.0    # Max 30% of frame distance
```
The algorithm picks the **spatially closest** match when multiple candidates exist.

#### Cross-Class NMS
Prevents duplicate detections when multiple classes overlap:
```bash
export CROSS_CLASS_NMS=true             # Enable
export CROSS_CLASS_IOU_THRESH=0.5       # 50% overlap threshold
```
**How it works**: If "HP" (90% conf) and "Defense" (70% conf) overlap by >50%, only "HP" is kept.

## Usage

Once connected to Label Studio, the backend automatically generates predictions when you:
- Open an unannotated task
- Click "Retrieve Predictions" in the task view
- Enable auto-annotation in project settings

## Troubleshooting

### Duplicate Detections (Different Classes)

**Symptom**: "HP" and "Defense" detected at same location  
**Solution**:
```bash
export CROSS_CLASS_NMS=true              # Enable (default)
export CROSS_CLASS_IOU_THRESH=0.3        # Lower threshold = more aggressive
python kartector_backend.py --debug      # See suppressions in logs
```

### Tracks Extending to End of Video

**Symptom**: Tracks don't end when objects disappear  
**Solution**:
```bash
export MAX_DISAPPEAR_FRAMES=5  # Close after 5 frames missing
# Restart backend for changes to take effect
```

### Wrong Tracks Being Merged

**Symptom**: Separate objects incorrectly combined into one track  
**Solution**:
```bash
export MERGE_DISTANCE_THRESHOLD=10.0     # Stricter spatial requirement
export MERGE_TIME_THRESHOLD=15           # Smaller time gap
# Or disable: export AUTO_MERGE_TRACKS=false
```

### Tracks Ending Too Early

**Symptom**: Tracks close even though object is still visible  
**Solution**:
```bash
export MAX_DISAPPEAR_FRAMES=10  # Increase tolerance
export CONF_THRESH=0.15         # Lower confidence threshold
```

### Backend Stuck in Loop

**Symptom**: Keeps re-processing same tasks  
**Check**: 
```bash
# Watch logs for:
# "Task 123: Already annotated, skipping prediction"
# "Task 124: Already being processed, skipping duplicate request"

# If needed, disable skip:
export SKIP_ANNOTATED_TASKS=false
```

### Important Detections Suppressed

**Symptom**: Correct detections removed by cross-class NMS  
**Solution**:
```bash
export CROSS_CLASS_IOU_THRESH=0.7        # Only suppress heavy overlap
# Or disable: export CROSS_CLASS_NMS=false
# Note: Model may need retraining if confidences are wrong
```

## Monitoring

Watch logs to see processing:

```
INFO:__main__:Config: MAX_DISAPPEAR_FRAMES=5, CROSS_CLASS_NMS=true, CROSS_CLASS_IOU_THRESH=0.5
INFO:__main__:Received prediction request for 1 task(s)
INFO:__main__:Task 123: Starting prediction for /data/local-files/...
INFO:__main__:Tracking complete. Found 15 track(s).
INFO:__main__:Merging track 5 into 3 (gap=3 frames, distance=3.6%, best of 2 candidates)
INFO:__main__:Merged 5 track(s). Tracks: 15 → 10
INFO:__main__:Track 1 (HP): frames 1-26 (6 total, avg_conf=0.61)
INFO:__main__:Task 123: Prediction completed successfully
```

With `--debug` flag:
```
DEBUG:__main__:Frame 42: Suppressed Defense (conf=0.72) due to overlap with HP (conf=0.85, IoU=0.80)
```

