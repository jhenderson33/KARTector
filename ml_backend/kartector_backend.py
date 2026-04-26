"""
KARTector — Label Studio ML Backend (Video Object Tracking)
============================================================
Runs YOLOv8 + BoTSORT tracking on each video task and returns
per-instance videorectangle annotations for the Label Studio timeline.

Usage
-----
    export MODEL_PATH=../runs/kartector_v1/weights/best.pt
    export LABEL_STUDIO_HOST=http://localhost:8080
    export LABEL_STUDIO_API_KEY=<your-token>
    python kartector_backend.py --port 9090
"""

import os, logging, tempfile
from pathlib import Path

import cv2, requests, numpy as np
from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase

logger = logging.getLogger(__name__)

MODEL_PATH   = os.getenv("MODEL_PATH",
                         str(Path(__file__).parent.parent / "runs/kartector_v1/weights/best.pt"))
CONF_THRESH  = float(os.getenv("CONF_THRESH",  "0.20"))
IOU_THRESH   = float(os.getenv("IOU_THRESH",   "0.7"))
TRACKER      = os.getenv("TRACKER",            "botsort.yaml")
FRAME_STRIDE = int(os.getenv("FRAME_STRIDE",   "1"))   # 1=every frame, 2=every other, etc.
MAX_DISAPPEAR_FRAMES = int(os.getenv("MAX_DISAPPEAR_FRAMES", "5"))  # 0=disabled, >0=close track after N frames
AUTO_MERGE_TRACKS = os.getenv("AUTO_MERGE_TRACKS", "true").lower() in ("true", "1", "yes")  # Merge nearby tracks
MERGE_TIME_THRESHOLD = int(os.getenv("MERGE_TIME_THRESHOLD", "30"))  # Max frames between tracks to merge
MERGE_DISTANCE_THRESHOLD = float(os.getenv("MERGE_DISTANCE_THRESHOLD", "30.0"))  # Max distance % to merge
SKIP_ANNOTATED_TASKS = os.getenv("SKIP_ANNOTATED_TASKS", "true").lower() in ("true", "1", "yes")  # Skip tasks that already have annotations
CROSS_CLASS_NMS = os.getenv("CROSS_CLASS_NMS", "true").lower() in ("true", "1", "yes")  # Apply NMS across different classes
CROSS_CLASS_IOU_THRESH = float(os.getenv("CROSS_CLASS_IOU_THRESH", "0.5"))  # IoU threshold for cross-class NMS
VIDEO_KEY    = os.getenv("VIDEO_KEY",          "video")
LS_BOX_NAME  = os.getenv("LS_BOX_NAME",        "box")
LS_VID_NAME  = os.getenv("LS_VID_NAME",        "video")


def _download_video(url, ls_host, api_key):
    """Download or access video file from Label Studio.
    
    For local files (/data/local-files/?d=...), tries to access directly from filesystem.
    For remote URLs, downloads via HTTP.
    """
    original_url = url
    
    # Check if this is a local file reference
    if "/data/local-files/?d=" in url or "?d=" in url:
        # Extract the filesystem path from the URL parameter
        import urllib.parse
        if "?d=" in url:
            path_param = url.split("?d=", 1)[1].split("&")[0]
            # URL decode the path
            file_path = urllib.parse.unquote(path_param)
            # Convert to absolute path if needed
            if not file_path.startswith("/"):
                file_path = "/" + file_path
            
            logger.info(f"Local file access: {original_url} -> {file_path}")
            
            if Path(file_path).exists():
                # Return the local path directly - no download needed
                return file_path
            else:
                logger.warning(f"Local file not found: {file_path}, falling back to HTTP download")
    
    # Fall back to HTTP download
    headers = {"Authorization": f"Token {api_key}"} if api_key else {}
    
    # Ensure ls_host has a scheme
    if not ls_host.startswith(("http://", "https://")):
        ls_host = "http://" + ls_host
    
    # Handle relative URLs - prepend host with proper scheme
    if url.startswith("/"):
        url = ls_host.rstrip("/") + url
    elif not url.startswith(("http://", "https://")):
        # URL is malformed (e.g., "localhost:8080/..." instead of "http://localhost:8080/...")
        # Extract the path part after the host
        if "localhost" in url or url.count(":") > 0:
            # Remove "host:port" prefix - find first / after host
            idx = url.find("/")
            if idx > 0:
                # Take everything after the first /
                path = url[idx:]
                url = ls_host.rstrip("/") + path
            else:
                # No path, just prepend scheme
                url = "http://" + url
        else:
            # Generic case: prepend ls_host
            url = ls_host.rstrip("/") + "/" + url.lstrip("/")

    logger.info(f"Downloading video: {original_url} -> {url}")
    logger.debug(f"API key present: {bool(api_key)}, Headers: {list(headers.keys())}")

    # Try with auth first, fallback to no auth for local files
    try:
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            suffix = Path(url.split("?")[0]).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            for chunk in r.iter_content(1 << 20):
                tmp.write(chunk)
            tmp.close()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401 and "/data/local-files/" in url:
            # Local files might not require auth - try without token
            logger.warning(f"401 with auth token. Retrying without auth for local file...")
            with requests.get(url, headers={}, stream=True, timeout=120) as r:
                r.raise_for_status()
                suffix = Path(url.split("?")[0]).suffix or ".mp4"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                for chunk in r.iter_content(1 << 20):
                    tmp.write(chunk)
                tmp.close()
        else:
            raise

    return tmp.name


def _calculate_distance(box1, box2):
    """Calculate center-to-center distance between two boxes (in percentage of frame)."""
    cx1 = box1["x"] + box1["width"] / 2
    cy1 = box1["y"] + box1["height"] / 2
    cx2 = box2["x"] + box2["width"] / 2
    cy2 = box2["y"] + box2["height"] / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def _calculate_iou(box1, box2):
    """Calculate IoU (Intersection over Union) between two boxes.

    Boxes are in format: {"x": x%, "y": y%, "width": w%, "height": h%}
    Returns IoU value between 0 and 1.
    """
    # Convert to absolute coordinates (x1, y1, x2, y2)
    x1_1, y1_1 = box1["x"], box1["y"]
    x2_1, y2_1 = box1["x"] + box1["width"], box1["y"] + box1["height"]

    x1_2, y1_2 = box2["x"], box2["y"]
    x2_2, y2_2 = box2["x"] + box2["width"], box2["y"] + box2["height"]

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0  # No intersection

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _merge_tracks(track_data):
    """Merge tracks of the same class that are temporally and spatially close.

    Returns merged track_data dict with combined tracks.
    """
    if not AUTO_MERGE_TRACKS:
        return track_data

    # Build list of tracks with metadata
    tracks = []
    for tid, data in track_data.items():
        if not data.get("frames"):
            continue
        tracks.append({
            "id": tid,
            "label": data["label"],
            "frames": data["frames"][:],  # Copy the list (already truncated if closed)
            "confs": data["confs"][:],    # Copy the list (already truncated if closed)
            "start_frame": data["frames"][0]["frame"],
            "end_frame": data["frames"][-1]["frame"],  # Use actual last frame, not metadata
            "last_seen_frame": data["frames"][-1]["frame"],  # Use actual last frame from truncated list
            "closed": data.get("closed", False),
            "merged_into": None  # Track ID this was merged into
        })

    if not tracks:
        return track_data

    # Sort by start frame
    tracks.sort(key=lambda t: t["start_frame"])

    # Try to merge each track with later tracks
    merged_count = 0
    for j in range(len(tracks)):
        if tracks[j]["merged_into"] is not None:
            continue  # Already merged into something
        
        # Find all potential merge candidates (earlier tracks that could merge into track j)
        merge_candidates = []
        
        for i in range(j):
            if tracks[i]["merged_into"] is not None:
                continue  # Already merged
            
            # Only merge same class
            if tracks[i]["label"] != tracks[j]["label"]:
                continue
            
            # Check temporal proximity (using actual end frames from truncated data)
            time_gap = tracks[j]["start_frame"] - tracks[i]["end_frame"]
            if time_gap < 0:  # Overlapping tracks - skip
                continue
            if time_gap > MERGE_TIME_THRESHOLD:
                continue  # Too far apart in time
            
            # Check spatial proximity - compare last frame of track i with first frame of track j
            last_box_i = tracks[i]["frames"][-1]
            first_box_j = tracks[j]["frames"][0]
            distance = _calculate_distance(last_box_i, first_box_j)
            
            if distance <= MERGE_DISTANCE_THRESHOLD:
                # This is a valid merge candidate
                merge_candidates.append({
                    "index": i,
                    "distance": distance,
                    "time_gap": time_gap
                })
        
        # If we have multiple candidates, pick the closest one spatially
        if len(merge_candidates) > 0:
            # Sort by distance (closest first), then by time gap (smallest first)
            merge_candidates.sort(key=lambda c: (c["distance"], c["time_gap"]))
            best_candidate = merge_candidates[0]
            i = best_candidate["index"]
            
            # Merge track i into track j
            logger.info(f"Merging track {tracks[i]['id']} into {tracks[j]['id']} "
                       f"(gap={best_candidate['time_gap']} frames, distance={best_candidate['distance']:.1f}%"
                       f"{', best of ' + str(len(merge_candidates)) + ' candidates' if len(merge_candidates) > 1 else ''})")
            
            # Combine the tracks: prepend track i's frames to track j
            tracks[j]["frames"] = tracks[i]["frames"] + tracks[j]["frames"]
            tracks[j]["confs"] = tracks[i]["confs"] + tracks[j]["confs"]
            tracks[j]["start_frame"] = tracks[i]["start_frame"]  # Update to earlier start
            tracks[j]["last_seen_frame"] = max(tracks[i]["last_seen_frame"], tracks[j]["last_seen_frame"])
            # Keep track j's closed status (it's the later one)
            
            tracks[i]["merged_into"] = tracks[j]["id"]
            merged_count += 1

    # Rebuild track_data with merged tracks
    merged_track_data = {}
    for track in tracks:
        if track["merged_into"] is None:  # Not merged into another track
            merged_track_data[track["id"]] = {
                "label": track["label"],
                "frames": track["frames"],
                "confs": track["confs"],
                "last_seen_frame": track["last_seen_frame"],
                "closed": track["closed"]
            }

    if merged_count > 0:
        logger.info(f"Merged {merged_count} track(s). Tracks: {len(track_data)} → {len(merged_track_data)}")

    return merged_track_data


class KARTectorBackend(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Weights not found: {MODEL_PATH}")
        self.model       = YOLO(MODEL_PATH)
        self.class_names = self.model.names
        self._processing_tasks = set()  # Track tasks currently being processed
        logger.info(f"Loaded model. Classes: {self.class_names}")
        logger.info(f"Config: MAX_DISAPPEAR_FRAMES={MAX_DISAPPEAR_FRAMES}, "
                   f"AUTO_MERGE_TRACKS={AUTO_MERGE_TRACKS}, "
                   f"MERGE_TIME_THRESHOLD={MERGE_TIME_THRESHOLD}, "
                   f"MERGE_DISTANCE_THRESHOLD={MERGE_DISTANCE_THRESHOLD}, "
                   f"SKIP_ANNOTATED_TASKS={SKIP_ANNOTATED_TASKS}, "
                   f"CROSS_CLASS_NMS={CROSS_CLASS_NMS}, "
                   f"CROSS_CLASS_IOU_THRESH={CROSS_CLASS_IOU_THRESH}")

    def predict(self, tasks, **kwargs):
        ls_host = os.getenv("LABEL_STUDIO_HOST", "http://localhost:8080")
        api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
        out = []

        logger.info(f"Received prediction request for {len(tasks)} task(s)")

        for task in tasks:
            task_id = task.get("id", "unknown")
            url = task.get("data", {}).get(VIDEO_KEY, "")

            if not url:
                logger.warning(f"Task {task_id}: No video URL found")
                out.append({"result": []})
                continue

            # Check if this task is already being processed
            if task_id in self._processing_tasks:
                logger.warning(f"Task {task_id}: Already being processed, skipping duplicate request")
                out.append({"result": []})
                continue

            # Check if task already has annotations (completed) - skip to avoid reprocessing
            if SKIP_ANNOTATED_TASKS and (task.get("annotations") or task.get("completed_at")):
                logger.info(f"Task {task_id}: Already annotated, skipping prediction")
                out.append({"result": []})
                continue

            # Mark task as being processed
            self._processing_tasks.add(task_id)
            logger.info(f"Task {task_id}: Starting prediction for {url}")

            video_path = None
            is_temp_file = False
            try:
                video_path = _download_video(url, ls_host, api_key)
                # Check if it's a temp file (starts with temp directory)
                is_temp_file = video_path.startswith(tempfile.gettempdir())
                result = self._track_video(video_path)
                out.append(result)
                logger.info(f"Task {task_id}: Prediction completed successfully")
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}", exc_info=True)
                out.append({"result": []})
            finally:
                # Only delete if it's a temporary downloaded file, not a local file
                if video_path and is_temp_file and Path(video_path).exists():
                    Path(video_path).unlink()
                # Remove from processing set
                self._processing_tasks.discard(task_id)

        return out

    def _track_video(self, path):
        cap         = cv2.VideoCapture(path)
        total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fw          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        track_data = {}   # track_id -> {label, frames[], confs[], last_seen_frame, closed}
        active_tracks = set()  # Track IDs seen in current frame

        for i, r in enumerate(self.model.track(
                source=path, conf=CONF_THRESH, iou=IOU_THRESH,
                tracker=TRACKER, persist=True, verbose=False,
                stream=True, vid_stride=FRAME_STRIDE,
        )):
            frame_1idx = i * FRAME_STRIDE + 1          # Label Studio is 1-indexed
            t_sec      = (frame_1idx - 1) / fps

            # Track which IDs are active in this frame
            active_tracks.clear()

            if r.boxes is not None and r.boxes.id is not None:
                # Collect all detections in this frame for cross-class NMS
                frame_detections = []
                for box in r.boxes:
                    if box.id is None: continue
                    tid   = int(box.id.item())

                    # Skip adding frames to already-closed tracks
                    if tid in track_data and track_data[tid].get("closed", False):
                        continue

                    label = self.class_names[int(box.cls.item())]
                    conf  = float(box.conf.item())
                    x1,y1,x2,y2 = box.xyxy[0].tolist()

                    frame_detections.append({
                        "tid": tid,
                        "label": label,
                        "conf": conf,
                        "box": {"x": x1, "y": y1, "x2": x2, "y2": y2},
                    })

                # Apply cross-class NMS if enabled
                if CROSS_CLASS_NMS and len(frame_detections) > 1:
                    # Sort by confidence (highest first)
                    frame_detections.sort(key=lambda d: d["conf"], reverse=True)

                    # Keep track of which detections to skip
                    suppressed = set()

                    for i, det1 in enumerate(frame_detections):
                        if i in suppressed:
                            continue

                        # Convert to percentage format for IoU calculation
                        box1 = {
                            "x": round(det1["box"]["x"]/fw*100, 4),
                            "y": round(det1["box"]["y"]/fh*100, 4),
                            "width": round((det1["box"]["x2"]-det1["box"]["x"])/fw*100, 4),
                            "height": round((det1["box"]["y2"]-det1["box"]["y"])/fh*100, 4),
                        }

                        # Check against all lower-confidence detections
                        for j in range(i + 1, len(frame_detections)):
                            if j in suppressed:
                                continue

                            det2 = frame_detections[j]

                            # Only apply NMS if different classes
                            if det1["label"] == det2["label"]:
                                continue

                            # Convert to percentage format
                            box2 = {
                                "x": round(det2["box"]["x"]/fw*100, 4),
                                "y": round(det2["box"]["y"]/fh*100, 4),
                                "width": round((det2["box"]["x2"]-det2["box"]["x"])/fw*100, 4),
                                "height": round((det2["box"]["y2"]-det2["box"]["y"])/fh*100, 4),
                            }

                            # Calculate IoU
                            iou = _calculate_iou(box1, box2)

                            # Suppress lower confidence detection if IoU is high
                            if iou >= CROSS_CLASS_IOU_THRESH:
                                suppressed.add(j)
                                logger.debug(f"Frame {frame_1idx}: Suppressed {det2['label']} (conf={det2['conf']:.2f}) "
                                           f"due to overlap with {det1['label']} (conf={det1['conf']:.2f}, IoU={iou:.2f})")

                    # Remove suppressed detections
                    frame_detections = [det for i, det in enumerate(frame_detections) if i not in suppressed]

                # Add non-suppressed detections to tracks
                for det in frame_detections:
                    tid = det["tid"]
                    label = det["label"]
                    conf = det["conf"]
                    x1, y1, x2, y2 = det["box"]["x"], det["box"]["y"], det["box"]["x2"], det["box"]["y2"]

                    if tid not in track_data:
                        track_data[tid] = {
                            "label": label,
                            "frames": [],
                            "confs": [],
                            "last_seen_frame": frame_1idx,
                            "closed": False
                        }

                    track_data[tid]["label"] = label   # keep most recent class
                    track_data[tid]["confs"].append(conf)
                    track_data[tid]["last_seen_frame"] = frame_1idx
                    track_data[tid]["frames"].append({
                        "frame":    frame_1idx,
                        "x":        round(x1/fw*100, 4),
                        "y":        round(y1/fh*100, 4),
                        "width":    round((x2-x1)/fw*100, 4),
                        "height":   round((y2-y1)/fh*100, 4),
                        "time":     round(t_sec, 4),
                        "enabled":  True,
                        "rotation": 0,
                    })
                    active_tracks.add(tid)

                    # Debug logging for first few frames of each track
                    if len(track_data[tid]["frames"]) <= 3:
                        logger.debug(f"Track {tid} ({label}): added frame {frame_1idx} (conf={conf:.2f}, total frames={len(track_data[tid]['frames'])})")

            # If MAX_DISAPPEAR_FRAMES is enabled, close tracks that haven't been seen recently
            if MAX_DISAPPEAR_FRAMES > 0:
                for tid in list(track_data.keys()):
                    # Skip already closed tracks
                    if track_data[tid].get("closed", False):
                        continue

                    # If track wasn't seen in this frame, check how long it's been missing
                    if tid not in active_tracks:
                        frames_missing = frame_1idx - track_data[tid]["last_seen_frame"]
                        # If object has been missing for too long, close the track
                        if frames_missing >= MAX_DISAPPEAR_FRAMES * FRAME_STRIDE:
                            track_data[tid]["closed"] = True
                            last_seen = track_data[tid]["last_seen_frame"]

                            # Truncate frames to only include up to last_seen_frame
                            original_frame_count = len(track_data[tid]["frames"])
                            track_data[tid]["frames"] = [
                                f for f in track_data[tid]["frames"] if f["frame"] <= last_seen
                            ]
                            track_data[tid]["confs"] = track_data[tid]["confs"][:len(track_data[tid]["frames"])]

                            logger.info(f"Closing track {tid} ({track_data[tid]['label']}) at frame {last_seen} "
                                      f"(missing for {frames_missing} frames, truncated {original_frame_count} → {len(track_data[tid]['frames'])} frames)")

        logger.info(f"Tracking complete. Found {len(track_data)} track(s).")


        # Merge tracks that are spatially and temporally close
        track_data = _merge_tracks(track_data)

        logger.info(f"Final count: {len(track_data)} track(s) over {total_frames} frames.")

        results = []
        for tid, d in track_data.items():
            if not d["frames"]: continue

            # Log the frame range for each track
            start_frame = d["frames"][0]["frame"]
            end_frame = d["frames"][-1]["frame"]
            logger.info(f"Track {tid} ({d['label']}): frames {start_frame}-{end_frame} "
                       f"({len(d['frames'])} total, avg_conf={np.mean(d['confs']):.2f})")

            # Create sequence with all frames
            sequence = d["frames"][:]

            # Add a final disabled keyframe to prevent Label Studio from extending track to end of video
            # Only add if the track doesn't already end at the last frame
            if end_frame < total_frames:
                last_box = d["frames"][-1]
                sequence.append({
                    "frame":    end_frame + 1,
                    "x":        last_box["x"],
                    "y":        last_box["y"],
                    "width":    last_box["width"],
                    "height":   last_box["height"],
                    "time":     round((end_frame) / fps, 4),
                    "enabled":  False,  # This tells Label Studio to hide the box from this frame onward
                    "rotation": 0,
                })

            results.append({
                "id":        f"track_{tid}",
                "from_name": LS_BOX_NAME,
                "to_name":   LS_VID_NAME,
                "type":      "videorectangle",
                "value": {
                    "sequence":    sequence,
                    "labels":      [d["label"]],
                    "framesCount": total_frames,
                },
                "score": float(np.mean(d["confs"])),
            })

        return {
            "result": results,
            "score":  float(np.mean([r["score"] for r in results])) if results else 0.0,
            "model_version": MODEL_PATH,
        }

    def fit(self, completions, workdir=None, **kwargs):
        return {}

    def predict_one_task(self, task_id, **kwargs):
        """Predict for a single task by ID.
        
        This is useful for manually triggering predictions for specific tasks.
        """
        logger.info(f"Manual prediction request for task ID: {task_id}")
        
        # Fetch task data from Label Studio
        ls_host = os.getenv("LABEL_STUDIO_HOST", "http://localhost:8080")
        api_key = os.getenv("LABEL_STUDIO_API_KEY", "")
        
        if not api_key:
            logger.error("LABEL_STUDIO_API_KEY not set - cannot fetch task")
            return {"result": [], "error": "API key required"}
        
        # Fetch task from Label Studio API
        headers = {"Authorization": f"Token {api_key}"}
        task_url = f"{ls_host}/api/tasks/{task_id}/"
        
        try:
            response = requests.get(task_url, headers=headers)
            response.raise_for_status()
            task = response.json()
            
            # Use the regular predict method
            results = self.predict([task], **kwargs)
            return results[0] if results else {"result": []}
            
        except Exception as e:
            logger.error(f"Failed to fetch/predict task {task_id}: {e}", exc_info=True)
            return {"result": [], "error": str(e)}

    def process_event(self, event, data, job_id, additional_params):
        """Handle Label Studio webhook events (PROJECT_UPDATED, etc.)"""
        logger.info(f"Received event: {event} (job_id={job_id})")
        # No action needed for project updates - just acknowledge
        return {"status": "ok", "event": event}


if __name__ == "__main__":
    import argparse
    from label_studio_ml.api import init_app
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--host",  default="0.0.0.0")
    p.add_argument("--port",  default=9090, type=int)
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    model_dir = os.getenv("MODEL_DIR", "./ml_backend_data")
    init_app(model_class=KARTectorBackend, model_dir=model_dir).run(host=a.host, port=a.port, debug=a.debug)

