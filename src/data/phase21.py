import argparse
import gc
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from tqdm import tqdm


MAP_RESOLUTION = 0.25
PATCH_SIZE_METERS = 50.0
CANVAS_SIZE = 200
FRAME_DT = 0.5
thread_local_storage = {}


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def get_dna_vector(category, attribute):
    dna = np.zeros(12, dtype=np.float32)
    attr = (attribute or "").lower()
    cat = (category or "").lower()

    if not attr:
        dna[11] = 1.0
    else:
        if "moving" in attr:
            dna[0] = 1.0
        if "stopped" in attr:
            dna[1] = 1.0
        if "parked" in attr:
            dna[2] = 1.0
        if "with_rider" in attr:
            dna[3] = 1.0
        if "without_rider" in attr:
            dna[4] = 1.0
        if "standing" in attr:
            dna[5] = 1.0
        if "walking" in attr:
            dna[6] = 1.0
        if "running" in attr:
            dna[7] = 1.0

    if any(name in cat for name in ("car", "truck", "bus", "trailer", "construction", "emergency")):
        dna[8] = 1.0
    elif "pedestrian" in cat:
        dna[9] = 1.0
    elif "bicycle" in cat or "motorcycle" in cat:
        dna[10] = 1.0
    else:
        dna[11] = 1.0

    return dna


def is_hard_obstacle(neighbor):
    attribute = (neighbor.get("attribute") or "").lower()
    category = (neighbor.get("category") or "").lower()
    return ("parked" in attribute) or ("bicycle" in category and "without_rider" in attribute)


def render_layers(nmap, patch_box, patch_angle, layers):
    mask = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    for layer_name in layers:
        try:
            layer_mask = nmap.get_map_mask(
                patch_box,
                patch_angle,
                layer_names=[layer_name],
                canvas_size=(CANVAS_SIZE, CANVAS_SIZE),
            )[0]
            mask = np.maximum(mask, layer_mask.astype(np.uint8))
        except Exception:
            continue
    return mask


def get_channel_layers(primary_category):
    category = primary_category.lower()
    if "pedestrian" in category:
        primary_layers = ["walkway", "ped_crossing"]
        flexible_layers = ["lane", "road_segment", "drivable_area", "terrain", "carpark_area"]
    else:
        primary_layers = ["lane", "road_segment", "drivable_area", "ped_crossing"]
        flexible_layers = ["walkway", "terrain", "carpark_area"]
    return primary_layers, flexible_layers


def local_xy_to_pixel(point_xy):
    px = int(round((CANVAS_SIZE / 2.0) + (point_xy[0] / MAP_RESOLUTION)))
    py = int(round((CANVAS_SIZE / 2.0) - (point_xy[1] / MAP_RESOLUTION)))
    return px, py


def burn_rotated_box(mask, center_xy, size_xy, local_yaw):
    px, py = local_xy_to_pixel(center_xy)
    width_px = max(3.0, float(size_xy[0] / MAP_RESOLUTION))
    length_px = max(3.0, float(size_xy[1] / MAP_RESOLUTION))
    rect = ((px, py), (length_px, width_px), float(-np.degrees(local_yaw)))
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(mask, [box], color=1)


def generate_v6_map_tensor(nmap, scene):
    patch_box = (
        float(scene["t0_global_pos"][0]),
        float(scene["t0_global_pos"][1]),
        PATCH_SIZE_METERS,
        PATCH_SIZE_METERS,
    )
    patch_angle = np.degrees(scene["t0_global_heading"]) - 90.0

    primary_layers, flexible_layers = get_channel_layers(scene["category"])
    primary_mask = render_layers(nmap, patch_box, patch_angle, primary_layers)
    flexible_mask = render_layers(nmap, patch_box, patch_angle, flexible_layers)
    divider_mask = render_layers(nmap, patch_box, patch_angle, ["road_divider", "lane_divider", "road_block", "stop_line"])

    passable = np.clip(primary_mask + flexible_mask, 0, 1)
    forbidden = np.maximum(1 - passable, divider_mask).astype(np.uint8)

    for neighbor in scene["neighbors"]:
        if is_hard_obstacle(neighbor):
            burn_rotated_box(
                forbidden,
                center_xy=np.asarray(neighbor["rel_pos"], dtype=np.float32),
                size_xy=np.asarray(neighbor.get("size", [1.0, 1.0]), dtype=np.float32),
                local_yaw=float(neighbor.get("local_yaw", 0.0)),
            )

    map_tensor = np.stack([primary_mask, flexible_mask, forbidden], axis=-1).astype(np.float32)
    return map_tensor


def build_social_graph(scene):
    active_neighbors = []
    for neighbor in scene["neighbors"]:
        if is_hard_obstacle(neighbor):
            continue

        history = np.asarray(neighbor["history"], dtype=np.float32)
        future = np.asarray(neighbor["future"], dtype=np.float32)
        current = history[-1]
        prev = history[-2]
        velocity = (current - prev) / FRAME_DT
        prev_velocity = velocity
        if history.shape[0] >= 3:
            prev_velocity = (history[-2] - history[-3]) / FRAME_DT
        yaw_rate = wrap_angle(np.arctan2(velocity[1], velocity[0]) - np.arctan2(prev_velocity[1], prev_velocity[0])) / FRAME_DT

        active_neighbors.append(
            {
                "distance": float(np.linalg.norm(current)),
                "node": np.concatenate(
                    [
                        current,
                        velocity,
                        np.asarray([yaw_rate], dtype=np.float32),
                        get_dna_vector(neighbor["category"], neighbor.get("attribute", "")),
                    ]
                ).astype(np.float32),
                "history": history,
                "future": future,
            }
        )

    active_neighbors.sort(key=lambda item: item["distance"])
    if not active_neighbors:
        empty_graph = np.zeros((0, 17), dtype=np.float32)
        empty_future = np.zeros((0, 6, 2), dtype=np.float32)
        empty_history = np.zeros((0, 5, 2), dtype=np.float32)
        return empty_graph, empty_future, empty_history

    social_graph = np.stack([item["node"] for item in active_neighbors]).astype(np.float32)
    neighbor_futures = np.stack([item["future"] for item in active_neighbors]).astype(np.float32)
    neighbor_histories = np.stack([item["history"] for item in active_neighbors]).astype(np.float32)
    return social_graph, neighbor_futures, neighbor_histories


def process_single_scene(args):
    scene, dataroot = args
    map_name = scene["location"]

    if thread_local_storage.get("current_map") != map_name:
        thread_local_storage["nmap"] = NuScenesMap(dataroot=dataroot, map_name=map_name)
        thread_local_storage["current_map"] = map_name
        gc.collect()

    nmap = thread_local_storage["nmap"]

    try:
        social_graph, neighbor_futures, neighbor_histories = build_social_graph(scene)
        map_tensor = generate_v6_map_tensor(nmap, scene)
        return {
            "sample_token": scene["sample_token"],
            "scene_token": scene["scene_token"],
            "primary_category": scene["category"],
            "primary_attribute": scene.get("attribute", ""),
            "map_tensor": map_tensor,
            "primary_dna": get_dna_vector(scene["category"], scene.get("attribute", "")),
            "primary_history": np.asarray(scene["primary_history"], dtype=np.float32),
            "primary_future": np.asarray(scene["primary_future"], dtype=np.float32),
            "social_graph": social_graph,
            "neighbor_histories": neighbor_histories,
            "neighbor_futures": neighbor_futures,
        }
    except Exception as exc:
        return {"error": str(exc), "sample_token": scene.get("sample_token", "unknown")}


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dataroot = os.environ.get(
        "NUSCENES_DATAROOT",
        os.path.join(project_root, "data", "sets", "nuscenes"),
    )
    default_input = os.path.join(project_root, "data", "processed", "v6_unified_p1.pkl")
    default_output = os.path.join(project_root, "data", "processed", "v6_feature_set.pkl")

    parser = argparse.ArgumentParser(description="Phase 2.1: rasterize semantic maps and social graph features.")
    parser.add_argument("--dataroot", default=default_dataroot)
    parser.add_argument("--input", default=default_input)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    args = parser.parse_args()

    with open(args.input, "rb") as handle:
        data = pickle.load(handle)

    if args.limit is not None:
        data = data[: args.limit]
    data = sorted(data, key=lambda item: item["location"])

    print(f"[INFO] Phase 2.1 using {args.workers} worker(s) on {len(data)} scenes.")
    tasks = [(scene, args.dataroot) for scene in data]

    if args.workers == 1:
        results = [process_single_scene(task) for task in tqdm(tasks, desc="Processing v6 features")]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_single_scene, tasks, chunksize=8),
                    total=len(tasks),
                    desc="Processing v6 features",
                )
            )

    feature_set = [item for item in results if item is not None and "error" not in item]
    failures = [item for item in results if item is not None and "error" in item]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump(feature_set, handle)

    print(f"[SUCCESS] Phase 2.1 complete: wrote {len(feature_set)} scenes to {args.output}")
    if failures:
        print(f"[WARN] Skipped {len(failures)} scenes due to rasterization errors.")
        for failed in failures[:5]:
            print(f"        sample {failed['sample_token']}: {failed['error']}")


if __name__ == "__main__":
    main()
