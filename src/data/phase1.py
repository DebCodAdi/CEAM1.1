import argparse
import os
import pickle

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from tqdm import tqdm


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class V6UnifiedPipeline:
    def __init__(
        self,
        dataroot,
        version="v1.0-mini",
        history_steps=4,
        future_steps=6,
        frame_dt=0.5,
        radius=50.0,
    ):
        print(f"[INFO] Initializing nuScenes {version} from {dataroot}...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.frame_dt = frame_dt
        self.radius = radius
        self.target_categories = ("human.pedestrian", "vehicle.bicycle")

    def is_target_category(self, category_name):
        return any(category_name.startswith(prefix) for prefix in self.target_categories)

    def get_attribute_name(self, ann):
        if ann["attribute_tokens"]:
            attr = self.nusc.get("attribute", ann["attribute_tokens"][0])
            return attr["name"]
        return ""

    def collect_window(self, ann):
        history = [np.asarray(ann["translation"][:2], dtype=np.float32)]
        current = ann
        for _ in range(self.history_steps):
            prev_token = current["prev"]
            if not prev_token:
                return None
            current = self.nusc.get("sample_annotation", prev_token)
            history.append(np.asarray(current["translation"][:2], dtype=np.float32))

        future = []
        current = ann
        for _ in range(self.future_steps):
            next_token = current["next"]
            if not next_token:
                return None
            current = self.nusc.get("sample_annotation", next_token)
            future.append(np.asarray(current["translation"][:2], dtype=np.float32))

        history = np.stack(history[::-1]).astype(np.float32)
        future = np.stack(future).astype(np.float32)
        return history, future

    def estimate_heading(self, history, future, fallback_yaw):
        candidates = [
            history[-1] - history[-2],
            future[0] - history[-1],
            history[-2] - history[-3],
            future[min(1, len(future) - 1)] - future[0],
        ]
        for vec in candidates:
            if np.linalg.norm(vec) > 0.15:
                return float(np.arctan2(vec[1], vec[0]))
        return float(fallback_yaw)

    def make_rotation(self, heading):
        rotation_angle = (np.pi / 2.0) - heading
        c, s = np.cos(rotation_angle), np.sin(rotation_angle)
        rot_mat = np.asarray([[c, -s], [s, c]], dtype=np.float32)
        return rotation_angle, rot_mat

    def transform_points(self, points, origin, rot_mat):
        points = np.asarray(points, dtype=np.float32)
        origin = np.asarray(origin[:2], dtype=np.float32)
        rel_points = points - origin
        return rel_points @ rot_mat.T

    def build_neighbor_record(self, ann, primary_origin, rot_mat, rotation_angle):
        rollout = self.collect_window(ann)
        if rollout is None:
            return None

        history, future = rollout
        annotation_yaw = Quaternion(ann["rotation"]).yaw_pitch_roll[0]
        history_local = self.transform_points(history, primary_origin, rot_mat)
        future_local = self.transform_points(future, primary_origin, rot_mat)

        return {
            "instance_token": ann["instance_token"],
            "category": ann["category_name"],
            "attribute": self.get_attribute_name(ann),
            "size": np.asarray(ann["size"][:2], dtype=np.float32),
            "global_yaw": float(annotation_yaw),
            "local_yaw": float(wrap_angle(annotation_yaw + rotation_angle)),
            "distance": float(np.linalg.norm(history_local[-1])),
            "rel_pos": history_local[-1].astype(np.float32),
            "history": history_local.astype(np.float32),
            "future": future_local.astype(np.float32),
        }

    def run(self, output_path, limit=None):
        processed_data = []
        samples = self.nusc.sample[:limit] if limit else self.nusc.sample
        print(f"[INFO] Phase 1: Processing {len(samples)} samples...")

        for sample in tqdm(samples):
            scene = self.nusc.get("scene", sample["scene_token"])
            log = self.nusc.get("log", scene["log_token"])
            location = log["location"]
            all_anns = [self.nusc.get("sample_annotation", token) for token in sample["anns"]]

            for primary_ann in all_anns:
                if not self.is_target_category(primary_ann["category_name"]):
                    continue

                primary_rollout = self.collect_window(primary_ann)
                if primary_rollout is None:
                    continue

                primary_history_global, primary_future_global = primary_rollout
                t0_pos = np.asarray(primary_ann["translation"][:2], dtype=np.float32)
                fallback_yaw = Quaternion(primary_ann["rotation"]).yaw_pitch_roll[0]
                t0_heading = self.estimate_heading(primary_history_global, primary_future_global, fallback_yaw)
                rotation_angle, rot_mat = self.make_rotation(t0_heading)

                primary_history = self.transform_points(primary_history_global, t0_pos, rot_mat)
                primary_future = self.transform_points(primary_future_global, t0_pos, rot_mat)

                neighbors = []
                for other_ann in all_anns:
                    if other_ann["token"] == primary_ann["token"]:
                        continue

                    distance = np.linalg.norm(np.asarray(other_ann["translation"][:2], dtype=np.float32) - t0_pos)
                    if distance > self.radius:
                        continue

                    neighbor_record = self.build_neighbor_record(other_ann, t0_pos, rot_mat, rotation_angle)
                    if neighbor_record is not None:
                        neighbors.append(neighbor_record)

                neighbors.sort(key=lambda item: item["distance"])
                processed_data.append(
                    {
                        "sample_token": sample["token"],
                        "scene_token": sample["scene_token"],
                        "location": location,
                        "t0_global_pos": t0_pos.astype(np.float32),
                        "t0_global_heading": float(t0_heading),
                        "t0_rotation": float(rotation_angle),
                        "primary_instance": primary_ann["instance_token"],
                        "category": primary_ann["category_name"],
                        "attribute": self.get_attribute_name(primary_ann),
                        "primary_history": primary_history.astype(np.float32),
                        "primary_future": primary_future.astype(np.float32),
                        "neighbors": neighbors,
                    }
                )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as handle:
            pickle.dump(processed_data, handle)

        print(f"[SUCCESS] Phase 1 complete: wrote {len(processed_data)} scenes to {output_path}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dataroot = os.environ.get(
        "NUSCENES_DATAROOT",
        os.path.join(project_root, "data", "sets", "nuscenes"),
    )
    default_output = os.path.join(project_root, "data", "processed", "v6_unified_p1.pkl")

    parser = argparse.ArgumentParser(description="Phase 1: build agent-centric nuScenes windows.")
    parser.add_argument("--dataroot", default=default_dataroot)
    parser.add_argument("--version", default=os.environ.get("NUSCENES_VERSION", "v1.0-mini"))
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pipeline = V6UnifiedPipeline(dataroot=args.dataroot, version=args.version)
    pipeline.run(output_path=args.output, limit=args.limit)


if __name__ == "__main__":
    main()
