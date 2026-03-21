"""Periodic simulation snapshots to disk for offline analysis."""

import json
import os
import time

import numpy as np

from analysis.metrics import (
    get_population_stats, get_genome_diversity, get_movement_stats,
    get_predation_stats, get_light_attenuation_stats, get_zone_stats,
    get_waste_stats, get_spatial_snapshot, get_genome_weight_snapshot,
    get_burst_spatial_snapshot, get_crn_snapshot,
    get_cluster_stats, get_division_stats,
)
from config import (
    SPATIAL_SNAPSHOT_INTERVAL, BURST_SNAPSHOT_INTERVAL,
    BURST_SNAPSHOT_LENGTH, GENOME_WEIGHT_SNAPSHOT_INTERVAL,
    GENOME_TYPE,
)


def _np_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


class SimulationLogger:
    def __init__(self, run_dir: str = "runs"):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(run_dir, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.spatial_dir = os.path.join(self.log_dir, "spatial")
        os.makedirs(self.spatial_dir, exist_ok=True)
        self.burst_dir = os.path.join(self.log_dir, "burst")
        os.makedirs(self.burst_dir, exist_ok=True)
        self.genomes_dir = os.path.join(self.log_dir, "genomes")
        os.makedirs(self.genomes_dir, exist_ok=True)

        self.log_path = os.path.join(self.log_dir, "metrics.jsonl")
        self._file = open(self.log_path, "w")

        self.lineage_path = os.path.join(self.log_dir, "lineage.jsonl")
        self._lineage_file = open(self.lineage_path, "w")
        self._lineage_buffer = []

        self.oee_path = os.path.join(self.log_dir, "oee_metrics.jsonl")
        self._oee_file = open(self.oee_path, "w")

        # CRN metrics
        self._crn_file = None
        if GENOME_TYPE == "crn":
            self.crn_path = os.path.join(self.log_dir, "crn_metrics.jsonl")
            self._crn_file = open(self.crn_path, "w")

        # CTRNN metrics
        self._ctrnn_file = None
        if GENOME_TYPE == "ctrnn":
            self.ctrnn_path = os.path.join(self.log_dir, "ctrnn_metrics.jsonl")
            self._ctrnn_file = open(self.ctrnn_path, "w")

        # Burst snapshot state machine
        self._burst_active = False
        self._burst_start_tick = -1
        self._burst_frame = 0
        self._burst_subdir = None

    def snapshot(self, tick: int):
        """Record a snapshot of simulation metrics."""
        pop_stats = get_population_stats()
        div_stats = get_genome_diversity()
        move_stats = get_movement_stats()
        pred_stats = get_predation_stats()

        light_stats = get_light_attenuation_stats()
        zone_stats = get_zone_stats()
        waste_stats = get_waste_stats()
        cluster_stats = get_cluster_stats()
        division_stats = get_division_stats()

        record = {
            "tick": tick,
            **pop_stats,
            **div_stats,
            **move_stats,
            **pred_stats,
            **light_stats,
            **zone_stats,
            **waste_stats,
            **cluster_stats,
            **division_stats,
        }

        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

        # CRN metrics
        if self._crn_file is not None:
            crn_snap = get_crn_snapshot()
            if crn_snap is not None:
                crn_snap["tick"] = tick
                self._crn_file.write(
                    json.dumps(crn_snap, default=_np_serializer) + "\n")
                self._crn_file.flush()

        # CTRNN metrics
        if self._ctrnn_file is not None:
            from analysis.metrics import get_ctrnn_snapshot
            ctrnn_snap = get_ctrnn_snapshot()
            if ctrnn_snap is not None:
                ctrnn_snap["tick"] = tick
                self._ctrnn_file.write(
                    json.dumps(ctrnn_snap, default=_np_serializer) + "\n")
                self._ctrnn_file.flush()

        # Save spatial snapshot at lower frequency
        if tick % SPATIAL_SNAPSHOT_INTERVAL == 0:
            spatial = get_spatial_snapshot()
            path = os.path.join(self.spatial_dir, f"spatial_{tick:08d}.npz")
            np.savez_compressed(path, **spatial)

        # Save genome weights at configured interval
        if tick > 0 and tick % GENOME_WEIGHT_SNAPSHOT_INTERVAL == 0:
            self.save_genome_weights(tick)

    def log_lineage_events(self, events):
        """Append lineage events to lineage.jsonl. Flushes every 100 events."""
        for parent_gid, child_gid, tick in events:
            self._lineage_buffer.append(
                json.dumps({"parent": parent_gid, "child": child_gid, "tick": tick})
            )
        if len(self._lineage_buffer) >= 100:
            self._flush_lineage()

    def _flush_lineage(self):
        if self._lineage_buffer:
            self._lineage_file.write("\n".join(self._lineage_buffer) + "\n")
            self._lineage_file.flush()
            self._lineage_buffer.clear()

    def check_burst_snapshot(self, tick: int):
        """State machine: start burst at interval, capture N consecutive frames."""
        if self._burst_active:
            # Currently capturing — save this frame
            snap = get_burst_spatial_snapshot()
            path = os.path.join(self._burst_subdir,
                                f"frame_{self._burst_frame:02d}.npz")
            np.savez_compressed(path, **snap)
            self._burst_frame += 1
            if self._burst_frame >= BURST_SNAPSHOT_LENGTH:
                self._burst_active = False
                print(f"  [burst] Captured {BURST_SNAPSHOT_LENGTH} frames at tick {self._burst_start_tick}")
        elif tick > 0 and tick % BURST_SNAPSHOT_INTERVAL == 0:
            # Start a new burst
            self._burst_active = True
            self._burst_start_tick = tick
            self._burst_frame = 0
            self._burst_subdir = os.path.join(
                self.burst_dir, f"burst_{tick:08d}")
            os.makedirs(self._burst_subdir, exist_ok=True)

    def save_genome_weights(self, tick: int):
        """Save active genome weights, lineage info, and ref counts."""
        snap = get_genome_weight_snapshot()
        path = os.path.join(self.genomes_dir, f"genomes_{tick:08d}.npz")
        np.savez_compressed(path, **snap)

    def log_oee_metrics(self, metrics: dict):
        """Write OEE metrics to oee_metrics.jsonl."""
        self._oee_file.write(json.dumps(metrics) + "\n")
        self._oee_file.flush()

    def close(self):
        self._flush_lineage()
        self._file.close()
        self._lineage_file.close()
        self._oee_file.close()
        if self._crn_file is not None:
            self._crn_file.close()
        if self._ctrnn_file is not None:
            self._ctrnn_file.close()
