"""Unit tests — all agents, runnable without GPU or API key."""
from __future__ import annotations
import json
import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.pop("OPENROUTER_API_KEY", None)  # force fallback mode for all tests

from core.event_bus import EventBus, EventType, Event
from core.experiment_registry import ExperimentRegistry
from core.shared_context import SharedContext, DatasetProfile
from utils.metrics import (
    EpochMetrics, ClassMetrics, SizeMetrics, AnchorMetrics,
    AnalysisReport, ConfigDelta, text_to_feature_vector, compute_cosine_similarity,
)


def make_config(tmp_dir: str) -> dict:
    return {
        "system": {"log_level": "WARNING", "gpu_memory_gb": 16.0, "max_iterations": 5},
        "llm": {"model": "anthropic/claude-sonnet-4-5", "max_tokens": 512, "temperature": 0.1},
        "monitor": {"overfitting_patience": 3, "overfitting_threshold": 0.005,
                    "plateau_patience": 4, "plateau_threshold": 0.001,
                    "grad_norm_max": 100.0, "lr_collapse_threshold": 1e-7},
        "analyzer": {"weak_class_ap_threshold": 0.3, "small_obj_size": 32,
                     "medium_obj_size": 96, "fp_analysis_iou": 0.5, "anchor_assignment_min_ratio": 0.5},
        "planner": {"model": "anthropic/claude-sonnet-4-5", "max_tokens": 1024,
                    "temperature": 0.2, "max_suggestions": 3, "confidence_threshold": 0.5},
        "memory": {"max_entries": 50, "similarity_threshold": 0.5, "embedding_dim": 64,
                   "top_k_recall": 8, "top_k_final": 3, "store_path": "memory_store.pkl"},
        "trainer": {"max_parallel_jobs": 1, "vram_buffer_gb": 2.0, "dedup_check": True},
    }


def make_context(tmp_dir: str) -> SharedContext:
    return SharedContext(
        dataset=DatasetProfile(num_classes=5, class_names=["person", "car", "bicycle", "dog", "cat"]),
        experiment_dir=tmp_dir,
        max_iterations=5,
    )


# ─────────────────────────────────────────────────────────────────────────────
class TestSharedContext(unittest.TestCase):
    def test_start_end_iteration(self):
        ctx = SharedContext()
        ctx.start_iteration(1, "run_001", {})
        ctx.end_iteration(0.5, 0.3, {"lr0": 0.005})
        self.assertEqual(ctx.best_map50, 0.5)
        self.assertEqual(ctx.iteration, 1)
        self.assertEqual(len(ctx.iteration_history), 1)

    def test_trend_summary_first_iter(self):
        ctx = SharedContext()
        s = ctx.trend_summary()
        self.assertIn("no trend", s)

    def test_trend_improving(self):
        ctx = SharedContext()
        for i, m in enumerate([0.3, 0.4, 0.5]):
            ctx.start_iteration(i+1, f"r{i}", {})
            ctx.end_iteration(m, m*0.6, {})
        self.assertIn("improving", ctx.trend_summary())

    def test_consecutive_no_improvement(self):
        ctx = SharedContext()
        ctx.start_iteration(1, "r1", {})
        ctx.end_iteration(0.5, 0.3, {})
        ctx.start_iteration(2, "r2", {})
        ctx.end_iteration(0.4, 0.2, {})  # worse
        self.assertEqual(ctx.consecutive_no_improvement, 1)

    def test_to_prompt_block(self):
        ctx = SharedContext(dataset=DatasetProfile(num_classes=3, class_names=["a", "b", "c"]))
        block = ctx.to_prompt_block()
        self.assertIn("Shared Context", block)
        self.assertIn("Dataset Profile", block)

    def test_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = SharedContext(experiment_dir=tmp)
            ctx.start_iteration(1, "r1", {})
            ctx.end_iteration(0.5, 0.3, {})
            ctx.save()
            self.assertTrue((Path(tmp) / "shared_context.json").exists())


# ─────────────────────────────────────────────────────────────────────────────
class TestEventBus(unittest.TestCase):
    def setUp(self): self.bus = EventBus()

    def test_publish_subscribe(self):
        received = []
        self.bus.subscribe(EventType.EPOCH_END, lambda e: received.append(e))
        self.bus.publish(Event(type=EventType.EPOCH_END, source="t", data={"epoch": 1}))
        self.assertEqual(len(received), 1)

    def test_history(self):
        self.bus.publish(Event(type=EventType.EPOCH_END, source="s", data=None))
        self.bus.publish(Event(type=EventType.MONITOR_ALERT, source="s", data=None))
        self.assertEqual(len(self.bus.get_history(EventType.EPOCH_END)), 1)
        self.assertEqual(len(self.bus.get_history()), 2)

    def test_unsubscribe(self):
        received = []
        h = lambda e: received.append(e)
        self.bus.subscribe(EventType.EPOCH_END, h)
        self.bus.unsubscribe(EventType.EPOCH_END, h)
        self.bus.publish(Event(type=EventType.EPOCH_END, source="s", data=None))
        self.assertEqual(len(received), 0)


# ─────────────────────────────────────────────────────────────────────────────
class TestExperimentRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry = ExperimentRegistry(Path(self.tmp.name) / "registry.db")
    def tearDown(self): self.tmp.cleanup()

    def test_register_get(self):
        self.registry.register("run_001", {"training": {"model": "yolov8n.pt", "imgsz": 640, "batch": 16}})
        self.assertEqual(len(self.registry.get_all()), 1)

    def test_dedup(self):
        cfg = {"training": {"model": "yolov8n.pt", "imgsz": 640, "batch": 16}}
        self.registry.register("run_001", cfg)
        self.registry.update_status("run_001", "completed")
        self.assertIsNotNone(self.registry.is_duplicate(cfg["training"]))

    def test_no_dedup_different(self):
        self.registry.register("run_001", {"training": {"imgsz": 640, "batch": 16}})
        self.registry.update_status("run_001", "completed")
        self.assertIsNone(self.registry.is_duplicate({"imgsz": 1280, "batch": 16}))

    def test_get_best(self):
        for i, m in enumerate([0.5, 0.7, 0.6]):
            self.registry.register(f"r{i}", {"training": {"imgsz": 640+i*10, "batch": 16}})
            self.registry.update_status(f"r{i}", "completed", final_map50=m)
        self.assertEqual(self.registry.get_best(1)[0].final_map50, 0.7)


# ─────────────────────────────────────────────────────────────────────────────
class TestMetrics(unittest.TestCase):
    def test_epoch_total_loss(self):
        m = EpochMetrics(epoch=1, train_box_loss=0.5, train_cls_loss=0.3, train_dfl_loss=0.2)
        self.assertAlmostEqual(m.train_loss_total, 1.0)

    def test_f1(self):
        m = ClassMetrics(class_name="cat", precision=0.8, recall=0.6)
        self.assertAlmostEqual(m.f1, 2*0.8*0.6/(0.8+0.6), places=5)

    def test_config_delta_apply(self):
        base = {"training": {"lr0": 0.01, "imgsz": 640}}
        new = ConfigDelta(changes={"lr0": 0.005, "imgsz": 1280}).apply_to(base)
        self.assertEqual(new["training"]["lr0"], 0.005)
        self.assertEqual(base["training"]["lr0"], 0.01)  # immutable

    def test_anchor_ratio(self):
        am = AnchorMetrics(total_objects=100, assigned_objects=80, unassigned_objects=20)
        self.assertAlmostEqual(am.assignment_ratio, 0.8)


# ─────────────────────────────────────────────────────────────────────────────
class TestEmbedding(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(text_to_feature_vector("test", 128).shape, (128,))

    def test_normalized(self):
        v = text_to_feature_vector("test", 64)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_identical(self):
        v = text_to_feature_vector("same text", 64)
        self.assertAlmostEqual(compute_cosine_similarity(v, v), 1.0, places=5)

    def test_zero_vector(self):
        self.assertEqual(compute_cosine_similarity(np.zeros(64, dtype=np.float32),
                                                   text_to_feature_vector("t", 64)), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
class TestMonitorAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from agents.monitor_agent import MonitorAgent
        cfg = make_config(self.tmp.name)
        ctx = make_context(self.tmp.name)
        self.agent = MonitorAgent(cfg, EventBus(), Path(self.tmp.name), ctx)
    def tearDown(self): self.tmp.cleanup()

    def test_fallback_gradient_explosion(self):
        # New logic: requires sustained grad norm in window (3+ epochs)
        for _ in range(3):
            self.agent._grad_norm_window.append(999.0)
        d = self.agent._fallback_decision(EpochMetrics(epoch=10, grad_norm=999.0))
        self.assertEqual(d["assessment"], "gradient_explosion")
        self.assertTrue(d["should_alert"])

    def test_single_epoch_grad_spike_not_alert(self):
        # Single spike should NOT trigger — prevents false positives at epoch 1
        self.agent._grad_norm_window.append(999.0)
        d = self.agent._fallback_decision(EpochMetrics(epoch=5, grad_norm=999.0))
        self.assertEqual(d["assessment"], "healthy")

    def test_metrics_corruption_detection(self):
        from agents.monitor_agent import MonitorAgent
        m = EpochMetrics(epoch=5, map50=0.0)
        m.train_box_loss = 0.0
        self.assertTrue(MonitorAgent._metrics_look_corrupted(m))
        m2 = EpochMetrics(epoch=0)
        self.assertFalse(MonitorAgent._metrics_look_corrupted(m2))

    def test_fallback_lr_collapse(self):
        d = self.agent._fallback_decision(EpochMetrics(epoch=1, lr=1e-10))
        self.assertEqual(d["assessment"], "lr_collapse")
        self.assertTrue(d["should_stop"])

    def test_fallback_plateau(self):
        for _ in range(5):
            self.agent._map_window.append(0.4500)
        d = self.agent._fallback_decision(EpochMetrics(epoch=10, map50=0.4500))
        self.assertEqual(d["assessment"], "plateau")

    def test_fallback_healthy(self):
        for i in range(3):
            self.agent._map_window.append(0.4 + i*0.05)
        d = self.agent._fallback_decision(EpochMetrics(epoch=5, map50=0.5))
        self.assertEqual(d["assessment"], "healthy")

    def test_reset(self):
        self.agent.history.append(EpochMetrics(epoch=1))
        self.agent.reset()
        self.assertEqual(len(self.agent.history), 0)
        self.assertFalse(self.agent.should_stop())


# ─────────────────────────────────────────────────────────────────────────────
class TestAnalyzerAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from agents.analyzer_agent import AnalyzerAgent
        cfg = make_config(self.tmp.name)
        ctx = make_context(self.tmp.name)
        self.agent = AnalyzerAgent(cfg, EventBus(), Path(self.tmp.name), ctx)
    def tearDown(self): self.tmp.cleanup()

    def test_mock_report(self):
        r = self.agent._mock_report("r1")
        self.assertGreater(len(r.class_metrics), 0)
        self.assertIsNotNone(r.confusion_matrix)

    def test_rule_based_weaknesses(self):
        r = self.agent._mock_report("r1")
        ws = self.agent._rule_based_weaknesses(r)
        self.assertGreater(len(ws), 0)
        sev = [w["severity"] for w in ws]
        self.assertTrue(any(s in ("critical", "high") for s in sev))

    def test_priority_ordering(self):
        r = self.agent._mock_report("r1")
        ws = self.agent._rule_based_weaknesses(r)
        priorities = [w["priority"] for w in ws]
        self.assertEqual(priorities, sorted(priorities))

    def test_run_mock_emits_event(self):
        events = []
        bus = EventBus()
        from agents.analyzer_agent import AnalyzerAgent
        agent = AnalyzerAgent(make_config(self.tmp.name), bus, Path(self.tmp.name), make_context(self.tmp.name))
        bus.subscribe(EventType.ANALYSIS_COMPLETE, lambda e: events.append(e))
        agent.run(run_id="test_run")
        self.assertEqual(len(events), 1)

    def test_fallback_decision(self):
        r = self.agent._mock_report("r1")
        d = self.agent._fallback_decision(r)
        self.assertIn("weakness_interpretations", d)
        self.assertIn("suggested_priority_focus", d)


# ─────────────────────────────────────────────────────────────────────────────
class TestMemoryAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from agents.memory_agent import MemoryAgent
        cfg = make_config(self.tmp.name)
        ctx = make_context(self.tmp.name)
        self.agent = MemoryAgent(cfg, EventBus(), Path(self.tmp.name), ctx)
    def tearDown(self): self.tmp.cleanup()

    def test_store_recall(self):
        ws = [{"type": "small_object_weakness", "severity": "high", "affected": "all"}]
        self.agent.store(ws, ConfigDelta(changes={"imgsz": 1280}), 0.05, "r1")
        result = self.agent.run(ws)
        self.assertGreater(result.get("raw_count", 0), 0)

    def test_recall_empty(self):
        result = self.agent.run([{"type": "unknown"}])
        self.assertEqual(result.get("raw_count", 0), 0)

    def test_update_existing_entry(self):
        ws = [{"type": "weak_class", "severity": "high", "affected": "cat"}]
        self.agent.store(ws, ConfigDelta(changes={"lr0": 0.005}), 0.04, "r1")
        self.agent.store(ws, ConfigDelta(changes={"lr0": 0.005}), 0.06, "r2")
        self.assertEqual(len(self.agent._entries), 1)
        self.assertAlmostEqual(self.agent._entries[0].actual_map_improvement, 0.05)
        self.assertEqual(self.agent._entries[0].iteration_count, 2)

    def test_max_entries_eviction(self):
        for i in range(60):
            ws = [{"type": f"t{i}", "severity": "low", "affected": f"c{i}"}]
            self.agent.store(ws, ConfigDelta(changes={"lr0": 0.001*i}), float(i)*0.001, f"r{i}")
        self.assertLessEqual(len(self.agent._entries), 50)

    def test_get_stats(self):
        ws = [{"type": "test", "severity": "medium", "affected": "a"}]
        self.agent.store(ws, ConfigDelta(changes={"lr0": 0.005}), 0.03, "r1")
        s = self.agent.get_stats()
        self.assertEqual(s["total"], 1)

    def test_fallback_curation(self):
        raw = [{"similarity": 0.8, "weakness_signature": "t:c:h", "config_delta": {"changes": {"lr0": 0.005}},
                "actual_map_improvement": 0.04, "run_id": "r1", "timestamp": "2024", "iteration_count": 1}]
        result = self.agent._fallback_curation(raw)
        self.assertIn("curated_memories", result)
        self.assertGreater(len(result["curated_memories"]), 0)


# ─────────────────────────────────────────────────────────────────────────────
class TestTrainerAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from agents.trainer_agent import TrainerAgent
        cfg = make_config(self.tmp.name)
        ctx = make_context(self.tmp.name)
        self.agent = TrainerAgent(cfg, EventBus(), Path(self.tmp.name), ctx)
        self.base_config = {"training": {"model": "yolov8n.pt", "imgsz": 640, "batch": 4,
                                          "lr0": 0.01, "weight_decay": 0.0005, "epochs": 1, "data": "data.yaml"}}
    def tearDown(self): self.tmp.cleanup()

    def test_vram_check_pass(self):
        # 640/16 = reference → should pass (same as reference minus buffer)
        self.assertTrue(self.agent._vram_check({"imgsz": 320, "batch": 4}))

    def test_vram_check_fail(self):
        # imgsz=1280 batch=16 = 4x reference = 52GB >> 14GB available
        self.assertFalse(self.agent._vram_check({"imgsz": 1280, "batch": 16}))

    def test_vram_estimate_scales_correctly(self):
        # 640 batch=16 → ref_gb*1.2 = 13*1.2 = 15.6
        est_ref = self.agent._estimate_vram_gb(640, 16)
        self.assertAlmostEqual(est_ref, 13.0 * 1.2, places=1)
        # 1280 batch=16 → 4x pixels → 15.6*4 = 62.4
        est_2x = self.agent._estimate_vram_gb(1280, 16)
        self.assertAlmostEqual(est_2x, 13.0 * 4 * 1.2, places=1)

    def test_json_truncation_recovery(self):
        from agents.base_agent import BaseAgent
        import types
        # Create minimal concrete BaseAgent to test _parse_json
        class ConcreteAgent(BaseAgent):
            AGENT_NAME = "TestAgent"
            def run(self): pass
        import tempfile, pathlib
        from core.event_bus import EventBus
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"system": {"log_level": "WARNING"}, "llm": {"model": "x", "temperature": 0.1,
                   "monitor_max_tokens": 512, "analyzer_max_tokens": 3000,
                   "memory_max_tokens": 1024, "trainer_max_tokens": 1024, "planner_max_tokens": 2048}}
            agent = ConcreteAgent(cfg, EventBus(), pathlib.Path(tmp))
        # Complete JSON parses fine
        result = agent._parse_json('{"assessment": "healthy", "should_alert": false}')
        self.assertEqual(result["assessment"], "healthy")
        # Truncated JSON — should recover what it can or return {}
        truncated = '{"assessment": "healthy", "should_alert": false, "message": "Training is going well so far'
        result2 = agent._parse_json(truncated)
        # May fail, but should not raise
        self.assertIsInstance(result2, dict)

    def test_fallback_decision(self):
        proposals = [
            ConfigDelta(changes={"lr0": 0.005}, confidence=0.8, expected_map_improvement=0.04, priority=1),
            ConfigDelta(changes={"imgsz": 1280}, confidence=0.3, expected_map_improvement=0.01, priority=2),
        ]
        d = self.agent._fallback_decision(proposals, self.base_config)
        self.assertIn("accepted", d)
        self.assertIn("rejected", d)

    def test_apply_decision_no_merge(self):
        proposals = [
            ConfigDelta(changes={"lr0": 0.005}, confidence=0.8, expected_map_improvement=0.04, priority=1),
        ]
        decision = {"accepted": [{"proposal_index": 0, "priority": 1, "risk_level": "low",
                                   "risk_reason": "", "scheduling_note": "", "estimated_vram_ok": True}],
                    "rejected": [], "merge_suggestion": None, "execution_reasoning": "ok"}
        result = self.agent._apply_decision(decision, proposals, self.base_config)
        self.assertEqual(len(result), 1)

    def test_apply_decision_merge(self):
        proposals = [
            ConfigDelta(changes={"lr0": 0.005}, confidence=0.8, expected_map_improvement=0.03, priority=1),
            ConfigDelta(changes={"dropout": 0.1}, confidence=0.7, expected_map_improvement=0.02, priority=2),
        ]
        decision = {
            "accepted": [], "rejected": [],
            "merge_suggestion": {"indices": [0, 1], "merged_changes": {"lr0": 0.005, "dropout": 0.1},
                                  "rationale": "compatible changes"},
            "execution_reasoning": "merge"
        }
        result = self.agent._apply_decision(decision, proposals, self.base_config)
        self.assertEqual(len(result), 1)
        self.assertIn("lr0", result[0].changes)
        self.assertIn("dropout", result[0].changes)


# ─────────────────────────────────────────────────────────────────────────────
class TestPlannerFallback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from agents.planner_agent import PlannerAgent
        cfg = make_config(self.tmp.name)
        ctx = make_context(self.tmp.name)
        self.agent = PlannerAgent(cfg, EventBus(), Path(self.tmp.name), ctx)
        self.base_config = {"training": {"imgsz": 640, "batch": 16, "lr0": 0.01, "weight_decay": 0.0005}}
    def tearDown(self): self.tmp.cleanup()

    def test_small_object_proposal(self):
        report = AnalysisReport(run_id="r1", epoch=50, overall_map50=0.35,
                                weaknesses=[{"type": "small_object_weakness", "severity": "high",
                                             "message": "small objects weak"}])
        proposals = self.agent._plan_rule_based(report, self.base_config)
        self.assertGreater(len(proposals), 0)
        self.assertIn("imgsz", proposals[0].changes)
        self.assertGreater(proposals[0].changes["imgsz"], 640)

    def test_proposals_valid(self):
        report = AnalysisReport(run_id="r2", epoch=50, overall_map50=0.4,
                                weaknesses=[{"type": "weak_class", "severity": "high", "message": "x"}])
        proposals = self.agent._plan_rule_based(report, self.base_config)
        for p in proposals:
            self.assertIsInstance(p.changes, dict)
            self.assertGreater(len(p.rationale), 0)
            self.assertGreaterEqual(p.confidence, 0.0)
            self.assertLessEqual(p.confidence, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
