from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    torch = None

from src.data_loader import build_graph_tensors


ROOT = Path(__file__).resolve().parents[1]


class LoaderTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch not installed in test environment")
    def test_dataloader_batch_has_expected_shapes(self):
        from src.data_loader import build_dataloader

        loader, meta = build_dataloader(ROOT / "data" / "clean", batch_size=2, sequence_length=3)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["x"].shape), (2, 3, meta["num_skills"]))
        self.assertEqual(tuple(batch["y"].shape), (2, meta["num_skills"]))
        self.assertEqual(len(batch["user_id"]), 2)

    @unittest.skipIf(torch is None, "PyTorch not installed in test environment")
    def test_graph_tensors_have_edges_and_stable_maps(self):
        graph = build_graph_tensors(ROOT / "data" / "clean")
        self.assertGreater(graph["occupation_skill_edge_index"].shape[1], 0)
        self.assertGreater(graph["transition_edge_index"].shape[1], 0)
        self.assertEqual(len(graph["skill_ids"]), 35)
        self.assertEqual(len(graph["occupation_ids"]), 12)

    @unittest.skipIf(torch is None, "PyTorch not installed in test environment")
    def test_temporal_gat_forward_returns_skill_scores(self):
        from src.data_loader import build_dataloader, build_graph_tensors
        from src.temporal_gat import TemporalGAT

        loader, meta = build_dataloader(ROOT / "data" / "clean", batch_size=2, sequence_length=3)
        graph = build_graph_tensors(ROOT / "data" / "clean")
        batch = next(iter(loader))
        model = TemporalGAT(num_skills=meta["num_skills"], hidden_dim=16, heads=2)
        output = model(batch["x"], graph["skill_edge_index"], graph["skill_edge_weight"])
        self.assertEqual(tuple(output.shape), (2, meta["num_skills"]))
        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
