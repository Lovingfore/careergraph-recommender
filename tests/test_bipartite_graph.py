from pathlib import Path
import tempfile
import unittest

from src.bipartite_graph import build_bipartite_graph, write_bipartite_outputs


ROOT = Path(__file__).resolve().parents[1]


class BipartiteGraphTests(unittest.TestCase):
    def test_builds_two_disjoint_node_sets_and_weighted_edges(self):
        graph = build_bipartite_graph(ROOT / "data" / "clean")

        self.assertEqual(len(graph["occupation_nodes"]), 12)
        self.assertEqual(len(graph["skill_nodes"]), 35)
        self.assertEqual(len(graph["edges"]), 420)
        self.assertTrue(all(edge["source_type"] == "occupation" for edge in graph["edges"]))
        self.assertTrue(all(edge["target_type"] == "skill" for edge in graph["edges"]))
        self.assertTrue(all(0 <= edge["weight"] <= 1 for edge in graph["edges"]))
        self.assertEqual(
            {node["id"] for node in graph["occupation_nodes"]}
            & {node["id"] for node in graph["skill_nodes"]},
            set(),
        )

    def test_writes_json_edge_table_and_summary(self):
        graph = build_bipartite_graph(ROOT / "data" / "clean", min_demand_weight=0.45)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_bipartite_outputs(graph, Path(tmp), render=False)
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["edges_csv"].exists())
            self.assertTrue(paths["summary_json"].exists())
            self.assertEqual(graph["filtered_edge_count"], 24)

    def test_rendered_visual_is_viewable(self):
        graph = build_bipartite_graph(ROOT / "data" / "clean", min_demand_weight=0.30)
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_bipartite_outputs(graph, Path(tmp), render=True)
            self.assertTrue(paths["png"].suffix in {".png", ".svg"})
            self.assertGreater(paths["png"].stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
