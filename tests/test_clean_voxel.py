import unittest
from interpolate_cellwise import clean_voxel, make_layers, MAX_LAYERS, MAX_EVENTS, DWELL_DT

class CleanVoxelTests(unittest.TestCase):
    def test_layer_limit(self):
        events = [
            {'tm': 18.0},
            {'tm': 27.0},
            {'tm': 36.0},
            {'tm': 45.0},
            {'tm': 55.0},
        ]
        cleaned = clean_voxel(events.copy())
        times = [e['tm'] for e in cleaned]
        self.assertEqual(times, [18.0, 27.0, 36.0])

    def test_event_limit_per_layer(self):
        # two layers with three events each
        events = [
            {'tm': 1.0},
            {'tm': 1.5},
            {'tm': 2.0},
            {'tm': 10.0},
            {'tm': 10.5},
            {'tm': 11.0},
        ]
        cleaned = clean_voxel(events.copy())
        times = [e['tm'] for e in cleaned]
        # each layer should only keep earliest MAX_EVENTS=2 hits
        expected = [1.0, 1.5, 10.0, 10.5]
        self.assertEqual(times, expected)

if __name__ == '__main__':
    unittest.main()
