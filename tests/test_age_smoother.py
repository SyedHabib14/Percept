from pdc.age_smoother import AgeSmoother


def test_buffer_fill_eviction_and_raw_smoothed_values():
    smoother = AgeSmoother(window=5)
    values = []
    for index in range(6):
        track, value = smoother.update((0, 0, 10, 10), float(index), True, True, float(index) / 10)
        values.append(value)
    assert track == 1
    assert values[-1] == 3.0  # average of 1,2,3,4,5 after eviction


def test_quality_gate_rejects_missing_or_degenerate_face():
    smoother = AgeSmoother()
    _, accepted = smoother.update((0, 0, 10, 10), 30, True, True, 0)
    _, rejected = smoother.update((0, 0, 10, 10), 90, True, False, .1)
    assert accepted == rejected == 30


def test_tracks_expire_after_timeout():
    smoother = AgeSmoother(max_age_seconds=1.0)
    first, _ = smoother.update((0, 0, 10, 10), 30, True, True, 0)
    second, _ = smoother.update((0, 0, 10, 10), 31, True, True, 1.1)
    assert (first, second) == (1, 2)
