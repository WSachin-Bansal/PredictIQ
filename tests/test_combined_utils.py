from predictiq.training.combined_utils import composite_score, remap_state, source_schedule


def test_schedule_hits_requested_number_of_batches():
    s = source_schedule(330, 15, 1)            # 15 image batches spread over 330 steps
    assert len(s) == 330 and sum(s) == 15
    assert sum(source_schedule(330, 40, 2)) == 80
    assert sum(source_schedule(10, 40, 2)) == 10   # never more than one per step


def test_schedule_spread_evenly():
    idx = [i for i, x in enumerate(source_schedule(300, 10, 1)) if x]
    gaps = {b - a for a, b in zip(idx, idx[1:])}
    assert gaps <= {30}


def test_composite_orders_sensibly():
    assert composite_score(0.9, 0.9, 10) > composite_score(0.9, 0.9, 20)
    assert composite_score(0.99, 0.5, 15) < composite_score(0.99, 0.9, 15)
    assert 0 <= composite_score(0, 0, 999) <= 3


def test_remap_state():
    st = {"enc.sensor.a": 1, "enc.audio.b": 2, "fusion.0.weight": 3, "fault_head.bias": 4}
    out = remap_state(st, [("enc.sensor.", "enc.vib."), ("enc.audio.", "enc.audio.")])
    assert out == {"enc.vib.a": 1, "enc.audio.b": 2}
    assert remap_state({"0.lstm.w": 1, "1.weight": 2}, [("0.", "enc.engine.")]) == {"enc.engine.lstm.w": 1}
