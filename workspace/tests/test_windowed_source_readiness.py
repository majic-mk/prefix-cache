import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from probekv.cacheblend_v6_online_engine import LayerwiseLoadTicket
from probekv.v8_schema10_staging import PhysicalLayerwiseSourceLoader


class Event:
    def __init__(self, ready=True):
        self.ready = ready

    def query(self):
        return self.ready

    def synchronize(self):
        self.ready = True


def ticket(**changes):
    row = dict(segment_id="C", source_id="S", started_host_ms=0,
               requested_bytes=8, layer_tensors={1: ("k1", "v1")},
               start_event=Event(), layer_events={1: Event()},
               pending_layers={2: ("k2", "v2")},
               source_digest_before="", source_digest_after="",
               segment_positions=(2, 3), integrity_mode="online_immutable",
               expected_artifact_digest="digest", expected_layer_count=2)
    row.update(changes)
    return LayerwiseLoadTicket(**row)


class WindowedSourceReadinessTests(unittest.TestCase):
    def resident_loader(self, mode):
        from contextlib import nullcontext
        loader = object.__new__(PhysicalLayerwiseSourceLoader)
        loader.integrity_mode = mode
        loader.stream, loader.authorize = Mock(), Mock()
        loader.events = []
        loader.torch = SimpleNamespace(cuda=SimpleNamespace(
            Event=lambda **kw: Mock(query=lambda: True), stream=lambda s: nullcontext(),
            current_stream=lambda: 'consumer-stream'))
        tensor = SimpleNamespace(device=SimpleNamespace(type='cuda'),numel=lambda:4,element_size=lambda:2)
        pairs = [(tensor,tensor),(tensor,tensor)]
        args = dict(segment_id='C',source_id='S',canonical_layers=pairs,
            segment_positions=(0,),expected_artifact_digest='digest',
            resident_layers={1:pairs[0],2:pairs[1]})
        return loader,args

    def test_resident_qualification_fences_then_verifies_all_three_digests(self):
        loader,args=self.resident_loader('qualification_full')
        calls=[]
        def digest(tensors):
            list(tensors)
            calls.append(loader.stream.synchronize.call_count)
            return 'digest'
        with patch('probekv.v8_schema10_staging.tensor_digest',side_effect=digest):
            t=loader.begin(**args)
        self.assertEqual(calls,[0,1,1])
        self.assertEqual((t.source_digest_before,t.destination_digest,t.source_digest_after),('digest',)*3)
        self.assertTrue(t.per_request_full_digest_verified)
        self.assertTrue(t.fully_ready())
        loader.authorize.assert_called_once_with(source_id='S',segment_id='C',bytes_required=32)
        loader.stream.wait_stream.assert_called_once_with('consumer-stream')

    def test_resident_qualification_rejects_corrupt_destination_or_changed_source(self):
        for digests in (['digest','corrupt','digest'],['digest','digest','changed']):
            loader,args=self.resident_loader('qualification_full')
            with patch('probekv.v8_schema10_staging.tensor_digest',side_effect=digests):
                with self.assertRaisesRegex(RuntimeError,'resident Source/destination'):
                    loader.begin(**args)

    def test_resident_online_immutable_does_not_hash_or_synchronize(self):
        loader,args=self.resident_loader('online_immutable')
        with patch('probekv.v8_schema10_staging.tensor_digest',side_effect=AssertionError('online full hash forbidden')):
            t=loader.begin(**args)
        self.assertFalse(t.per_request_full_digest_verified)
        loader.stream.synchronize.assert_not_called()
        loader.authorize.assert_called_once()

    def test_resident_qualification_still_rejects_missing_authority_before_hash(self):
        loader,args=self.resident_loader('qualification_full')
        loader.authorize.side_effect=RuntimeError('no live lease')
        with patch('probekv.v8_schema10_staging.tensor_digest') as digest:
            with self.assertRaisesRegex(RuntimeError,'no live lease'):loader.begin(**args)
        digest.assert_not_called()

    def test_submitted_ready_is_not_full_ready(self):
        t = ticket()
        self.assertTrue(t.layer_ready(1))
        self.assertFalse(t.fully_ready())

    def test_wait_all_enqueues_pending_not_only_waits_existing_events(self):
        t = ticket()
        def submit(observed, end):
            self.assertIs(t, observed)
            self.assertEqual(end, 2)
            t.layer_tensors[2] = t.pending_layers.pop(2)
            t.layer_events[2] = Event(False)
        t.wait_all(SimpleNamespace(prefetch_pending=submit))
        self.assertTrue(t.fully_ready())

    def test_wait_all_rejects_missing_submission(self):
        with self.assertRaises(RuntimeError):
            ticket().wait_all(SimpleNamespace(prefetch_pending=lambda *_: None))

    def test_inventory_must_not_have_gaps_or_duplicates(self):
        for pending in ({3: ("k", "v")}, {1: ("k", "v")}):
            with self.assertRaises(ValueError):
                ticket(pending_layers=pending)

    def test_qualification_stays_unverified_until_all_layers_and_three_digests(self):
        t = ticket(integrity_mode="qualification_full", source_digest_before="digest",
                   integrity_verification_pending=True)
        with self.assertRaises(RuntimeError):
            t.finalize_integrity([], lambda _: "digest")
        t.layer_tensors[2] = t.pending_layers.pop(2)
        t.layer_events[2] = Event()
        self.assertFalse(t.per_request_full_digest_verified)
        digest = Mock(return_value="digest")
        t.finalize_integrity([("k1", "v1"), ("k2", "v2")], digest)
        self.assertEqual(digest.call_count, 2)
        self.assertTrue(t.per_request_full_digest_verified)
        self.assertFalse(t.integrity_verification_pending)

    def test_corrupted_destination_fails_and_cannot_be_ready(self):
        t = ticket(integrity_mode="qualification_full", source_digest_before="digest",
                   integrity_verification_pending=True)
        t.layer_tensors[2] = t.pending_layers.pop(2)
        t.layer_events[2] = Event()
        with self.assertRaises(RuntimeError):
            t.finalize_integrity([], Mock(side_effect=["corrupt", "digest"]))
        self.assertFalse(t.fully_ready())

    def test_online_immutable_never_calls_full_digest(self):
        t = ticket()
        digest = Mock(side_effect=AssertionError("online hashing forbidden"))
        t.finalize_integrity([], digest)
        digest.assert_not_called()

    def test_bad_pending_tensor_is_not_lost_on_validation_failure(self):
        from contextlib import nullcontext
        t = ticket()
        bad = SimpleNamespace(device=SimpleNamespace(type="cpu"), is_pinned=lambda: False)
        t.pending_layers[2] = (bad, bad)
        loader = object.__new__(PhysicalLayerwiseSourceLoader)
        loader.stream = Mock()
        loader.torch = SimpleNamespace(cuda=SimpleNamespace(stream=lambda _: nullcontext()))
        with self.assertRaises(ValueError):
            loader.prefetch_pending(t, 2)
        self.assertIn(2, t.pending_layers)
        self.assertNotIn(2, t.layer_events)


if __name__ == "__main__":
    unittest.main()
