import json
from pathlib import Path
import tempfile
import unittest

from vrc_ardy_agent.observation_benchmark import balanced_schedule, load_cases, run_benchmark
from vrc_ardy_agent.benchmark_report import summarize

ROOT = Path(__file__).resolve().parents[1]

class RunnerTests(unittest.TestCase):
    def test_schedule_pairs_are_balanced_and_reproducible(self):
        cases = load_cases(ROOT/'benchmarks/observation/cases.json')['cases']
        order = balanced_schedule(cases, repeats=3, seed=91)
        self.assertEqual(len(order), 120)
        self.assertEqual(order, balanced_schedule(cases, repeats=3, seed=91))
        self.assertEqual(sum(order[i]['policy']=='always' for i in range(0,120,2)),30)
        for i in range(0,120,2):
            self.assertEqual(order[i]['case_id'],order[i+1]['case_id'])
            self.assertNotEqual(order[i]['policy'],order[i+1]['policy'])

    def test_dry_run_and_resume_do_not_duplicate_turns(self):
        with tempfile.TemporaryDirectory() as folder:
            kwargs=dict(cases_path=ROOT/'benchmarks/observation/cases.json', output=Path(folder),
                        repeats=1, seed=91, real_api=False, limit=4)
            run_benchmark(**kwargs)
            initial=(Path(folder)/'events.jsonl').read_text()
            run_benchmark(**kwargs)
            rows=[json.loads(x) for x in initial.splitlines()]
            self.assertEqual(sum(x['event']=='turn.finished' for x in rows),4)
            self.assertEqual(initial,(Path(folder)/'events.jsonl').read_text())
            summary=summarize(Path(folder))
            self.assertEqual(summary['completed_turns'],4)
            self.assertEqual(summary['mode'],'dry_run')
            self.assertEqual(summary['human_graded'],0)

    def test_real_requires_budget_and_human_rubric(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'approval|rubric|budget'):
                run_benchmark(cases_path=ROOT/'benchmarks/observation/cases.json', output=Path(folder),
                              repeats=3, seed=91, real_api=True)
            self.assertFalse((Path(folder)/'events.jsonl').exists())

    def test_tampered_fixture_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source=ROOT/'benchmarks/observation/cases.json'
            import shutil
            shutil.copytree(source.parent,Path(folder)/'fixtures')
            p=Path(folder)/'fixtures/images/mixed.jpg';p.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'hash'):
                load_cases(Path(folder)/'fixtures/cases.json')

class ReportEvidenceTests(unittest.TestCase):
    def test_late_completion_and_failure_are_not_lost_or_double_counted(self):
        from vrc_ardy_agent.benchmark_store import atomic_json
        from vrc_ardy_agent.benchmark_report import turn_records
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            atomic_json(root/'manifest.json', {'run_id':'r','mode':'real_api','schedule':[],'cases':[]})
            common=dict(schedule_index=0,case_id='x',repeat=0,policy='selective',category='visual',warmup=False,turn_id='t')
            rows=[dict(common,event='turn.input'),
                  dict(common,event='call.requested',call_id='c',kind='model',status='requested',started_ns=None),
                  dict(common,event='call.started',call_id='c',kind='model',status='running',started_ns=1),
                  dict(common,event='call.wait_ended',call_id='c',wait_status='timeout',status='running',started_ns=1),
                  dict(common,event='turn.finished',status='FAILED',plan_ms=None,elapsed_ms=30),
                  dict(common,event='call.finished',call_id='c',status='completed',late=True,usage={'prompt_tokens':2})]
            (root/'events.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            _, turns=turn_records(root)
            self.assertEqual(turns[0]['model_calls'],1)
            self.assertEqual(turns[0]['pending_calls'],0)
            self.assertIsNone(turns[0]['plan_ms'])
            self.assertEqual(turns[0]['calls'][0]['usage'],{'prompt_tokens':2})
            self.assertTrue(turns[0]['calls'][0]['late'])

    def test_writer_ownership_and_corrupt_log_are_explicit(self):
        from vrc_ardy_agent.benchmark_store import EventStore, read_events
        with tempfile.TemporaryDirectory() as folder:
            store=EventStore(folder)
            try:
                with self.assertRaisesRegex(RuntimeError,'owns'):
                    EventStore(folder)
            finally:store.close()
            (Path(folder)/'events.jsonl').write_text('{bad')
            with self.assertRaisesRegex(ValueError,'corrupt'):
                read_events(folder)
