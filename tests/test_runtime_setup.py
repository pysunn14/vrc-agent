import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from vrc_ardy_agent.runner.runtime_setup.paths import absolute_path, discover_python, installation_paths
from vrc_ardy_agent.runner.runtime_setup.jobs import SetupJobs
from vrc_ardy_agent.runner.runtime_setup.plan import make_plan


class RuntimePathsTests(unittest.TestCase):
    def test_relative_unicode_path_keeps_interpreter_symlink(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / '환경 폴더').mkdir()
            real = root / 'python-real'
            real.touch()
            link = root / '환경 폴더' / 'python'
            link.symlink_to(real)
            self.assertEqual(absolute_path('환경 폴더/python', root), link)
            self.assertNotEqual(absolute_path(str(link), root), real)

    def test_discovery_requires_unambiguous_environment(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            python = root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            python.parent.mkdir(parents=True)
            python.touch()
            self.assertEqual(discover_python(root), python)
            second = root / 'runtime' / python.relative_to(root / '.venv')
            second.parent.mkdir(parents=True)
            second.touch()
            with self.assertRaisesRegex(ValueError, 'multiple'):
                discover_python(root)

    def test_models_are_separate_from_environment(self):
        with TemporaryDirectory() as folder:
            paths = installation_paths(Path(folder), Path(folder) / 'model files')
            self.assertEqual(paths['models'], str(Path(folder) / 'model files'))
            self.assertNotEqual(Path(paths['models']).parent, Path(paths['environment']))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                installation_paths(Path(folder), Path(folder) / 'runtime' / 'models')


class RuntimeJobsTests(unittest.TestCase):
    def test_missing_or_reused_pid_is_interrupted_not_running(self):
        with TemporaryDirectory() as folder:
            jobs = SetupJobs(Path(folder))
            jobs.directory.mkdir(parents=True)
            path = jobs.directory / 'example.json'
            path.write_text(json.dumps({'id': 'example', 'state': 'running', 'pid': os.getpid(),
                                        'created': -1, 'request': {}, 'heartbeat_at': 99999999999}))
            self.assertEqual(jobs.status('example')['state'], 'interrupted')

    def test_install_plan_does_not_create_files_or_accept_foreign_environment(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / 'new root'
            request = {'mode': 'install', 'root': str(root), 'base_dir': folder}
            with patch('vrc_ardy_agent.runner.runtime_setup.plan.host_info', return_value={
                'os': 'macos', 'arch': 'arm64', 'device': 'mps', 'tools': {}, 'memory_bytes': 64*1024**3}):
                plan = make_plan(request)
            self.assertFalse(root.exists())
            self.assertEqual(plan['device'], 'mps')
            self.assertTrue(plan['recipe'])
            self.assertGreater(plan['model_bytes'], 0)
            (root / 'runtime').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'owned'):
                make_plan(request)


if __name__ == '__main__': unittest.main()

class RuntimeIntegrityTests(unittest.TestCase):
    def test_cache_copy_fallback_is_verified_and_corruption_is_detected(self):
        import errno
        import hashlib
        from vrc_ardy_agent.runner.runtime_setup.download import reuse_file, valid_file
        with TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'cache'; target = root / 'models' / 'weights'
            source.write_bytes(b'valid model')
            spec = {'bytes': 11, 'sha256': hashlib.sha256(b'valid model').hexdigest()}
            with patch('os.link', side_effect=OSError(errno.EXDEV, 'different disks')):
                reuse_file(source, target)
            self.assertTrue(valid_file(target, spec))
            target.write_bytes(b'wrong model')
            self.assertFalse(valid_file(target, spec))
            self.assertEqual(source.read_bytes(), b'valid model')

    def test_readiness_becomes_stale_when_interpreter_or_source_changes(self):
        from vrc_ardy_agent.runner.runtime_setup.verify import evidence_current, fingerprint
        with TemporaryDirectory() as folder:
            source = Path(folder) / 'source.py'; source.write_text('a = 1')
            result = {'evidence': [fingerprint(source)]}
            self.assertTrue(evidence_current(result))
            source.write_text('a = 234')
            self.assertFalse(evidence_current(result))
            self.assertFalse(evidence_current({}))

    def test_runtime_environment_has_no_network_or_text_encoder_fallback(self):
        from vrc_ardy_agent.runner.runtime_setup.verify import runtime_environment
        with patch.dict(os.environ, {'TEXT_ENCODERS_DIR': '/unrelated', 'TEXT_ENCODER_MODE': 'auto', 'TEXT_ENCODER_DEVICE': 'cpu'}):
            env = runtime_environment({'hf_cache_dir': '/selected/cache'})
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertEqual(env['TRANSFORMERS_OFFLINE'], '1')
        self.assertEqual(env['TEXT_ENCODER_MODE'], 'local')
        self.assertEqual(env['HF_HUB_CACHE'], '/selected/cache')
        self.assertNotIn('TEXT_ENCODERS_DIR', env)
        self.assertNotIn('TEXT_ENCODER_DEVICE', env)

    def test_nonfinite_generated_coordinates_fail_verification(self):
        import numpy as np
        from vrc_ardy_agent.ardy_runtime import ArdyMotionChunk
        from vrc_ardy_agent.runner.runtime_setup.verify import check_chunk
        chunk = ArdyMotionChunk(np.zeros((2, 27, 3)), np.zeros((2, 27, 3, 3)),
            np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 2)), np.zeros((2, 4)), 20., 'sample', .1)
        self.assertEqual(check_chunk(chunk, 27)['frames'], 2)
        chunk.posed_joints[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'non-finite'): check_chunk(chunk, 27)

    def test_model_download_pins_revisions_and_reuses_cache_without_network(self):
        import hashlib
        from vrc_ardy_agent.runner.runtime_setup.download import prepare_models
        from types import ModuleType
        with TemporaryDirectory() as folder:
            root = Path(folder); cached = root / 'cached'; cached.write_bytes(b'abc')
            hub = ModuleType('huggingface_hub'); calls = []
            def download(repo, filename, **kwargs):
                calls.append(kwargs)
                self.assertTrue(kwargs['local_files_only'])
                return str(cached)
            hub.hf_hub_download = download
            constants = ModuleType('huggingface_hub.constants'); constants.HF_HUB_CACHE = str(root / 'cache')
            errors = ModuleType('huggingface_hub.errors')
            errors.LocalEntryNotFoundError = type('LocalEntryNotFoundError', (Exception,), {})
            errors.GatedRepoError = type('GatedRepoError', (Exception,), {})
            manifest = [{'repo': 'example/model', 'revision': 'fixed-revision', 'files': [
                {'name': 'weights', 'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}]}]
            with patch.dict('sys.modules', {'huggingface_hub': hub, 'huggingface_hub.constants': constants, 'huggingface_hub.errors': errors}), \
                 patch('vrc_ardy_agent.runner.runtime_setup.download.models', return_value=manifest):
                plan = {'hf_cache_dir': str(root / 'selected'), 'checkpoints_dir': str(root / 'checkpoints')}
                self.assertEqual(prepare_models(plan)['bytes'], 3)
                self.assertEqual(prepare_models(plan)['bytes'], 3)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['revision'], 'fixed-revision')
            self.assertEqual((root / 'selected/models--example--model/refs/main').read_text(), 'fixed-revision')

    def test_unauthorized_model_download_is_actionable_and_does_not_set_ready_refs(self):
        from types import ModuleType
        from vrc_ardy_agent.runner.runtime_setup.download import prepare_models
        with TemporaryDirectory() as folder:
            root = Path(folder)
            hub = ModuleType('huggingface_hub')
            errors = ModuleType('huggingface_hub.errors')
            errors.LocalEntryNotFoundError = type('LocalEntryNotFoundError', (Exception,), {})
            errors.GatedRepoError = type('GatedRepoError', (Exception,), {})
            def download(*args, **kwargs):
                if kwargs.get('local_files_only'): raise errors.LocalEntryNotFoundError()
                raise errors.GatedRepoError()
            hub.hf_hub_download = download
            constants = ModuleType('huggingface_hub.constants'); constants.HF_HUB_CACHE = str(root / 'cache')
            manifest = [{'repo': 'example/model', 'revision': 'fixed', 'files': [{'name': 'weights', 'bytes': 1, 'sha256': 'x'}]}]
            with patch.dict('sys.modules', {'huggingface_hub': hub, 'huggingface_hub.constants': constants, 'huggingface_hub.errors': errors}), \
                 patch('vrc_ardy_agent.runner.runtime_setup.download.models', return_value=manifest), \
                 self.assertRaisesRegex(RuntimeError, 'authentication required'):
                prepare_models({'hf_cache_dir': str(root / 'models')})
            self.assertFalse((root / 'models/models--example--model/refs/main').exists())
