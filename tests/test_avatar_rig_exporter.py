"""Static checks for the Unity-to-Python hand landmark export contract.

These inspect the export declaration; they do not replace Unity compilation or
an export from a loaded Humanoid avatar.
"""
from pathlib import Path
import re
import unittest

from tests.test_avatar_rig_profile import _bone, _profile_document
from vrc_ardy_agent.avatar_rig_profile import AvatarRigProfile


class AvatarRigExporterTests(unittest.TestCase):
    def test_exports_bilateral_hand_landmarks_with_correct_humanoid_mapping(self):
        source = (Path(__file__).parents[1] / 'native/unity_avatar_rig_exporter/Editor/AvatarRigProfileExporter.cs').read_text()
        entries = re.findall(r'new BoneSpec\("([^"]+)", HumanBodyBones\.(\w+), (true|false)\)', source)
        self.assertEqual(len(entries), len({name for name, _, _ in entries}))
        mappings = {name: (human, required) for name, human, required in entries}
        for side in ('left', 'right'):
            for finger in ('Thumb', 'Index', 'Middle', 'Little'):
                name = f'{side}{finger}Proximal'
                with self.subTest(bone=name):
                    self.assertEqual(mappings.get(name), (f'{side.title()}{finger}Proximal', 'false'))

    def test_profile_loader_preserves_exported_finger_landmarks(self):
        document = _profile_document()
        for side, sign in (('left', -1), ('right', 1)):
            for finger, z in (('Thumb', .04), ('Index', .02), ('Middle', 0), ('Little', -.02)):
                document['bones'].append(_bone(f'{side}{finger}Proximal', (sign * .45, -.10, z)))
        profile = AvatarRigProfile.from_mapping(document)
        for side in ('left', 'right'):
            for finger in ('Thumb', 'Index', 'Middle', 'Little'):
                self.assertEqual(len(profile.bone(f'{side}{finger}Proximal').position_m), 3)
