"""Tests for the verified-API-surface reference (anti-hallucination).

Guards that the reference delivered to the agent (SKILL.md §7) stays honest:
every listed CLI command / Python symbol / module / facade method must actually
resolve, and the module names agents keep inventing (``harness.agents``,
``harness.task_config``) must remain absent so the "correct name" hints keep
their meaning.
"""
from __future__ import annotations

import importlib
import unittest

import harness.api_reference as api_ref


class ValidateReferenceTests(unittest.TestCase):
    def test_reference_is_fully_valid(self):
        problems = api_ref.validate_reference()
        self.assertEqual(
            problems, [],
            msg="stale reference entries:\n" + "\n".join(problems),
        )

    def test_all_cli_commands_registered(self):
        commands = api_ref._ai_harness_commands()
        for name, _ in api_ref.CLI_COMMANDS:
            self.assertIn(name, commands, "%r not a real ai-harness command" % name)

    def test_facade_methods_exist_on_client(self):
        client = api_ref._resolve("harness.kaggle_compat.KaggleClient")
        for method, _ in api_ref.FACADE_METHODS:
            self.assertTrue(hasattr(client, method), "KaggleClient.%s missing" % method)

    def test_key_modules_import(self):
        for dotted, _ in api_ref.KEY_MODULES:
            importlib.import_module(dotted)


class RenderReferenceTests(unittest.TestCase):
    def test_render_includes_verified_names(self):
        md = api_ref.render_reference()
        for needle in (
            "## 7. Verified API surface",
            "harness.kaggle_compat.KaggleClient",
            "harness.task_spec",
            "list_submissions",
            "submission",
            "dispatch-kernel",
            "ai-harness-agent",
        ):
            self.assertIn(needle, md)

    def test_render_warns_against_guessing(self):
        md = api_ref.render_reference()
        self.assertIn("do NOT invent", md)
        self.assertIn("never guess", md)


class SkillInjectionTests(unittest.TestCase):
    def test_skill_md_contains_reference(self):
        from harness.skill_injector import generate_skill_content

        md = generate_skill_content({"name": "T", "platform": "kaggle"})
        self.assertIn("## 7. Verified API surface", md)
        self.assertIn("harness.task_spec", md)


class HallucinatedNamesAbsentTests(unittest.TestCase):
    """Pin the motivation: the module names agents invent are NOT real."""

    def test_invented_module_names_do_not_exist(self):
        for bad in ("harness.agents", "harness.task_config"):
            with self.assertRaises(
                ModuleNotFoundError, msg="%s should not exist" % bad
            ):
                importlib.import_module(bad)


class SubmissionFieldsTests(unittest.TestCase):
    """The normalised Submission fields listed in §7 actually resolve.

    These pin the micro-gap fix: the agent used to guess ``.date`` /
    ``.submittedDate``; the reference now advertises the real canonical names and
    the test proves each one resolves through the normalising proxy.
    """

    def test_submission_fields_resolve_on_normalised_object(self):
        from harness.kaggle_compat import _NormalisedObject

        sub = _NormalisedObject(api_ref._submission_fixture(), "kaggle")
        for name, _ in api_ref.SUBMISSION_FIELDS:
            with self.subTest(field=name):
                getattr(sub, name)  # must not raise AttributeError

    def test_created_at_is_the_normalised_date_field(self):
        from harness.kaggle_compat import _NormalisedObject

        names = [n for n, _ in api_ref.SUBMISSION_FIELDS]
        self.assertIn("created_at", names)
        sub = _NormalisedObject(api_ref._submission_fixture(), "kaggle")
        self.assertEqual(sub.created_at, "2026-10-17T21:14:27Z")


class RenderSubmissionFieldTests(unittest.TestCase):
    def test_render_lists_submission_fields(self):
        md = api_ref.render_reference()
        self.assertIn("normalised fields", md)
        for name in ("ref", "status", "public_score", "created_at"):
            self.assertIn("`%s`" % name, md)


if __name__ == "__main__":
    unittest.main()
