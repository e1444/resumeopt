import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from render_resume import (
    build_sectioned_skills,
    inject_skills_into_template,
    render_pdf_with_pdflatex,
    render_skills_lines,
    write_tex_from_template,
)


class FakeGroupingLLMProvider:
    def call(self, *args, **kwargs):  # pragma: no cover - not used in tests
        raise NotImplementedError

    def call_json(self, prompt: str, **kwargs):
        if "Group the following canonical skills" in prompt:
            return {
                "active_sections": ["Languages", "ML & Data", "Tools"],
                "grouped_skills": {
                    "Languages": ["python"],
                    "ML & Data": ["pytorch", "jupyter"],
                    "Tools": ["git"],
                },
            }
        return {}


class FakePrunedGroupingLLMProvider:
    def call(self, *args, **kwargs):  # pragma: no cover - not used in tests
        raise NotImplementedError

    def call_json(self, prompt: str, **kwargs):
        if "Group the following canonical skills" in prompt:
            return {
                "active_sections": ["Languages", "Tools"],
                "grouped_skills": {
                    "Languages": ["python"],
                    "Tools": ["git", "docker"],
                },
            }
        return {}


class RenderResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.template_path = self.repo_root / "data" / "template.tex"

    def test_llm_grouping_into_three_sections(self) -> None:
        grouped = build_sectioned_skills(
            canonical_skills=["python", "pytorch", "jupyter", "git"],
            llm_provider=FakeGroupingLLMProvider(),
        )

        self.assertEqual(grouped["Languages"], ["python"])
        self.assertEqual(grouped["ML & Data"], ["pytorch", "jupyter"])
        self.assertEqual(grouped["Tools"], ["git"])

    def test_template_injection_replaces_marker(self) -> None:
        sample_template = "before\n[INSERT SKILLS HERE]\nafter\n"
        skills_block = "\\textbf{Languages}: Python\\\\"

        rendered = inject_skills_into_template(sample_template, skills_block)

        self.assertNotIn("[INSERT SKILLS HERE]", rendered)
        self.assertIn(skills_block, rendered)

    def test_llm_can_omit_irrelevant_section(self) -> None:
        grouped = build_sectioned_skills(
            canonical_skills=["python", "git", "docker"],
            llm_provider=FakePrunedGroupingLLMProvider(),
        )

        self.assertIn("Languages", grouped)
        self.assertIn("Tools", grouped)
        self.assertNotIn("ML & Data", grouped)

        rendered = render_skills_lines(grouped)
        self.assertIn("\\textbf{Languages}", rendered)
        self.assertIn("\\textbf{Tools}", rendered)
        self.assertNotIn("ML \\& Data", rendered)

    def test_render_skills_lines_has_no_trailing_linebreak_on_last_line(self) -> None:
        grouped = {
            "Languages": ["python"],
            "Tools": ["git"],
        }

        rendered = render_skills_lines(grouped)

        self.assertIn("\\\\\n", rendered)
        self.assertFalse(rendered.endswith("\\\\"))

    def test_render_skills_lines_capitalizes_skill_display(self) -> None:
        grouped = {
            "Languages": ["python"],
            "ML & Data": ["machine learning"],
        }

        rendered = render_skills_lines(grouped)

        self.assertIn("Python", rendered)
        self.assertIn("Machine Learning", rendered)

    def test_write_tex_from_template_and_render_pdf(self) -> None:
        grouped = {
            "Languages": ["python"],
            "ML & Data": ["pytorch", "jupyter"],
            "Tools": ["git"],
        }
        skills_block = render_skills_lines(grouped)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_tex = tmp_path / "tailored_resume.tex"
            output_pdf = tmp_path / "tailored_resume.pdf"

            write_tex_from_template(self.template_path, output_tex, skills_block)
            self.assertTrue(output_tex.exists())

            rendered_pdf = render_pdf_with_pdflatex(output_tex, output_pdf)
            self.assertTrue(rendered_pdf.exists())
            self.assertGreater(rendered_pdf.stat().st_size, 0)

    def test_render_pdf_writes_pdflatex_logs(self) -> None:
        grouped = {
            "Languages": ["python"],
            "Tools": ["git"],
        }
        skills_block = render_skills_lines(grouped)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_tex = tmp_path / "tailored_resume.tex"
            output_pdf = tmp_path / "tailored_resume.pdf"
            logs_dir = tmp_path / "logs"

            write_tex_from_template(self.template_path, output_tex, skills_block)
            render_pdf_with_pdflatex(output_tex, output_pdf, logs_dir=logs_dir)

            self.assertTrue((logs_dir / "pdflatex.command.log").exists())
            self.assertTrue((logs_dir / "pdflatex.stdout.log").exists())
            self.assertTrue((logs_dir / "pdflatex.stderr.log").exists())
            self.assertTrue((logs_dir / "pdflatex.engine.log").exists())


if __name__ == "__main__":
    unittest.main()
