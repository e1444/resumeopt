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
    validate_pdf,
    write_tex_from_template,
)


class FakeGroupingLLMProvider:
    def call(self, *args, **kwargs):  # pragma: no cover - not used in tests
        raise NotImplementedError

    def call_json(self, prompt: str, **kwargs):
        if "resume-appropriate section names" in prompt:
            return {
                "sections": [
                    {"name": "Languages", "skills": ["python"]},
                    {"name": "ML & Data", "skills": ["pytorch", "jupyter"]},
                    {"name": "Tools", "skills": ["git"]},
                ]
            }
        return {}


class FakePrunedGroupingLLMProvider:
    def call(self, *args, **kwargs):  # pragma: no cover - not used in tests
        raise NotImplementedError

    def call_json(self, prompt: str, **kwargs):
        if "resume-appropriate section names" in prompt:
            return {
                "sections": [
                    {"name": "Languages", "skills": ["python"]},
                    {"name": "Tools", "skills": ["git", "docker"]},
                ]
            }
        return {}


class RenderResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.template_path = self.repo_root / "data" / "template.tex"

    def test_llm_grouping_into_dynamic_sections(self) -> None:
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

    def test_llm_can_propose_arbitrary_posting_specific_section_names(self) -> None:
        class FakeCustomNamesProvider:
            def call(self, *args, **kwargs):  # pragma: no cover - not used
                raise NotImplementedError

            def call_json(self, prompt: str, **kwargs):
                if "resume-appropriate section names" in prompt:
                    return {
                        "sections": [
                            {"name": "Security", "skills": ["nmap", "wireshark"]},
                            {"name": "Cloud & DevOps", "skills": ["kubernetes"]},
                        ]
                    }
                return {}

        grouped = build_sectioned_skills(
            canonical_skills=["nmap", "wireshark", "kubernetes"],
            llm_provider=FakeCustomNamesProvider(),
        )

        self.assertEqual(grouped["Security"], ["nmap", "wireshark"])
        self.assertEqual(grouped["Cloud & DevOps"], ["kubernetes"])

    def test_more_than_max_sections_are_merged_down(self) -> None:
        class FakeTooManySectionsProvider:
            def call(self, *args, **kwargs):  # pragma: no cover - not used
                raise NotImplementedError

            def call_json(self, prompt: str, **kwargs):
                if "resume-appropriate section names" in prompt:
                    return {
                        "sections": [
                            {"name": "A", "skills": ["python", "java", "c#"]},
                            {"name": "B", "skills": ["git"]},
                            {"name": "C", "skills": ["docker"]},
                            {"name": "D", "skills": ["kubernetes"]},
                            {"name": "E", "skills": ["nmap"]},
                        ]
                    }
                return {}

        grouped = build_sectioned_skills(
            canonical_skills=["python", "java", "c#", "git", "docker", "kubernetes", "nmap"],
            llm_provider=FakeTooManySectionsProvider(),
        )

        self.assertLessEqual(len(grouped), 4)
        all_skills = {skill for skills in grouped.values() for skill in skills}
        self.assertEqual(
            all_skills, {"python", "java", "c#", "git", "docker", "kubernetes", "nmap"}
        )

    def test_skill_omitted_by_llm_is_still_covered(self) -> None:
        class FakeIncompleteProvider:
            def call(self, *args, **kwargs):  # pragma: no cover - not used
                raise NotImplementedError

            def call_json(self, prompt: str, **kwargs):
                if "resume-appropriate section names" in prompt:
                    return {"sections": [{"name": "Languages", "skills": ["python"]}]}
                return {}

        grouped = build_sectioned_skills(
            canonical_skills=["python", "git"],
            llm_provider=FakeIncompleteProvider(),
        )

        all_skills = {skill for skills in grouped.values() for skill in skills}
        self.assertIn("git", all_skills)

    def test_posting_context_is_included_in_the_prompt(self) -> None:
        seen_prompts = []

        class FakeCapturingProvider:
            def call(self, *args, **kwargs):  # pragma: no cover - not used
                raise NotImplementedError

            def call_json(self, prompt: str, **kwargs):
                seen_prompts.append(prompt)
                return {"sections": [{"name": "Languages", "skills": ["python"]}]}

        build_sectioned_skills(
            canonical_skills=["python"],
            llm_provider=FakeCapturingProvider(),
            posting_context="Role: Security Engineer\nDomain: cybersecurity",
        )

        self.assertTrue(any("Security Engineer" in prompt for prompt in seen_prompts))


    def test_render_skills_lines_has_no_trailing_linebreak_on_last_line(self) -> None:
        grouped = {
            "Languages": ["python"],
            "Tools": ["git"],
        }

        rendered = render_skills_lines(grouped)

        self.assertIn("\\\\\n", rendered)
        self.assertFalse(rendered.endswith("\\\\"))

    def test_render_skills_lines_does_not_alter_casing(self) -> None:
        """Capitalization is owned entirely by the cache-write layer now
        (`webapp.skills_cache_io.add_skill` / `main.py`'s missing-skill
        collection, both via `render_resume.capitalize_skill_name`) - by the
        time a name reaches `render_skills_lines`, it's already in its final
        display form, so rendering must be a pure pass-through: whatever
        casing `sectioned_skills` holds is rendered verbatim, not
        re-capitalized (previously this function re-applied
        `capitalize_skill_name` defensively, which is now redundant and
        removed per explicit design decision)."""

        grouped = {
            "Languages": ["python", "Java"],
            "ML & Data": ["machine learning"],
        }

        rendered = render_skills_lines(grouped)

        self.assertIn("python", rendered)
        self.assertIn("Java", rendered)
        self.assertIn("machine learning", rendered)
        self.assertNotIn("Python", rendered)
        self.assertNotIn("Machine", rendered)

    def test_render_skills_lines_preserves_acronyms_and_stylized_names(self) -> None:
        """Regression test for the "Sql" display bug: canonical names that
        already carry internal capitalization (acronyms, stylized brand
        names) must be preserved verbatim, not naively re-capitalized."""

        grouped = {
            "Languages": ["SQL", "PostgreSQL", "JavaScript"],
        }

        rendered = render_skills_lines(grouped)

        self.assertIn("SQL", rendered)
        self.assertIn("PostgreSQL", rendered)
        self.assertIn("JavaScript", rendered)
        self.assertNotIn("Sql", rendered)
        self.assertNotIn("Postgresql", rendered)

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

    def test_validate_pdf_passes_for_one_page_short_skills_section(self) -> None:
        grouped = {
            "Languages": ["python"],
            "Tools": ["git"],
        }
        skills_block = render_skills_lines(grouped)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_tex = tmp_path / "tailored_resume.tex"
            output_pdf = tmp_path / "tailored_resume.pdf"

            write_tex_from_template(self.template_path, output_tex, skills_block)
            render_pdf_with_pdflatex(output_tex, output_pdf)

            report = validate_pdf(output_pdf)

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["issues"], [])
            self.assertEqual(report["page_count"], 1)
            self.assertEqual(report["skills_section_line_count"], 2)

    def test_validate_pdf_flags_skills_section_over_line_limit(self) -> None:
        grouped = {
            "Languages": ["python"],
            "Tools": ["git"],
        }
        skills_block = render_skills_lines(grouped)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_tex = tmp_path / "tailored_resume.tex"
            output_pdf = tmp_path / "tailored_resume.pdf"

            write_tex_from_template(self.template_path, output_tex, skills_block)
            render_pdf_with_pdflatex(output_tex, output_pdf)

            report = validate_pdf(output_pdf, max_skills_section_lines=1)

            self.assertEqual(report["status"], "fail")
            issue_types = {issue["type"] for issue in report["issues"]}
            self.assertIn("skills_section_too_long", issue_types)

    def test_validate_pdf_fails_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_pdf = Path(tmp_dir) / "does_not_exist.pdf"

            report = validate_pdf(missing_pdf)

            self.assertEqual(report["status"], "fail")
            issue_types = {issue["type"] for issue in report["issues"]}
            self.assertIn("missing_pdf", issue_types)

    def test_validate_pdf_fails_for_page_count_exceeded(self) -> None:
        grouped = {
            "Languages": ["python"],
            "Tools": ["git"],
        }
        skills_block = render_skills_lines(grouped)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_tex = tmp_path / "tailored_resume.tex"
            output_pdf = tmp_path / "tailored_resume.pdf"

            write_tex_from_template(self.template_path, output_tex, skills_block)
            render_pdf_with_pdflatex(output_tex, output_pdf)

            report = validate_pdf(output_pdf, max_pages=0)

            self.assertEqual(report["status"], "fail")
            issue_types = {issue["type"] for issue in report["issues"]}
            self.assertIn("page_count_exceeded", issue_types)


if __name__ == "__main__":
    unittest.main()
