"""Tests for the distribution content generators (X threads, Substack, LinkedIn)."""

from pathlib import Path

import pytest

from vidgen.distribution import (
    generate_linkedin_post,
    generate_newsletter,
    generate_thread,
    _clean_narration,
    _extract_hook_line,
    _extract_section_insight,
    _split_sentences,
    _trim_to_limit,
)
from vidgen.models import Script, ScriptSection


@pytest.fixture
def sample_script() -> Script:
    """Create a sample script for testing."""
    return Script(
        title="Why 90% of AI Startups Fail",
        hook=ScriptSection(
            id="hook",
            title="Hook",
            narration_text=(
                "[SCENE: startup logos fading] "
                "Jasper AI raised 125 million dollars at a 1.5 billion valuation in 2022. "
                "By 2025, they'd laid off most of their staff. "
                "We are watching the exact same pattern unfold right now."
            ),
            scene_marker="[SCENE: startup graveyard]",
        ),
        introduction=ScriptSection(
            id="intro",
            title="Introduction",
            narration_text="In this video we identify five failure patterns and map them onto today's AI startups.",
            scene_marker="[SCENE: timeline]",
        ),
        body_sections=[
            ScriptSection(
                id="section_1",
                title="Section 1: The Funding Mirage",
                narration_text=(
                    "The first failure pattern is the funding mirage. "
                    "In 1999, raising a hundred million dollars felt like winning. "
                    "CB Insights reports that 400 of 1200 funded AI startups have shut down. "
                    "That is a thirty-three percent failure rate in two years."
                ),
                scene_marker="[SCENE: champagne]",
            ),
            ScriptSection(
                id="section_2",
                title="Section 2: The Thin Wrapper Problem",
                narration_text=(
                    "A thin wrapper startup takes an existing model and adds a prompt plus UI. "
                    "The problem is obvious: no moat. Your competitor can copy you in a weekend. "
                    "Jasper built on GPT-3. Then ChatGPT launched and did the same thing for free. "
                    "Revenue dropped forty percent in six months."
                ),
                scene_marker="[SCENE: wrapper]",
            ),
            ScriptSection(
                id="section_3",
                title="Section 3: The Capability Treadmill",
                narration_text=(
                    "The third pattern is the capability treadmill. "
                    "Every model upgrade from every lab simultaneously improves all competitors. "
                    "Eighteen months of specialized training wiped out by a general model upgrade."
                ),
                scene_marker="[SCENE: treadmill]",
            ),
        ],
        conclusion=ScriptSection(
            id="conclusion",
            title="Conclusion",
            narration_text=(
                "The ninety percent failure rate is a prediction of concentration. "
                "The value concentrates into survivors who become dominant. "
                "Subscribe for weekly AI business analysis."
            ),
            scene_marker="[SCENE: graveyard pullback]",
        ),
        total_word_count=800,
    )


class TestGenerateThread:
    def test_produces_5_to_6_tweets(self, sample_script: Script):
        thread = generate_thread(sample_script)
        tweet_markers = [l for l in thread.split("\n") if l.startswith("--- Tweet")]
        # 1 hook + up to 4 body sections + 1 CTA = 3-6 tweets
        assert 5 <= len(tweet_markers) <= 6

    def test_first_tweet_has_hook(self, sample_script: Script):
        thread = generate_thread(sample_script)
        assert "125 million" in thread or "Jasper" in thread

    def test_last_tweet_has_cta(self, sample_script: Script):
        thread = generate_thread(sample_script, video_url="https://youtu.be/abc")
        assert "@TokenEconomyAI" in thread
        assert "https://youtu.be/abc" in thread

    def test_tweets_under_280_chars(self, sample_script: Script):
        thread = generate_thread(sample_script)
        # Extract individual tweets
        tweets = []
        current = []
        for line in thread.split("\n"):
            if line.startswith("--- Tweet"):
                if current:
                    tweets.append("\n".join(current).strip())
                current = []
            else:
                current.append(line)
        if current:
            tweets.append("\n".join(current).strip())

        for tweet in tweets:
            assert len(tweet) <= 280, f"Tweet too long ({len(tweet)} chars): {tweet[:50]}..."

    def test_no_scene_markers_in_output(self, sample_script: Script):
        thread = generate_thread(sample_script)
        assert "[SCENE:" not in thread

    def test_no_bold_markers_in_output(self, sample_script: Script):
        thread = generate_thread(sample_script)
        assert "**" not in thread

    def test_empty_url_uses_link_in_bio(self, sample_script: Script):
        thread = generate_thread(sample_script, video_url="")
        assert "[link in bio]" in thread


class TestGenerateNewsletter:
    def test_has_title(self, sample_script: Script):
        nl = generate_newsletter(sample_script)
        assert "# Why 90% of AI Startups Fail" in nl

    def test_has_tldr_section(self, sample_script: Script):
        nl = generate_newsletter(sample_script)
        assert "## TL;DR" in nl

    def test_has_body_section_headers(self, sample_script: Script):
        nl = generate_newsletter(sample_script)
        assert "## The Funding Mirage" in nl
        assert "## The Thin Wrapper Problem" in nl
        assert "## The Capability Treadmill" in nl

    def test_has_conclusion(self, sample_script: Script):
        nl = generate_newsletter(sample_script)
        assert "## The Bottom Line" in nl

    def test_has_subscribe_cta(self, sample_script: Script):
        nl = generate_newsletter(sample_script)
        assert "tokeneconomyai.substack.com" in nl

    def test_has_video_embed_when_url_provided(self, sample_script: Script):
        nl = generate_newsletter(sample_script, video_url="https://youtu.be/xyz")
        assert "https://youtu.be/xyz" in nl

    def test_no_scene_markers(self, sample_script: Script):
        nl = generate_newsletter(sample_script)
        assert "[SCENE:" not in nl

    def test_no_bold_markers(self, sample_script: Script):
        nl = generate_newsletter(sample_script)
        assert "**" not in nl or "**Watch" in nl  # Allow bold in embed callout


class TestGenerateLinkedinPost:
    def test_has_hook(self, sample_script: Script):
        post = generate_linkedin_post(sample_script)
        assert "Jasper" in post or "125 million" in post

    def test_has_hashtags(self, sample_script: Script):
        post = generate_linkedin_post(sample_script)
        assert "#AI" in post
        assert "#TokenEconomyAI" in post

    def test_has_video_url_when_provided(self, sample_script: Script):
        post = generate_linkedin_post(sample_script, video_url="https://youtu.be/abc")
        assert "https://youtu.be/abc" in post

    def test_reasonable_length(self, sample_script: Script):
        post = generate_linkedin_post(sample_script)
        assert len(post) < 3000  # LinkedIn limit


class TestUtilities:
    def test_clean_narration_removes_scene_markers(self):
        text = "[SCENE: something] Hello world [SCENE: another] end."
        assert "[SCENE:" not in _clean_narration(text)

    def test_clean_narration_removes_bold(self):
        assert _clean_narration("This is **bold** text") == "This is bold text"

    def test_split_sentences_basic(self):
        text = "First sentence. Second sentence. Third one."
        sentences = _split_sentences(text)
        assert len(sentences) == 3

    def test_trim_to_limit_short_text(self):
        assert _trim_to_limit("Hello", 280) == "Hello"

    def test_trim_to_limit_long_text(self):
        long = "word " * 100
        result = _trim_to_limit(long, 50)
        assert len(result) <= 50
        assert result.endswith("...")

    def test_extract_hook_line_prefers_numbers(self):
        text = "This is generic. Company X raised 500 million dollars in 2024. More generic stuff."
        result = _extract_hook_line(text)
        assert "500 million" in result


class TestRealScripts:
    """Test against actual scripts in the repo."""

    @pytest.mark.parametrize("script_file", [
        "scripts/003_why_ai_startups_fail.md",
        "scripts/005_open_vs_closed_ai.md",
        "scripts/002_fifty_cent_revolution.md",
    ])
    def test_thread_generates_for_real_scripts(self, script_file: str):
        path = Path(script_file)
        if not path.exists():
            pytest.skip(f"Script not found: {path}")

        from vidgen.parsers import parse_script
        script = parse_script(path)
        thread = generate_thread(script)

        # Should have 6 tweets
        markers = [l for l in thread.split("\n") if l.startswith("--- Tweet")]
        assert len(markers) == 6

    @pytest.mark.parametrize("script_file", [
        "scripts/003_why_ai_startups_fail.md",
        "scripts/005_open_vs_closed_ai.md",
    ])
    def test_newsletter_generates_for_real_scripts(self, script_file: str):
        path = Path(script_file)
        if not path.exists():
            pytest.skip(f"Script not found: {path}")

        from vidgen.parsers import parse_script
        script = parse_script(path)
        nl = generate_newsletter(script)

        assert "## TL;DR" in nl
        assert "## The Bottom Line" in nl
        assert "tokeneconomyai.substack.com" in nl
