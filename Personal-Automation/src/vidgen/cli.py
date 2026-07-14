"""CLI entry point for the vidgen video generation pipeline."""

import logging
import sys
from pathlib import Path

import click

from vidgen.config import PipelineConfig, load_config, validate_config
from vidgen.parsers import ParseError, parse_scene_plan, parse_script, validate_script_scene_alignment
from vidgen.pipeline import PipelineOrchestrator
from vidgen.queue import QueueManager

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("config.yaml")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """vidgen - Automated YouTube video generation pipeline."""
    setup_logging(verbose)


@main.command()
@click.argument("script", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--scene-plan", "-p",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to scene_plan.json (auto-detected if not provided)",
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to config.yaml",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory override",
)
@click.option("--resume/--no-resume", default=False, help="Resume from last completed stage")
def run(script: Path, scene_plan: Path | None, config: Path | None, output: Path | None, resume: bool) -> None:
    """Run pipeline for a single script file."""
    config_path = config or DEFAULT_CONFIG

    try:
        # Load config with optional output override
        overrides: dict[str, str] = {}
        if output:
            overrides["output_dir"] = str(output)

        if config_path.exists():
            pipeline_config = load_config(config_path, overrides if overrides else None)
        else:
            pipeline_config = PipelineConfig()
            if output:
                pipeline_config = pipeline_config.model_copy(update={"output_dir": output})

        # Validate config
        errors = validate_config(pipeline_config)
        if errors:
            click.echo("Configuration errors:", err=True)
            for err in errors:
                click.echo(f"  - {err}", err=True)
            sys.exit(1)

        # Parse inputs
        parsed_script = parse_script(script)

        # Auto-detect scene plan if not provided
        if scene_plan is None:
            scene_plan = script.parent / "scene_plan.json"
            if not scene_plan.exists():
                click.echo(f"Error: No scene plan found at {scene_plan}", err=True)
                click.echo("Provide one with --scene-plan/-p", err=True)
                sys.exit(1)

        parsed_plan = parse_scene_plan(scene_plan)

        # Validate alignment
        alignment_errors = validate_script_scene_alignment(parsed_script, parsed_plan)
        if alignment_errors:
            click.echo("Script/Scene plan alignment errors:", err=True)
            for err in alignment_errors:
                click.echo(f"  - {err}", err=True)
            sys.exit(1)

        # Set up job directory
        job_dir = pipeline_config.jobs_dir / parsed_plan.topic_slug
        job_dir.mkdir(parents=True, exist_ok=True)

        # Run pipeline
        orchestrator = PipelineOrchestrator(pipeline_config, job_dir)
        estimated = orchestrator.estimate_duration(parsed_script)
        click.echo(f"Starting pipeline for: {parsed_script.title}")
        click.echo(f"Estimated duration: {estimated}")

        result = orchestrator.run(parsed_script, parsed_plan, resume=resume)

        if result.success:
            click.echo(f"Pipeline completed in {result.total_duration_seconds:.1f}s")
            click.echo(f"  Output: {result.output_dir}")
        else:
            click.echo(f"Pipeline failed: {result.error}", err=True)
            sys.exit(1)

    except ParseError as e:
        click.echo(f"Error parsing input: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command("queue-add")
@click.argument("script", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--scene-plan", "-p",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to scene_plan.json",
)
@click.option("--priority", type=int, default=0, help="Job priority (higher = processed first)")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to config.yaml",
)
def queue_add(script: Path, scene_plan: Path, priority: int, config: Path | None) -> None:
    """Add a job to the processing queue."""
    config_path = config or DEFAULT_CONFIG
    pipeline_config = load_config(config_path) if config_path.exists() else PipelineConfig()

    queue_file = pipeline_config.jobs_dir / "queue.json"
    mgr = QueueManager(queue_file, pipeline_config.jobs_dir)

    job_id = mgr.add_job(script, scene_plan, priority)
    click.echo(f"Added job {job_id} to queue (priority={priority})")


@main.command("queue-run")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to config.yaml",
)
def queue_run(config: Path | None) -> None:
    """Process all queued jobs sequentially."""
    config_path = config or DEFAULT_CONFIG
    pipeline_config = load_config(config_path) if config_path.exists() else PipelineConfig()

    queue_file = pipeline_config.jobs_dir / "queue.json"
    mgr = QueueManager(queue_file, pipeline_config.jobs_dir)

    click.echo("Processing queue...")
    summary = mgr.process_queue(pipeline_config)

    click.echo("\nQueue complete:")
    click.echo(f"  Total: {summary.total_jobs}")
    click.echo(f"  Completed: {summary.completed}")
    click.echo(f"  Failed: {summary.failed}")
    click.echo(f"  Duration: {summary.total_duration_seconds:.1f}s")


@main.command("queue-status")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to config.yaml",
)
def queue_status(config: Path | None) -> None:
    """Print current queue state."""
    config_path = config or DEFAULT_CONFIG
    pipeline_config = load_config(config_path) if config_path.exists() else PipelineConfig()

    queue_file = pipeline_config.jobs_dir / "queue.json"
    mgr = QueueManager(queue_file, pipeline_config.jobs_dir)

    state = mgr.get_status()

    if not state.entries:
        click.echo("Queue is empty.")
        return

    click.echo(f"Queue ({len(state.entries)} jobs):")
    status_icons = {
        "queued": "o",
        "in-progress": "-",
        "completed": "*",
        "failed": "x",
    }
    for entry in state.entries:
        icon = status_icons.get(entry.status, "?")
        click.echo(f"  {icon} [{entry.status}] {entry.job_id[:8]}... (priority={entry.priority})")
        if entry.error:
            click.echo(f"    Error: {entry.error}")


@main.command()
@click.argument("script", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--scene-plan", "-p",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to scene_plan.json",
)
def validate(script: Path, scene_plan: Path) -> None:
    """Validate input files without running generation."""
    try:
        parsed_script = parse_script(script)
        click.echo(f"Script valid: '{parsed_script.title}' ({parsed_script.total_word_count} words)")

        parsed_plan = parse_scene_plan(scene_plan)
        click.echo(f"Scene plan valid: {len(parsed_plan.scenes)} scenes, {parsed_plan.total_duration:.0f}s")

        errors = validate_script_scene_alignment(parsed_script, parsed_plan)
        if errors:
            click.echo("Alignment errors:", err=True)
            for err in errors:
                click.echo(f"  - {err}", err=True)
            sys.exit(1)
        else:
            click.echo("Script and scene plan are aligned")

    except ParseError as e:
        click.echo(f"Parse error: {e}", err=True)
        sys.exit(1)


@main.command("short")
@click.argument("scripts", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to config.yaml",
)
def short(scripts: tuple[Path, ...], config: Path | None) -> None:
    """Generate YouTube Shorts from script markdown files."""
    from vidgen.shorts_pipeline import ShortsPipeline

    if not scripts:
        click.echo("No script files provided.", err=True)
        click.echo("Usage: vidgen short shorts/003_short_*.md", err=True)
        sys.exit(1)

    config_path = config or DEFAULT_CONFIG
    pipeline = ShortsPipeline(config_path if config_path.exists() else None)

    results = pipeline.run_batch(list(scripts))

    # Print summary
    click.echo("")
    completed = 0
    for script_path, result in zip(scripts, results):
        icon = "✓" if result.success else "✗"
        if result.success:
            click.echo(f"  {icon} {script_path.name} → {result.output_path}")
            completed += 1
        else:
            click.echo(f"  {icon} {script_path.name} — {result.error}")

    total = len(scripts)
    total_time = sum(r.duration_seconds for r in results)
    click.echo(f"\n  {completed}/{total} completed in {total_time:.0f}s")

    if completed < total:
        sys.exit(1)


@main.command("produce")
@click.argument("script", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--scene-plan", "-p",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to scene_plan.json",
)
@click.option("--url", "-u", default="", help="YouTube video URL for distribution CTAs")
@click.option("--shorts-dir", "-s", type=click.Path(exists=True, path_type=Path), default=None, help="Directory with short scripts")
@click.option("--resume", is_flag=True, help="Skip stages whose output already exists")
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), default=None, help="Path to config.yaml")
def produce(script: Path, scene_plan: Path, url: str, shorts_dir: Path | None, resume: bool, config: Path | None) -> None:
    """Generate complete content package from a single script.

    Produces: long-form video, shorts, X thread, Substack newsletter,
    LinkedIn post, and metadata — all in one organized folder.

    Example:
        vidgen produce scripts/003_why_ai_startups_fail.md -p scripts/003_plan.json
    """
    from vidgen.producer import ContentProducer

    config_path = config or DEFAULT_CONFIG
    producer = ContentProducer(config_path=config_path if config_path.exists() else None)

    click.echo(f"Producing: {script.name}")
    click.echo(f"Scene plan: {scene_plan.name}")
    if url:
        click.echo(f"Video URL: {url}")
    click.echo("")

    result = producer.produce(
        script_path=script,
        scene_plan_path=scene_plan,
        shorts_dir=shorts_dir,
        video_url=url,
        resume=resume,
    )

    # Summary
    click.echo("")
    click.echo(f"{'═' * 50}")
    click.echo(f"  Output: {result.output_dir}")
    click.echo(f"  Video:        {'✓' if result.video_ok else '✗'}")
    click.echo(f"  Shorts:       {result.shorts_completed}/{result.shorts_total}")
    click.echo(f"  Distribution: {'✓' if result.distribution_ok else '✗'}")
    click.echo(f"  Time:         {result.total_duration_seconds:.0f}s")
    click.echo(f"{'═' * 50}")

    if not result.success:
        sys.exit(1)


@main.command("thread")
@click.argument("script", type=click.Path(exists=True, path_type=Path))
@click.option("--url", "-u", default="", help="YouTube video URL for CTA")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output file path")
def thread(script: Path, url: str, output: Path | None) -> None:
    """Generate X/Twitter thread from a video script."""
    from vidgen.distribution import generate_thread

    parsed = parse_script(script)
    thread_text = generate_thread(parsed, video_url=url)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(thread_text)
        click.echo(f"Thread saved: {output}")
    else:
        click.echo(thread_text)


@main.command("newsletter")
@click.argument("script", type=click.Path(exists=True, path_type=Path))
@click.option("--url", "-u", default="", help="YouTube video URL for embed")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output file path")
def newsletter(script: Path, url: str, output: Path | None) -> None:
    """Generate Substack newsletter from a video script."""
    from vidgen.distribution import generate_newsletter

    parsed = parse_script(script)
    newsletter_text = generate_newsletter(parsed, video_url=url)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(newsletter_text)
        click.echo(f"Newsletter saved: {output}")
    else:
        click.echo(newsletter_text)


@main.command("linkedin")
@click.argument("script", type=click.Path(exists=True, path_type=Path))
@click.option("--url", "-u", default="", help="YouTube video URL")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output file path")
def linkedin(script: Path, url: str, output: Path | None) -> None:
    """Generate LinkedIn post from a video script."""
    from vidgen.distribution import generate_linkedin_post

    parsed = parse_script(script)
    post_text = generate_linkedin_post(parsed, video_url=url)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(post_text)
        click.echo(f"LinkedIn post saved: {output}")
    else:
        click.echo(post_text)


@main.command("distribute")
@click.argument("script", type=click.Path(exists=True, path_type=Path))
@click.option("--url", "-u", default="", help="YouTube video URL")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=None, help="Output directory")
def distribute(script: Path, url: str, output_dir: Path | None) -> None:
    """Generate all distribution content (thread + newsletter + LinkedIn) from a script."""
    from vidgen.distribution import generate_linkedin_post, generate_newsletter, generate_thread

    parsed = parse_script(script)
    slug = script.stem

    out = output_dir or Path("output/distribution") / slug
    out.mkdir(parents=True, exist_ok=True)

    # Generate all formats
    thread_text = generate_thread(parsed, video_url=url)
    (out / "thread.txt").write_text(thread_text)

    newsletter_text = generate_newsletter(parsed, video_url=url)
    (out / "newsletter.md").write_text(newsletter_text)

    linkedin_text = generate_linkedin_post(parsed, video_url=url)
    (out / "linkedin_post.txt").write_text(linkedin_text)

    click.echo(f"Distribution package generated:")
    click.echo(f"  {out}/thread.txt")
    click.echo(f"  {out}/newsletter.md")
    click.echo(f"  {out}/linkedin_post.txt")


@main.command("metadata")
@click.argument("script", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--scene-plan", "-p",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to scene_plan.json",
)
@click.option("--url", "-u", required=True, help="YouTube video URL")
@click.option("--shorts-dir", "-s", type=click.Path(exists=True, path_type=Path), default=None, help="Directory with short scripts")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output directory path")
def metadata(script: Path, scene_plan: Path, url: str, shorts_dir: Path | None, output: Path | None) -> None:
    """Generate per-platform upload metadata files.

    Creates a metadata/ folder with separate files:
      youtube.md, twitter.md, linkedin.md, substack.md, community.md

    Example:
        vidgen metadata scripts/001.md -p scripts/001_plan.json -u https://youtu.be/abc
    """
    from vidgen.parsers import parse_scene_plan
    from vidgen.upload_package import generate_upload_package

    parsed_script = parse_script(script)
    parsed_plan = parse_scene_plan(scene_plan)

    # Find matching short scripts
    number = script.stem.split("_", 1)[0]
    search_dir = shorts_dir or Path("shorts")
    short_scripts = sorted(search_dir.glob(f"{number}_short_*.md")) if search_dir.exists() else []

    # Default output path: output/{script_stem}/metadata/
    if output is None:
        output = Path("output") / script.stem / "metadata"

    package = generate_upload_package(
        script=parsed_script,
        scene_plan=parsed_plan,
        video_url=url,
        short_scripts=short_scripts if short_scripts else None,
        output_path=output,
    )

    metadata_dir = output if output.name == "metadata" else output / "metadata"
    click.echo(f"Upload package generated: {metadata_dir}/")
    click.echo(f"  youtube.md   — titles, description, tags, chapters, shorts")
    click.echo(f"  twitter.md   — X/Twitter thread")
    click.echo(f"  linkedin.md  — LinkedIn post")
    click.echo(f"  substack.md  — newsletter article")
    click.echo(f"  community.md — 3 community tab posts")
    click.echo(f"  Shorts: {len(short_scripts)} with metadata")


@main.command("captions")
@click.argument("narration_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output .srt file path")
def captions(narration_dir: Path, output: Path | None) -> None:
    """Generate SRT captions from narration boundary data.

    NARRATION_DIR is the directory containing scene_NNN.wav and
    scene_NNN.boundaries.json files (e.g. jobs/produce_xyz/narration/).

    Example:
        vidgen captions jobs/produce_one-person-billion-dollar-company/narration/
        vidgen captions jobs/produce_youtube-algorithm-2026-explained/narration/ -o output/006/captions.srt
    """
    from vidgen.captions import generate_srt

    if output is None:
        output = narration_dir.parent / "captions.srt"

    result = generate_srt(narration_dir=narration_dir, output_path=output)
    if result:
        click.echo(f"Captions generated: {result}")
    else:
        click.echo("No boundary data found — cannot generate captions.", err=True)
        sys.exit(1)


@main.command("news-scan")
@click.option("--sources", "-s", default=None, help="Comma-separated source list (hackernews,techcrunch,arxiv,reddit,producthunt)")
@click.option("--min-score", type=float, default=None, help="Only show candidates above this score")
@click.option("--limit", "-n", type=int, default=10, help="Max candidates to display")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON instead of formatted table")
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), default=None, help="Config file path")
def news_scan(sources: str | None, min_score: float | None, limit: int, as_json: bool, config: Path | None) -> None:
    """Scan AI/tech news sources and rank candidates for reactive Shorts."""
    import asyncio
    import json as json_module
    from vidgen.news_monitor import NewsConfig, format_candidates_terminal, scan_news

    # Load config
    news_config = NewsConfig()  # Uses sensible defaults
    if config:
        import yaml
        with open(config) as f:
            raw = yaml.safe_load(f)
        if raw and "news_scanner" in raw:
            news_config = NewsConfig.model_validate(raw["news_scanner"])
    elif DEFAULT_CONFIG.exists():
        import yaml
        with open(DEFAULT_CONFIG) as f:
            raw = yaml.safe_load(f)
        if raw and "news_scanner" in raw:
            news_config = NewsConfig.model_validate(raw["news_scanner"])

    # Parse source filter
    source_filter = None
    if sources:
        source_filter = [s.strip() for s in sources.split(",")]

    # Run scan
    click.echo("Scanning news sources...")
    candidates = asyncio.run(scan_news(news_config, source_filter=source_filter))

    # Apply filters
    if min_score is not None:
        candidates = [c for c in candidates if c.score >= min_score]

    # Output
    if as_json:
        data = [c.model_dump(mode="json") for c in candidates[:limit]]
        click.echo(json_module.dumps(data, indent=2, default=str))
    else:
        output = format_candidates_terminal(candidates, limit=limit)
        click.echo(output)


if __name__ == "__main__":
    main()
