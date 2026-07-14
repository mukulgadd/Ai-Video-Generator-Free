"""Reactive News Scanner — fetches, scores, and ranks AI/tech news for reactive Shorts production.

Scans multiple AI/tech news sources concurrently (Hacker News, TechCrunch, ArXiv,
Reddit r/MachineLearning, Product Hunt), scores stories against a weighted algorithm
derived from channel analytics, deduplicates cross-source stories, and presents
ranked candidates for editorial decision-making.

Usage:
    vidgen news-scan              # Scan all sources, show top 10
    vidgen news-scan --json       # Output as JSON
    vidgen news-scan -n 5         # Limit to top 5
"""

import asyncio
import json
import logging
import re
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import feedparser
import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Story(BaseModel):
    """A single news item from any source."""

    id: str  # Unique ID (source-specific: HN item id, RSS guid, etc.)
    headline: str  # Title/headline text
    url: str  # Link to original article
    source: str  # Source identifier: "hackernews", "techcrunch", "arxiv", "reddit", "producthunt"
    published_at: datetime  # Publication timestamp (UTC)
    metadata: dict[str, Any] = {}  # Source-specific extras (HN score, Reddit upvotes, etc.)


class ScoreBreakdown(BaseModel):
    """Individual factor contributions to a story's score."""

    brand_name: float = 0.0
    numbers_stats: float = 0.0
    conflict_tension: float = 0.0
    multi_source: float = 0.0
    hn_trending: float = 0.0
    recency: float = 0.0
    niche_score: float = 0.0  # "Who profits/dies" economics angle
    total: float = 0.0


class Candidate(BaseModel):
    """A scored, deduplicated story ready for editorial review."""

    headline: str  # Best headline (from highest-scoring source)
    urls: list[str]  # All source URLs
    sources: list[str]  # All source identifiers
    score: float  # Final composite score
    score_breakdown: ScoreBreakdown  # Factor-by-factor breakdown
    published_at: datetime  # Earliest publication timestamp
    hook_angle: str  # Suggested title: [Brand/Number] + [Verb] + [Consequence]
    story_ids: list[str]  # IDs of merged stories


class SourceConfig(BaseModel):
    """Per-source configuration."""

    enabled: bool = True
    url: str = ""  # Custom feed URL (override default)
    timeout: float = 10.0  # Per-source timeout in seconds


class ScoreWeights(BaseModel):
    """Scoring weights — tuned from analytics data."""

    brand_name: float = 3.0
    numbers_stats: float = 2.0
    conflict_tension: float = 2.5
    multi_source: float = 2.0
    hn_trending: float = 1.5
    recency: float = 2.0
    niche_economics: float = 3.0  # High weight — our niche filter is critical


class NewsConfig(BaseModel):
    """Top-level news scanner configuration."""

    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    brand_list: list[str] = Field(default_factory=lambda: [
        "OpenAI", "Cursor", "Google", "Meta", "Anthropic", "Microsoft",
        "Apple", "NVIDIA", "Tesla", "Amazon", "DeepSeek", "Mistral",
        "Hugging Face", "Stability AI", "Midjourney", "Perplexity",
        "GitHub Copilot", "Claude", "GPT", "Gemini", "Llama",
    ])
    conflict_terms: list[str] = Field(default_factory=lambda: [
        "vs", "kills", "fails", "dies", "destroying", "crashes",
        "layoffs", "shuts down", "collapses", "disrupts", "threatens",
        "replaces", "eliminates", "overtakes", "dominates",
        "sues", "sued", "lawsuit", "drops", "blocks", "bans",
        "banned", "cuts", "slashes", "loses", "lost", "plummets",
        "tanks", "fires", "fired", "axes", "scraps", "halts",
    ])
    niche_terms: list[str] = Field(default_factory=lambda: [
        # Revenue/money signals
        "revenue", "profit", "valuation", "funding", "ipo", "acquisition",
        "billion", "million", "arr", "mrr", "arpu", "margin",
        "burn rate", "roi", "market cap", "stock", "shares",
        # Business impact signals
        "layoffs", "fired", "hiring", "headcount", "workforce",
        "market share", "pricing", "subscription", "enterprise",
        "startup", "bankruptcy", "shut down", "pivot",
        # Economic structure signals
        "monopoly", "moat", "disruption", "commoditiz", "consolidat",
        "per-seat", "usage-based", "unit economics", "cost",
        "infrastructure", "capex", "data center", "compute",
    ])
    sources: dict[str, SourceConfig] = Field(default_factory=lambda: {
        "hackernews": SourceConfig(url="https://hacker-news.firebaseio.com/v0"),
        "techcrunch": SourceConfig(url="https://techcrunch.com/category/artificial-intelligence/feed/"),
        "arxiv": SourceConfig(url="http://export.arxiv.org/rss/cs.AI"),
        "reddit": SourceConfig(url="https://www.reddit.com/r/MachineLearning/hot.json"),
        "producthunt": SourceConfig(url="https://www.producthunt.com/feed"),
    })
    candidates_file: Path = Path("news/candidates.json")
    max_story_age_hours: int = 24
    dedup_similarity_threshold: float = 0.6
    total_timeout: float = 30.0


class ScoreEngine:
    """Scores stories using weighted algorithm derived from channel analytics."""

    def __init__(self, config: NewsConfig):
        self.config = config
        self.weights = config.weights

    def score(self, story: Story, all_stories: list[Story]) -> tuple[float, ScoreBreakdown]:
        """Score a single story against all factors. Returns (total_score, breakdown)."""
        breakdown = ScoreBreakdown()
        headline_lower = story.headline.lower()

        # Brand names (cumulative, capped at 2x single weight)
        brand_hits = sum(1 for b in self.config.brand_list if b.lower() in headline_lower)
        breakdown.brand_name = min(brand_hits * self.weights.brand_name, 2 * self.weights.brand_name)

        # Numbers/stats detection (regex: $X, X%, Xm, Xb, Xx, X.Xx)
        if re.search(r'(\$[\d,.]+[mbk]?|\d+%|\d+x|\d+\.\d+x)', headline_lower):
            breakdown.numbers_stats = self.weights.numbers_stats

        # Conflict/tension terms
        if any(term in headline_lower for term in self.config.conflict_terms):
            breakdown.conflict_tension = self.weights.conflict_tension

        # Multi-source: count stories with similar headline from different sources
        source_count = self._count_related_sources(story, all_stories)
        if source_count >= 2:
            breakdown.multi_source = self.weights.multi_source

        # HN trending (>100 points)
        if story.source == "hackernews" and story.metadata.get("score", 0) > 100:
            breakdown.hn_trending = self.weights.hn_trending

        # Recency (linear decay: full weight 0-3h, decay 3-24h, zero after 24h)
        age_hours = (datetime.now(UTC) - story.published_at).total_seconds() / 3600
        if age_hours < 0:
            age_hours = 0  # Future timestamps get full recency
        if age_hours < 3:
            breakdown.recency = self.weights.recency
        elif age_hours < 24:
            breakdown.recency = self.weights.recency * (1 - (age_hours - 3) / 21)
        # else: 0.0 (default)

        # Niche economics: "who profits/dies" angle — AI + money/business impact
        niche_hits = sum(1 for term in self.config.niche_terms if term.lower() in headline_lower)
        if niche_hits >= 3:
            breakdown.niche_score = self.weights.niche_economics
        elif niche_hits >= 2:
            breakdown.niche_score = self.weights.niche_economics * 0.7
        elif niche_hits >= 1:
            breakdown.niche_score = self.weights.niche_economics * 0.4

        breakdown.total = (
            breakdown.brand_name + breakdown.numbers_stats +
            breakdown.conflict_tension + breakdown.multi_source +
            breakdown.hn_trending + breakdown.recency +
            breakdown.niche_score
        )
        return breakdown.total, breakdown

    def _count_related_sources(self, story: Story, all_stories: list[Story]) -> int:
        """Count how many distinct sources have a similar story."""
        sources_seen = {story.source}
        story_tokens = set(story.headline.lower().split())

        for other in all_stories:
            if other.id == story.id:
                continue
            if other.source in sources_seen:
                continue
            other_tokens = set(other.headline.lower().split())
            if not story_tokens or not other_tokens:
                continue
            overlap = len(story_tokens & other_tokens) / min(len(story_tokens), len(other_tokens))
            if overlap >= 0.5:  # Looser threshold for multi-source detection
                sources_seen.add(other.source)

        return len(sources_seen)


async def fetch_hackernews(client: httpx.AsyncClient, config: SourceConfig) -> list[Story]:
    """Fetch top stories from Hacker News Firebase API."""
    base_url = config.url or "https://hacker-news.firebaseio.com/v0"
    stories: list[Story] = []

    try:
        # Get top story IDs
        resp = await client.get(f"{base_url}/topstories.json", timeout=config.timeout)
        resp.raise_for_status()
        story_ids = resp.json()[:30]  # Top 30 stories

        # Fetch individual story details concurrently
        tasks = [
            client.get(f"{base_url}/item/{sid}.json", timeout=config.timeout)
            for sid in story_ids
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        for resp in responses:
            if isinstance(resp, Exception):
                continue
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not data or not data.get("title"):
                continue

            # Parse timestamp
            timestamp = datetime.fromtimestamp(data.get("time", 0), tz=UTC)

            stories.append(Story(
                id=f"hn_{data['id']}",
                headline=data["title"],
                url=data.get("url", f"https://news.ycombinator.com/item?id={data['id']}"),
                source="hackernews",
                published_at=timestamp,
                metadata={
                    "score": data.get("score", 0),
                    "comments": data.get("descendants", 0),
                },
            ))

    except (httpx.HTTPError, httpx.TimeoutException) as e:
        logger.warning(f"Hacker News fetch failed: {e}")
    except Exception as e:
        logger.warning(f"Hacker News unexpected error: {e}")

    logger.info(f"Fetched {len(stories)} stories from Hacker News")
    return stories


async def fetch_rss(client: httpx.AsyncClient, url: str, source_id: str, timeout: float = 10.0) -> list[Story]:
    """Fetch and parse an RSS/Atom feed, returning normalized Story objects."""
    stories: list[Story] = []

    try:
        resp = await client.get(url, timeout=timeout)
        resp.raise_for_status()

        feed = feedparser.parse(resp.text)

        for entry in feed.entries:
            # Extract headline
            headline = entry.get("title", "").strip()
            if not headline:
                continue

            # Extract URL
            link = entry.get("link", "")
            if not link:
                continue

            # Extract publication time
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if published:
                timestamp = datetime.fromtimestamp(time.mktime(published), tz=UTC)
            else:
                timestamp = datetime.now(UTC)

            # Generate unique ID from source + link
            story_id = f"{source_id}_{hash(link) & 0xFFFFFFFF:08x}"

            stories.append(Story(
                id=story_id,
                headline=headline,
                url=link,
                source=source_id,
                published_at=timestamp,
                metadata={
                    "summary": entry.get("summary", "")[:200],
                },
            ))

    except (httpx.HTTPError, httpx.TimeoutException) as e:
        logger.warning(f"{source_id} RSS fetch failed: {e}")
    except Exception as e:
        logger.warning(f"{source_id} RSS unexpected error: {e}")

    logger.info(f"Fetched {len(stories)} stories from {source_id}")
    return stories


class Deduplicator:
    """Merges duplicate stories from different sources into single candidates."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def similarity(self, a: str, b: str) -> float:
        """Normalized case-insensitive token overlap between two headlines."""
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        return len(intersection) / min(len(tokens_a), len(tokens_b))

    def _same_domain(self, url_a: str, url_b: str) -> bool:
        """Check if two URLs share the same domain."""
        try:
            from urllib.parse import urlparse
            domain_a = urlparse(url_a).netloc.replace("www.", "")
            domain_b = urlparse(url_b).netloc.replace("www.", "")
            return domain_a == domain_b and domain_a != ""
        except Exception:
            return False

    def _are_duplicates(self, a: Story, b: Story) -> bool:
        """Determine if two stories refer to the same news event."""
        if a.source == b.source:
            return False  # Same source can't be duplicates in our context

        headline_sim = self.similarity(a.headline, b.headline)

        # High headline similarity alone
        if headline_sim >= self.threshold:
            return True

        # Same domain + lower threshold
        if self._same_domain(a.url, b.url) and headline_sim >= 0.4:
            return True

        return False

    def deduplicate(
        self, stories: list[Story], scores: dict[str, tuple[float, ScoreBreakdown]]
    ) -> list[Candidate]:
        """Group duplicate stories and merge into Candidate objects."""
        # Build groups of related stories
        groups: list[list[Story]] = []
        assigned: set[str] = set()

        for story in stories:
            if story.id in assigned:
                continue

            group = [story]
            assigned.add(story.id)

            for other in stories:
                if other.id in assigned:
                    continue
                if self._are_duplicates(story, other):
                    group.append(other)
                    assigned.add(other.id)

            groups.append(group)

        # Convert groups to Candidates
        candidates: list[Candidate] = []
        for group in groups:
            # Pick best headline (from highest-scoring story in group)
            best_story = max(group, key=lambda s: scores.get(s.id, (0.0, ScoreBreakdown()))[0])

            # Merge all URLs and sources
            all_urls = list(dict.fromkeys(s.url for s in group))  # Preserve order, deduplicate
            all_sources = list(dict.fromkeys(s.source for s in group))
            all_ids = [s.id for s in group]

            # Earliest timestamp
            earliest = min(s.published_at for s in group)

            # Use the best story's score and breakdown
            score, breakdown = scores.get(best_story.id, (0.0, ScoreBreakdown()))

            # If multi-source, ensure the multi_source bonus is applied
            if len(all_sources) >= 2 and breakdown.multi_source == 0.0:
                # This will be recalculated by the orchestrator if needed
                pass

            candidates.append(Candidate(
                headline=best_story.headline,
                urls=all_urls,
                sources=all_sources,
                score=score,
                score_breakdown=breakdown,
                published_at=earliest,
                hook_angle="",  # Filled in by presentation layer
                story_ids=all_ids,
            ))

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates


async def fetch_reddit(client: httpx.AsyncClient, config: SourceConfig) -> list[Story]:
    """Fetch hot posts from Reddit r/MachineLearning JSON API."""
    url = config.url or "https://www.reddit.com/r/MachineLearning/hot.json"
    stories: list[Story] = []

    try:
        # Reddit requires a User-Agent header
        headers = {"User-Agent": "TokenEconomyAI-NewsScanner/1.0"}
        resp = await client.get(url, timeout=config.timeout, headers=headers)
        resp.raise_for_status()

        data = resp.json()
        posts = data.get("data", {}).get("children", [])

        for post_wrapper in posts:
            post = post_wrapper.get("data", {})
            if not post:
                continue

            headline = post.get("title", "").strip()
            if not headline:
                continue

            # Skip stickied/pinned posts
            if post.get("stickied", False):
                continue

            # Extract URL (use external link if available, else Reddit permalink)
            link = post.get("url", "")
            if not link or link.startswith("/r/"):
                link = f"https://www.reddit.com{post.get('permalink', '')}"

            # Parse timestamp
            created_utc = post.get("created_utc", 0)
            timestamp = datetime.fromtimestamp(created_utc, tz=UTC) if created_utc else datetime.now(UTC)

            stories.append(Story(
                id=f"reddit_{post.get('id', '')}",
                headline=headline,
                url=link,
                source="reddit",
                published_at=timestamp,
                metadata={
                    "score": post.get("score", 0),
                    "upvote_ratio": post.get("upvote_ratio", 0),
                    "num_comments": post.get("num_comments", 0),
                    "subreddit": post.get("subreddit", "MachineLearning"),
                },
            ))

    except (httpx.HTTPError, httpx.TimeoutException) as e:
        logger.warning(f"Reddit fetch failed: {e}")
    except Exception as e:
        logger.warning(f"Reddit unexpected error: {e}")

    logger.info(f"Fetched {len(stories)} stories from Reddit")
    return stories


def load_candidates(path: Path) -> list[Candidate]:
    """Load candidates from JSON file. Returns empty list on missing/malformed file."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return [Candidate.model_validate(item) for item in data]
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Malformed candidates file {path}: {e}. Starting fresh.")
        return []


def save_candidates(path: Path, candidates: list[Candidate]) -> None:
    """Save candidates to JSON file atomically. Creates parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.model_dump(mode="json") for c in candidates]
    # Write atomically via temp file
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def merge_candidates(
    existing: list[Candidate], new: list[Candidate], max_age_hours: int = 24
) -> list[Candidate]:
    """Merge new candidates with existing, prune entries older than max_age_hours."""
    cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)

    # Index existing by headline (lowercase) for merging
    merged: dict[str, Candidate] = {}

    for c in existing:
        if c.published_at >= cutoff:
            merged[c.headline.lower()] = c

    for c in new:
        if c.published_at >= cutoff:
            key = c.headline.lower()
            if key in merged:
                old = merged[key]
                if c.score > old.score:
                    merged[key] = c
                else:
                    # Merge sources/URLs from new into old
                    updated_urls = list(dict.fromkeys(old.urls + c.urls))
                    updated_sources = list(dict.fromkeys(old.sources + c.sources))
                    merged[key] = old.model_copy(update={
                        "urls": updated_urls,
                        "sources": updated_sources,
                    })
            else:
                merged[key] = c

    # Sort by score descending
    return sorted(merged.values(), key=lambda c: c.score, reverse=True)


# --- Presentation Layer ---


def format_age(timestamp: datetime) -> str:
    """Format a timestamp as human-readable relative time."""
    now = datetime.now(UTC)
    # Ensure timestamp is timezone-aware
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    delta = now - timestamp
    seconds = delta.total_seconds()

    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        mins = int(seconds // 60)
        return f"{mins}m ago"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours}h ago"
    else:
        days = int(seconds // 86400)
        return f"{days}d ago"


def generate_hook_angle(headline: str, brands: list[str], conflict_terms: list[str]) -> str:
    """Generate a suggested Short title hook from a headline.

    Follows the channel formula: [Brand/Number] + [Strong Verb] + [Consequence]
    Extracts actual verbs and context from the headline for specificity.
    """
    headline_lower = headline.lower()

    # 1. Extract brand mention (first match)
    brand = next((b for b in brands if b.lower() in headline_lower), None)

    # 2. Extract number/stat if present
    stat_match = re.search(r'(\$[\d,.]+[MBK]?|\d+%|\d+x|\d+\.\d+x)', headline, re.IGNORECASE)

    # 3. Extract actual verb from headline (prefer conflict terms, then headline verbs)
    verb = None
    for term in conflict_terms:
        if term in headline_lower:
            verb = term.upper()
            break

    if not verb:
        # Try to extract action verbs from the headline itself
        action_verbs = [
            "launches", "raises", "acquires", "buys", "sells", "releases",
            "announces", "reveals", "introduces", "ships", "builds", "develops",
            "expands", "pivots", "partners", "invests", "surpasses", "hits",
            "reaches", "breaks", "opens", "closes", "merges", "splits",
            "bets", "targets", "challenges", "enters", "exits", "adopts",
        ]
        for v in action_verbs:
            if v in headline_lower:
                verb = v.upper()
                break

    if not verb:
        verb = "MOVES"  # Minimal fallback — still implies action

    # 4. Extract consequence/context from headline (words after the verb or key phrase)
    consequence = ""
    # Try to grab a meaningful snippet from the headline for specificity
    words = headline.split()
    if len(words) > 5:
        # Take the last 3-4 meaningful words as consequence context
        tail = " ".join(words[-4:])
        # Clean up if it starts with common prepositions
        for prefix in ["in ", "on ", "for ", "to ", "with ", "as ", "after "]:
            if tail.lower().startswith(prefix):
                tail = tail[len(prefix):]
                break
        consequence = tail

    # 5. Compose hook — use specific consequence if available
    lead = brand or (stat_match.group(0) if stat_match else words[0] if words else "This")

    if consequence and len(consequence) < 40:
        return f"{lead} {verb} — {consequence}"
    else:
        return f"{lead} {verb} — Why It Matters"


def format_candidates_terminal(candidates: list[Candidate], limit: int = 10) -> str:
    """Format candidates as a rich terminal display."""
    if not candidates:
        return "No candidates found. Try again later or check your sources."

    lines: list[str] = []
    shown = candidates[:limit]

    lines.append(f"\n{'━' * 60}")
    lines.append(f" NEWS SCAN — Top {len(shown)} Candidates")
    lines.append(f"{'━' * 60}")

    for i, c in enumerate(shown, 1):
        age = format_age(c.published_at)
        sources_str = ", ".join(c.sources)

        lines.append(f"\n {'━' * 58}")
        lines.append(f"  #{i}  {c.headline}")
        lines.append(f" {'━' * 58}")
        lines.append(f"  Score: {c.score:.1f}  │  Sources: {sources_str}")
        lines.append(f"  Age: {age}  │  Brand: +{c.score_breakdown.brand_name:.1f}  "
                     f"Numbers: +{c.score_breakdown.numbers_stats:.1f}  "
                     f"Conflict: +{c.score_breakdown.conflict_tension:.1f}")
        lines.append(f"  {'':11}│  Multi-src: +{c.score_breakdown.multi_source:.1f}  "
                     f"HN: +{c.score_breakdown.hn_trending:.1f}  "
                     f"Recency: +{c.score_breakdown.recency:.1f}")
        lines.append(f"  Hook: \"{c.hook_angle}\"")

        # Show first URL only
        if c.urls:
            lines.append(f"  URL: {c.urls[0]}")
            if len(c.urls) > 1:
                lines.append(f"       + {len(c.urls) - 1} more source(s)")

    lines.append(f"\n{'─' * 60}")
    lines.append(f" Total candidates: {len(candidates)} │ Showing: {len(shown)}")
    lines.append(f"{'─' * 60}\n")

    return "\n".join(lines)


# --- Orchestrator ---


async def scan_news(config: NewsConfig, source_filter: list[str] | None = None) -> list[Candidate]:
    """Orchestrator: fetch from all sources, score, deduplicate, persist, return ranked candidates."""

    async with httpx.AsyncClient() as client:
        # Build fetch tasks for enabled sources
        tasks: list = []

        for source_id, source_config in config.sources.items():
            if not source_config.enabled:
                continue
            if source_filter and source_id not in source_filter:
                continue

            if source_id == "hackernews":
                tasks.append(asyncio.create_task(fetch_hackernews(client, source_config)))
            elif source_id == "reddit":
                tasks.append(asyncio.create_task(fetch_reddit(client, source_config)))
            elif source_id in ("techcrunch", "arxiv", "producthunt"):
                url = source_config.url
                tasks.append(asyncio.create_task(
                    fetch_rss(client, url, source_id, timeout=source_config.timeout)
                ))

        if not tasks:
            logger.warning("No sources enabled or all filtered out.")
            return []

        # Fetch all sources concurrently with total timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=config.total_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Total fetch timeout ({config.total_timeout}s) exceeded.")
            results = []

        # Flatten stories from all sources
        all_stories: list[Story] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Source fetch failed: {result}")
                continue
            all_stories.extend(result)

        if not all_stories:
            logger.info("No stories fetched from any source.")
            return []

        logger.info(f"Total stories fetched: {len(all_stories)}")

        # Score all stories
        engine = ScoreEngine(config)
        scores: dict[str, tuple[float, ScoreBreakdown]] = {}
        for story in all_stories:
            score, breakdown = engine.score(story, all_stories)
            scores[story.id] = (score, breakdown)

        # Deduplicate into candidates
        dedup = Deduplicator(threshold=config.dedup_similarity_threshold)
        candidates = dedup.deduplicate(all_stories, scores)

        # Generate hook angles for each candidate
        for i, c in enumerate(candidates):
            hook = generate_hook_angle(c.headline, config.brand_list, config.conflict_terms)
            candidates[i] = c.model_copy(update={"hook_angle": hook})

        # Persist with rolling window
        existing = load_candidates(config.candidates_file)
        merged = merge_candidates(existing, candidates, max_age_hours=config.max_story_age_hours)
        save_candidates(config.candidates_file, merged)

        logger.info(f"Scan complete: {len(candidates)} candidates ({len(merged)} total in file)")
        return candidates
