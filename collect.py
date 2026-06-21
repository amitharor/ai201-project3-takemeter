"""
collect.py: Pull public r/nba comments into data/raw_comments.csv for labeling.

Three backends, auto selected:
  1. arctic: the arctic-shift public reddit archive (no auth). DEFAULT when no Reddit
     creds are set, because reddit.com blocks unauthenticated requests from many IPs.
  2. PRAW: if REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are set in .env (the official API).
  3. .json: public reddit.com JSON endpoints, no auth. Often blocked (403); kept as a
     last resort for networks where reddit is reachable directly.

Strategy: sample from a *mix* of sources so we get all three labels
(reaction / hot_take / analysis), not just reaction heavy game threads. r/nba gives the
broad mix; r/nbadiscussion boosts analysis. See planning.md §4.

Usage:
    python collect.py                  # default: ~300 comments via the arctic archive
    python collect.py --target 300     # how many clean comments to keep
    python collect.py --source arctic  # force a backend: arctic|praw|json|auto
    python collect.py --sort top       # listing sort for the reddit backends

Output: data/raw_comments.csv with columns: id, text, score, permalink, thread_type
You then label it into data/labeled_data.csv (add a `label` column).
"""

import argparse
import csv
import html
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

OUT_PATH = Path(__file__).parent / "data" / "raw_comments.csv"
USER_AGENT = os.getenv("REDDIT_USER_AGENT") or "takemeter-collector/0.1 (by u/anon)"

# Thread type buckets to which r/nba listings to pull posts from. The goal is variety:
# game/post game threads are reaction rich; discussion threads are analysis/hot_take rich.
SUBREDDIT_FEEDS = {
    "game_thread": [("nba", "hot"), ("nba", "new")],      # GAME THREAD / POST GAME THREAD
    "discussion": [("nbadiscussion", "hot"), ("nbadiscussion", "top")],
    "hot_take": [("nba", "top")],                          # top posts incl. opinion bait
}

# Comments matching these are dropped (bots, deleted, link only, etc.)
JUNK_PATTERNS = re.compile(
    r"^\s*(\[deleted\]|\[removed\])\s*$"
    r"|i am a bot"
    r"|^!",                                                # bot command invocations
    re.IGNORECASE,
)
BOT_AUTHORS = {"AutoModerator", "NBA_MOD", "SortingHat_bot"}

# arctic-shift archive: a no auth public mirror of reddit data. We pull from r/nba (broad
# discourse mix) and r/nbadiscussion (analysis heavy) so the labels do not collapse onto one
# source. Each tuple is (subreddit, thread_type tag, fraction of the target to pull there).
ARCTIC_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"
ARCTIC_SOURCES = [
    ("nba", "nba_general", 0.65),
    ("nbadiscussion", "discussion", 0.35),
]

MIN_WORDS = 3          # a 1 to 2 word burst is hard to label; allow short emotional reactions though
MAX_CHARS = 1500       # DistilBERT truncates ~512 tokens anyway; keep comments readable


# =========================================================================== #
# Filtering
# =========================================================================== #
def clean_text(body: str) -> str:
    body = html.unescape(body or "")            # &gt; &amp; &nbsp; etc. to real characters
    body = body.replace("\r", " ").replace("\n", " ").strip()
    body = re.sub(r"\s+", " ", body)            # collapse runs of whitespace (incl. nbsp)
    return body


def keep(body: str, author: str) -> bool:
    if not body or author in BOT_AUTHORS:
        return False
    if JUNK_PATTERNS.search(body):
        return False
    if len(body.split()) < MIN_WORDS:
        return False
    if len(body) > MAX_CHARS:
        return False
    # drop comments that are basically just a URL or a quote reply with no content
    if body.startswith(">") and len(body.split()) < 8:
        return False
    if re.fullmatch(r"https?://\S+", body):
        return False
    return True


# =========================================================================== #
# Backend 1: PRAW
# =========================================================================== #
def collect_praw(target, sort):
    import praw  # imported lazily so the .json fallback works without praw installed

    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=USER_AGENT,
    )
    reddit.read_only = True
    print(f"[praw] authenticated read only as app; collecting up to {target} comments")

    rows, seen = [], set()
    for thread_type, feeds in SUBREDDIT_FEEDS.items():
        per_type = max(target // len(SUBREDDIT_FEEDS), 1)
        for sub, listing in feeds:
            if len([r for r in rows if r["thread_type"] == thread_type]) >= per_type:
                break
            subreddit = reddit.subreddit(sub)
            posts = getattr(subreddit, listing)(limit=15)
            for post in posts:
                try:
                    post.comments.replace_more(limit=0)
                except Exception:
                    continue
                for c in post.comments.list()[:60]:
                    body = clean_text(getattr(c, "body", ""))
                    author = str(getattr(c, "author", "") or "")
                    if not keep(body, author) or body in seen:
                        continue
                    seen.add(body)
                    rows.append({
                        "id": c.id,
                        "text": body,
                        "score": getattr(c, "score", 0),
                        "permalink": f"https://reddit.com{c.permalink}",
                        "thread_type": thread_type,
                    })
                if len([r for r in rows if r["thread_type"] == thread_type]) >= per_type:
                    break
            time.sleep(1)
    return rows


# =========================================================================== #
# Backend 2: public .json endpoints (no auth)
# =========================================================================== #
def _get_json(url, params=None):
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _walk_comments(children, out):
    """Recursively flatten a reddit comment tree from the .json listing."""
    for child in children:
        if child.get("kind") != "t1":
            continue
        d = child.get("data", {})
        out.append(d)
        replies = d.get("replies")
        if isinstance(replies, dict):
            _walk_comments(replies.get("data", {}).get("children", []), out)


def collect_json(target, sort):
    print(f"[json] no Reddit creds found, using public .json endpoints (rate limited)")
    rows, seen = [], set()
    for thread_type, feeds in SUBREDDIT_FEEDS.items():
        per_type = max(target // len(SUBREDDIT_FEEDS), 1)
        for sub, listing in feeds:
            if len([r for r in rows if r["thread_type"] == thread_type]) >= per_type:
                break
            try:
                listing_json = _get_json(
                    f"https://www.reddit.com/r/{sub}/{listing}.json",
                    params={"limit": 15, "t": "week"},
                )
            except Exception as e:
                print(f"  ! failed to list r/{sub}/{listing}: {e}")
                continue
            posts = listing_json.get("data", {}).get("children", [])
            for post in posts:
                pid = post.get("data", {}).get("id")
                if not pid:
                    continue
                try:
                    thread = _get_json(
                        f"https://www.reddit.com/r/{sub}/comments/{pid}.json",
                        params={"limit": 100, "sort": sort},
                    )
                except Exception as e:
                    print(f"  ! failed to fetch comments for {pid}: {e}")
                    time.sleep(2)
                    continue
                flat = []
                if len(thread) > 1:
                    _walk_comments(thread[1].get("data", {}).get("children", []), flat)
                for d in flat:
                    body = clean_text(d.get("body", ""))
                    author = str(d.get("author", "") or "")
                    if not keep(body, author) or body in seen:
                        continue
                    seen.add(body)
                    rows.append({
                        "id": d.get("id", ""),
                        "text": body,
                        "score": d.get("score", 0),
                        "permalink": f"https://reddit.com{d.get('permalink', '')}",
                        "thread_type": thread_type,
                    })
                time.sleep(2)  # be polite to the unauthenticated endpoint
                if len([r for r in rows if r["thread_type"] == thread_type]) >= per_type:
                    break
    return rows


# =========================================================================== #
# Backend 3 (default no auth): arctic-shift public reddit archive
# =========================================================================== #
def collect_arctic(target, sort):
    print("[arctic] using the arctic-shift public reddit archive (no auth)")
    rows, seen = [], set()
    for sub, thread_type, frac in ARCTIC_SOURCES:
        want = max(int(target * frac), 1)
        before, pages, kept = None, 0, 0
        while kept < want and pages < 80:
            params = {"subreddit": sub, "limit": 100, "sort": "desc"}
            if before is not None:
                params["before"] = before
            try:
                data = _get_json(ARCTIC_URL, params=params).get("data", [])
            except Exception as e:
                print(f"  ! arctic fetch failed for r/{sub}: {e}")
                break
            if not data:
                break
            for d in data:
                body = clean_text(d.get("body", ""))
                author = str(d.get("author", "") or "")
                if not keep(body, author) or body in seen:
                    continue
                seen.add(body)
                rows.append({
                    "id": d.get("id", ""),
                    "text": body,
                    "score": d.get("score", 0),
                    "permalink": f"https://reddit.com{d.get('permalink', '')}",
                    "thread_type": thread_type,
                })
                kept += 1
                if kept >= want:
                    break
            before = min(c["created_utc"] for c in data)  # page to older comments
            pages += 1
            time.sleep(0.5)  # be polite to the archive
        print(f"  r/{sub}: kept {kept} ({thread_type})")
    return rows


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser(description="Collect r/nba comments for TakeMeter.")
    ap.add_argument("--target", type=int, default=300, help="clean comments to keep")
    ap.add_argument("--sort", default="top", help="comment sort: top|best|new")
    ap.add_argument("--source", default="auto", choices=["auto", "arctic", "praw", "json"],
                    help="data backend; auto picks praw if creds are set, else arctic")
    args = ap.parse_args()

    has_creds = bool(os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET"))
    source = args.source
    if source == "auto":
        source = "praw" if has_creds else "arctic"

    backends = {"arctic": collect_arctic, "praw": collect_praw, "json": collect_json}
    try:
        rows = backends[source](args.target, args.sort)
    except ImportError:
        print("[warn] praw not installed; falling back to the arctic-shift archive")
        rows = collect_arctic(args.target, args.sort)
    except Exception as e:
        print(f"[warn] {source} backend failed ({e}); falling back to the arctic-shift archive")
        rows = collect_arctic(args.target, args.sort)

    if not rows and source != "arctic":
        print("[warn] no comments from that backend; trying the arctic-shift archive")
        rows = collect_arctic(args.target, args.sort)

    if not rows:
        print("No comments collected. Check your network and try again.")
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "score", "permalink", "thread_type"])
        writer.writeheader()
        writer.writerows(rows)

    # quick summary so you can sanity check the thread mix
    by_type = {}
    for r in rows:
        by_type[r["thread_type"]] = by_type.get(r["thread_type"], 0) + 1
    print(f"\nWrote {len(rows)} comments to {OUT_PATH}")
    print("By thread type:", ", ".join(f"{k}={v}" for k, v in by_type.items()))
    print("\nNext: open the CSV, add a `label` column (analysis|hot_take|reaction)")
    print("and a `notes` column, then save as data/labeled_data.csv.")


if __name__ == "__main__":
    main()
