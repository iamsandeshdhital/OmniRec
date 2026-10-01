"""
Platform definitions — each social platform has its own content types,
interaction surfaces, and signal characteristics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Platform:
    name: str
    display_name: str
    content_types: List[str]
    surfaces: List[str]
    max_aspects: int
    description: str = ""
    primary_modality: str = "mixed"
    signals: List[str] = field(default_factory=list)


PLATFORMS: Dict[str, Platform] = {
    "facebook": Platform(
        name="facebook",
        display_name="Facebook",
        content_types=["post", "video", "story", "reel", "group_post", "event", "marketplace"],
        surfaces=["feed", "watch", "groups", "marketplace", "stories", "reels"],
        max_aspects=600,
        primary_modality="mixed",
        signals=["reaction", "comment", "share", "save", "follow"],
    ),
    "instagram": Platform(
        name="instagram",
        display_name="Instagram",
        content_types=["post", "story", "reel", "live", "guide", "carousel"],
        surfaces=["feed", "explore", "reels", "stories", "shop"],
        max_aspects=550,
        primary_modality="visual",
        signals=["like", "comment", "save", "share", "follow"],
    ),
    "snapchat": Platform(
        name="snapchat",
        display_name="Snapchat",
        content_types=["snap", "story", "spotlight", "map", "lens"],
        surfaces=["camera", "chat", "stories", "spotlight", "map", "discover"],
        max_aspects=400,
        primary_modality="ephemeral",
        signals=["screenshot", "reply", "share", "view"],
    ),
    "youtube": Platform(
        name="youtube",
        display_name="YouTube",
        content_types=["video", "short", "live", "community", "playlist"],
        surfaces=["home", "shorts", "subscriptions", "search", "library"],
        max_aspects=700,
        primary_modality="video",
        signals=["like", "dislike", "subscribe", "comment", "share", "save"],
    ),
    "twitter": Platform(
        name="twitter",
        display_name="Twitter / X",
        content_types=["tweet", "thread", "space", "article", "video"],
        surfaces=["timeline", "explore", "search", "notifications", "messages"],
        max_aspects=500,
        primary_modality="text",
        signals=["like", "retweet", "reply", "quote", "bookmark", "follow"],
    ),
    "netflix": Platform(
        name="netflix",
        display_name="Netflix",
        content_types=["movie", "series", "documentary", "standup", "anime"],
        surfaces=["home", "search", "my_list", "top10", "new_releases"],
        max_aspects=450,
        primary_modality="video",
        signals=["watch", "rate", "add_list", "continue", "trailer"],
    ),
    "tiktok": Platform(
        name="tiktok",
        display_name="TikTok",
        content_types=["video", "live", "story", "duet", "stitch"],
        surfaces=["foryou", "following", "search", "live", "shop"],
        max_aspects=500,
        primary_modality="video",
        signals=["like", "comment", "share", "follow", "duet", "stitch"],
    ),
    "twitch": Platform(
        name="twitch",
        display_name="Twitch",
        content_types=["stream", "clip", "vod", "event"],
        surfaces=["home", "browse", "search", "following", "esports"],
        max_aspects=400,
        primary_modality="live",
        signals=["follow", "subscribe", "clip", "chat", "raid"],
    ),
    "reddit": Platform(
        name="reddit",
        display_name="Reddit",
        content_types=["post", "comment", "ama", "gallery", "video"],
        surfaces=["home", "popular", "all", "subreddit", "search"],
        max_aspects=500,
        primary_modality="mixed",
        signals=["upvote", "downvote", "award", "comment", "share", "save"],
    ),
    "linkedin": Platform(
        name="linkedin",
        display_name="LinkedIn",
        content_types=["post", "article", "job", "newsletter", "video"],
        surfaces=["feed", "jobs", "notifications", "messaging", "search"],
        max_aspects=400,
        primary_modality="professional",
        signals=["like", "comment", "share", "follow", "connect"],
    ),
    "pinterest": Platform(
        name="pinterest",
        display_name="Pinterest",
        content_types=["pin", "board", "idea", "product"],
        surfaces=["home", "search", "following", "shop"],
        max_aspects=350,
        primary_modality="visual",
        signals=["save", "click", "follow", "buy"],
    ),
    "spotify": Platform(
        name="spotify",
        display_name="Spotify",
        content_types=["track", "album", "playlist", "podcast", "artist"],
        surfaces=["home", "search", "radio", "library", "discover"],
        max_aspects=400,
        primary_modality="audio",
        signals=["play", "save", "share", "follow", "skip"],
    ),
    "discord": Platform(
        name="discord",
        display_name="Discord",
        content_types=["message", "server", "event", "stage"],
        surfaces=["server", "dm", "discover", "stage"],
        max_aspects=300,
        primary_modality="community",
        signals=["react", "reply", "join", "mention"],
    ),
}


def get_platform(name: str) -> Optional[Platform]:
    return PLATFORMS.get(name.lower())


def all_platforms() -> List[Platform]:
    return list(PLATFORMS.values())


def platform_names() -> List[str]:
    return list(PLATFORMS.keys())


if __name__ == "__main__":
    for p in all_platforms():
        print(f"{p.display_name:14} | {p.primary_modality:12} | "
              f"{len(p.content_types)} types | {len(p.signals)} signals")
