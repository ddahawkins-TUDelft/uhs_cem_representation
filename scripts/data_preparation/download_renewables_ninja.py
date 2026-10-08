#!/usr/bin/env python3
"""
Download country-aggregated Renewables.ninja PV and wind capacity factors.

Target period
-------------
2009-04-01 00:00 UTC <= time < 2024-04-01 00:00 UTC

This yields exactly 15 complete April-March model years at hourly resolution.

Outputs
-------
One Parquet file per country:

    <output_dir>/<ISO3>.parquet

with columns:

    solar
    onshore_wind
    offshore_wind

and a timezone-aware UTC DatetimeIndex named "timesteps".

The script also writes:

    qa_summary.csv
    qa/<ISO3>.json
    metadata/<ISO3>.json
    raw_cache/

Data-quality behaviour
----------------------
* Reindex every series to a complete hourly UTC index.
* Interpolate ONLY missing runs of 1 or 2 consecutive hours.
* Do not fill longer gaps.
* Flag duplicates, remaining missing values, and values outside [0, 1].
* Keep the processed Parquet even when QA warnings remain.
* For landlocked Austria and Hungary, if Renewables.ninja supplies no offshore
  series, offshore_wind is set to 0 and this is recorded in QA.

API-rate behaviour
------------------
The script deliberately enforces conservative local limits:
* maximum 1 request per 1.05 seconds
* maximum 50 HTTP requests per rolling hour

Request timestamps are persisted to disk, so an immediate rerun still respects
the hourly cap. Metadata and downloaded files are cached to avoid unnecessary
requests.

Renewables.ninja documentation:
https://www.renewables.ninja/documentation/api
https://www.renewables.ninja/documentation/datasets

Authentication
--------------
Preferred:
    set RENEWABLES_NINJA_TOKEN=<your-token>

or:
    python download_renewables_ninja.py --token <your-token>

The token is never written to disk or printed.

Dependencies
------------
    pandas
    requests
    pyarrow

Optional, only if a returned download is a 7z archive:
    py7zr
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests


BASE_URL = "https://www.renewables.ninja"
COUNTRY_API = f"{BASE_URL}/api/countries"

START = pd.Timestamp("2009-04-01 00:00:00", tz="UTC")
END_EXCLUSIVE = pd.Timestamp("2024-04-01 00:00:00", tz="UTC")
EXPECTED_INDEX = pd.date_range(
    START,
    END_EXCLUSIVE,
    freq="h",
    inclusive="left",
    name="timesteps",
)

COUNTRIES = {
    "AUT": "AT",
    "DEU": "DE",
    "DNK": "DK",
    "ESP": "ES",
    "FIN": "FI",
    "FRA": "FR",
    "GBR": "GB",
    "HUN": "HU",
    "NLD": "NL",
    "POL": "PL",
    "SWE": "SE",
}

# If no offshore series is available for these countries, zero is physically
# appropriate rather than treating it as a data-quality gap.
LANDLOCKED_ISO3 = {"AUT", "HUN"}

MIN_SECONDS_BETWEEN_REQUESTS = 1.05
DEFAULT_MAX_REQUESTS_PER_HOUR = 50
MAX_RETRIES = 4


@dataclass
class SeriesQA:
    source_url: str | None = None
    source_column: str | None = None
    raw_rows: int = 0
    duplicate_timestamps: int = 0
    missing_before_interpolation: int = 0
    interpolated_short_gap_hours: int = 0
    missing_after_interpolation: int = 0
    below_zero_count: int = 0
    above_one_count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    note: str | None = None


class DiscoveryError(RuntimeError):
    pass


class PersistentRateLimiter:
    """
    Conservative rolling-window limiter persisted across script invocations.
    """

    def __init__(
        self,
        state_path: Path,
        max_per_hour: int = DEFAULT_MAX_REQUESTS_PER_HOUR,
        min_interval_seconds: float = MIN_SECONDS_BETWEEN_REQUESTS,
    ) -> None:
        self.state_path = state_path
        self.max_per_hour = max_per_hour
        self.min_interval_seconds = min_interval_seconds
        self.timestamps = self._load()

    def _load(self) -> list[float]:
        if not self.state_path.exists():
            return []
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return [float(x) for x in payload.get("request_timestamps", [])]
        except Exception:
            return []

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"request_timestamps": self.timestamps}, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.state_path)

    def _prune(self, now: float) -> None:
        cutoff = now - 3600.0
        self.timestamps = [x for x in self.timestamps if x > cutoff]

    def wait_for_slot(self) -> None:
        while True:
            now = time.time()
            self._prune(now)

            # Burst limit.
            if self.timestamps:
                since_last = now - self.timestamps[-1]
                if since_last < self.min_interval_seconds:
                    time.sleep(self.min_interval_seconds - since_last)
                    continue

            # Rolling hourly limit.
            if len(self.timestamps) >= self.max_per_hour:
                wait = (self.timestamps[0] + 3600.0) - now + 0.25
                if wait > 0:
                    print(
                        f"[rate-limit] Local {self.max_per_hour}/hour ceiling "
                        f"reached; sleeping {wait:.1f} s.",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue

            return

    def record(self) -> None:
        now = time.time()
        self._prune(now)
        self.timestamps.append(now)
        self._save()


def build_session(token: str | None) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "UHS-energy-system-research/1.0 "
                "(Renewables.ninja country-data downloader)"
            )
        }
    )
    if token:
        session.headers["Authorization"] = f"Token {token}"
    return session


def request_with_limits(
    session: requests.Session,
    limiter: PersistentRateLimiter,
    url: str,
    *,
    timeout: int = 120,
) -> requests.Response:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        limiter.wait_for_slot()
        limiter.record()

        try:
            response = session.get(url, timeout=timeout)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(60 * attempt, 300)
                print(
                    f"[429] Server rate limit for {url}; sleeping {wait:.0f} s.",
                    flush=True,
                )
                time.sleep(wait)
                continue

            if response.status_code >= 500:
                wait = min(5 * attempt, 30)
                print(
                    f"[{response.status_code}] Temporary server error for {url}; "
                    f"retrying in {wait} s.",
                    flush=True,
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response

        except requests.RequestException as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait = min(5 * attempt, 30)
            print(f"[request-error] {exc}; retrying in {wait} s.", flush=True)
            time.sleep(wait)

    raise RuntimeError(f"Request failed after {MAX_RETRIES} attempts: {url}") from last_error


def cached_json_request(
    session: requests.Session,
    limiter: PersistentRateLimiter,
    url: str,
    cache_path: Path,
    *,
    refresh: bool,
) -> Any:
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    response = request_with_limits(session, limiter, url)
    payload = response.json()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def looks_like_download_reference(key: str, value: str) -> bool:
    """
    Return True only for plausible dataset-download references.

    Country metadata also contains citation_url and license_url fields. Those
    must NOT enter dataset discovery simply because their surrounding metadata
    mentions PV or wind.
    """
    key_lower = key.lower()
    val_lower = value.lower()

    # Explicitly reject non-data links commonly present in metadata.
    if any(
        token in key_lower
        for token in ("citation", "license", "doi", "paper", "reference")
    ):
        return False
    if any(
        token in val_lower
        for token in (
            "creativecommons.org",
            "doi.org",
            "sciencedirect.com",
            "researchgate.net",
        )
    ):
        return False

    # Renewables.ninja country downloads use this path.
    if "/country_downloads/" in val_lower:
        return True

    # Also permit direct archive / CSV links in case their URL layout changes.
    clean = val_lower.split("?", 1)[0]
    if clean.endswith((".csv", ".zip", ".gz", ".7z")):
        return True

    # Relative paths explicitly stored under download/file-like keys.
    if any(token in key_lower for token in ("download", "file")):
        return value.startswith(("/", "http://", "https://"))

    return False


def extract_download_candidates(payload: Any) -> list[dict[str, str]]:
    """
    Recursively inspect country metadata without assuming a fixed API JSON schema.

    Each candidate keeps nearby scalar values as context, which lets us identify
    the v1.4 national MERRA-2 PV, current-onshore wind, and current-offshore wind downloads.
    """
    candidates: list[dict[str, str]] = []

    def walk(obj: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(obj, dict):
            sibling_context_parts: list[str] = []
            for k, v in obj.items():
                if isinstance(v, (str, int, float, bool)) and not (
                    isinstance(v, str) and looks_like_download_reference(str(k), v)
                ):
                    sibling_context_parts.append(f"{k}={v}")
            sibling_context = " ".join(sibling_context_parts)

            for key, value in obj.items():
                key_s = str(key)
                if isinstance(value, str) and looks_like_download_reference(key_s, value):
                    # Some metadata strings are not URLs despite key names like "file".
                    # urljoin handles relative download paths.
                    resolved = urljoin(BASE_URL + "/", value)
                    candidates.append(
                        {
                            "url": resolved,
                            "context": " ".join(
                                [*path, key_s, sibling_context, value]
                            ).strip(),
                        }
                    )
                else:
                    walk(value, (*path, key_s))

        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(value, (*path, str(i)))

    walk(payload)

    # Deduplicate by URL while preserving richest context.
    merged: dict[str, str] = {}
    for item in candidates:
        url = item["url"]
        context = item["context"]
        if url not in merged or len(context) > len(merged[url]):
            merged[url] = context

    return [{"url": u, "context": c} for u, c in merged.items()]


def score_candidate(
    candidate: dict[str, str],
    kind: str,
    *,
    wind_subtype: str | None = None,
) -> int:
    text = f"{candidate['url']} {candidate['context']}".lower()
    score = 0

    # Prefer current country-aggregated MERRA-2 datasets.
    for token, weight in (
        ("v1.4", 8),
        ("version=1.4", 8),
        ("version 1.4", 8),
        ("merra", 8),
        ("national", 4),
        ("country", 3),
    ):
        if token in text:
            score += weight

    # Reject clearly unrelated downloads.
    for token, penalty in (
        ("nuts", -30),
        ("subnational", -30),
        ("weather", -20),
        ("demand", -20),
    ):
        if token in text:
            score += penalty

    if kind == "pv":
        if "pv" in text:
            score += 20
        if "solar" in text:
            score += 10
        if "sarah" in text:
            score -= 10  # Prefer MERRA-2 for consistency with wind.
        if "wind" in text:
            score -= 50

    elif kind == "wind":
        if "wind" in text:
            score += 20
        if "current" in text:
            score += 10
        if "pv" in text or "solar" in text:
            score -= 50

        if wind_subtype == "onshore":
            if "onshore" in text:
                score += 40
            if "offshore" in text or "total" in text:
                score -= 40

        elif wind_subtype == "offshore":
            if "offshore" in text:
                score += 40
            if "onshore" in text or "total" in text:
                score -= 40

    return score


def select_download_candidate(
    payload: Any,
    kind: str,
    *,
    country_iso3: str,
    wind_subtype: str | None = None,
    allow_missing: bool = False,
) -> dict[str, str] | None:
    candidates = extract_download_candidates(payload)

    scored = [
        (
            score_candidate(
                candidate,
                kind,
                wind_subtype=wind_subtype,
            ),
            candidate,
        )
        for candidate in candidates
    ]
    scored = [(score, candidate) for score, candidate in scored if score > 0]
    scored.sort(key=lambda item: item[0], reverse=True)

    label = kind if wind_subtype is None else f"{wind_subtype} wind"

    if not scored:
        if allow_missing:
            return None
        raise DiscoveryError(
            f"{country_iso3}: could not discover a plausible {label} download URL."
        )

    best_score = scored[0][0]
    best = [candidate for score, candidate in scored if score == best_score]

    if len(best) > 1:
        details = "\n".join(
            f"  score={best_score}: {candidate['url']} | {candidate['context']}"
            for candidate in best
        )
        raise DiscoveryError(
            f"{country_iso3}: ambiguous {label} dataset discovery.\n{details}"
        )

    return best[0]


def cache_filename_from_url(url: str, prefix: str) -> str:
    parsed = urlparse(url)
    basename = Path(parsed.path).name or "download"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}_{basename}"


def cached_binary_request(
    session: requests.Session,
    limiter: PersistentRateLimiter,
    url: str,
    cache_dir: Path,
    prefix: str,
    *,
    refresh: bool,
) -> tuple[bytes, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / cache_filename_from_url(url, prefix)

    if path.exists() and not refresh:
        return path.read_bytes(), path

    response = request_with_limits(session, limiter, url)
    data = response.content
    path.write_bytes(data)
    return data, path


def unpack_download(data: bytes, filename_hint: str) -> tuple[bytes, str]:
    """
    Return CSV bytes and an informative inner filename.
    """
    lower = filename_hint.lower()

    if data.startswith(b"PK\x03\x04") or lower.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = [x for x in zf.namelist() if x.lower().endswith(".csv")]
            if not members:
                raise ValueError("ZIP archive contains no CSV file.")
            # Per-country archives should contain one CSV; if not, use the largest.
            member = max(members, key=lambda x: zf.getinfo(x).file_size)
            return zf.read(member), member

    if data.startswith(b"\x1f\x8b") or lower.endswith(".gz"):
        return gzip.decompress(data), filename_hint.removesuffix(".gz")

    if data.startswith(b"7z\xbc\xaf'\x1c") or lower.endswith(".7z"):
        try:
            import py7zr  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "A 7z archive was returned. Install optional dependency `py7zr` "
                "and rerun; the cached archive will be reused."
            ) from exc

        with py7zr.SevenZipFile(io.BytesIO(data), mode="r") as archive:
            extracted = archive.readall()
            csv_items = [
                (name, stream)
                for name, stream in extracted.items()
                if name.lower().endswith(".csv")
            ]
            if not csv_items:
                raise ValueError("7z archive contains no CSV file.")
            name, stream = csv_items[0]
            return stream.read(), name

    return data, filename_hint


def parse_country_csv(csv_bytes: bytes) -> tuple[pd.DataFrame, list[str]]:
    """
    Parse Renewables.ninja country CSV while preserving header comments.
    """
    text = csv_bytes.decode("utf-8-sig")
    lines = text.splitlines()
    comments = [line for line in lines if line.lstrip().startswith("#")]

    # Country files historically have two comment rows, while API-format files
    # may have three. Find the actual header instead of hard-coding a skip count.
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip().strip('"').lower()
        if stripped == "time" or stripped.startswith("time,"):
            header_idx = i
            break

    if header_idx is None:
        # Fallback: pandas comment handling, useful if the exact CSV formatting
        # differs slightly from historical examples.
        frame = pd.read_csv(io.StringIO(text), comment="#")
    else:
        frame = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))

    if "time" not in frame.columns:
        # Be forgiving about whitespace/case.
        mapping = {
            col: col.strip().lower()
            for col in frame.columns
        }
        frame = frame.rename(columns=mapping)

    if "time" not in frame.columns:
        raise ValueError(f"Could not find a `time` column. Columns: {list(frame.columns)}")

    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    if frame["time"].isna().any():
        raise ValueError("Some Renewables.ninja timestamps could not be parsed.")

    frame = frame.set_index("time").sort_index()
    frame.index.name = "timesteps"
    return frame, comments


def normalise_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def choose_column(
    frame: pd.DataFrame,
    *,
    exact_names: tuple[str, ...] = (),
    contains: tuple[str, ...] = (),
    fallback_single: bool = False,
) -> str:
    normalised = {col: normalise_column_name(col) for col in frame.columns}

    for wanted in exact_names:
        wanted_n = normalise_column_name(wanted)
        for col, norm in normalised.items():
            if norm == wanted_n:
                return col

    for token in contains:
        token_n = normalise_column_name(token)
        for col, norm in normalised.items():
            if token_n in norm:
                return col

    if fallback_single and len(frame.columns) == 1:
        return frame.columns[0]

    raise KeyError(
        f"Could not identify requested column. Available columns: "
        f"{list(frame.columns)}"
    )


def extract_pv_series(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    col = choose_column(
        frame,
        exact_names=("national", "pv", "solar"),
        contains=("national",),
        fallback_single=True,
    )
    return pd.to_numeric(frame[col], errors="coerce"), str(col)


def extract_national_series(
    frame: pd.DataFrame,
    *,
    preferred_names: tuple[str, ...] = ("national",),
) -> tuple[pd.Series, str]:
    """
    Extract the capacity-factor series from one country-aggregated file.

    Renewables.ninja currently provides current onshore and current offshore
    wind as separate downloads. Each file is therefore expected to contain a
    single `national` capacity-factor column (plus the time column).
    """
    col = choose_column(
        frame,
        exact_names=preferred_names,
        contains=("national",),
        fallback_single=True,
    )
    return pd.to_numeric(frame[col], errors="coerce"), str(col)


def collapse_duplicate_timestamps(series: pd.Series) -> tuple[pd.Series, int]:
    duplicate_count = int(series.index.duplicated(keep=False).sum())
    if duplicate_count:
        # Mean is safe for capacity factors and preserves information if the
        # duplicate values differ. The issue is still flagged in QA.
        series = series.groupby(level=0).mean()
    return series.sort_index(), duplicate_count


def interpolate_only_short_gaps(
    series: pd.Series,
    max_gap_hours: int = 2,
) -> tuple[pd.Series, int]:
    """
    Interpolate complete missing runs only when the run length <= max_gap_hours.

    Unlike pandas `limit=2`, this does NOT fill the first two hours of a longer
    missing run.
    """
    s = series.astype(float).copy()
    missing = s.isna()

    if not missing.any():
        return s, 0

    run_id = missing.ne(missing.shift(fill_value=False)).cumsum()
    run_lengths = missing.groupby(run_id).transform("sum")
    eligible = missing & (run_lengths <= max_gap_hours)

    candidate = s.interpolate(method="time", limit_area="inside")
    fillable = eligible & candidate.notna()

    s.loc[fillable] = candidate.loc[fillable]
    return s, int(fillable.sum())


def process_series(
    raw: pd.Series,
    *,
    source_url: str | None,
    source_column: str | None,
    note: str | None = None,
) -> tuple[pd.Series, SeriesQA]:
    qa = SeriesQA(
        source_url=source_url,
        source_column=source_column,
        raw_rows=len(raw),
        note=note,
    )

    raw, qa.duplicate_timestamps = collapse_duplicate_timestamps(raw)

    # Restrict and then reindex to a canonical hourly UTC horizon.
    raw = raw.loc[(raw.index >= START) & (raw.index < END_EXCLUSIVE)]
    aligned = raw.reindex(EXPECTED_INDEX)

    qa.missing_before_interpolation = int(aligned.isna().sum())

    aligned, qa.interpolated_short_gap_hours = interpolate_only_short_gaps(
        aligned,
        max_gap_hours=2,
    )

    qa.missing_after_interpolation = int(aligned.isna().sum())
    qa.below_zero_count = int((aligned < 0).sum())
    qa.above_one_count = int((aligned > 1).sum())

    finite = aligned.dropna()
    if not finite.empty:
        qa.minimum = float(finite.min())
        qa.maximum = float(finite.max())

    return aligned, qa


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def check_server_limits(
    session: requests.Session,
    limiter: PersistentRateLimiter,
    cache_dir: Path,
    *,
    refresh: bool,
) -> None:
    """
    Query /api/limits once for visibility, while still enforcing our conservative
    local 50/hour and 1/s ceilings regardless of the returned account limits.
    """
    try:
        payload = cached_json_request(
            session,
            limiter,
            f"{BASE_URL}/api/limits",
            cache_dir / "api_limits.json",
            refresh=refresh,
        )
        print("[limits] Renewables.ninja reports:")
        print(json.dumps(payload, indent=2))
        print(
            f"[limits] This script will nevertheless enforce <= "
            f"{limiter.max_per_hour}/hour and >= "
            f"{limiter.min_interval_seconds:.2f} s between requests."
        )
    except Exception as exc:
        print(
            f"[warning] Could not query /api/limits ({exc}). "
            "Continuing with conservative local limits.",
            file=sys.stderr,
        )


def download_country(
    *,
    iso3: str,
    iso2: str,
    session: requests.Session,
    limiter: PersistentRateLimiter,
    root: Path,
    refresh: bool,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    metadata_dir = root / "metadata"
    raw_dir = root / "raw_cache"
    qa_dir = root / "qa"

    qa_payload: dict[str, Any] = {
        "country_iso3": iso3,
        "country_iso2": iso2,
        "expected_start_utc": str(START),
        "expected_end_exclusive_utc": str(END_EXCLUSIVE),
        "expected_hours": len(EXPECTED_INDEX),
        "status": "ok",
        "issues": [],
        "series": {},
    }

    print(f"\n[{iso3}] Fetching/discovering country datasets...")

    try:
        metadata = cached_json_request(
            session,
            limiter,
            f"{COUNTRY_API}/{iso2}",
            metadata_dir / f"{iso3}_country_api.json",
            refresh=refresh,
        )

        pv_candidate = select_download_candidate(
            metadata,
            "pv",
            country_iso3=iso3,
        )
        onshore_candidate = select_download_candidate(
            metadata,
            "wind",
            country_iso3=iso3,
            wind_subtype="onshore",
        )
        if iso3 in LANDLOCKED_ISO3:
            # Austria and Hungary are landlocked: there is no meaningful
            # offshore-wind availability profile to discover.
            offshore_candidate = None
        else:
            offshore_candidate = select_download_candidate(
                metadata,
                "wind",
                country_iso3=iso3,
                wind_subtype="offshore",
            )

        assert pv_candidate is not None
        assert onshore_candidate is not None

        qa_payload["pv_candidate"] = pv_candidate
        qa_payload["onshore_wind_candidate"] = onshore_candidate
        qa_payload["offshore_wind_candidate"] = offshore_candidate

        # Download PV and onshore wind for every country.
        pv_bytes, pv_cache = cached_binary_request(
            session,
            limiter,
            pv_candidate["url"],
            raw_dir,
            f"{iso3}_pv",
            refresh=refresh,
        )
        onshore_bytes, onshore_cache = cached_binary_request(
            session,
            limiter,
            onshore_candidate["url"],
            raw_dir,
            f"{iso3}_wind_onshore",
            refresh=refresh,
        )

        pv_csv, pv_inner_name = unpack_download(pv_bytes, pv_cache.name)
        onshore_csv, onshore_inner_name = unpack_download(
            onshore_bytes,
            onshore_cache.name,
        )

        pv_frame, pv_comments = parse_country_csv(pv_csv)
        onshore_frame, onshore_comments = parse_country_csv(onshore_csv)

        solar_raw, solar_col = extract_pv_series(pv_frame)
        onshore_raw, onshore_col = extract_national_series(onshore_frame)

        solar, solar_qa = process_series(
            solar_raw,
            source_url=pv_candidate["url"],
            source_column=solar_col,
        )
        onshore, onshore_qa = process_series(
            onshore_raw,
            source_url=onshore_candidate["url"],
            source_column=onshore_col,
        )

        offshore_comments: list[str] = []
        offshore_cache: Path | None = None
        offshore_inner_name: str | None = None

        if offshore_candidate is None and iso3 in LANDLOCKED_ISO3:
            # No meaningful offshore resource for landlocked countries.
            offshore = pd.Series(
                0.0,
                index=EXPECTED_INDEX,
                name="offshore_wind",
            )
            offshore_qa = SeriesQA(
                source_url=None,
                source_column=None,
                raw_rows=0,
                minimum=0.0,
                maximum=0.0,
                note=(
                    "Country is landlocked and no offshore dataset was supplied; "
                    "offshore_wind set to 0 for the full horizon."
                ),
            )
        else:
            if offshore_candidate is None:
                raise DiscoveryError(
                    f"{iso3}: no offshore wind dataset found for a "
                    "non-landlocked country."
                )

            offshore_bytes, offshore_cache = cached_binary_request(
                session,
                limiter,
                offshore_candidate["url"],
                raw_dir,
                f"{iso3}_wind_offshore",
                refresh=refresh,
            )
            offshore_csv, offshore_inner_name = unpack_download(
                offshore_bytes,
                offshore_cache.name,
            )
            offshore_frame, offshore_comments = parse_country_csv(offshore_csv)
            offshore_raw, offshore_col = extract_national_series(offshore_frame)

            offshore, offshore_qa = process_series(
                offshore_raw,
                source_url=offshore_candidate["url"],
                source_column=offshore_col,
            )

        output = pd.DataFrame(
            {
                "solar": solar,
                "onshore_wind": onshore,
                "offshore_wind": offshore,
            },
            index=EXPECTED_INDEX,
        )
        output.index.name = "timesteps"

        qa_payload["series"] = {
            "solar": asdict(solar_qa),
            "onshore_wind": asdict(onshore_qa),
            "offshore_wind": asdict(offshore_qa),
        }
        qa_payload["source_metadata_comments"] = {
            "pv": pv_comments,
            "onshore_wind": onshore_comments,
            "offshore_wind": offshore_comments,
        }
        qa_payload["cached_files"] = {
            "pv": str(pv_cache),
            "onshore_wind": str(onshore_cache),
            "offshore_wind": (
                str(offshore_cache) if offshore_cache is not None else None
            ),
            "pv_inner_name": pv_inner_name,
            "onshore_wind_inner_name": onshore_inner_name,
            "offshore_wind_inner_name": offshore_inner_name,
        }

        # Flag, but do not reject, residual quality issues.
        for name, q in (
            ("solar", solar_qa),
            ("onshore_wind", onshore_qa),
            ("offshore_wind", offshore_qa),
        ):
            if q.duplicate_timestamps:
                qa_payload["issues"].append(
                    f"{name}: {q.duplicate_timestamps} duplicate timestamp rows "
                    "collapsed by mean."
                )
            if q.missing_after_interpolation:
                qa_payload["issues"].append(
                    f"{name}: {q.missing_after_interpolation} hourly values "
                    "remain missing after <=2 h interpolation."
                )
            if q.below_zero_count or q.above_one_count:
                qa_payload["issues"].append(
                    f"{name}: values outside [0,1] detected "
                    f"(below={q.below_zero_count}, above={q.above_one_count})."
                )

        if qa_payload["issues"]:
            qa_payload["status"] = "warning"

        output_path = root / f"{iso3}.parquet"
        output.to_parquet(output_path, engine="pyarrow", index=True)
        qa_payload["output_path"] = str(output_path)

        write_json(qa_dir / f"{iso3}.json", qa_payload)

        print(
            f"[{iso3}] saved {output_path.name}: "
            f"{len(output):,} hourly rows, "
            f"status={qa_payload['status']}"
        )
        if qa_payload["issues"]:
            for issue in qa_payload["issues"]:
                print(f"  WARNING: {issue}")

        return output, qa_payload

    except Exception as exc:
        qa_payload["status"] = "error"
        qa_payload["issues"].append(f"{type(exc).__name__}: {exc}")
        write_json(qa_dir / f"{iso3}.json", qa_payload)
        print(f"[{iso3}] ERROR: {exc}", file=sys.stderr)
        return None, qa_payload


def build_qa_summary(all_qa: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for item in all_qa:
        row: dict[str, Any] = {
            "country": item["country_iso3"],
            "status": item["status"],
            "issues": " | ".join(item.get("issues", [])),
            "expected_hours": item.get("expected_hours"),
        }

        for series_name in ("solar", "onshore_wind", "offshore_wind"):
            q = item.get("series", {}).get(series_name, {})
            for field in (
                "missing_before_interpolation",
                "interpolated_short_gap_hours",
                "missing_after_interpolation",
                "duplicate_timestamps",
                "below_zero_count",
                "above_one_count",
                "minimum",
                "maximum",
            ):
                row[f"{series_name}_{field}"] = q.get(field)

        rows.append(row)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Renewables.ninja national solar/onshore/offshore capacity "
            "factors for Apr-2009 through Mar-2024."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/renewables_ninja"),
        help="Output/cache directory (default: data/renewables_ninja).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Renewables.ninja API token. Prefer setting "
            "RENEWABLES_NINJA_TOKEN instead so the token is not in shell history."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached API metadata/downloads and fetch them again.",
    )
    parser.add_argument(
        "--skip-limit-check",
        action="store_true",
        help="Do not make the optional /api/limits request.",
    )
    parser.add_argument(
        "--max-requests-per-hour",
        type=int,
        default=DEFAULT_MAX_REQUESTS_PER_HOUR,
        help=(
            "Local rolling request ceiling. Default is deliberately conservative "
            f"at {DEFAULT_MAX_REQUESTS_PER_HOUR}/hour."
        ),
    )
    parser.add_argument(
        "--countries",
        nargs="*",
        choices=sorted(COUNTRIES),
        default=sorted(COUNTRIES),
        help="Optional subset of ISO3 countries to download.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.output_dir
    root.mkdir(parents=True, exist_ok=True)

    token = args.token or os.getenv("RENEWABLES_NINJA_TOKEN")
    if not token:
        print(
            "[warning] No API token found. Attempting anonymous access. "
            "For reproducible full-history access, set RENEWABLES_NINJA_TOKEN.",
            file=sys.stderr,
        )

    limiter = PersistentRateLimiter(
        root / ".renewables_ninja_rate_state.json",
        max_per_hour=args.max_requests_per_hour,
        min_interval_seconds=MIN_SECONDS_BETWEEN_REQUESTS,
    )
    session = build_session(token)

    print(
        f"Target: {START} <= time < {END_EXCLUSIVE} "
        f"({len(EXPECTED_INDEX):,} hourly UTC timesteps)"
    )
    print(
        f"Local request limits: <= {args.max_requests_per_hour}/hour, "
        f">= {MIN_SECONDS_BETWEEN_REQUESTS:.2f} s between HTTP requests."
    )

    if not args.skip_limit_check:
        check_server_limits(
            session,
            limiter,
            root / "metadata",
            refresh=args.refresh,
        )

    all_qa: list[dict[str, Any]] = []

    for iso3 in args.countries:
        _, qa = download_country(
            iso3=iso3,
            iso2=COUNTRIES[iso3],
            session=session,
            limiter=limiter,
            root=root,
            refresh=args.refresh,
        )
        all_qa.append(qa)

    summary = build_qa_summary(all_qa)
    summary_path = root / "qa_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nQA summary")
    print("----------")
    print(summary[["country", "status", "issues"]].to_string(index=False))
    print(f"\nSaved QA summary: {summary_path}")

    error_count = int((summary["status"] == "error").sum())
    warning_count = int((summary["status"] == "warning").sum())

    print(
        f"Completed: {len(summary) - error_count} country files created, "
        f"{warning_count} with QA warnings, {error_count} errors."
    )

    # Do not fail merely because QA warnings remain: the user explicitly wants
    # the downloads retained for later repair. Return non-zero only when an
    # entire country failed to download/process.
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
