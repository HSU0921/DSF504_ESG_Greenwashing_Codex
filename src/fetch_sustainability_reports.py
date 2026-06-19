"""
Fetch external corporate sustainability report PDFs for inference examples.

This script is intentionally additive: it downloads report PDFs into
data_external/sustainability_reports/ and writes a small manifest. It does
not retrain models and does not touch the formal dataset or holdout split.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(PROJECT_ROOT, "data_external", "sustainability_reports")
MANIFEST_PATH = os.path.join(REPORT_DIR, "download_manifest.csv")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36 DSF504-ESG-Academic-Project"
)
TIMEOUT = 30
MIN_PDF_BYTES = 200 * 1024
PDF_KEYWORDS = ("sustainability", "impact", "esg", "climate", "report")

TARGETS = [
    ("NVDA", "NVIDIA", "https://www.nvidia.com/en-us/sustainability/"),
    ("XOM", "Exxon Mobil", "https://corporate.exxonmobil.com/sustainability-and-reports"),
    ("MSFT", "Microsoft", "https://www.microsoft.com/en-us/corporate-responsibility/sustainability/report"),
    ("TSLA", "Tesla", "https://www.tesla.com/impact"),
    ("KO", "Coca-Cola", "https://www.coca-colacompany.com/sustainability"),
    ("JPM", "JPMorgan Chase", "https://www.jpmorganchase.com/about/our-esg-approach"),
]


@dataclass
class PdfCandidate:
    url: str
    text: str
    year: int
    size_hint: int


def _session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7",
        }
    )
    return sess


def _candidate_year(text: str) -> int:
    years = [int(match) for match in re.findall(r"\b(20(?:1[8-9]|2[0-9]))\b", text)]
    return max(years) if years else 0


def _looks_like_report_pdf(href: str, link_text: str) -> bool:
    combined = f"{href} {link_text}".lower()
    if ".pdf" not in combined:
        return False
    return any(keyword in combined for keyword in PDF_KEYWORDS)


def _size_hint(session: requests.Session, url: str) -> int:
    try:
        response = session.head(url, timeout=TIMEOUT, allow_redirects=True)
        length = response.headers.get("content-length", "")
        return int(length) if str(length).isdigit() else 0
    except Exception:
        return 0


def _find_pdf_candidates(session: requests.Session, landing_url: str) -> list[PdfCandidate]:
    response = session.get(landing_url, timeout=TIMEOUT)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    candidates: dict[str, PdfCandidate] = {}
    for tag in soup.find_all("a", href=True):
        href = str(tag.get("href", "")).strip()
        text = " ".join(tag.get_text(" ", strip=True).split())
        if not _looks_like_report_pdf(href, text):
            continue
        pdf_url = urljoin(landing_url, href)
        combined = f"{pdf_url} {text}"
        candidates[pdf_url] = PdfCandidate(
            url=pdf_url,
            text=text,
            year=_candidate_year(combined),
            size_hint=_size_hint(session, pdf_url),
        )
    return sorted(candidates.values(), key=lambda item: (item.year, item.size_hint), reverse=True)


def _download_pdf(session: requests.Session, candidate: PdfCandidate, output_path: str) -> tuple[bool, int, str]:
    response = session.get(candidate.url, timeout=TIMEOUT, allow_redirects=True)
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"%PDF"):
        return False, len(content), "downloaded file does not start with %PDF"
    if len(content) < MIN_PDF_BYTES:
        return False, len(content), "downloaded PDF is smaller than 200 KB"
    with open(output_path, "wb") as f:
        f.write(content)
    return True, len(content), ""


def fetch_reports() -> pd.DataFrame:
    os.makedirs(REPORT_DIR, exist_ok=True)
    session = _session()
    rows = []
    for ticker, company, landing_url in TARGETS:
        output_path = os.path.join(REPORT_DIR, f"{ticker}.pdf")
        try:
            candidates = _find_pdf_candidates(session, landing_url)
            if not candidates:
                rows.append(
                    {
                        "ticker": ticker,
                        "company": company,
                        "status": "failed",
                        "file_size": 0,
                        "source_url": "",
                        "local_file": "",
                        "error": "no matching PDF link found on landing page",
                    }
                )
                continue

            last_error = ""
            for candidate in candidates:
                try:
                    ok, file_size, error = _download_pdf(session, candidate, output_path)
                    if ok:
                        rows.append(
                            {
                                "ticker": ticker,
                                "company": company,
                                "status": "downloaded",
                                "file_size": file_size,
                                "source_url": candidate.url,
                                "local_file": output_path,
                                "error": "",
                            }
                        )
                        break
                    last_error = f"{candidate.url}: {error}"
                except Exception as exc:
                    last_error = f"{candidate.url}: {type(exc).__name__}: {exc}"
            else:
                rows.append(
                    {
                        "ticker": ticker,
                        "company": company,
                        "status": "failed",
                        "file_size": 0,
                        "source_url": "",
                        "local_file": "",
                        "error": last_error or "candidate PDFs failed validation",
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "status": "failed",
                    "file_size": 0,
                    "source_url": "",
                    "local_file": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_PATH, index=False)
    return manifest


def main() -> None:
    manifest = fetch_reports()
    print("Sustainability report download status")
    printable = manifest[["ticker", "status", "file_size", "source_url", "error"]].copy()
    print(printable.to_string(index=False))
    print(f"\nWrote manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
