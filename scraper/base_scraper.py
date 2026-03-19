"""
Base scraper module for bank websites.
Abstract base class that handles:
- Playwright browser lifecycle
- Page rendering and tab clicking
- Content saving (JSON + plain text)
- Summary reporting and saving

Each bank subclass defines its own URLs and extraction logic.
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


class BaseBankScraper(ABC):
    """Abstract base class for scraping Armenian bank websites."""
    @property
    @abstractmethod
    def bank_name(self) -> str:
        """Bank name in Armenian."""
        pass

    @property
    @abstractmethod
    def bank_name_en(self) -> str:
        """Bank name in English."""
        pass

    @property
    @abstractmethod
    def website(self) -> str:
        """Bank website URL."""
        pass

    @abstractmethod
    def get_loan_urls(self) -> dict:
        """Return dict of {key: url} for all loan pages."""
        pass

    @abstractmethod
    def get_deposit_urls(self) -> dict:
        """Return dict of {key: url} for all deposit/savings pages."""
        pass

    @abstractmethod
    def get_branch_url(self) -> str:
        """Return the URL of the branch/service-network page."""
        pass

    @abstractmethod
    def extract_page_content(self, page_html: str, url: str) -> dict:
        """
        Extract content from a rendered page.
        returns dict with keys: title, url, content, tables
        """
        pass

    @abstractmethod
    def extract_branches(self, page_html: str) -> list:
        """
        Extract branch data from the rendered branch page.
        returns a list of dicts, each with keys:
        name, address, phone, schedule, extended_hours, description
        """
        pass

    # Shared utilities over child classes

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove excessive whitespace and blank lines."""
        text = re.sub(r'[^\S\n]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text.strip()

    @staticmethod
    def postprocess_content(text: str) -> str:
        """
        Post-process extracted content to make it more LLM-friendly.
        Called before results saving.

        Steps:
        1. Strip leftover HTML entities
        2. Remove orphan single-character lines
        3. Merge fragmented short lines into paragraphs
        4. Deduplicate identical content blocks
        5. Collapse excessive whitespace
        """
        if not text:
            return text

        # 1. Replace common HTML entities and their leftovers
        text = text.replace('\xa0', ' ')  # &nbsp;
        text = text.replace('\u200b', '')  # zero-width space
        text = text.replace('\u00ad', '')  # soft hyphen

        # 2. Remove lines that are only whitespace, single characters, or only punctuation
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip empty lines
            if not stripped:
                cleaned_lines.append('')
                continue
            # Skip lines that are just a single character
            if len(stripped) <= 1 and not stripped.isdigit():
                continue
            # Skip lines that are only non-breaking spaces or dots
            if all(c in ' .\u00a0' for c in stripped):
                continue
            cleaned_lines.append(stripped)

        text = '\n'.join(cleaned_lines)

        # 3. Merge fragmented short lines:
        lines = text.split('\n')
        merged = []
        i = 0
        while i < len(lines):
            line = lines[i]

            # Keep blank lines, section headers, and labeled data as-is
            is_header = line.startswith('[') or line.startswith('#')
            is_numbered = bool(re.match(r'^\d+\.', line))
            is_labeled = ':' in line and len(line) < 80
            is_bullet = line.startswith('Q:') or line.startswith('A:')

            if not line or is_header or is_numbered or is_labeled or is_bullet:
                merged.append(line)
                i += 1
                continue

            # For short lines, look ahead and merge with next short line
            if len(line) < 40 and i + 1 < len(lines):
                next_line = lines[i + 1]
                next_stripped = next_line.strip()
                # Merge if next line is also short and not a special line
                if (next_stripped
                        and len(next_stripped) < 40
                        and not next_stripped.startswith('[')
                        and not re.match(r'^\d+\.', next_stripped)
                        and ':' not in next_stripped):
                    merged.append(line + ' ' + next_stripped)
                    i += 2
                    continue

            merged.append(line)
            i += 1

        text = '\n'.join(merged)

        # 4. Deduplicate identical content blocks.
        #    Split by double-newline, remove exact dupes.
        blocks = text.split('\n\n')
        seen = set()
        unique_blocks = []
        for block in blocks:
            block_stripped = block.strip()
            if not block_stripped:
                continue
            if block_stripped in seen:
                continue
            seen.add(block_stripped)
            unique_blocks.append(block_stripped)

        text = '\n\n'.join(unique_blocks)

        # 5. Whitespace cleanup
        text = re.sub(r'[^\S\n]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        return text.strip()

    def _create_browser_context(self, playwright):
        """Create a Playwright browser and context with settings."""
        browser = playwright.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        context = browser.new_context(
            locale='hy-AM',
            viewport={'width': 1920, 'height': 1080},
            user_agent=(
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        )
        return browser, context

    def _render_page(self, context, url: str, click_tabs: bool = True) -> str:
        """
        Navigate to a URL, wait for JS rendering, click tabs,
        and return the fully rendered HTML.
        """
        page = context.new_page()

        page.goto(url, wait_until='domcontentloaded', timeout=30000)
        page.wait_for_timeout(5000)

        if click_tabs:
            # Click tab navigation links to trigger content rendering
            try:
                tabs = page.query_selector_all('.tabs-navigation a')
                for tab in tabs:
                    try:
                        tab.click()
                        page.wait_for_timeout(1000)
                    except Exception:
                        pass
            except Exception:
                pass

            # Expand accordion sections
            try:
                accordion_selectors = [
                    '.cs-accordion__title',
                    '.accordion-toggle',
                    '[data-toggle="collapse"]',
                    'details summary',
                ]
                for selector in accordion_selectors:
                    buttons = page.query_selector_all(selector)
                    for btn in buttons:
                        try:
                            btn.click()
                            page.wait_for_timeout(500)
                        except Exception:
                            pass
            except Exception:
                pass

        html_content = page.content()
        page.close()
        return html_content

    def _save_json(self, data: dict, filepath: str):
        """Save data as JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON: {filepath}")

    def _save_content_txt(self, pages_data: dict, category: str, filepath: str):
        """Save content pages as plain text."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Bank: {self.bank_name_en}\n")
            f.write(f"Website: {self.website}\n")
            f.write(f"Category: {category}\n")
            f.write("=" * 80 + "\n\n")

            for key, page_data in pages_data.items():
                f.write(f"--- {page_data['title']} ---\n")
                f.write(f"URL: {page_data['url']}\n\n")
                f.write(page_data['content'])
                f.write("\n\n" + "=" * 80 + "\n\n")

        print(f"Saved TXT:  {filepath}")

    def _save_branches_txt(self, branches: list, source_url: str, filepath: str):
        """Save branch data as plain text."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Bank: {self.bank_name_en}\n")
            f.write(f"Website: {self.website}\n")
            f.write(f"Category: Branches\n")
            f.write(f"Source: {source_url}\n")
            f.write("=" * 80 + "\n\n")

            for i, branch in enumerate(branches, 1):
                f.write(f"{i}. {branch['name']}\n")
                f.write(f"   Address: {branch['address']}\n")
                f.write(f"   Phone: {branch['phone']}\n")
                f.write(f"   Schedule: {branch['schedule']}\n")
                if branch.get('description'):
                    f.write(f"   Note: {branch['description']}\n")
                f.write("\n")

        print(f"Saved TXT:  {filepath}")

    def _print_content_summary(self, pages_data: dict, category: str):
        """Print summary stats for content scraping."""
        total_chars = sum(len(p.get('content', '')) for p in pages_data.values())
        total_tables = sum(len(p.get('tables', [])) for p in pages_data.values())
        errors = sum(1 for p in pages_data.values() if p.get('error'))

        print(f"\n{category.upper()} SUMMARY")
        print(f"Pages scraped: {len(pages_data)}")
        print(f"Total content: {total_chars:,} characters")
        print(f"Total tables: {total_tables}")
        print(f"Errors: {errors}")

    # Main scraping methods

    def _scrape_content_pages(self, urls: dict, category: str, output_dir: str):
        """
        Scrape a set of content pages (loans or deposits).
        Shared logic for any category that follows the pattern:
        loop over URLs -> render -> extract content.
        """
        os.makedirs(output_dir, exist_ok=True)
        slug = self.bank_name_en.lower().replace(' ', '_')

        all_data = {
            "bank_name": self.bank_name,
            "bank_name_en": self.bank_name_en,
            "website": self.website,
            "category": category,
            "scraped_at": datetime.now().isoformat(),
            "pages": {}
        }

        with sync_playwright() as p:
            browser, context = self._create_browser_context(p)

            total = len(urls)
            for idx, (key, url) in enumerate(urls.items(), 1):
                print(f"[{idx}/{total}] Scraping: {key}")
                print(f"URL: {url}")

                try:
                    html_content = self._render_page(context, url)
                    page_data = self.extract_page_content(html_content, url)

                    # Post-process content for LLM readability
                    page_data['content'] = self.postprocess_content(
                        page_data.get('content', '')
                    )

                    all_data["pages"][key] = page_data
                    print(f"  Extracted {len(page_data['content'])} chars, "
                          f"{len(page_data['tables'])} tables")
                except Exception as e:
                    print(f"  Error: {e}")
                    all_data["pages"][key] = {
                        "title": key,
                        "url": url,
                        "content": "",
                        "tables": [],
                        "error": str(e)
                    }

                time.sleep(1)

            browser.close()

        # Save outputs
        json_path = os.path.join(output_dir, f"{slug}_{category}.json")
        txt_path = os.path.join(output_dir, f"{slug}_{category}.txt")

        self._save_json(all_data, json_path)
        self._save_content_txt(all_data["pages"], category, txt_path)
        self._print_content_summary(all_data["pages"], category)

        return all_data

    def scrape_loans(self, output_dir: str = "data"):
        """Scrape all loan pages."""
        urls = self.get_loan_urls()
        if not urls:
            print(f"No loan URLs defined for {self.bank_name_en}, skipping.")
            return None
        print(f"\nSCRAPING LOANS: {self.bank_name_en}\n")
        return self._scrape_content_pages(urls, "loans", output_dir)

    def scrape_deposits(self, output_dir: str = "data"):
        """Scrape all deposit/savings pages."""
        urls = self.get_deposit_urls()
        if not urls:
            print(f"No deposit URLs defined for {self.bank_name_en}, skipping.")
            return None
        print(f"\nSCRAPING DEPOSITS: {self.bank_name_en}\n")
        return self._scrape_content_pages(urls, "deposits", output_dir)

    def scrape_branches(self, output_dir: str = "data"):
        """Scrape branch locations."""
        branch_url = self.get_branch_url()
        if not branch_url:
            print(f"No branch URL defined for {self.bank_name_en}, skipping.")
            return None

        os.makedirs(output_dir, exist_ok=True)
        slug = self.bank_name_en.lower().replace(' ', '_')

        print(f"\nSCRAPING BRANCHES: {self.bank_name_en}\n")
        print(f"[1/1] Scraping: branches")
        print(f"URL: {branch_url}")

        with sync_playwright() as p:
            browser, context = self._create_browser_context(p)
            page = context.new_page()
            page.goto(branch_url, wait_until='domcontentloaded', timeout=30000)
            page.wait_for_timeout(5000)

            # Give extra time for map/sidebar JS to populate
            try:
                page.wait_for_selector('.sidebar-item', timeout=10000)
            except Exception:
                page.wait_for_timeout(5000)

            html_content = page.content()
            browser.close()

        branches = self.extract_branches(html_content)
        print(f"  Extracted {len(branches)} branches")

        all_data = {
            "bank_name": self.bank_name,
            "bank_name_en": self.bank_name_en,
            "website": self.website,
            "category": "branches",
            "source_url": branch_url,
            "scraped_at": datetime.now().isoformat(),
            "total_branches": len(branches),
            "branches": branches,
        }

        json_path = os.path.join(output_dir, f"{slug}_branches.json")
        txt_path = os.path.join(output_dir, f"{slug}_branches.txt")

        self._save_json(all_data, json_path)
        self._save_branches_txt(branches, branch_url, txt_path)

        standard = sum(1 for b in branches if not b.get('extended_hours'))
        extended = sum(1 for b in branches if b.get('extended_hours'))
        print(f"\nBRANCHES SUMMARY")
        print(f"Total branches: {len(branches)}")
        print(f"Standard hours: {standard}")
        print(f"Extended hours: {extended}")

        return all_data

    def scrape_all(self, output_dir: str = "data"):
        print(f"\n  {self.bank_name_en.upper()} - FULL SCRAPE\n")

        results = {}
        results['loans'] = self.scrape_loans(output_dir)
        results['deposits'] = self.scrape_deposits(output_dir)
        results['branches'] = self.scrape_branches(output_dir)

        print(f"\n  {self.bank_name_en.upper()} - COMPLETE\n")

        return results