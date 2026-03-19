"""
Mellat Bank scraper implementation.

Mellat Bank uses an Angular SPA with server-side rendering.
Content is inside #inner divs (app-reach-text components).
Tabs use #businessMain with .nav-tabs and .tab-pane.
Tables are standard HTML tables inside #inner.
PDF links are in #package .download a[download].

Mellat Bank has only one branch (HQ), so branch info is hardcoded.
"""

import os
import re
import time
from datetime import datetime

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from base_scraper import BaseBankScraper


class MellatBankScraper(BaseBankScraper):

    @property
    def bank_name(self) -> str:
        return "Mellat Bank"

    @property
    def bank_name_en(self) -> str:
        return "MellatBank"

    @property
    def website(self) -> str:
        return "https://mellatbank.am"

    def get_loan_urls(self) -> dict:
        return {
            "loans_overview": "https://mellatbank.am/hy/loans_individual",
            "mortgage_loan": "https://mellatbank.am/hy/Mortgage-loan1",
            "repair_loan": "https://mellatbank.am/hy/repair_loan",
            "student_loan": "https://mellatbank.am/hy/student_loan",
            "car_loan": "https://mellatbank.am/hy/Car_loan",
            "express_consumer_loan": "https://mellatbank.am/hy/Express_consumer_loan",
            "privileged_loan": "https://mellatbank.am/hy/Privileged_Loan%20",
            "consumer_guarantee_1": "https://mellatbank.am/hy/Consumer_guarantee_1",
            "consumer_guarantee_2": "https://mellatbank.am/hy/Consumer_guarantee_2",
        }

    def get_deposit_urls(self) -> dict:
        # Single URL - tabs are handled via Playwright clicking in _scrape_content_pages
        return {
            "deposits": "https://mellatbank.am/hy/Deposits",
        }

    def get_branch_url(self) -> str:
        return ""

    def _render_page(self, context, url: str, click_tabs: bool = True) -> str:
        """
        Override for MellatBank Angular SPA.
        Basic rendering without tab clicking. Used as fallback.
        """
        page = context.new_page()
        page.goto(url, wait_until='domcontentloaded', timeout=30000)
        page.wait_for_timeout(5000)

        try:
            page.wait_for_selector('#inner, #loanHeader, #businessMain', timeout=10000)
        except Exception:
            page.wait_for_timeout(3000)

        html_content = page.content()
        page.close()
        return html_content

    def _scrape_angular_page(self, context, url: str) -> dict:
        """
        Scrape a single MellatBank page, handling Angular tab clicks.
        Returns a page_data dict with content from ALL tabs.
        """
        page = context.new_page()
        page.goto(url, wait_until='domcontentloaded', timeout=30000)
        page.wait_for_timeout(5000)

        try:
            page.wait_for_selector('#inner, #loanHeader, #businessMain', timeout=10000)
        except Exception:
            page.wait_for_timeout(3000)

        # Get page title
        title = page.title() or ""
        title = re.sub(r'^MellatBank:\s*', '', title)

        # Detect tabs
        tab_selector = '#businessMain .nav-tabs .nav-link'
        tab_links = page.query_selector_all(tab_selector)
        tab_names = [tab.inner_text().strip() for tab in tab_links]

        content_parts = []
        tables = []

        if tab_links:
            # Multi-tab page: click each tab and capture content
            for i, tab in enumerate(tab_links):
                tab_name = tab_names[i] if i < len(tab_names) else f"Tab {i+1}"

                try:
                    tab.click()
                    page.wait_for_timeout(3000)

                    try:
                        page.wait_for_selector('.tab-pane.active #inner', timeout=5000)
                    except Exception:
                        page.wait_for_timeout(2000)

                    html = page.content()
                    tab_content, tab_tables = self._extract_active_tab(html, tab_name)
                    content_parts.extend(tab_content)
                    tables.extend(tab_tables)

                except Exception as e:
                    print(f"    Tab click error ({tab_name}): {e}")
        else:
            # No tabs: extract from full page
            html = page.content()
            page_data = self.extract_page_content(html, url)
            page.close()
            return page_data

        # Extract PDF links
        html = page.content()
        soup = BeautifulSoup(html, 'html.parser')
        pdf_links = []
        for link in soup.select('#package .download a[download]'):
            href = link.get('href', '')
            link_text = link.get_text(strip=True)
            if href and link_text:
                pdf_links.append(f"{link_text}: {href}")

        if pdf_links:
            content_parts.append("\n[PDF Documents]\n" + "\n".join(pdf_links))

        # Also extract loan card info if present
        for card in soup.select('#loanCard .info'):
            label_el = card.select_one('label')
            value_el = card.select_one('h4')
            if label_el and value_el:
                label = label_el.get_text(strip=True)
                value = value_el.get_text(strip=True)
                content_parts.append(f"{label}: {value}")

        page.close()

        content = self.clean_text('\n\n'.join(content_parts))

        return {
            "title": title,
            "url": url,
            "content": content,
            "tables": tables,
        }

    def _extract_active_tab(self, html: str, tab_name: str):
        """
        Extract content from the currently active tab pane in the HTML.
        Returns (content_parts, tables) tuple.
        """
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup.find_all(['script', 'style', 'noscript']):
            tag.decompose()

        content_parts = []
        tables = []

        active_pane = soup.select_one('.tab-pane.active, .tab-pane.show.active')
        if active_pane:
            inner_blocks = active_pane.select('#inner')
            for block in inner_blocks:
                text = block.get_text(separator='\n', strip=True)
                if text and len(text) > 10:
                    content_parts.append(f"\n[{tab_name}]\n{text}")

                for table in block.find_all('table'):
                    table_data = self._extract_table(table)
                    if table_data:
                        tables.append(table_data)

        return content_parts, tables

    def _scrape_content_pages(self, urls: dict, category: str, output_dir: str):
        """
        Override to use Angular tab-clicking approach for all Mellat pages.
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
                print(f"  URL: {url}")

                try:
                    page_data = self._scrape_angular_page(context, url)
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

        json_path = os.path.join(output_dir, f"{slug}_{category}.json")
        txt_path = os.path.join(output_dir, f"{slug}_{category}.txt")

        self._save_json(all_data, json_path)
        self._save_content_txt(all_data["pages"], category, txt_path)
        self._print_content_summary(all_data["pages"], category)

        return all_data

    def scrape_branches(self, output_dir: str = "data"):
        """
        Mellat Bank has only one branch (HQ at Charents 19).
        """
        os.makedirs(output_dir, exist_ok=True)
        slug = self.bank_name_en.lower().replace(' ', '_')

        branches = self.extract_branches("")
        all_data = {
            "bank_name": self.bank_name,
            "bank_name_en": self.bank_name_en,
            "website": self.website,
            "category": "branches",
            "source_url": "https://mellatbank.am",
            "scraped_at": datetime.now().isoformat(),
            "total_branches": len(branches),
            "branches": branches,
        }

        json_path = os.path.join(output_dir, f"{slug}_branches.json")
        txt_path = os.path.join(output_dir, f"{slug}_branches.txt")

        self._save_json(all_data, json_path)
        self._save_branches_txt(branches, self.website, txt_path)

        print(f"\n  BRANCHES SUMMARY")
        print(f"  Total branches:  {len(branches)}")

        return all_data

    def extract_page_content(self, page_html: str, url: str) -> dict:
        """
        Extract content from Mellat Bank pages (used for loan pages).

        For deposit pages, scrape_deposits handles extraction directly
        because it needs to click tabs with Playwright.
        """
        soup = BeautifulSoup(page_html, 'html.parser')

        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else ""
        title = re.sub(r'^MellatBank:\s*', '', title)

        for tag in soup.find_all(['script', 'style', 'noscript']):
            tag.decompose()

        content_parts = []
        tables = []

        # Layer 1: Tab structure (#businessMain)
        tab_container = soup.select_one('#businessMain')
        if tab_container:
            tab_names = [
                a.get_text(strip=True)
                for a in tab_container.select('.nav-tabs .nav-link')
            ]

            tab_panes = tab_container.select('.tab-content > .tab-pane')
            for i, pane in enumerate(tab_panes):
                tab_label = tab_names[i] if i < len(tab_names) else f"Tab {i+1}"

                inner_blocks = pane.select('#inner')
                for block in inner_blocks:
                    block_text = block.get_text(separator='\n', strip=True)
                    if block_text and len(block_text) > 10:
                        content_parts.append(f"\n[{tab_label}]\n{block_text}")

                    for table in block.find_all('table'):
                        table_data = self._extract_table(table)
                        if table_data:
                            tables.append(table_data)

        # Layer 2: #inner blocks outside tabs
        if not content_parts:
            main_section = soup.select_one('section#main')
            if main_section:
                for inner in main_section.select('#inner'):
                    text = inner.get_text(separator='\n', strip=True)
                    if text and len(text) > 10:
                        content_parts.append(text)

                    for table in inner.find_all('table'):
                        table_data = self._extract_table(table)
                        if table_data:
                            tables.append(table_data)

        # Layer 3: Loan header banner
        if not content_parts:
            banner = soup.select_one('#loanHeader')
            if banner:
                blur_div = banner.select_one('div.blur')
                if blur_div:
                    text = blur_div.get_text(separator='\n', strip=True)
                    if text:
                        content_parts.append(text)

        # Layer 4: Loan card info sections
        loan_cards = soup.select('#loanCard .info')
        for card in loan_cards:
            label_el = card.select_one('label')
            value_el = card.select_one('h4')
            if label_el and value_el:
                label = label_el.get_text(strip=True)
                value = value_el.get_text(strip=True)
                content_parts.append(f"{label}: {value}")

        # Layer 5: Body fallback
        if not content_parts:
            body = soup.find('body')
            if body:
                for sel in ['nav', 'footer', '#footer', '.navbar']:
                    for el in body.select(sel):
                        el.decompose()
                text = body.get_text(separator='\n', strip=True)
                if text:
                    content_parts.append(text)

        # Extract PDF links
        pdf_links = []
        for link in soup.select('#package .download a[download]'):
            href = link.get('href', '')
            link_text = link.get_text(strip=True)
            if href and link_text:
                pdf_links.append(f"{link_text}: {href}")

        if pdf_links:
            content_parts.append("\n[PDF Documents]\n" + "\n".join(pdf_links))

        combined = '\n\n'.join(content_parts)
        content_text = self.clean_text(combined)

        return {
            "title": title,
            "url": url,
            "content": content_text,
            "tables": tables,
        }

    def extract_branches(self, page_html: str) -> list:
        """
        Mellat Bank has only one branch (HQ).
        """
        return [
            {
                "name": "Mellat Bank - Glkhamasnavorutyun",
                "address": "ք. Երևան, Չարենցի 19",
                "phone": "(374) 60 38 88 88",
                "schedule": "",
                "extended_hours": False,
                "description": None,
            }
        ]

    @staticmethod
    def _extract_table(table_element) -> list:
        """Extract rows from a table element as a list of lists."""
        table_data = []
        for row in table_element.find_all('tr'):
            cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
            if any(cells):
                table_data.append(cells)
        return table_data


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    scraper = MellatBankScraper()
    scraper.scrape_all(output_dir=os.getenv("DATA_PATH", "data"))