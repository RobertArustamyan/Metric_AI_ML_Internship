"""
IDBank scraper implementation.

All tab/accordion content is in the DOM regardless of CSS visibility,
so no clicking is needed — BeautifulSoup reads hidden elements.
"""

import re

from bs4 import BeautifulSoup

from base_scraper import BaseBankScraper


# Domains that cause networkidle hangs — blocked for speed
BLOCKED_DOMAINS = [
    "widget.beesender.com",
    "balance.beesender.com",
    "connect.facebook.net",
    "www.facebook.com",
    "www.google.com/recaptcha",
    "www.gstatic.com/recaptcha",
    "googletagmanager.com",
    "google-analytics.com",
    "mc.yandex.ru",
    "abt.s3.yandex.net",
    "static.cloudflareinsights.com",
]


class IDBankScraper(BaseBankScraper):

    @property
    def bank_name(self) -> str:
        return "ԱյԴի Բանկ"

    @property
    def bank_name_en(self) -> str:
        return "IDBank"

    @property
    def website(self) -> str:
        return "https://idbank.am"

    def get_loan_urls(self) -> dict:
        base = self.website
        return {
            # Loans without collateral
            "rocket_line": f"{base}/credits/loans-without-collateral/rocket-line/",
            "rocket_loan": f"{base}/credits/loans-without-collateral/rocket-loan/",
            "credit_lines": f"{base}/credits/loans-without-collateral/credit-lines/",
            "profi": f"{base}/credits/loans-without-collateral/profi/",
            "student_loan": f"{base}/credits/loans-without-collateral/student-loan/",
            "micro_loans_jerm_ojakh": f"{base}/credits/loans-without-collateral/micro-loans-jerm-ojakh/",
            "renovation_furnishing_credit": f"{base}/credits/loans-without-collateral/renovation-furnishing-credit/",

            # Collateral loans
            "consumer_loan": f"{base}/credits/collateral-loans/consumer-loan/",
            "refinancing_loans": f"{base}/credits/collateral-loans/refinancing-loans/",
            "loans_secured_by_financial_assets": f"{base}/credits/collateral-loans/loans-secured-by-financial-assets/",
            "invest_loan": f"{base}/credits/collateral-loans/-invest-loan-/",

            # Gold-pledged loans
            "idgold": f"{base}/credits/loans-with-plegde-of-gold/IDGold/",
            "gold_standart": f"{base}/credits/loans-with-plegde-of-gold/gold-standart/",

            # Mortgage
            "idhome_package": f"{base}/credits/mortgage/idhome-package/",
            "mortgage_own_resources": f"{base}/credits/mortgage/for-the-purchase-own-resources/",
            "mortgage_renovation": f"{base}/credits/mortgage/mortgage-loans-renovation/",
            "duryan_house": f"{base}/credits/mortgage/duryan-house-residential-area/",
            "mortgage_national_company": f"{base}/credits/mortgage/for-the-purchase-national-mortgage-company/",
            "mortgage_young_families": f"{base}/credits/mortgage/for-the-purchase-young-families/",
            "renovation_national_company": f"{base}/credits/mortgage/renovation-mortgage-loans-national-mortgage-company/",
            "mortgage_diaspora": f"{base}/credits/mortgage/mortgage-loans-for-the-diaspora/",
            "mortgage_refinancing": f"{base}/credits/mortgage/mortgage-loan-refinancing/",
        }

    def get_deposit_urls(self) -> dict:
        base = self.website
        return {
            "deposit_safe_replenish": f"{base}/deposits/deposit_safe/replenish/",
            "deposit_safe_noreplenish": f"{base}/deposits/deposit_safe/noreplenish/",
            "demand_deposit": f"{base}/deposits/demand/demand/",
        }

    def get_branch_url(self) -> str:
        return f"{self.website}/information/about/branches-and-atms/"

    def _render_page(self, context, url: str, click_tabs: bool = False) -> str:
        """
        Fast page renderer for IDBank.
        Blocks images/fonts/trackers, uses domcontentloaded, no clicking needed.
        """
        page = context.new_page()

        def block_resources(route):
            req_url = route.request.url
            resource_type = route.request.resource_type

            # Block heavy resource types we don't need for text extraction
            if resource_type in ("image", "font", "media", "stylesheet"):
                route.abort()
                return

            # Block known tracker/widget domains that prevent networkidle
            for domain in BLOCKED_DOMAINS:
                if domain in req_url:
                    route.abort()
                    return

            route.continue_()

        page.route("**/*", block_resources)
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        page.wait_for_timeout(1500)

        html_content = page.content()
        page.close()
        return html_content

    def extract_page_content(self, page_html: str, url: str) -> dict:
        """
        Extract content from IDBank product pages.

        Structure:
        - Banner panel: .main-banner__panel-item (rate, term, min amount summary)
        - Tabs: .big-tabs__section (Overview, Tariffs, Documents, FAQ)
        - Tariff tables: .tariffs__table > .tariffs__table-row (key-value pairs)
        - FAQ: .faq__item (question + answer accordions)
        - PDF links: .tariffs__load-link, .archive__list-item
        """
        soup = BeautifulSoup(page_html, "html.parser")

        # Page title
        title = ""
        h1 = soup.find("h1", class_="main-banner__slide-title")
        if h1:
            title = self.clean_text(h1.get_text())
        if not title:
            title_tag = soup.find("title")
            title = self.clean_text(title_tag.get_text()) if title_tag else ""

        # Remove noise elements
        for tag in soup.find_all(["script", "style", "noscript", "iframe"]):
            tag.decompose()
        for cls_fragment in [
            "beesenderchat", "modal-fs", "overlay", "svg-sprite",
            "page-subfooter", "m-menu", "store-breadcrumb",
        ]:
            for el in soup.find_all(
                class_=lambda c: c and cls_fragment in str(c)
            ):
                el.decompose()
        for tag_name in ["header", "footer"]:
            for el in soup.find_all(tag_name):
                el.decompose()

        content_parts = []
        tables = []

        # 1. Banner summary panel
        panel_items = soup.find_all("div", class_="main-banner__panel-item")
        if panel_items:
            panel_text = []
            for item in panel_items:
                label = item.find("div", class_="main-banner__panel-title")
                value = item.find("div", class_="main-banner__panel-subtitle")
                if label and value:
                    panel_text.append(
                        f"{self.clean_text(label.get_text())}: "
                        f"{self.clean_text(value.get_text())}"
                    )
            if panel_text:
                content_parts.append("[Summary]\n" + "\n".join(panel_text))

        # 2. Product overview (first tab — preview section)
        preview_title = soup.find("h4", class_="product-preview__title")
        preview_article = soup.find("p", class_="product-preview__article")
        if preview_title or preview_article:
            overview_text = ""
            if preview_title:
                overview_text += self.clean_text(preview_title.get_text()) + "\n"
            if preview_article:
                overview_text += self.clean_text(preview_article.get_text())
            content_parts.append(f"[Overview]\n{overview_text}")

        # 3. Feature highlights (product-plus slides)
        features = []
        for slide in soup.find_all("div", class_="product-plus__slide"):
            feat_title = slide.find("div", class_="product-plus__slide-title")
            feat_sub = slide.find("div", class_="product-plus__slide-subtitle")
            if feat_title:
                feat = self.clean_text(feat_title.get_text())
                if feat_sub:
                    feat += " — " + self.clean_text(feat_sub.get_text())
                features.append(feat)
        if features:
            content_parts.append("[Features]\n" + "\n".join(features))

        # 4. Tariff tables (the main structured data — rates, terms, fees)
        for table_div in soup.find_all("div", class_="tariffs__table"):
            head_label = table_div.find("span", class_="tariffs__table-head-label")
            section_name = (
                self.clean_text(head_label.get_text()) if head_label else "Tariffs"
            )

            rows = []
            for row in table_div.find_all("div", class_="tariffs__table-row"):
                cells = row.find_all("div", class_="tariffs__table-cell")
                if len(cells) == 2:
                    key = self.clean_text(cells[0].get_text())
                    value = self.clean_text(cells[1].get_text())
                    rows.append([key, value])

            if rows:
                tables.append(rows)
                table_text = "\n".join(f"  {r[0]}: {r[1]}" for r in rows)
                content_parts.append(f"[{section_name}]\n{table_text}")

        # 5. FAQ
        faq_parts = []
        for item in soup.find_all("div", class_="faq__item"):
            q_el = item.find("span", class_="faq__item-btn-label")
            a_el = item.find("div", class_="faq__item-body-text")
            if q_el and a_el:
                q = self.clean_text(q_el.get_text())
                a = self.clean_text(a_el.get_text())
                faq_parts.append(f"Q: {q}\nA: {a}")
        if faq_parts:
            content_parts.append("[FAQ]\n" + "\n\n".join(faq_parts))

        # 6. PDF/document links
        doc_links = []
        for link_wrap in soup.find_all("div", class_="tariffs__load-link-wrap"):
            a_tag = link_wrap.find("a", class_="tariffs__load-link")
            if a_tag:
                href = a_tag.get("href", "")
                if not href.startswith("http"):
                    href = self.website + href
                text = self.clean_text(a_tag.get_text())
                doc_links.append(f"{text}: {href}")
        if doc_links:
            content_parts.append("[Documents]\n" + "\n".join(doc_links))

        # Fallback: if nothing was extracted, grab main content text
        if not content_parts:
            main = soup.find("main", class_="page-main")
            if main:
                main_text = main.get_text(separator="\n", strip=True)
                if main_text and len(main_text) > 50:
                    content_parts.append(main_text)

        combined = "\n\n".join(content_parts)
        content_text = self.clean_text(combined)

        return {
            "title": title,
            "url": url,
            "content": content_text,
            "tables": tables,
        }

    def extract_branches(self, page_html: str) -> list:
        """
        Extract branch data from the branches & ATMs page.
        Branches are in .map__point elements with structured fields.
        Each has: name, address, lat/lon data attrs, hours section, phones section.
        """
        soup = BeautifulSoup(page_html, "html.parser")
        branches = []

        for point in soup.find_all("div", class_="map__point"):
            name_el = point.find("div", class_="map__point-name")
            addr_el = point.find("div", class_="map__point-address")

            name = self.clean_text(name_el.get_text()) if name_el else ""
            address = self.clean_text(addr_el.get_text()) if addr_el else ""

            if not name:
                continue

            # Coordinates from data attributes
            lat = point.get("data-lat", "")
            lon = point.get("data-lon", "")

            # Sections: position 0 = working hours, position 1 = phones
            sections = point.find_all("div", class_="map__point-section")

            # Working hours
            schedule = ""
            if len(sections) >= 1:
                hours_section = sections[0]
                hours_texts = []
                for el in hours_section.find_all(
                    "div", class_="map__point-section-text"
                ):
                    hours_texts.append(self.clean_text(el.get_text()))
                if not hours_texts:
                    # Fallback: some branches use raw text with <br> tags
                    raw = hours_section.get_text()
                    title_el = hours_section.find(
                        "div", class_="map__point-section-title"
                    )
                    if title_el:
                        raw = raw.replace(title_el.get_text(), "", 1)
                    hours_texts = [
                        self.clean_text(line)
                        for line in raw.split("\n")
                        if self.clean_text(line)
                    ]
                schedule = " | ".join(hours_texts)

            # Phones
            phones = []
            if len(sections) >= 2:
                for a_tag in sections[1].find_all(
                    "a", class_="map__point-section-text"
                ):
                    phone = self.clean_text(a_tag.get_text())
                    if phone:
                        phones.append(phone)
            phone_str = ", ".join(phones)

            # Extended hours detection
            is_extended = False
            close_time = point.get("data-close", "")
            if close_time and close_time >= "20:00":
                is_extended = True

            branches.append({
                "name": name,
                "address": address,
                "phone": phone_str,
                "schedule": schedule,
                "extended_hours": is_extended,
                "description": f"Coordinates: {lat}, {lon}" if lat else None,
            })

        return branches


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    scraper = IDBankScraper()
    scraper.scrape_all(output_dir=os.getenv("DATA_PATH", "data"))