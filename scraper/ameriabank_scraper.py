"""
Ameriabank scraper implementation.

Ameriabank uses DotNetNuke CMS with WSC content manager modules.
Content lives inside .wsc_content_manager_module_container divs.
Tabbed content uses .tab-pane (all tabs in DOM regardless of CSS visibility).
Branch data uses .sidebar-item elements on the service-network page.
"""

import re

from bs4 import BeautifulSoup

from base_scraper import BaseBankScraper


class AmeriabankScraper(BaseBankScraper):

    @property
    def bank_name(self) -> str:
        return "Ամերիաբանկ"

    @property
    def bank_name_en(self) -> str:
        return "Ameriabank"

    @property
    def website(self) -> str:
        return "https://ameriabank.am"

    def get_loan_urls(self) -> dict:
        return {
            # Consumer Loans
            "consumer_loans_overview": "https://ameriabank.am/loans/consumer-loans",
            "consumer_loan": "https://ameriabank.am/personal/loans/consumer-loans/consumer-loans",
            "overdraft": "https://ameriabank.am/personal/loans/consumer-loans/overdraft",
            "credit_line": "https://ameriabank.am/personal/loans/consumer-loans/credit-line",
            "consumer_finance": "https://ameriabank.am/personal/loans/consumer-loans/consumer-finance",
            "online_consumer_finance": "https://ameriabank.am/personal/loans/consumer-loan/online-consumer-finance",

            # Secured Loans
            "secured_loans_overview": "https://ameriabank.am/loans/secured-loans",
            "secured_consumer_loan": "https://ameriabank.am/loans/secured-loans/consumer-loan",
            "secured_overdraft": "https://ameriabank.am/loans/secured-loans/overdraft",
            "secured_credit_line": "https://ameriabank.am/loans/secured-loans/credit-line",
            "investment_loan": "https://ameriabank.am/personal/loans/other-loans/investment-loan",

            # Car Loans
            "car_loans_overview": "https://ameriabank.am/personal/loans/car-loans",
            "car_loan_primary": "https://ameriabank.am/personal/loans/car-loan/primary",
            "car_loan_secondary": "https://ameriabank.am/personal/loans/car-loan/secondary-market",
            "car_loan_secondary_unused": "https://ameriabank.am/personal/loans/car-loan/secondary-market-unused",
            "car_loan_online_primary": "https://ameriabank.am/personal/loans/car-loan/without-bank-visit",
            "car_loan_online_secondary": "https://ameriabank.am/personal/loans/car-loan/online-secondary-market",

            # Mortgage Loans
            "mortgage_overview": "https://ameriabank.am/personal/loans/mortgage-loans",
            "mortgage_online": "https://ameriabank.am/personal/loans/mortgage/online",
            "mortgage_primary": "https://ameriabank.am/personal/loans/mortgage/primary-market-loan",
            "mortgage_secondary": "https://ameriabank.am/personal/loans/mortgage/secondary-market",
            "mortgage_commercial": "https://ameriabank.am/personal/loans/mortgage/commercial-mortgage",
            "mortgage_renovation": "https://ameriabank.am/personal/loans/mortgage/renovation-mortgage",
            "mortgage_construction": "https://ameriabank.am/personal/loans/mortgage/construction-mortgage",

            # Info
            "payment_options": "https://ameriabank.am/personal/loans/other-loans/payment-options",
            "loan_support_info": "https://ameriabank.am/personal/loans/more/support/information",
        }

    def get_deposit_urls(self) -> dict:
        return {
            "deposits_overview": "https://ameriabank.am/personal/saving",
            "ameria_deposit": "https://ameriabank.am/personal/saving/deposits/ameria-deposit",
            "cumulative_deposit": "https://ameriabank.am/personal/saving/deposits/cumulative-deposit",
            "kids_deposit": "https://ameriabank.am/personal/saving/deposits/kids-deposit",
            "saving_account": "https://ameriabank.am/personal/accounts/accounts/saving-account",
            "bonds": "https://ameriabank.am/personal/saving/saving/bonds",
            "deposit_support": "https://ameriabank.am/personal/saving/other-services/support",
            "deposit_terms": "https://ameriabank.am/personal/saving/other-services/terms",
            "deposits_see_all": "https://ameriabank.am/personal/saving/deposits/see-all",
        }

    def get_branch_url(self) -> str:
        return "https://ameriabank.am/service-network"

    def extract_page_content(self, page_html: str, url: str) -> dict:
        """
        Extract content from pages.
        """
        soup = BeautifulSoup(page_html, 'html.parser')

        # Get page title
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else ""
        title = re.sub(r'\s*\|.*$', '', title)

        # Remove scripts, styles, noscript only
        for tag in soup.find_all(['script', 'style', 'noscript']):
            tag.decompose()

        content_parts = []
        tables = []

        # Layer 1: WSC content manager modules (bank-specific)
        wsc_containers = soup.select('.wsc_content_manager_module_container')
        for container in wsc_containers:
            tab_panes = container.select('.tab-pane')
            if tab_panes:
                tab_names = [
                    a.get_text(strip=True)
                    for a in container.select('.tabs-navigation a')
                ]

                for i, pane in enumerate(tab_panes):
                    tab_label = tab_names[i] if i < len(tab_names) else f"Tab {i+1}"
                    pane_text = pane.get_text(separator='\n', strip=True)
                    if pane_text:
                        content_parts.append(f"\n[{tab_label}]\n{pane_text}")

                    for table in pane.find_all('table'):
                        table_data = self._extract_table(table)
                        if table_data:
                            tables.append(table_data)
            else:
                # No tabs: extract descriptions, text blocks, and accordions
                descriptions = container.select('.description, .mng-text-theme')
                for desc in descriptions:
                    desc_text = desc.get_text(separator='\n', strip=True)
                    if desc_text:
                        content_parts.append(desc_text)

                # FAQ accordions
                accordions = container.select('.cs-accordion__body')
                for acc in accordions:
                    acc_title = acc.select_one('.cs-accordion__title')
                    acc_panel = acc.select_one('.cs-accordion__panel')
                    if acc_title and acc_panel:
                        q = acc_title.get_text(strip=True)
                        a = acc_panel.get_text(separator='\n', strip=True)
                        content_parts.append(f"\nQ: {q}\nA: {a}")

                for table in container.find_all('table'):
                    table_data = self._extract_table(table)
                    if table_data:
                        tables.append(table_data)

        # Layer 2: standalone .description divs
        if not content_parts:
            for desc in soup.select('.description'):
                desc_text = desc.get_text(separator='\n', strip=True)
                if desc_text:
                    content_parts.append(desc_text)

        # Layer 3: DNN content panes
        if not content_parts:
            for selector in ['#dnn_ContentPane', '#dnn_ContentBottom']:
                for el in soup.select(selector):
                    el_text = el.get_text(separator='\n', strip=True)
                    if el_text and len(el_text) > 50:
                        content_parts.append(el_text)

        # Layer 4: generic fallbacks
        if not content_parts:
            for selector in ['main', 'article', '.content-area']:
                el = soup.select_one(selector)
                if el:
                    el_text = el.get_text(separator='\n', strip=True)
                    if el_text and len(el_text) > 50:
                        content_parts.append(el_text)
                        break

        # Layer 5: body with nav/footer stripped
        if not content_parts:
            body = soup.find('body')
            if body:
                for selector in ['nav', 'header', 'footer', '.navbar',
                                 '.menu', '.sidebar', '.cookie-banner']:
                    for element in body.select(selector):
                        element.decompose()
                body_text = body.get_text(separator='\n', strip=True)
                if body_text:
                    content_parts.append(body_text)

                for table in body.find_all('table'):
                    table_data = self._extract_table(table)
                    if table_data:
                        tables.append(table_data)

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
        Extract branch data from the service-network page
        which are in .sidebar-item elements with structured fields.
        """
        soup = BeautifulSoup(page_html, 'html.parser')
        sidebar_items = soup.select('.sidebar-item')
        branches = []

        for item in sidebar_items:
            name_el = item.select_one('.sidebar-item__title')
            location_el = item.select_one('.sidebar-item__location')
            phone_el = item.select_one('.sidebar-item__phone')
            tag_el = item.select_one('.sidebar-item__tag')
            desc_el = item.select_one('.sidebar-item__description')

            name = self.clean_text(name_el.get_text()) if name_el else ""
            address = self.clean_text(location_el.get_text()) if location_el else ""
            phone = self.clean_text(phone_el.get_text()) if phone_el else ""
            schedule = self.clean_text(tag_el.get_text()) if tag_el else ""
            description = self.clean_text(desc_el.get_text()) if desc_el else ""

            is_extended = bool(
                tag_el and 'sidebar-item__tag--green' in tag_el.get('class', [])
            )

            if name:
                branches.append({
                    "name": name,
                    "address": address,
                    "phone": phone,
                    "schedule": schedule,
                    "extended_hours": is_extended,
                    "description": description if description else None,
                })

        return branches

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
    import os
    from dotenv import load_dotenv

    load_dotenv()

    scraper = AmeriabankScraper()
    scraper.scrape_all(output_dir=os.getenv("DATA_PATH", "data"))