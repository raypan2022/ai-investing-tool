import requests
import os
import json
from typing import Optional, Dict, Any
from datetime import datetime
import time

class SECFilingsFetcher:
    def __init__(self, config):
        self.config = config
        self.base_url = "https://data.sec.gov/submissions/CIK"
        
    def get_latest_10k(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch the latest 10-K filing for a given ticker"""
        try:
            print(f"Fetching 10-K for {ticker}...")
            
            # Get company CIK
            cik = self._get_cik_from_ticker(ticker)
            if not cik:
                print(f"Could not find CIK for {ticker}")
                return None
            
            # For MVP, we'll use mock data
            # In production, you'd fetch from SEC EDGAR API
            filing_text = self._get_mock_10k_text(ticker)
            
            if filing_text:
                result = {
                    'ticker': ticker.upper(),
                    'cik': cik,
                    'accession_number': 'mock-123456789',
                    'filing_date': datetime.now().strftime('%Y-%m-%d'),
                    'text': filing_text,
                    'downloaded_at': datetime.now().isoformat(),
                    'file_size': len(filing_text)
                }
                
                print(f"Successfully fetched 10-K for {ticker}")
                return result
            
        except Exception as e:
            print(f"Error fetching 10-K for {ticker}: {str(e)}")
            return None
    
    def _get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """Get CIK number from ticker symbol"""
        # Simplified mapping for common stocks
        ticker_to_cik = {
            'AAPL': '0000320193',
            'MSFT': '0000789019',
            'GOOGL': '0001652044',
            'AMZN': '0001018724',
            'TSLA': '0001318605',
            'META': '0001326801',
            'NVDA': '0001045810',
            'NFLX': '0001065280',
            'JPM': '0000019617',
            'JNJ': '0000200404'
        }
        
        return ticker_to_cik.get(ticker.upper())
    
    def _get_mock_10k_text(self, ticker: str) -> str:
        """Return mock 10-K text for development"""
        company_names = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla, Inc.',
            'META': 'Meta Platforms, Inc.',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix, Inc.',
            'JPM': 'JPMorgan Chase & Co.',
            'JNJ': 'Johnson & Johnson'
        }
        
        company_name = company_names.get(ticker.upper(), f'{ticker} Corporation')
        
        return f"""
ITEM 1. BUSINESS

{company_name} ("the Company") is a leading technology company that designs, manufactures, and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services. The Company's fiscal year is the 52 or 53-week period that ends on the last Saturday of September.

The Company's business strategy leverages its unique ability to design and develop its own operating systems, hardware, application software, and services to provide its customers products and solutions with innovative design, superior ease-of-use, and seamless integration. As part of its strategy, the Company continues to expand its platform for the discovery and delivery of digital content and applications through its Digital Content and Services segment, which allows customers to discover and download or stream digital content, iOS, macOS, watchOS and tvOS applications, and books through either a Mac or Windows personal computer or through iPhone, iPad and iPod touch devices ("iOS devices"), Apple TV, Apple Watch and HomePod.

The Company believes a high-quality user experience is a key differentiator for its products that drives customer loyalty and helps support the retention of users in the Company's ecosystem of products and services. The Company's ability to develop products and services that integrate seamlessly with each other and with other third-party products and services is a key competitive advantage.

ITEM 1A. RISK FACTORS

The Company's business, financial condition, results of operations, and stock price can be affected by a number of factors, whether currently known or unknown, including but not limited to those described below, any one or more of which could, directly or indirectly, cause the Company's actual financial condition, results of operations, and stock price to differ materially from historical results or from any future results expressed or implied by such forward-looking statements.

The Company's business depends substantially on the Company's ability to continue to develop and sell or license new products, services, and technologies on a timely and cost-effective basis. The Company's ability to compete effectively depends heavily on the timely and successful introduction of new products, services, and technologies. The development of new products, services, and technologies is a complex and uncertain process requiring high levels of innovation and investment, as well as the accurate anticipation of technological and market trends.

ITEM 2. PROPERTIES

The Company owns and leases various facilities worldwide, including corporate offices, retail stores, data centers, and research and development facilities. The Company's corporate headquarters are located in Cupertino, California, where the Company owns and leases approximately 3.4 million square feet of space. The Company also owns and leases significant space in Austin, Texas; Cork, Ireland; and various other locations worldwide.

ITEM 3. LEGAL PROCEEDINGS

The Company is involved in various legal proceedings, including patent infringement lawsuits, antitrust investigations, and other litigation. The Company believes that the ultimate outcome of these proceedings will not have a material adverse effect on the Company's financial condition or results of operations.

ITEM 4. MINE SAFETY DISCLOSURES

Not applicable.

ITEM 5. MARKET FOR REGISTRANT'S COMMON EQUITY, RELATED STOCKHOLDER MATTERS AND ISSUER PURCHASES OF EQUITY SECURITIES

The Company's common stock is traded on The NASDAQ Global Select Market under the symbol "{ticker}". The Company has not paid any cash dividends on its common stock and does not anticipate paying any cash dividends in the foreseeable future. The Company intends to retain all available funds for use in the operation and expansion of its business.

ITEM 6. SELECTED FINANCIAL DATA

The following selected financial data should be read in conjunction with the Company's consolidated financial statements and related notes thereto and "Management's Discussion and Analysis of Financial Condition and Results of Operations" included elsewhere in this Annual Report on Form 10-K.

(In millions, except per share amounts)
Year Ended September 28, 2024:
Net sales: $394,328
Net income: $96,995
Earnings per share: $6.16

Year Ended September 30, 2023:
Net sales: $383,285
Net income: $96,995
Earnings per share: $6.16

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS

The following discussion and analysis should be read in conjunction with the consolidated financial statements and related notes thereto included elsewhere in this Annual Report on Form 10-K. This discussion contains forward-looking statements that involve risks and uncertainties. The Company's actual results could differ materially from those anticipated in these forward-looking statements as a result of various factors, including those discussed in "Risk Factors" and elsewhere in this Annual Report on Form 10-K.

The Company's net sales increased 2.9% during 2024 compared to 2023. The increase in net sales was primarily due to higher net sales of iPhone, Mac, and Services, partially offset by lower net sales of iPad and Wearables, Home and Accessories.

The Company's gross margin percentage was 44.1% during 2024 compared to 43.3% during 2023. The increase in gross margin percentage was primarily due to favorable foreign currency exchange rates and cost savings, partially offset by higher costs associated with the Company's supply chain and inflationary pressures.

The Company's operating income increased 2.9% during 2024 compared to 2023. The increase in operating income was primarily due to higher net sales and gross margin, partially offset by higher research and development expenses and selling, general and administrative expenses.
""" 