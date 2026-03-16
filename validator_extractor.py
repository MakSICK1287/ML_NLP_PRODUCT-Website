import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import validators
from requests.exceptions import RequestException, Timeout, TooManyRedirects
import time
import random
import re
from bs4 import Comment
import json
from fake_useragent import UserAgent
import socket
from urllib3.exceptions import NewConnectionError


CSV_FILE_PATH = 'URL_list.csv'
URL_COLUMN_NAME = 'max(page)'
NUMBER_OF_URLS_NEEDED = 200


class ImprovedURLValidator:
    def __init__(self, timeout=15, max_retries=3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.ua = UserAgent()

        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]

        self.session = self._create_session()

        self.bad_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.mp4',
                               '.mp3', '.zip', '.rar', '.exe', '.doc', '.docx',
                               '.xml', '.json', '.css', '.js', '.svg']

        # Successful response codes
        self.success_codes = [200, 301, 302, 307, 308]

    def _create_session(self):
        session = requests.Session()
        session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        return session

    def _rotate_user_agent(self):
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents)
        })

    def check_domain_exists(self, url):
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            if not domain:
                return False

            domain = domain.split(':')[0]
            socket.gethostbyname(domain)
            return True
        except:
            return False

    def is_valid_url_format(self, url):
        if not url or not isinstance(url, str):
            return False

        url = url.strip()

        if not url:
            return False

        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        try:
            if not validators.url(url):
                return False
        except:
            parsed = urlparse(url)
            if not parsed.netloc or '.' not in parsed.netloc:
                return False

        parsed = urlparse(url)
        path = parsed.path.lower()

        if any(path.endswith(ext) for ext in self.bad_extensions):
            return False

        return url

    def check_url_with_multiple_attempts(self, url):
        """Check URL using multiple strategies"""

        strategies = [
            {'timeout': 10, 'verify': True, 'allow_redirects': True},
            {'timeout': 15, 'verify': False, 'allow_redirects': True},
            {'timeout': 20, 'verify': False, 'allow_redirects': True, 'stream': True},
        ]

        for attempt in range(self.max_retries):

            self._rotate_user_agent()

            strategy = strategies[attempt % len(strategies)]

            try:
                response = self.session.get(
                    url,
                    timeout=strategy['timeout'],
                    verify=strategy.get('verify', True),
                    allow_redirects=strategy.get('allow_redirects', True),
                    stream=strategy.get('stream', False)
                )

                if response.status_code in self.success_codes:

                    content_type = response.headers.get('Content-Type', '').lower()

                    allowed_types = ['text/html', 'text/plain', 'application/xhtml+xml']

                    if any(allowed in content_type for allowed in allowed_types):

                        final_url = response.url

                        if response.status_code == 200:

                            if strategy.get('stream', False):
                                content_preview = response.raw.read(500).decode('utf-8', errors='ignore')
                            else:
                                content_preview = response.text[:500]

                            error_indicators = [
                                '404 not found',
                                'page not found',
                                'access denied'
                            ]

                            if not any(indicator in content_preview.lower() for indicator in error_indicators):
                                return True, final_url, "OK"
                            else:
                                return False, url, "Error page"

                        return True, final_url, "OK"

                    else:
                        return False, url, f"Not HTML content: {content_type}"

                elif response.status_code == 403:
                    continue

                elif response.status_code == 429:
                    time.sleep(5)

                else:
                    return False, url, f"HTTP {response.status_code}"

            except Timeout:
                if attempt == self.max_retries - 1:
                    return False, url, "timeout"
                time.sleep(2)

            except TooManyRedirects:
                return False, url, "Too many redirects"

            except ConnectionError:
                if attempt == self.max_retries - 1:
                    return False, url, "Connection error"
                time.sleep(2)

            except Exception as e:
                if attempt == self.max_retries - 1:
                    return False, url, str(e)[:50]
                time.sleep(2)

        return False, url, "All tries failed"

    def validate_urls(self, urls, quick_check_first=True):

        valid_urls = []
        invalid_urls = []
        need_full_check = []

        print("\nChecking URLs...")

        if quick_check_first:

            print("\nFast check...")

            for i, url in enumerate(urls):

                print(f"{i+1}/{len(urls)}: {url[:50]}...", end=" ")

                formatted_url = self.is_valid_url_format(url)

                if not formatted_url:
                    print("Invalid format")
                    invalid_urls.append({'url': url, 'reason': 'Invalid format'})
                    continue

                result, final_url, reason = self.quick_check(formatted_url)

                if result is True:
                    print("OK")
                    valid_urls.append(final_url)

                elif result is False:
                    print(reason)
                    invalid_urls.append({'url': url, 'reason': reason})

                else:
                    print("Full check")
                    need_full_check.append(formatted_url)

                time.sleep(0.3)

        return valid_urls, invalid_urls


def main():

    print("=" * 60)
    print("URL Extraction and Validation")
    print("=" * 60)

    try:

        df = pd.read_csv(CSV_FILE_PATH)

        print(f"\nCSV loaded: {df.shape[0]} rows")
        print(f"Columns: {list(df.columns)}")

        if URL_COLUMN_NAME not in df.columns:

            print(f"Column '{URL_COLUMN_NAME}' not found")
            return

        all_urls = df[URL_COLUMN_NAME].dropna().tolist()

        print(f"Found URLs: {len(all_urls)}")

    except Exception as e: 

        print(f"CSV loading failed: {e}")
        return

    validator = ImprovedURLValidator()

    valid_urls, invalid_urls = validator.validate_urls(all_urls)

    print("\nValidation results:")
    print(f"Valid URLs: {len(valid_urls)}")
    print(f"Invalid URLs: {len(invalid_urls)}")


if __name__ == "__main__":
    main()