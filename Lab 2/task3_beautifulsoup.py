"""
Lab Task 3:
Install beautifulsoup4 with PIP. Scrape a website for headlines using requests
to fetch HTML and BeautifulSoup to parse/extract data.
"""

import requests
from bs4 import BeautifulSoup


def scrape_headlines(url: str = "https://news.ycombinator.com/") -> None:
    """Fetch a news page and extract article headlines."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        print(f"Fetched: {url}")
        print(f"Status Code: {response.status_code}\n")

        soup = BeautifulSoup(response.text, "html.parser")

        # Hacker News stores titles in <span class="titleline"><a>...</a></span>
        title_links = soup.select("span.titleline > a")

        if not title_links:
            # Fallback: generic headline tags on other sites
            title_links = soup.find_all(["h1", "h2", "h3"], limit=20)

        print(f"Found {len(title_links)} headlines:\n")
        print("-" * 60)

        for i, link in enumerate(title_links[:15], start=1):
            title = link.get_text(strip=True)
            href = link.get("href", "")
            if href and not href.startswith("http"):
                href = f"https://news.ycombinator.com/{href}"
            print(f"{i:2d}. {title}")
            if href:
                print(f"    Link: {href}")
            print()

        print("-" * 60)
        print("Scraping completed successfully.")

    except requests.exceptions.Timeout:
        print("Error: Request timed out.")
    except requests.exceptions.ConnectionError:
        print("Error: Connection failed. Check your internet connection.")
    except requests.exceptions.HTTPError as exc:
        print(f"Error: HTTP error — {exc}")
    except requests.exceptions.RequestException as exc:
        print(f"Error: Request failed — {exc}")
    except Exception as exc:
        print(f"Error while parsing HTML — {exc}")


def main():
    print("Scraping headlines from Hacker News...\n")
    scrape_headlines()


if __name__ == "__main__":
    main()
