import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import re

def parse_dadkhast_page(url):
    """Parse the dadkhast website to extract h3 texts and their associated links"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        container = soup.find('div', class_='container container--xl')
        
        dadkhast_texts = []
        urls = []
        
        if container:
            petition_cards = container.find_all('article', class_='petition-card')
            
            for card in petition_cards:
                h3_elem = card.find('h3', class_='heading-3')
                if h3_elem:
                    dadkhast_text = h3_elem.get_text(strip=True)
                    dadkhast_texts.append(dadkhast_text)
                    
                    link = card.find('a', href=True)
                    if link:
                        # Fix URL construction
                        href = link['href']
                        if href.startswith('/'):
                            full_url = f"https://www.daadkhast.org{href}"
                        else:
                            full_url = f"https://www.daadkhast.org/{href}"
                        urls.append(full_url)
                    else:
                        urls.append(None)
        
        return pd.DataFrame({
            'dadkhast': dadkhast_texts,
            'url': urls
        })
    
    except Exception as e:
        print(f"Error in parse_dadkhast_page: {e}")
        return None

def extract_petition_data(url):
    """Extract detailed petition information from a single petition page"""
    try:
        # Verify and fix URL if needed
        if 'orgfa/' in url:
            url = url.replace('orgfa/', 'org/fa/')
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        data = {}
        
        # Extract pubdate from script tag
        script_tag = soup.find('script', string=re.compile('window.pageTrackingData'))
        if script_tag:
            json_str = script_tag.string.split('window.pageTrackingData = ')[1].strip()
            tracking_data = json.loads(json_str)
            data['pubdate'] = tracking_data.get('pubdate')
        else:
            data['pubdate'] = None
            
        # Basic petition info
        title = soup.find('h1', class_='heading-1')
        data['title'] = title.text.strip() if title else None
        
        author = soup.find('address', class_='author')
        if author:
            author_text = author.find('span', class_='author__text')
            data['organizer'] = author_text.text.strip() if author_text else None
        
        progress = soup.find('progress', class_='progress__input')
        if progress:
            data['current_signatures'] = progress.get('value')
            data['signature_goal'] = progress.get('max')
        
        # Extract province data
        province_data = {}
        province_table = soup.find('table', class_='visually-hidden')
        if province_table:
            for row in province_table.find_all('tr')[1:]:
                cols = row.find_all('td')
                if len(cols) == 2:
                    province_name = cols[0].text.strip()
                    signatures = cols[1].text.strip()
                    signatures = int(signatures.translate(str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')))
                    province_data[province_name] = {'signatures': signatures}
        
        # Get opacity values from SVG
        svg = soup.find('svg', attrs={'viewBox': '0 0 860 860'})
        if svg and svg.find('style'):
            style_text = svg.find('style').text
            opacity_matches = re.findall(r'#([^,\s]+)[^{]*{[^}]*--province-opacity:\s*([\d.]+)', style_text)
            
            id_to_persian = {
                'alborz': 'استان البرز',
                'tehran': 'استان تهران',
                'qom': 'استان قم',
                'markazi': 'استان مرکزی',
                'mazandaran': 'استان مازندران',
                'golestan': 'استان گلستان',
                'gilan': 'استان گیلان',
                'ardabil': 'استان اردبیل',
                'east-azerbaijan': 'آذربایجان شرقی',
                'west-azerbaijan': 'استان آذربایجان غربی',
                'kurdistan': 'استان کردستان',
                'zanjan': 'استان زنجان',
                'kermanshah': 'استان کرمانشاه',
                'ilam': 'استان ایلام',
                'lorestan': 'استان لرستان',
                'hamadan': 'استان همدان',
                'qazvin': 'استان قزوین',
                'isfahan': 'استان اصفهان',
                'yazd': 'استان یزد',
                'fars': 'استان فارس',
                'bushehr': 'استان بوشهر',
                'hormozgan': 'استان هرمزگان',
                'kerman': 'استان کرمان',
                'sistan-and-baluchestan': 'استان سیستان و بلوچستان',
                'razavi-khoresan': 'استان خراسان رضوی',
                'north-khoresan': 'استان خراسان شمالی',
                'south-khoresan': 'استان خراسان جنوبی',
                'samnan': 'استان سمنان',
                'khuzestan': 'استان خوزستان',
                'chahar-mahaal-and-bakhtiari': 'استان چهارمحال و بختیاری',
                'kohgiluyeh-and-boyer-ahmad': 'استان کهگیلویه و بویراحمد'
            }
            
            for province_id, opacity in opacity_matches:
                if province_id in id_to_persian:
                    persian_name = id_to_persian[province_id]
                    opacity_value = float(opacity)
                    if persian_name in province_data:
                        province_data[persian_name]['opacity'] = opacity_value
        
        data['active_provinces'] = json.dumps(province_data, ensure_ascii=False)
        
        # Extract main petition content
        content_blocks = []
        for p in soup.find_all('p', class_='editor-content blocks__unit'):
            content = p.text.strip()
            if content:
                content_blocks.append(content)
        data['content'] = '\n'.join(content_blocks) if content_blocks else None
        
        # Extract signatures
        signatures_data = []
        signatures = soup.find('ol', class_='timeline-latest-signatures')
        if signatures:
            for sig in signatures.find_all('li'):
                signature_dict = {}
                name_elem = sig.find('span', class_='timeline-signature__author-name')
                signature_dict['name'] = name_elem.text.strip() if name_elem else None
                time_elem = sig.find('time')
                signature_dict['time'] = time_elem.text.strip() if time_elem else None
                comment_elem = sig.find('div', class_='timeline-signature__comment')
                signature_dict['comment'] = comment_elem.get_text(strip=True) if comment_elem else None
                signatures_data.append(signature_dict)
        
        data['comments'] = json.dumps(signatures_data, ensure_ascii=False)
        
        return data
    
    except Exception as e:
        print(f"Error in extract_petition_data for URL {url}: {e}")
        return None

def scrape_all_petitions(base_url, start_page=1, end_page=29):
    """Scrape all petitions including their detailed information"""
    all_petition_data = []
    failed_urls = []
    
    for page_num in range(start_page, end_page + 1):
        print(f"Scanning page {page_num} for petition URLs...")
        url = f"{base_url}{page_num}/"
        
        df = parse_dadkhast_page(url)
        if df is not None and not df.empty:
            for index, row in df.iterrows():
                petition_url = row['url']
                if petition_url:
                    if 'orgfa/' in petition_url:
                        petition_url = petition_url.replace('orgfa/', 'org/fa/')
                    
                    print(f"Scraping detailed data from: {petition_url}")
                    
                    petition_data = extract_petition_data(petition_url)
                    if petition_data:
                        petition_data['url'] = petition_url
                        all_petition_data.append(petition_data)
                    else:
                        failed_urls.append(petition_url)
                    
                    time.sleep(2)  # Be respectful to the server
        
        time.sleep(1)
    
    if failed_urls:
        print("\nFailed to scrape the following URLs:")
        for url in failed_urls:
            print(url)
    
    if all_petition_data:
        df = pd.DataFrame(all_petition_data)
        df.insert(0, 'index', range(1, len(df) + 1))
        return df, failed_urls
    else:
        return None, failed_urls

# Main execution
if __name__ == "__main__":
    base_url = "https://www.daadkhast.org/fa/petitions/"
    df, failed_urls = scrape_all_petitions(base_url)

    if df is not None:
        print(f"\nTotal petitions scraped: {len(df)}")
        print("\nColumns in the dataset:")
        print(df.columns.tolist())
        print("\nSample of first few records:")
        print(df.head())
        
        # Save to CSV
        output_file = 'complete_dadkhast_data2.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nData saved to {output_file}")
        
        # Save failed URLs if any
        if failed_urls:
            with open('failed_urls.txt', 'w', encoding='utf-8') as f:
                for url in failed_urls:
                    f.write(f"{url}\n")
            print("Failed URLs have been saved to 'failed_urls.txt'")
