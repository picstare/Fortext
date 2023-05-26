import re
import scrapy
from scrapy.spiders import SitemapSpider

class DetiknewsSpider(SitemapSpider):
    name = "detiknews"
    allowed_domains = ["detik.com"]

    custom_settings = {
        'USER_AGENT': 'Googlebot',
    }
    
    def start_requests(self):
        query = getattr(self, 'query', None)
        sitemap_url = 'https://news.detik.com/sitemap.xml'
        yield scrapy.Request(url=sitemap_url, callback=self.parse_sitemap, meta={'query': query})

    def parse_sitemap(self, response):
        query = response.meta.get('query')
        pattern = re.sub(r'\s+', '-', query) if query else None

        sitemap_urls = response.xpath('//loc/text()').getall()
        for sitemap_url in sitemap_urls:
            yield scrapy.Request(url=sitemap_url, callback=self.parse_sitemap_item, meta={'pattern': pattern})
    def parse_sitemap_item(self, response):
        pattern = response.meta.get('pattern')
        if pattern:
            sitemap_rules = [(pattern, 'parse_article')]
        else:
            sitemap_rules = [('.*', 'parse_article')]

        for rule in sitemap_rules:
            rule_pattern, callback = rule
            if re.search(rule_pattern, response.url):
                callback_method = getattr(self, callback)
                if callback_method:
                    return callback_method(response)

    def parse_article(self, response):
        query = response.meta.get('query')
        url = response.url
        print('Parsing article:', url)



        # Extract data from the article page
        
        title = response.css('h1.detail__title::text').get()
        clean_title = title.strip() if title else ""
        author = response.css('div.detail__author::text').get()
        image_url = response.css('.detail__media > figure > img').get()
        date = response.css('div.detail__date::text').get()
        content = response.css('div.detail__body-text.itp_bodycontent').get()
        content = ' '.join(content).strip()
        tags=response.css('div.detail__body-tag.mgt-16 a::text').getall()
        # Process and yield the extracted data
        yield {
            'query': query,
            'title': clean_title,
            'date': date,
            'image_url': image_url,
            'content': content,
            'author': author,
            'tags': tags,
            'url': url
        }
       
