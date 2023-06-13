
import re
from scrapy.spiders import SitemapSpider

class TempoSpider(SitemapSpider):
    name = "tempo"
    allowed_domains = ["tempo.co"]
    

    def __init__(self, query, *args, **kwargs):
        super(TempoSpider, self).__init__(*args, **kwargs)
        self.query = query

        # Convert query to pattern with dashes
        self.pattern = re.sub(r'\s+', '-', self.query)

        self.sitemap_urls = ['https://www.tempo.co/sitemap.xml']
        self.sitemap_rules = [(self.pattern, self.parse)]

    def parse(self, response):
        url = response.url
        print('Parsing article:', url)

         # Extract desired data from the article page
        title = response.css('div.detail-title > h1::text').get()
        author = response.css('main article:nth-child(1) p.title.bold a span::text').get()
        image_url = response.css('main article:first-child img::attr(src)').get()
        date = response.css('div.detail-title > p::text').get()
        content = ' '.join(response.css('.detail-konten p::text').getall())
        tags = ''

        yield {
            'query': self.query,
            'title': title,
            'date': date,
            'image_url': image_url,
            'content': content,
            'author': author,
            'tags': tags,
            'url': url,
        }