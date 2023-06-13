import scrapy
import re
from scrapy.spiders import SitemapSpider


class InfogresikSpider(SitemapSpider):
    name = "infogresik"
    allowed_domains = ["infogresik.id"]
    start_urls = ["https://infogresik.id"]

    
    def __init__(self, query, *args, **kwargs):
        super(InfogresikSpider, self).__init__(*args, **kwargs)
        self.query = query

        # Convert query to pattern with dashes
        self.pattern = re.sub(r'\s+', '-', self.query)

        self.sitemap_urls = ['https://infogresik.id/post-sitemap.xml']
        self.sitemap_rules = [(self.pattern, self.parse)]

    def parse(self, response):
        url = response.url
        print('Parsing article:', url)

         # Extract desired data from the article page
        title = response.css('h1.post-title.single-post-title.entry-title::text').get()
        author =  response.css('a.tdb-author-name::text').get()
        image_url = response.css('.post-image > a > img').get()
        date = response.css('.entry-date::text').get()
        content = ''.join(response.css('#penci-post-entry-inner p::text').getall())
        tags = ''

        yield {
            'query': self.query,  # Include the query field
            'title': title,
            'date': date,
            'image_url': image_url,
            'content': content,
            'author': author,
            'tags': tags,
            'url': url
        }

