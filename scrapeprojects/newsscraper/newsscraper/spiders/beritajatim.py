import re
from scrapy.spiders import SitemapSpider


class BeritajatimSpider(SitemapSpider):
    name = "beritajatim"
    allowed_domains = ["beritajatim.com"]

    def __init__(self, query, *args, **kwargs):
        super(BeritajatimSpider, self).__init__(*args, **kwargs)
        self.query = query

        # Convert query to pattern with dashes
        self.pattern = re.sub(r'\s+', '-', self.query)

        self.sitemap_urls = ['https://beritajatim.com/sitemap.xml']
        self.sitemap_rules = [(self.pattern, self.parse)]

    def parse(self, response):
        url = response.url
        print('Parsing article:', url)

         # Extract desired data from the article page
        title = response.css('h1.entry-title::text').get()
        author = response.css('.date-time p::text').get()
        image_url = response.css('.post-thumbnail > img::attr(src)').get()
        date = response.css('.date-time time::text').get()
        content = ''.join(response.css('.entry-content p::text').getall())
        tags = ''

        yield {
            'query': self.query,  # Include the query field
            'title': title,
            'date': date,
            'image_url': image_url,
            'content': content,
            'author': author,
            'tags': tags,
            'url': url,
        }