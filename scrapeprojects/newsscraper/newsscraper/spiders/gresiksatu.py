import re
from scrapy.spiders import SitemapSpider


class GresiksatuSpider(SitemapSpider):
    name = "gresiksatu"
    allowed_domains = ['gresiksatu.com']
    

    def __init__(self, query, *args, **kwargs):
        super(GresiksatuSpider, self).__init__(*args, **kwargs)
        self.query = query

        # Convert query to pattern with dashes
        self.pattern = re.sub(r'\s+', '-', self.query)

        self.sitemap_urls = ['https://www.gresiksatu.com/sitemaps.xml']
        self.sitemap_rules = [(self.pattern, self.parse)]


    def parse(self, response):
        url = response.url
        print('Parsing article:', url)

         # Extract desired data from the article page
        title = response.css('#tdi_94 h1::text').get()
        author = response.css('#tdi_94 a::attr(href)').get()
        image_url = response.css('#tdi_94 img::attr(src)').get()
        date = response.css('time.entry-date::text').get()
        content = ' '.join(response.css('div.tdb-block-inner.td-fix-index p span.s1::text').getall())
        tags = ''

        yield {
            # 'query': query,  # Include the query field
            'title': title,
            'date': date,
            'image_url': image_url,
            'content': content,
            'author': author,
            'tags': tags,
            'url': url,
        }