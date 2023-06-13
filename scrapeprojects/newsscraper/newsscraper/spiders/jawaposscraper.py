


# class JawaposscraperSpider(SitemapSpider):
#     name = "jawaposscraper"
#     allowed_domains = ["jawapos.com"]
#     sitemap_urls = ["https://jawapos.com/sitemap.xml"]
    

#     def parse(self, response):
#         query = getattr(self, 'query', None)
#         pattern = re.sub(r'\s+', '-', query) if query else None

#         sitemap_urls = response.xpath('//loc/text()').getall()

#         for sitemap_url in sitemap_urls:
#             yield scrapy.Request(url=sitemap_url, callback=self.parse_sitemap_item, meta={'pattern': pattern})

#     def parse_sitemap_item(self, response):
#         pattern = response.meta.get('pattern')
#         if pattern:
#             sitemap_rules = [(pattern, self.parse_article)]
#         else:
#             sitemap_rules = [('.*', self.parse_article)]

#         for rule in sitemap_rules:
#             rule_pattern, callback = rule
#             if re.search(rule_pattern, response.url):
#                 callback_method = getattr(self, callback)
#                 if callback_method:
#                     return callback_method(response)

#     def parse_article(self, response):
#         query = getattr(self, 'query', None)
#         url = response.url
#         print('Parsing article:', url)

#         # Extract desired data from the article page
#         title = response.css('h1.read__title::text').get()
#         author = response.css('.read__info__author::text').get()
#         image_url = response.css('.photo__wrap > img::attr(src)').get()
#         date = response.css('.read__info_date::text').get()
#         content = response.css('.read__content').get()
#         tags = ''

#         yield {
#             'query': query,  # Include the query field
#             'title': title,
#             'date': date,
#             'image_url': image_url,
#             'content': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > div').get(default=''),
#             'author': author,
#             'tags': tags,
#             'url': url,
#         }


import re
from scrapy.spiders import SitemapSpider


class JawaposscraperSpider(SitemapSpider):
    name = "jawaposscraper"
    allowed_domains = ["jawapos.com", 'https://www.jawapos.com/sitemap-news.xml']
    
    def __init__(self, query, *args, **kwargs):
        super(JawaposscraperSpider, self).__init__(*args, **kwargs)
        self.query = query

        # Convert query to pattern with dashes
        self.pattern = re.sub(r'\s+', '-', self.query)

        self.sitemap_urls = ['https://jawapos.com/sitemap.xml']
        self.sitemap_rules = [(self.pattern, self.parse)]

    def parse(self, response):
        url = response.url
        print('Parsing article:', url)

         # Extract desired data from the article page
        title = response.css('h1.read__title::text').get()
        author = response.css('.read__info__author::text').get().strip()
        image_url = response.css('.photo__img > img::attr(src)').get()
        date = response.css('div.read__info__date::text').get().replace('-', '').strip()
        content = ''.join(response.css('.read__content.clearfix p::text').getall())
        tags = response.css('.tag.tag--article li h4 a::text').getall()

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