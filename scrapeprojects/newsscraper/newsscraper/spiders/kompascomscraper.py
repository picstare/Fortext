import re
import scrapy
from scrapy.spiders import SitemapSpider


class KompascomscraperSpider(SitemapSpider):
    name = "kompascomscraper"
    allowed_domains = ["kompas.com"]
    sitemap_urls = ['https://www.kompas.com/sitemap.xml']

    def parse(self, response):
        query = getattr(self, 'query', None)
        pattern = re.sub(r'\s+', '-', query) if query else None

        sitemap_urls = response.xpath('//loc/text()').getall()

        for sitemap_url in sitemap_urls:
            yield scrapy.Request(url=sitemap_url, callback=self.parse_sitemap_item, meta={'pattern': pattern})

    def parse_sitemap_item(self, response):
        pattern = response.meta.get('pattern')
        if pattern:
            sitemap_rules = [(pattern, self.parse_article)]
        else:
            sitemap_rules = [('.*', self.parse_article)]

        for rule in sitemap_rules:
            rule_pattern, callback = rule
            if re.search(rule_pattern, response.url):
                callback_method = getattr(self, callback)
                if callback_method:
                    return callback_method(response)

    def parse_article(self, response):
        query = getattr(self, 'query', None)
        url = response.url
        print('Parsing article:', url)

        # Extract desired data from the article page
        title = response.css('h1.read__title::text').get()
        author = response.css('.read__credit__item > div#penulis > a::text').get()
        image_url = response.css('.photo__wrap > img::attr(src)').get()
        date = response.css('.read__time::text').get()
        content = response.css('.read__content').get()
        tags = ''

        yield {
            'query': query,  # Include the query field
            'title': title,
            'author': author,
            'image_url': image_url,
            'date': date,
            'content': content,
            'tags': tags,
            'url': url,
        }



# import re
# from scrapy.spiders import SitemapSpider


# class KompascomscraperSpider(SitemapSpider):
#     name = "kompascomscraper"
#     allowed_domains = ["kompas.com"]
#     query = "ganjar pranowo"

#     # Convert query to pattern with dashes
#     pattern = re.sub(r'\s+', '-', query)

#     sitemap_urls = ['https://www.kompas.com/sitemap.xml']
#     sitemap_rules = [(pattern, 'parse_article')]

#     def parse_article(self, response):
#         url = response.url
#         print('Parsing article:', url)

#         # Extract desired data from the article page
#         title = response.css('h1.read__title::text').get()
#         author = response.css('.read__credit__item > div#penulis > a').get()
#         image_url = response.css('.photo__wrap > img::attr(src)').get()
#         date = response.css('.read__time::text').get()
#         content = response.css('.read__content').get()

#         yield {
#             'url': url,
#             'title': title,
#             'author': author,
#             'image_url':image_url,
#             'date': date,
#             'content': content,
#         }

