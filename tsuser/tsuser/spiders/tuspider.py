import scrapy


class TuspiderSpider(scrapy.Spider):
    name = 'tuspider'
    allowed_domains = ['twitter.com']
    start_urls = ['http://twitter.com/']

    def parse(self, response):
        pass
