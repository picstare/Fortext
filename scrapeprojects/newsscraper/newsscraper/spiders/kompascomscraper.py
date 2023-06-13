import re
from scrapy.spiders import SitemapSpider


class KompascomscraperSpider(SitemapSpider):
    name = "kompascomscraper"
    allowed_domains = ["kompas.com", "megapolitan.kompas.com", "internasional.kompas.com", "nasional.kompas.com", "news.kompas.com", "regional.kompas.com",
                       "bandung.kompas.com", "sorotpolitik.kompas.com", "kilasdaerah.kompas.com", "kilaskementerian.kompas.com", "kilaskementerian.kompas.com/kemnaker",
                       "kilasparlemen.kompas.com/news/", "sorotpolitik.kompas.com/web/","kilasdaerah.kompas.com/jawa-barat/","kilaskementerian.kompas.com/kemenkes/web", 
                       "kilaskementerian.kompas.com/kemenko-perekonomian/web", "kilaskementerian.kompas.com/kemenkumham/web", "kilaskementerian.kompas.com/ditjen-ebtke/web",
                        "kilaskementerian.kompas.com/kemenko-perekonomian/video","kilaskementerian.kompas.com/kemenparekraf/video", "surabaya.kompas.com","bola.kompas.com"]
    
    def __init__(self, query, *args, **kwargs):
        super(KompascomscraperSpider, self).__init__(*args, **kwargs)
        self.query = query

        # Convert query to pattern with dashes
        self.pattern = re.sub(r'\s+', '-', self.query)

        self.sitemap_urls = ['https://www.kompas.com/sitemap.xml']
        self.sitemap_rules = [(self.pattern, self.parse)]

    def parse(self, response):
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
            'query': self.query,  # Include the query field
            'title': title,
            'date': date,
            'image_url': image_url,
            'content': content,
            'author': author,
            'tags': tags,
            'url': url,
        }



# import re
# from scrapy.spiders import SitemapSpider


# class KompascomscraperSpider(SitemapSpider):
#     name = "kompascomscraper"
#     allowed_domains = ["kompas.com"]
#     query = "anies baswedan"

#     # Convert query to pattern with dashes
#     pattern = re.sub(r'\s+', '-', query)

#     sitemap_urls = ['https://www.kompas.com/sitemap.xml']
#     sitemap_rules = [(pattern, 'parse_article')]

#     def parse_article(self, response):
#         url = response.url
#         print('Parsing article:', url)

#          # Extract desired data from the article page
#         title = response.css('h1.read__title::text').get()
#         author = response.css('.read__credit__item > div#penulis > a::text').get()
#         image_url = response.css('.photo__wrap > img::attr(src)').get()
#         date = response.css('.read__time::text').get()
#         content = response.css('.read__content').get()
#         tags = ''

#         yield {
#             # 'query': query,  # Include the query field
#             'title': title,
#             'date': date,
#             'image_url': image_url,
#             'content': content,
#             'author': author,
#             'tags': tags,
#             'url': url,
#         }


