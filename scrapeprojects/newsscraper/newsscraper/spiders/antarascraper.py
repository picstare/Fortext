import scrapy
import re
import time

class AntarascraperSpider(scrapy.Spider):
    name = "antarascraper"
    allowed_domains = ["antaranews.com"]

    def start_requests(self):
        query = getattr(self, 'query', None)
        if query:
            url = f"https://www.antaranews.com/search?q={query}"
            yield scrapy.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'},
                callback=self.parse,
                meta={'query': query}
            )

    def parse(self, response):
        query = response.meta.get('query')

        # Extract articles
        articles = response.css('article')
        for article in articles:
            article_url = article.css('h3 a::attr(href)').get()
            yield response.follow(article_url, callback=self.parse_article, meta={'query': query})
            time.sleep(1)

        # Extract total pages
        total_pages_element = response.css('ul.pagination.pagination-sm li:last-child a::attr(href)').get()
        total_pages = int(re.search(r'page=(\d+)', total_pages_element).group(1)) if total_pages_element else None

        # Extract current page
        current_page_element = response.css('ul.pagination.pagination-sm li.active a::text').get()
        current_page = int(current_page_element) if current_page_element else None

        if total_pages and current_page:
            for i in range(current_page + 1, total_pages + 1):
                next_page_url = f'https://www.antaranews.com/search?q={query}&page={i}'
                yield scrapy.Request(url=next_page_url, callback=self.parse_page, meta={'query': query})

    def parse_page(self, response):
        query = response.meta.get('query')

        # Extract articles from the current page
        articles = response.css('article')
        for article in articles:
            article_url = article.css('h3 a::attr(href)').get()
            yield response.follow(article_url, callback=self.parse_article, meta={'query': query})

    def parse_article(self, response):
        query = response.meta.get('query')
        article_url = response.url

        # Extract article details

        yield {
            'query': query,
            'title': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > header > h1::text').get(),
            'date': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > header > p > span > span::text').get(),
            'image_url': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > header > figure img::attr(data-src)').get(),
            'content': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > div').get(default=''),
            'author': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > div > p::text').get(),
            'tags': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > footer > div.tags-wrapper > ul > li > a::text').get(),
            'url': article_url
        }



































# import scrapy
# import re



# class AntarascraperSpider(scrapy.Spider):
#     name = "antarascraper"
#     allowed_domains = ["antaranews.com"]

#     def start_requests(self):
#         query = getattr(self, 'query', None)
#         if query:
#             url = f"https://www.antaranews.com/search?q={query}"
#             yield scrapy.Request(url, callback=self.parse, meta={'query': query})

#     def parse(self, response):
#         query = response.meta.get('query')

#         # Extract articles
#         articles = response.css('article')
#         for article in articles:
#             article_url = article.css('h3 a::attr(href)').get()
#             yield response.follow(article_url, callback=self.parse_article, meta={'query': query})

#         # Extract total pages
#         total_pages_element = response.css('ul.pagination.pagination-sm li:last-child a::attr(href)').get()
#         total_pages = int(re.search(r'page=(\d+)', total_pages_element).group(1)) if total_pages_element else None

#         # Extract current page
#         current_page_element = response.css('ul.pagination.pagination-sm li.active a::text').get()
#         current_page = int(current_page_element) if current_page_element else None

#         if total_pages and current_page:
#             for i in range(current_page + 1, total_pages + 1):
#                 next_page_url = f'https://www.antaranews.com/search?q={query}&page={i}'
#                 yield scrapy.Request(url=next_page_url, callback=self.parse)

#     def parse_article(self, response):
#         query = response.meta.get('query')
#         article_url = response.url

#         # Extract article details
    
#         yield {
#             'query': query,
#             'title': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > header > h1::text').get(),
#             'date': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > header > p > span > span::text').get(),
#             'image_url': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > header > figure img::attr(src)').get(),
#             'content': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > div').get(),
#             'author': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > div > p::text').get(),
#             'tags': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > footer > div.tags-wrapper > ul > li > a::text').get(),
#             'url': article_url,
#         }


        




































# # import scrapy
# # import re


# # class AntarascraperSpider(scrapy.Spider):
# #     name = "antarascraper"
# #     allowed_domains = ["antaranews.com"]

# #     def start_requests(self):
# #         url_template = "https://www.antaranews.com/search?q={query}"
# #         query = getattr(self, 'query', None)
# #         if query:
# #             url = url_template.format(query=query)
# #             yield scrapy.Request(url, callback=self.parse, meta={'query': query})

# #     def parse(self, response):
# #         query = response.meta.get('query')
        
# #         # Extract articles
# #         for article in response.css('article'):
# #             article_url = article.css('h3 a::attr(href)').get()
# #             yield response.follow(article_url, callback=self.parse_article, meta={'query': query})

# #         # Extract total pages
# #         total_pages = None
# #         total_pages_element = response.css('ul.pagination.pagination-sm li:last-child a::attr(href)').get()
# #         if total_pages_element:
# #             total_pages_match = re.search(r'page=(\d+)', total_pages_element)
# #             if total_pages_match:
# #                 total_pages = int(total_pages_match.group(1))

# #         # Extract current page
# #         current_page = None
# #         current_page_element = response.css('ul.pagination.pagination-sm li.active a::text').get()
# #         if current_page_element:
# #             current_page = int(current_page_element)

# #         if total_pages and current_page:
# #             for i in range(current_page + 1, total_pages + 1):
# #                 next_page_url = f'https://www.antaranews.com/search?q={query}&page={i}'
# #                 yield scrapy.Request(url=next_page_url, callback=self.parse) 

# #     def parse_article(self, response):
# #         query = response.meta.get('query')
        
# #         # Extract article details
# #         yield {
               
# #             'title': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > header > h1').get(),
# #             'date': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > header > p > span > span').get(),
# #             'image_url': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > header > figure').get(),
# #             'content': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > div').get(),
# #             'author': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > div > p').get(),
# #             'tags': response.css('#main-container > div.main-content.mag-content.clearfix > div > div.col-md-8 > article > footer > div.tags-wrapper').get(),
# #             }




            



# import scrapy
# import re


# class AntarascraperSpider(scrapy.Spider):
#     name = "antarascraper"
#     allowed_domains = ["antaranews.com"]

#     def start_requests(self):
#         query = getattr(self, 'query', None)
#         if query:
#             url = f"https://www.antaranews.com/search?q={query}"
#             yield scrapy.Request(url, callback=self.parse)

#     def parse(self, response):
#         # Extract articles
#         for article in response.css('article'):
#             yield {
#                 'title': article.css('h3 a::text').get(),
#                 'url': article.css('h3 a::attr(href)').get(),
#                 'image_url': article.css('img::attr(src)').get()
#             }

#         # Extract total pages
#         total_pages = None
#         total_pages_element = response.css('ul.pagination.pagination-sm li:last-child a::attr(href)').get()
#         if total_pages_element:
#             total_pages_match = re.search(r'page=(\d+)', total_pages_element)
#             if total_pages_match:
#                 total_pages = int(total_pages_match.group(1))

#         # Extract current page
#         current_page = None
#         current_page_element = response.css('ul.pagination.pagination-sm li.active a::text').get()
#         if current_page_element:
#             current_page = int(current_page_element)

#         if total_pages and current_page:
#             for i in range(current_page + 1, total_pages + 1):
#                 next_page_url = f'https://www.antaranews.com/search?q={query}={i}' # type: ignore
#                 yield scrapy.Request(url=next_page_url, callback=self.parse)

