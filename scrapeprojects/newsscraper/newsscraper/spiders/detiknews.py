import re
import time
import json

from scrapy.spiders import SitemapSpider

class DetiknewsSpider(SitemapSpider):
    name = "detiknews"
    allowed_domains = [
        "detik.com", "news.detik.com", "news.detik.com/berita", 
        "news.detik.com/jawabarat", "news.detik.com/jawatimur", 
        "news.detik.com/jawatengah", "news.detik.com/internasional", 
        "www.detik.com"
    ]

    custom_settings = {
        "USER_AGENT": "Googlebot",
    }

    def __init__(self, query, *args, **kwargs):
        super(DetiknewsSpider, self).__init__(*args, **kwargs)
        self.query = query

        # Convert query to pattern with dashes
        self.pattern = re.sub(r"\s+", "-", self.query)

        self.sitemap_urls = ["https://news.detik.com/berita/sitemap_news.xml"]
        self.sitemap_rules = [(self.pattern, self.parse)]

        # self.start_time = time.time()
        # self.time_limit = 10 * 60  # 10 minutes in seconds

        # self.output_file = "newscraped/tmp.json"
        # self.output_data = []

    def parse(self, response):
        url = response.url

        # Check if the URL belongs to the allowed domains
        if not any(domain in url for domain in self.allowed_domains):
            return

        # Check if the URL belongs to the ignored domain
        if "https://forum.detik.com" in url:
            return
        
        if "https://news.detik.com/x/detail/" in url:
            return

        print("Parsing article:", url)

        # Extract data from the article page
        title = response.css("h1.detail__title::text").get()
        clean_title = title.strip() if title else ""
        author = response.css("div.detail__author::text").get()
        image_url = response.css(".detail__media > figure > img").get()
        date = response.css("div.detail__date::text").get()
        content = response.css("div.detail__body-text.itp_bodycontent").getall()

        # Process the content list
        processed_content = " ".join(content).strip()

        tags = response.css("div.detail__body-tag.mgt-16 a::text").getall()

        # Process and add the extracted data to the output data list
        yield {
            "query": self.query,
            "title": clean_title,
            "date": date,
            "image_url": image_url,
            "content": processed_content,
            "author": author,
            "tags": tags,
            "url": url,
        }
        
    def should_follow(self, response):
        # Check if the URL belongs to the allowed domains
        if any(domain in response.url for domain in self.allowed_domains):
            return True
        return False

# # Check if the time limit has been reached
        # elapsed_time = time.time() - self.start_time
        # if elapsed_time >= self.time_limit:
        #     self.save_output_data()
        #     self.crawler.engine.close_spider(self, "Time limit reached")



    # def save_output_data(self):
    #     with open(self.output_file, "w") as f:
    #         for item in self.output_data:
    #             f.write(json.dumps(item) + "\n")
    #         f.write(json.dumps({"end_of_file": True}))





# class DetiknewsSpider(SitemapSpider):
#     name = "detiknews"
#     allowed_domains = ["detik.com"]

#     custom_settings = {
#         'USER_AGENT': 'Googlebot',
#     }
    
#     def start_requests(self):
#         query = getattr(self, 'query', None)
#         sitemap_url = 'https://news.detik.com/sitemap.xml'
#         yield scrapy.Request(url=sitemap_url, callback=self.parse_sitemap, meta={'query': query})

#     def parse_sitemap(self, response):
#         query = response.meta.get('query')
#         pattern = re.sub(r'\s+', '-', query) if query else None

#         sitemap_urls = response.xpath('//loc/text()').getall()
#         for sitemap_url in sitemap_urls:
#             yield scrapy.Request(url=sitemap_url, callback=self.parse_sitemap_item, meta={'pattern': pattern})

#     def parse_sitemap_item(self, response):
#         pattern = response.meta.get('pattern')
#         if pattern:
#             sitemap_rules = [(pattern, 'parse_article')]
#         else:
#             sitemap_rules = [('.*', 'parse_article')]

#         for rule in sitemap_rules:
#             rule_pattern, callback = rule
#             if re.search(rule_pattern, response.url):
#                 callback_method = getattr(self, callback)
#                 if callback_method:
#                     return callback_method(response)

#     def parse_article(self, response):
#         query = response.meta.get('query')
#         url = response.url
#         print('Parsing article:', url)



#         # Extract data from the article page
        
#         title = response.css('h1.detail__title::text').get()
#         clean_title = title.strip() if title else ""
#         author = response.css('div.detail__author::text').get()
#         image_url = response.css('.detail__media > figure > img').get()
#         date = response.css('div.detail__date::text').get()
#         content = response.css('div.detail__body-text.itp_bodycontent').get()
#         content = ' '.join(content).strip()
#         tags=response.css('div.detail__body-tag.mgt-16 a::text').getall()
#         # Process and yield the extracted data
#         yield {
#             'query': query,
#             'title': clean_title,
#             'date': date,
#             'image_url': image_url,
#             'content': content,
#             'author': author,
#             'tags': tags,
#             'url': url
#         }
    