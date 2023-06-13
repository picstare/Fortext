import scrapy
import json

class InstagramSpider(scrapy.Spider):
    name = 'instagram'

    def start_requests(self):
        username = getattr(self, 'username', None)
        if username is not None:
            url = f'https://www.instagram.com/{username}/'
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        if response.status == 403:
            self.logger.info("Access to profile page is forbidden. Check robots.txt file.")
            # Handle the forbidden response here, e.g., logging, raising exception, etc.
        else:
            json_data = response.xpath('//script[contains(text(), "window._sharedData")]/text()').get()
            profile_data = self.extract_profile_data(json_data)
            yield profile_data

    def extract_profile_data(self, json_data):
        data = json.loads(json_data.strip().split('= ')[1][:-1])
        user_data = data['entry_data']['ProfilePage'][0]['graphql']['user']
        
        username = user_data['username']
        full_name = user_data['full_name']
        biography = user_data['biography']
        follower_count = user_data['edge_followed_by']['count']
        following_count = user_data['edge_follow']['count']
        posts_count = user_data['edge_owner_to_timeline_media']['count']
        profile_picture_url = user_data['profile_pic_url_hd']
        
        profile_data = {
            'username': username,
            'full_name': full_name,
            'biography': biography,
            'follower_count': follower_count,
            'following_count': following_count,
            'posts_count': posts_count,
            'profile_picture_url': profile_picture_url
        }
        
        return profile_data