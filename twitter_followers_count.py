import tweepy
import configparser
import json

cfg_parser = configparser.ConfigParser()
cfg_parser.read('config.txt')
consumer_key = cfg_parser['twitterConfig']['api_key']
consumer_secret = cfg_parser['twitterConfig']['api_secret_key']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

api = tweepy.API(auth,  wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

user_twitter = json.load(open('twitter_author_name.json'))
twitter_username = list(user_twitter.values())

followers_number = {}
for username in twitter_username:
    followers_number[username] = 0

for username in followers_number.keys():
    try:
        user = api.get_user(username)
        followers_number[username] = user.followers_count
    except:
        pass

follower_copy = dict(followers_number)
for key, value in followers_number.items():
    if value == 0:
        del follower_copy[key]

followers_number = follower_copy
del follower_copy

with open("twitter_followers.json", 'w') as f:
    json.dump(followers_number, f)
    f.close()


