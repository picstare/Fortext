import re
import html.entities
import streamlit as st


def lmap(f, xs):
    return list(map(f, xs))


emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""

# The components of the tokenizer:
regex_strings = (
    # Phone numbers:
    r"""
    (?:
      (?:            # (international)
        \+?[01]
        [\-\s.]*
      )?            
      (?:            # (area code)
        [\(]?
        \d{3}
        [\-\s.\)]*
      )?    
      \d{3}          # exchange
      [\-\s.]*   
      \d{4}          # base
    )""",
    # Emoticons:
    emoticon_string,
    # HTML tags:
    r"""<[^>]+>""",
    # Twitter username:
    r"""(?:@[\w_]+)""",
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""",
    # Remaining word types:
    r"""
    (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """,
)

######################################################################
# This is the core tokenizing regex:

word_re = re.compile(
    r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE
)

# The emoticon string gets its own regex so that we can preserve case for them as needed:
emoticon_re = re.compile(regex_strings[1], re.VERBOSE | re.I | re.UNICODE)

# These are for regularizing HTML entities to Unicode:
html_entity_digit_re = re.compile(r"&#\d+;")
html_entity_alpha_re = re.compile(r"&\w+;")
amp = "&amp;"

######################################################################


class Tokenizer:
    def __init__(self, preserve_case=False):
        self.preserve_case = preserve_case

    def tokenize(self, s):
        """
        Argument: s -- any string or unicode object
        Value: a tokenize list of strings; conatenating this list returns the original string if preserve_case=False
        """
        # Try to ensure unicode:
        # try:
        # s = str(s,'utf-8')
        # except UnicodeDecodeError:
        # s = str(s).encode('string_escape')
        # s = str(s,'utf-8')
        # Fix HTML character entitites:
        s = self.__html2unicode(s)
        # Tokenize:
        words = word_re.findall(s)
        # Possible alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:
            words = lmap((lambda x: x if emoticon_re.search(x) else x.lower()), words)
        return words

    def __html2unicode(self, s):
        """
        Internal metod that seeks to replace all the HTML entities in
        s with their corresponding unicode characters.
        """
        # First the digits:
        ents = set(html_entity_digit_re.findall(s))
        if len(ents) > 0:
            for ent in ents:
                entnum = ent[2:-1]
                try:
                    entnum = int(entnum)
                    s = s.replace(ent, chr(entnum))
                except:
                    pass
        # Now the alpha versions:
        ents = set(html_entity_alpha_re.findall(s))
        ents = filter((lambda x: x != amp), ents)
        for ent in ents:
            entname = ent[1:-1]
            try:
                s = s.replace(
                    ent, chr(html.entities.entitydefs.name2codepoint[entname])
                )
            except:
                pass
            s = s.replace(amp, " and ")
        return s


######################################################################
TWEET_CRAP_RE = re.compile(r"\bRT\b", re.IGNORECASE)
URL_RE = re.compile(r"(^|\W)https?://[\w./&%]+\b", re.IGNORECASE)
PURE_NUMBERS_RE = re.compile(r"(^|\W)\$?[0-9]+\%?", re.IGNORECASE)
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002500-\U00002BEF"  # chinese char
    "\U00002702-\U000027B0"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"  # dingbats
    "\u3030"
    "]+",
    re.UNICODE,
)
OTHER_REMOVALS_RE = re.compile("[" "\u2026" "]+", re.UNICODE)  # Ellipsis
SHORTHAND_STOPWORDS_RE = re.compile(
    r"(?:^|\b)("
    "w|w/|"  # Short for "with"
    "bc|b/c|"  # Short for "because"
    "wo|w/o"  # Short for "without"
    r")(?:\b|$)",
    re.IGNORECASE,
)
AT_MENTION_RE = re.compile(r"(^|\W)@\w+\b", re.IGNORECASE)
HASH_TAG_RE = re.compile(r"(^|\W)#\w+\b", re.IGNORECASE)
PREFIX_CHAR_RE = re.compile(r"(^|\W)[#@]", re.IGNORECASE)


def clean_tweet_text(text):
    regexes = [
        EMOJI_RE,
        PREFIX_CHAR_RE,
        PURE_NUMBERS_RE,
        TWEET_CRAP_RE,
        OTHER_REMOVALS_RE,
        SHORTHAND_STOPWORDS_RE,
        URL_RE,
    ]

    for regex in regexes:
        text = regex.sub("", text)

        return text


#     def tokenization(text):
#         text = re.split('\W+', text)
#         return text
