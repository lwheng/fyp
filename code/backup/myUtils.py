import string
import math
import re

def hyphenated(word):
    tokens = word.split("-")
    for t in tokens:
        if not t.isalnum():
            return False
    return True

def apos(word):
    tokens = word.split("'")
    if len(tokens) != 2:
        return False
    for t in tokens:
        if not t.isalnum():
            return False
    return True

def removepunctuation(word):
    output = ""
    for w in word:
        if not w in string.punctuation:
            output += w
    return output

def removespecialcharacters(word):
    specialchars = ["\n", "\t", "\r"]
    output = word
    for e in specialchars:
        output = output.replace(e, "")
    return output

def strip_control_characters(input):
    if input:
        # unicode invalid characters
        RE_XML_ILLEGAL = u'([\u0000-\u0008\u000b-\u000c\u000e-\u001f\ufffe-\uffff])' + \
                        u'|' + \
                        u'([%s-%s][^%s-%s])|([^%s-%s][%s-%s])|([%s-%s]$)|(^[%s-%s])' % \
                        (unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
                        unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
                        unichr(0xd800),unichr(0xdbff),unichr(0xdc00),unichr(0xdfff),
                        )
        input = re.sub(RE_XML_ILLEGAL, "", input)

        # ascii control characters
        input = re.sub(r"[\x01-\x1F\x7F]", "", input)