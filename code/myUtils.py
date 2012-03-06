import string
import math

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