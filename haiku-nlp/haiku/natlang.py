
from itertools import product

from nltk.corpus import cmudict
import nltk


WORDS = [
    'CC', 'CD', 'DT', 'EX', 'FW',
    'IN', 'JJ', 'JJR', 'JJS',
    'LS', 'MD', 'NN', 'NNP',
    'NNPS', 'NNS', 'PDT',
    'POS', 'PRP', 'PRP$', 'RB',
    'RBR', 'RBS', 'RP', 'SYM',
    'TO', 'UH', 'VB', 'VBD',
    'VBG', 'VBN', 'VBP', 'VBZ',
    'WDT', 'WP', 'WP$', 'WRB'
]

PUNCS = [ '``', '$', "''", '(', ')', ',', '--', '.', ':' ]

def countsyl(word, tag, d = cmudict.dict()):

    def count(pronunciation):
        syllables = 0
        for phoneme in pronunciation:
            if phoneme[-1] in "1234567890":
                syllables += 1
        return syllables

    return sorted({
        count(pronunciation)
            for pronunciation
            in d[word.lower()]
    }) if tag in WORDS else [0]

def partial(l):
    fullsum = 0
    partsum = []
    for num in l:
        fullsum += num
        partsum.append(fullsum)
    return partsum

class form:

    __slots__ = [ "word", "post", "seqs", "syls", "mpos", "broken" ]


    syldict = nltk.corpus.cmudict.dict()

    def __init__(self, word):

        def destroy(self):
            self.broken = True


        def syld(word, tag):
            """
            returns a list of possible syllable counts for the given word
            """
            return countsyl(word, tag, form.syldict)

        def tokenize(word):
            tokens = nltk.word_tokenize(word)
            return nltk.pos_tag(tokens)

        def possyls(post):
            for word, tag in post:
                if tag in WORDS:
                    break
            return tag

        def initialize(self):
            self.word = ""
            self.post = []
            self.seqs = []
            self.syls = []
            self.mpos = ""
            self.broken = True

        def main(self, word):
            initialize(self)
            if len(word) > 500: return destroy(self)

            valid_a = lambda pair: pair[1] in PUNCS or pair[0].lower() in form.syldict
            valid_b = lambda pair: pair[1] in WORDS

            self.word = word                                            # the word
            self.post = tokenize(word)                                  # part of speech tag list
            if not self.post or not all(map(valid_a, self.post)):
                return destroy(self)
            if sum(map(valid_b, self.post)) > 1:
                return destroy(self)

            self.seqs = [syld(word, tag) for word, tag in self.post]    # sequence of syllables corresponding to self.post
            self.syls = sorted({*map(sum, product(*self.seqs))})        # all possible number of syllables
            self.mpos = possyls(self.post)                              # major part of speech

            self.broken = False

        return main(self, word)

    def __bool__(self):
        return not self.broken

    def __str__(self):
        return self.word

    def __repr__(self):
        return f'form("{self.word}")'


class haiku:

    __slots__ = ["texts", "words", "forms", "count", "lines", "broken"]

    def __init__(self, text):

        def initialize(self):
            self.texts = ""
            self.words = []
            self.forms = []
            self.count = []
            self.lines = []
            self.broken = True

        def destroy(self):
            self.broken = True

        def make(forms):

            def valid(pair):
                diff, seq = pair
                return 5 in seq and 12 in seq and seq[-1] == 17

            count = [word.syls for word in self.forms]
            prods = [*product(*count)]
            parts = [*map(partial, prods)]
            valid = [*filter(lambda x: valid(x), zip(prods, parts))]
            lines = []
            for diff, part in valid:
                part.reverse()
                a = len(part) - part.index(5)
                b = len(part) - part.index(12)
                part.reverse()
                string = [*map(str, forms)]
                lines.append((
                    " ".join(string[:a]),
                    " ".join(string[a:b]),
                    " ".join(string[b:])))
            return lines


        def main(self, text):
            initialize(self)

            if len(text) > 500:
                return destroy(self)

            self.texts = text
            self.words = text.split()
            self.forms = [*map(form, self.words)]

            if any(word.broken for word in self.forms):
                return destroy(self)

            self.count = [word.syls for word in self.forms]
            self.lines = make(self.forms)

            if not self.lines:
                return destroy(self)

        return main(self, text)

    def __bool__(self):
        return not self.broken

print("done import", flush=True)
text = """
Night; and once again, the while I wait for you, cold wind turns into rain.
"""

for i in range(1000):
    if i % 30 == 0: print(i)
    k = haiku(text)
