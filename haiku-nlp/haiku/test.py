
text = """
Such precious gemstones
Morning dew shines like diamonds
God’s tears from heaven
"""


text = """
Deep wounds that leave scars
One moment much more must flow
Shadows from the soul
"""


text = """
Please do not grieve more
And afterwards remember
Pearls of morning’s dew
"""

text = """
Her tears in silence
The strong arms that held her up
In the thoughts and deeds
"""

text = text.replace('\n', ' ')
while '  ' in text:
    text = text.replace('  ', ' ')
text = text.strip()
print(text.strip())
print(text.split(' '))

"""
Night; and once again,
the while I wait for you, cold wind
turns into rain.
"""


print([5, 17] < [5, 17, 18])
