import unittest
from reader import Reader


class TestReader(unittest.TestCase):
    def test_preprocess(self):
        text1 = "@User1 🚓 :) This a test message!!1! https://www.example.com/test/test #ThisIsAHashtag!"
        text2 = "This is another test message for @User2 :( #anotherbunchofwords #yetanotherone"
        text1_expected = "$MENTION$ $EMOJI$ $SMILEY$ This a test message!!1! $URL$ this is a hash tag!"
        text2_expected = "This is another test message for $MENTION$ $SMILEY$ another bunch of words yet another one"
        self.assertEqual(
            Reader.preprocess([text1, text2]), [text1_expected, text2_expected]
        )

    def test_split_hashtags(self):
        text1 = "@User1 This a test message!!1! #ThisIsAHashtag!"
        text1_expected = "@User1 This a test message!!1! this is a hash tag!"
        self.assertEqual(Reader.split_hashtags(text1), text1_expected)


if __name__ == "__main__":
    unittest.main()
