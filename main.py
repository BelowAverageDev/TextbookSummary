from io import TextIOWrapper
from typing import Dict, List, Union
from utils.summarizer import Summmarizer
from utils.parser import parse_book

def unpack(book: Dict[str, Union[str, List]], filename: str):
    """
    unpack book into file

    :param book: book in nested dict structure
    :param filename: filename to dump book to, will overwrite
    """

    def _unpack(curr: Dict[str, Union[str, List]], file: TextIOWrapper, depth: int=1):
        """
        recursive helper to unpack book

        :param curr: current element to unpack
        :param file: output file
        :param depth: recursion depth, necessary for formatting, defaults to 1
        """
        if file:
            if curr["title"]:
                file.write(f"<h{depth}>{curr['title']}</h{depth}>\n")
            for p in curr["text"].split("\n"):
                if p:
                    file.write(f"<p>{p}</p>\n")
        for sec in curr["sections"]:
            _unpack(sec, file, depth=depth + 1)

    with open(filename, 'w', encoding="utf-8") as f:
        for chapter in book:
            _unpack(chapter, f)
    


def main():
    # initilize summarizer
    summarizer = Summmarizer()

    # parse book
    book = parse_book(proc=summarizer)

    # unpack book to file
    unpack(book, "book.html")

    # view stats of how much we reduced size
    summarizer.print_stats()


if __name__ == "__main__":
    main()
