import re
import bs4
from bs4 import BeautifulSoup
from typing import Callable, Dict, List, Union

DOC_PATH = "xhtml/"
TOC_NAME = "toc.xhtml"


def sanitize(in_str: str) -> str:
    """
    Remove whitespace from string and fix some characters

    :param in_str: input string
    :return: processed string
    """
    whitespace_rem = re.sub(r"\n+", "\n", in_str.replace("\r", "").strip())
    quote_san = re.sub(u"\u2018|\u2019", "'", whitespace_rem)
    dub_quote_san = re.sub(u"\u201c|\u201d", '"', quote_san)
    dash_san = re.sub(u"\u2014", " - ", dub_quote_san)
    invis_san = re.sub(u"\u00A0", "", dash_san)
    return invis_san


def get_soup_doc(doc_name: str) -> BeautifulSoup:
    """
    Get BeautifulSoup document body for the given document name

    :param doc_name: document name
    :return: BeautifulSoup instance for the body of the document
    """
    doc = DOC_PATH + doc_name
    soup = None
    with open(doc, "r", encoding="utf8") as f:
        soup = BeautifulSoup(f.read(), "xml")

    return soup.body


def process_doc(
    doc: BeautifulSoup, proc: Callable = None, log: bool = True
) -> Union[None, Dict[str, Union[str, List]]]:
    """
    Process soup to extract text in sections recursively

    ex return:
    {
        "title": "Chapter 1",
        "text": "",
        "sections": [
            {
                "title": "Section 1",
                "text": "",
                "sections": [
                    {
                        "title": "Subsection 1",
                        "text" : "this is text\nthis is the second paragraph",
                        "sections": [],
                    }
                ]
            }
        ]
    }

    :param doc: bs4 document
    :param proc: processor function to apply to text before storage (summarizer or other)
    :param log: log completion of call
    :return: list of sections and all subsections represented as dictionaries
    """
    paragraphs = []
    output = {"title": None, "text": "", "sections": []}

    # populate title
    header_el = doc.find("header", recursive=False)
    if header_el:
        title_el = header_el.find("h1", recursive=False)
        output["title"] = get_text(title_el)

        # check if we are in dummy section
        if output["title"] == "Essential Learning Concepts":
            return None

    # populate paragraphs
    p_els = doc.findAll("p", recursive=False)
    for p_el in p_els:
        # extract text from p
        p_text = get_text(p_el)
        paragraphs.append(p_text)

    output["text"] = "\n".join(paragraphs)
    if proc and callable(proc) and output["text"]:
        output["text"] = proc(output["text"])

    # populate subsections
    section_els = doc.findAll("section", recursive=False)
    for sec in section_els:
        sub_sec = process_doc(sec, proc=proc)
        if sub_sec:
            output["sections"].append(sub_sec)

    # convenience so we can call on body of document instead of first section
    # if root has no title or paragraphs and has only one section, return that section
    if not output["title"] and not output["text"] and len(output["sections"]) == 1:
        return output["sections"][0]
    else:
        if log:
            print(output["title"])
        return output


def get_text(tag: bs4.Tag) -> str:
    """
    Get text from inside tag, including inline elements

    :param tag: BS4 tag to get text from inside of
    :return: text inside tag
    """
    _inline_elements = {
        "a",
        "span",
        "em",
        "strong",
        "u",
        "i",
        "font",
        "mark",
        "label",
        "s",
        "sub",
        "sup",
        "tt",
        "bdo",
        "button",
        "cite",
        "del",
        "b",
        "a",
        "font",
    }

    def _get_text(tag: bs4.Tag):
        """
        recursive helper to get text inside dom

        :param tag: tag to get text from inside of
        :yield: text
        """
        for child in tag.children:
            if type(child) is bs4.Tag:
                # if the tag is a block type tag then yield new lines before and after
                is_block_element = child.name not in _inline_elements
                if is_block_element:
                    continue
                else:
                    yield from ["\n"] if child.name == "br" else _get_text(child)
            elif type(child) is bs4.NavigableString:
                ret = child.string
                if not ret.isspace():
                    yield ret

    return sanitize("".join(_get_text(tag)))


def parse_toc():
    """
    Parse table of contents to get a list of documents and their titles

    :return: list of documents and their titles
    """

    toc = []

    toc_body = get_soup_doc(TOC_NAME)
    toc_entry_list = toc_body.find("ol", class_="tocentrylist")
    if toc_entry_list:
        for toc_entry in toc_entry_list.findChildren("li", recursive=False):
            # only get content of chapters, skip everything else
            link_el = toc_entry.find("a")
            if not link_el:
                continue
            link = link_el["href"]
            if not link or not link.startswith("ch") or not link.endswith(".xhtml"):
                continue

            # parse title
            title = get_text(link_el)
            toc.append({"title": title, "text": "", "sections": []})

            # parse sections
            for section_entry in toc_entry.find_all("li"):
                link_el = section_entry.find("a")
                if not link_el:
                    continue
                link = link_el["href"].split("#")[0]
                section_title = get_text(link_el)
                toc[-1]["sections"].append({"title": section_title, "file": link})
            # remove Essential Concepts Reviews
            toc[-1]["sections"].pop(-1)

    return toc


def parse_book(proc: Callable) -> List[Dict[str, Union[str, List]]]:
    """
    Parse entire book from toc

    :param proc: processing function, should accept and return str
    :return: book contents
    """

    toc = parse_toc()

    processed = []

    for chapter in toc:
        processed.append(chapter.copy())
        processed[-1]["sections"] = []
        for sec in chapter["sections"]:
            soup = get_soup_doc(sec["file"])
            processed[-1]["sections"].append(process_doc(soup, proc=proc))

    return processed


if __name__ == "__main__":
    """
    This code will process ch1 pg 2 and store it as json
    
    CHAPTER_1_2 = "ch01_pg0002.xhtml"
    doc = process_doc(get_soup_doc(CHAPTER_1_2))
    import json
    with open("ch_1.json", "w") as f:
        json.dump(doc, f, indent=2)
    """

    """
    This code will parse table of contents and print chapter/section titles

    toc = parse_toc()
    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(toc)
    """
