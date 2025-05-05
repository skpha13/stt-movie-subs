import re
import string
from datetime import time

from movie2sub.utils.custom_types import Segment


def clean_asr_text(text: str):
    # remove html files
    text = re.sub(r"<[^>]+>", "", text)

    # replace hyphens
    text = re.sub(r"^\s*-\s*", "", text, flags=re.MULTILINE)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # lowercase the text
    text = text.lower().strip()

    # split into lines where appropriate (based on punctuation)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # remove punctuation except apostrophes
    allowed_punct = "'"
    cleaned = "\n".join(
        re.sub(f"[{re.escape(string.punctuation.replace(allowed_punct, ''))}]", "", s).strip() for s in sentences if s
    )

    return cleaned


if __name__ == "__main__":
    text = """
        <i>I've got your final tests here.</i>
        <i>Everybody did pretty well.Uh, I'm gonna call your--</i>
        - I'm headed this way.- Oh.
        - Later.- So, what did we decide?
        - About what?- About tonight.
        Oh, uh, look, I'll probably get,you know, hung up with theguys maybe, you know, later.
        - Why don't we just meet at the party?- All right.
        - Wouldn't want you waitin' around for meall night anyway.- All right. Whatever.
        - Cool. I'll see ya later?- Bye.
        Bye.
    """
    print(clean_asr_text(text))
