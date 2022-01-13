from pathlib import PosixPath
from docutils import nodes
from myst_parser.sphinx_parser import MystParser
from markdown_it.main import MarkdownIt


class Parser(MystParser):
    def parse(self, inputstring: str, document: nodes.document) -> None:
        def new_init(self, *args, **kwargs):
            old_init(self, *args, **kwargs)
            cwd = PosixPath.cwd()
            self.options.setdefault('relative-images', "..")
            self.options.setdefault('relative-docs', ("docs/", cwd, cwd.parent))

        old_init = MarkdownIt.__init__
        try:
            MarkdownIt.__init__ = new_init
            return super().parse(inputstring, document)
        finally:
            MarkdownIt.__init__ = old_init
