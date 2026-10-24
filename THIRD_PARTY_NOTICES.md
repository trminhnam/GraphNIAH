# Third-party notices

The code in this repository is released under the MIT License (see
[`LICENSE`](LICENSE)). The MIT License does not cover the files and data
listed below. They keep their own licenses and terms.

## Files included in this repository

The BibTeX style files in `src/reference_parser/style_files/` are used by the
optional reference parsers. They are third-party files and are distributed
under their original terms, not under the MIT License:

| File | Original work | License |
| --- | --- | --- |
| `aaai.bst` | AAAI natbib style | LaTeX Project Public License (LPPL) |
| `acl.bst` | ACL natbib style (Patrick W. Daly, Norman Gray) | LPPL |
| `cvpr.bst` | natbib style used by CVPR (Patrick W. Daly) | LPPL |
| `iclr.bst` | natbib style used by ICLR (Hal Daumé III, Patrick W. Daly) | LPPL |
| `icml.bst` | natbib style used by ICML (Hal Daumé III, Patrick W. Daly) | LPPL |
| `ieee.bst` | IEEEtran BibTeX style (Michael Shell) | LPPL |
| `neurips.bst` | natbib style used by NeurIPS (Patrick W. Daly) | LPPL |
| `acm.bst` | ACM reference format (Nelson H. F. Beebe, Boris Veytsman, Gerald Murray) | Public domain |

Each file keeps its original copyright header. The LPPL is available at
<https://www.latex-project.org/lppl/>.

## Data used by the pipeline

This repository does **not** contain or redistribute any of the data below.
The pipeline downloads it on your machine, and you must follow each source's
terms when you download, use, or share it.

| Source | Used for | Terms |
| --- | --- | --- |
| [Semantic Scholar Academic Graph](https://www.semanticscholar.org/product/api) (`papers`, `citations`, `paper-ids`) | Paper metadata and citation graph | [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1.0/). Cite Kinney et al. (2023), "The Semantic Scholar Open Data Platform", [arXiv:2301.10140](https://arxiv.org/abs/2301.10140). Downloading requires an API key under the [Semantic Scholar API License Agreement](https://www.semanticscholar.org/product/api/license). |
| [Linux kernel](https://github.com/torvalds/linux) source files | `code` distractors | [GPL-2.0](https://github.com/torvalds/linux/blob/master/COPYING) |
| [Paul Graham essays](https://www.paulgraham.com/articles.html), via the Hugging Face dataset [`sgoel9/paul_graham_essays`](https://huggingface.co/datasets/sgoel9/paul_graham_essays) | `essays` distractors | The essays are copyrighted by Paul Graham. The MIT tag on the Hugging Face dataset is the uploader's and does not license the essays themselves. Use them only as your local law allows (for example, non-commercial research), and do not redistribute them. |
| [PG-19](https://github.com/google-deepmind/pg19), via the Hugging Face dataset [`emozilla/pg19`](https://huggingface.co/datasets/emozilla/pg19) | `books` distractors | The dataset is released by DeepMind under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). The books are Project Gutenberg texts published before 1919 and are in the public domain in the United States. |
| Wikipedia passages ([Pyserini](https://github.com/castorini/pyserini) prebuilt `wikipedia-dpr-100w` index) | `wikipedia` distractors | Wikipedia text: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Pyserini: Apache-2.0. |

## Model outputs

Running inference sends prompts to third-party model providers (for example,
OpenAI, Google, and OpenRouter). Model outputs and the use of each API are
governed by the provider's own terms of service.
