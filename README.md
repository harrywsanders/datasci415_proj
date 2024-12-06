# Citations are All You Need
## How To Navigate This Repository
I've split the repo into two parts: the PROJECT_CODE directory and the PROJECT_PAPER directory. Guess what's in them! The code and the paper, respectively. The code is written in Python and the paper is written in LaTeX. 

### PROJECT_CODE
The code uses Python 3.12. I used [Poetry](https://python-poetry.org/) to manage the dependencies. You can install the dependencies by running `poetry install` in the PROJECT_CODE directory, and then running `poetry shell` to activate the virtual environment. Then, just run the code with `python main.py`. It will train, but it takes a while, so be warned. The results are saved in the `results` directory, the models are saved in the `models` directory, and the plots are saved in the `plots` directory. 

### PROJECT_PAPER
The paper is written in LaTeX. You can compile it by running `pdflatex main.tex`, or by using your favorite LaTeX editor. The bibliography is in `references.bib`. The figures are (and I know this is bad practice) imported directly from the `PROJECT_CODE/plots` directory. 
