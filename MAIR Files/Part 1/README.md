#### Part1 - Methods in Artificial Intelligence Research: Group 2

# On the Right TrAKX
#### A state-based restaurant recommendation bot designed to understand what you want and show you where to find it! 
#### This repository aims to satisfy the assignment guidelines for Part 1 of the MAIR Course in the AI Master's Programme at Utrecht University.

## Installation
There are no local installation steps required aside from having the appropriate versions of the used packages and libraries. 
The packages and libraries required to operate this chatbot include the following:

Required packages (most are standard python packages):
  - os
  - sys
  - inspect
  - time
  - string
  - collections
  - copy
  - pandas 
  - numpy
  - sklearn
  - nltk (Natural Language Toolkit, english stopwords need to be downloaded, e.g. via "nltk.download('stopwords')")
  - itertools (efficient looping)
  - tqdm (progress bars representation)
  - pickle (object serialization to/from disk)
  - Python-Levenshtein
   
If the neural network is used:
  - tensorflow (a version including keras, e.g. from anaconda)
  
If text2speech is enabled:
  - pyttsx3

  
## Usage
To begin, open chatbot.py (Part1/chatbot.py) in any console that runs Python code and can access the other folders in this repository. 
To use the chatbot.py file, simply enter a request in English describing an establishment at which you might like to eat. 
The bot will correct misspellings and prompt you for the information it needs to give an appropriate recommendation, given the constraints 
of the Cambridge database provided.
It is generally the case that questions from the bot need to answered before the conversation can continue.
Please note that you can reset the conversation at any point by typing for example "start over".

## Credits
Contained in the title is an acronym of our names: Tarek Khellaf, Anneline Daggelinckx, Kuil Schoneveld, and Xinyu Zhang (TrAKX). 
The four of us constitute Group 2 of MAIR.

We can be found at the following email addresses respectively:
- t.khellaf@students.uu.nl
- a.daggelinckx@students.uu.nl
- k.g.schoneveld@students.uu.nl
- x.zhang16@students.uu.nl
