from a.decisionTree_and_logisticRegression import *
from b.utils import *
from b.states import *
import b.globals as globals


"""
On the Right TrAKX: Methods in Artificial Intelligence Research Group 2
A state-based dialogue system for interpreting natural language commands and recommending suitable restaurants.

This method must be called in order to start the chatbot. Please make sure that all required packages are installed
and that you understand the functionality of this program (see 'README.md' for details).

KHELLAF, TAREK (t.khellaf@students.uu.nl)
    Candidate for Master of Science in Artificial Intelligence, Universiteit Utrecht, 6234496
DAGGELINCKX, ANNELINE (a.daggelinckx@students.uu.nl) 
    Candidate for Master of Science in Artificial Intelligence, Universiteit Utrecht, 2060481
ZHANG, XINYU (x.zhang16@students.uu.nl) 
    Candidate for Master of Science in Artificial Intelligence, Universiteit Utrecht, 3056684
SCHONEVELD, G, KUIL (k.g.schoneveld@students.uu.nl) 
    Candidate for Master of Science in Artificial Intelligence, Universiteit Utrecht, 7129019

"""
if __name__ == '__main__':
    globals.init()
    globals.feature_matrix = CountVectorizer(max_features=700)
    if globals.switch_nn:
        print('agent: ' + "Please wait a moment..", end="")
        import a.nn as nn
        from tensorflow import keras
        
    make_feature_matrix()
    find_all_categories()

    str_hi = "Hi, how can I help you? Please describe somewhere you might like to eat."

    globals.system_state = 'hello'

    if globals.switch_CAP:
        str_hi = str_hi.upper()
    if globals.switch_text2speech:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say("Hi, how can I help you")
        engine.runAndWait()
    print('agent: ' + str_hi)


    while globals.system_state != 'end':
        user_sentence = input('user: ')
        globals.system_state, system_sentence = state_transition_function(globals.system_state, user_sentence)
        if globals.switch_delay:
            t = random.randint(3, 7)
            time.sleep(t)
        if globals.switch_CAP:
            system_sentence = system_sentence.upper()
        if globals.switch_text2speech:
            engine.say(system_sentence)
            engine.runAndWait()
        print('agent: ' + system_sentence)
