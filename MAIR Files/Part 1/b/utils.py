import os
import itertools
from string import punctuation
import pandas as pd
from Levenshtein import distance
import numpy as np

# import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

from a.decisionTree_and_logisticRegression import *
from a.preprocessing import *
from a.baselines import *
import b.states as states
import b.globals as globals



def find_all_categories():
    """
    Populate lists of available options by looking through the restaurant database.
    """
    df = pd.read_csv('b/data/restaurant_info.csv')

    globals.price_ranges = df['pricerange'].unique().tolist()
    globals.price_ranges = [x for x in globals.price_ranges if str(x) != 'nan']
    globals.areas = df['area'].unique().tolist()
    globals.areas = [x for x in globals.areas if str(x) != 'nan']
    globals.food_types = df['food'].unique().tolist()
    globals.food_types = [x for x in globals.food_types if str(x) != 'nan']


def make_feature_matrix():
    """
    Load feature_matrix for bag of words representation and - if the selected model type was not trained yet - trains the model.
    If the neural network model is selected, it in any case will be stored as a variable to reduce computational cost and avoid warnings by tensorflow. 
    """
    sentences, labels = split_x_y(read_data("a/dialog_acts.dat"))

    #   fixed random state to keep the test data unbiased
    x_train_full, x_test, y_train_full, y_test = train_test_split(sentences, labels, test_size=0.15, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.10, random_state=23)

    #   fit the bag of words representation on train set
    globals.feature_matrix.fit(x_train)
    
    x_train_bow, x_val_bow, x_test_bow = map(globals.feature_matrix.transform, [x_train, x_val, x_test])
    
    if globals.switch_dectree and not os.path.isfile("b/data/dec_tree_model.pkl"):
        dtlr.prepare_sklearn_models(x_train_bow, y_train, x_val_bow, y_val)
    elif not(globals.switch_baseline1 or globals.switch_baseline2 or globals.switch_dectree or globals.switch_nn) and not os.path.isfile("b/data/log_reg_model.pkl"):
        dtlr.prepare_sklearn_models(x_train_bow, y_train, x_val_bow, y_val)
    elif globals.switch_nn:        
        if not os.path.isfile("b/data/keras_model_nn.h5"):
            cats = ['inform', 'confirm', 'reqalts', 'request', 'null', 'affirm', 'thankyou', 'negate', 'hello', 'bye', 'repeat', 'deny', 'reqmore', 'restart', 'ack']
            
            Y = np.zeros((len(y_train), 15), dtype=np.int32)    
            for i in range(len(y_train)):          
                Y_i = np.zeros((1, 15), dtype=np.int32)
                Y_i[0,cats.index(y_train[i])] = 1.0
                Y[i,:] = Y_i   
            y_train_vec = Y
            
            Y = np.zeros((len(y_val), 15), dtype=np.int32)            
            for i in range(len(y_val)):
                Y_i = np.zeros((1, 15), dtype=np.int32)
                Y_i[0,cats.index(y_val[i])] = 1.0
                Y[i,:] = Y_i
            y_val_vec = Y    
            
            globals.model = nn.create_model(x_train_bow, y_train_vec, x_val_bow, y_val_vec)
            

        else:
            globals.model = keras.models.load_model("b/data/keras_model_nn.h5")
        
        print(" Okay, I'm ready!")


def find_alternatives(preference):
    """
    This method is used to find similar alternatives to a given user query passed as an argument.
    The method uses predefined similarity lists to check for alternative recommendations by exchanging one of the user preferences
    with one of the other entries in a given similarity list (if it is part of one). Only one user preference is exchanged at a time,
    so alternatives where two slots need to be altered are not considered.
    
    :param preference: The preference settings for which alternatives will be searched.
    :return: A new list containing recommendations similar to the users preference.
    """
    alternatives = {'price_range': [["cheap", "moderate"], ["moderate", "expensive"]],
                    'area': [["centre", "north", "west"],
                             ["centre", "north", "east"],
                             ["centre", "south", "west"],
                             ["centre", "south", "east"]],
                    'food_type': [["thai", "chinese", "korean", "vietnamese", "asian oriental"],
                                  ["mediterranean", "spanish", "portuguese", "italian", "romanian", "tuscan",
                                   "catalan"],
                                  ["french", "european", "bistro", "swiss", "gastropub", "traditional"],
                                  ["north american", "steakhouse", "british"],
                                  ["lebanese", "turkish", "persian"], ["internationa", "modern european", "fusion"]]}

    all_alternatives = []

    for slot in ['food_type', 'area', 'price_range']:
        for alt_list in alternatives.get(slot):
            if preference[slot] in alt_list:
                for alt in alt_list:
                    if alt == globals.preferences[slot]:
                        continue
                    elif lookup_restaurant({**preference, slot: alt}) is not None:
                        all_alternatives.extend(
                            [x for x in lookup_restaurant({**preference, slot: alt}) if x not in all_alternatives])
    return all_alternatives


def lookup_restaurant(preferences):
    """
    This function retrieves suitable restaurant suggestions from the CSV database.
    It does so by filtering the database according to the preferences unless their value is "any".
    
    :param preferences: The preference settings used to look up restaurants.    
    :return: List containing all suitable restaurants in dict form, is of length 0 if no suitable place was found.
    """
    df = pd.read_csv('b/data/restaurant_info.csv')

    if preferences.get('food_type') != "any":
        df = df.loc[(df['food'] == preferences.get('food_type'))]
    if preferences.get('area') != "any":
        df = df.loc[(df['area'] == preferences.get('area'))]
    if preferences.get('price_range') != "any":
        df = df.loc[(df['pricerange'] == preferences.get('price_range'))]

    return df.to_dict('records')


def check_remaining_matches():
    """
    This function checks the how many restaurants in the database are left given the current user preferences. Empty slots are filled with "any".
    If there are zero or one recommendations left, inform the user via the make_recommendation method. Otherwise continue the conversation.
      
    :return: False if there are still more than 2 matches left, otherwise calls method "make_recommendation".
    """
    pref_copy = dict(globals.preferences)
    for key in pref_copy.keys():
        if pref_copy[key] == None:
            pref_copy[key] = "any"

    matches = lookup_restaurant(pref_copy)

    if len(matches) == 1:
        globals.preferences = dict(pref_copy)
        globals.recommendations = list(matches)
        return make_recommendation("One suitable restaurant was found! ")
    elif len(matches) == 0:
        globals.preferences = dict(pref_copy)
        globals.recommendations = lookup_restaurant(globals.preferences)
        return make_recommendation("")
    else:
        return False


def get_match(specification, user_utterance, curr_list):
    """
    A method used to find trigger words and patterns given the user input and a list for comparison.
    Depending on the settings, spelling mistakes are also detected using the Levenshtein distance.
    The act is taken into consideration as well as the "specification", which influences which patterns are searched for.
    The system looks for individual words as well as simple patterns.
        Example for patterns: "i dont care which area" -> remove stopwords -> "care area" = match! (distance = 0)
                              "i dont mind any of the areas" -> remove stopwords -> "any areas" = match! (distance = 1)
    
    :param specification: Additional info regarding which strings the method should look for.
    :param user_utterance: List of user input words.
    :param curr_list: List of words to match.
    :return: match and Levenshtein distance
    """

    word_pairs = []
    pair_matches = {}
    split_doubles = {}
    any_matches = {}
    double_words = [word for word in curr_list if " " in word]
    split_doubles = {i: dw for dw in double_words for i in dw.split(" ")}
    for key in split_doubles.copy().keys():
        if key in globals.food_types + globals.areas + globals.price_ranges:
            del split_doubles[key]

    max_dist = 0

    if globals.switch_distance:
        max_dist = 2

    if len(user_utterance) > 1:
        word_pairs = [' '.join(user_utterance[i:i + 2]) for i in range(len(user_utterance) - 1)]
        pair_matches = {j: distance(i, j) for i, j in itertools.product(word_pairs, double_words) if
                        distance(i, j) <= max_dist}

    if specification is not None:
        if globals.system_state in ["hello"] and len(user_utterance) > 1:
            any_matches = {j: distance(i, j) for i, j in
                           itertools.product(word_pairs, ["any " + specification, "whatever " + specification, "care " + specification, 
                                                          "mind " + specification, "care " + specification]) if distance(i, j) <= max_dist}
        if globals.system_state in ["food_type", "area", "price_range"]:
            any_matches = {j: distance(i, j) for i, j in
                           itertools.product(user_utterance, ["any", "whatever", "anything", "anywhere"]) if distance(i, j) <= max_dist}
            
            if len(user_utterance) > 1:
                any_double_matches = {j: distance(i, j) for i, j in
                            itertools.product(word_pairs, ["any " + specification, "whatever " + specification, "care " + specification, 
                                                           "mind " + specification, "care " + specification, "dont care", "dont mind"]) if distance(i, j) <= max_dist}
                
                any_matches = {**any_matches, **any_double_matches}

    single_matches = {j: distance(i, j) for i, j in itertools.product(user_utterance, curr_list) if
                      distance(i, j) <= max_dist}

    split_matches = {j: distance(i, j) for i, j in itertools.product(user_utterance, split_doubles.keys()) if
                     distance(i, j) <= max_dist}

    if len(pair_matches) > 0:
        return min(pair_matches, key=pair_matches.get), pair_matches.get(min(pair_matches, key=pair_matches.get))
    elif len(single_matches) > 0:
        return min(single_matches, key=single_matches.get), single_matches.get(
            min(single_matches, key=single_matches.get))
    elif len(split_matches) > 0:
        return split_doubles.get(min(split_matches, key=split_matches.get)), 1
    elif len(any_matches) > 0:
        return "any " + specification, 1
    else:
        return None


def make_recommendation(ut=""):
    """
    Uses the saved user preferences to make a restaurant recommendation.
    If there are still restaurants that match the user preference, shows the next match.
    When there is no match left in the list, checks if similar alternatives are available
    and either returns the "hello" state if none were found or the "request_alternatives" state
    if at least one was found.
    
    :param ut: Optional system utterance, shown previous to the recommendation string.
    :return: The new state and the new system utterance.
    """
    if not globals.recommendations:
        return_string = "No "
        if globals.preferences['price_range'] != "any":
            return_string = return_string + globals.preferences['price_range'] + " "
        if globals.preferences['food_type'] != "any":
            return_string = return_string + globals.preferences['food_type'] + " "
        return_string = return_string + "restaurant was found"
        if globals.preferences['area'] != "any":
            return_string = return_string + " in the " + globals.preferences['area']
        globals.alternative_recommendations = find_alternatives(globals.preferences)
        if len(globals.alternative_recommendations) == 0 or globals.is_alternative == 1:
            return_string = return_string + "! What you would like to change to your query?"
            return "hello", ut + return_string
        else:
            return_string = return_string + "! Would you like to hear similar alternatives or make changes to your query?"
            return "request_alternatives", ut + return_string
    else:
        globals.current_recommendation = globals.recommendations[globals.current_recommendation_no]
        area_string = ""
        if globals.preferences['area'] != "any":
            area_string = "the " + globals.current_recommendation['area']
        else:
            area_string = "any area"
        return "handle_requests", ut + globals.current_recommendation['restaurantname'].capitalize() \
               + f" is a {globals.current_recommendation['food']} restaurant with {globals.current_recommendation['pricerange']} prices in {area_string}. What additional information do you need?"


def parse(user_ut):
    """
    Parse raw user utterance into list of words and detect user act using the model selected in the settings.
    The default model is logistic regression.
    
    :param user_ut: The user utterance.
    :return: The predicted dialog act and the list of words.
    """
    user_utterance = user_ut.lower()
    user_utterance = "".join(c for c in user_utterance if c not in punctuation)  # delete punctuations
    # predict dialog act with logistic regressionmatch
    if globals.switch_baseline1:
        act = baseline_1(user_utterance)
    elif globals.switch_baseline2:
        act = baseline_2(user_utterance)
    elif globals.switch_dectree:
        act = predict(globals.feature_matrix, "dec_tree", user_utterance)
    elif globals.switch_nn:
        act = nn.make_prediction(globals.model, user_utterance, globals.feature_matrix)
    else:
        act = predict(globals.feature_matrix, "log_reg", user_utterance)
    user_words = user_utterance.split(' ')

    return act, user_words


def state_transition_function(state, user_utterance):
    """
    Transitions the state using the current state and the raw user utterance by calling the respective method.
    The method additionally filters stopwords and other unwanted words from the parsed user utterance.
    
    :param state: The current system state.
    :param user_utterance: The raw user input.
    :return: The new state and the new system utterance.
    """
    state_transition = {
        "restart": states.restart_state,
        "hello": states.hello_state,
        "food_type": states.food_state,
        "price_range": states.price_state,
        "area": states.area_state,
        "handle_requests": states.handle_request_state,
        "make_sure": states.make_sure_state,
        "request_alternatives": states.request_alternatives_state,
        "additional_preferences": states.additional_req_state,
        None: lambda *_: (state, "Sorry, I don't understand that. Could you try to rephrase your query?")
    }
    act, user_words = parse(user_utterance)
    user_words_filtered = [word for word in user_words if ((word not in stopwords.words('english') or word == "any") and word not in ["hi", "want"])]

    if act == 'restart' and globals.switch_restart:
        return 'restart', "Do you really want to start over?"
    return state_transition[state](act, user_words_filtered)


def extract_property_consequences(restaurant, requested_prop):
    """
    A method testing which additional requirements a restaurant fulfills, particularly the
    requirement passed as an argument.
     
    
    :param restaurant: Dict with the restaurant properties.
    :param requested_prop: The property the restaurant should be checked for.
    :return: Dict of new properties with True or False.
    """
    properties = {
        'cheap': (restaurant['pricerange'] == 'cheap'),
        'expensive': (restaurant['pricerange'] == "expensive"),
        'good food': (restaurant['foodquality'] == 'good food'),
        'spanish': (restaurant['food'] == 'spanish'),
        'north american': (restaurant['food'] == 'north american'),
        'steakhouse': (restaurant['food'] == 'steakhouse'),
        'italian': (restaurant['food'] == 'italian'),
        'french': (restaurant['food'] == 'french'),
        'tuscan': (restaurant['food'] == 'tuscan'),
        'indian': (restaurant['food'] == 'indian'),
        'thai': (restaurant['food'] == 'thai'),
        'south': (restaurant['area'] == 'south'),
        'west': (restaurant['area'] == 'west'),
        'north': (restaurant['area'] == 'north'),
        'east': (restaurant['area'] == 'east'),
    }

    test_rules = [
        # new calculated property name, rule based on these properties ..., rule to compute new property
        ('busy', 'good food', 'cheap', lambda g, c: g and c),
        ('long time', 'spanish', lambda s: s),
        ('long time', 'busy', lambda b: b),
        ('children', 'long time', lambda c: not c),
        ('romantic', 'busy', lambda b: not b),
        ('romantic', 'long time', lambda l: l),

        ('noisy', 'north american', 'steakhouse', 'cheap', lambda a, s, c: (a or s) and c), #(north american OR steakhouse) AND cheap -> noisy
        #location != centre -> latenight(long drive)
        ('late night', 'south', lambda s: s),
        ('late night', 'north', lambda n: n),
        ('late night', 'west', lambda w: w),
        ('late night', 'east', lambda e: e),
        ('quiet', 'late night', 'expensive', lambda n, e: n and e), #late night AND expensive -> quiet
        ('cheap liquor', 'cheap', lambda c: c), #cheap prices -> cheap liquor
        ('need transit', 'late night', 'cheap liquor', lambda n, l: n and l), #late night AND cheap liquor -> need transit
        # (italian, french, OR tuscan) AND late night -> romance
        ('romantic', 'italian', 'late night', lambda i, n: i and n),
        ('romantic', 'french', 'late night', lambda i, n: i and n),
        ('romantic', 'tuscan', 'late night', lambda i, n: i and n),
    ]


    def propagate_consequences(props, rules, requested_prop):
        """
        A method testing the given restaurant property with the given rules, informing the user
        of the results of the tests.
        
        :param props: Dict with the restaurant properties.
        :param rules: A list rules used for testing.
        :param requested_prop: The property the restaurant should be checked for.
        :return: Dict of new properties with True or False.
        """
        i = 0
        conclusion = ""
        while True:
            new_props = {}
            i += 1
            change_happened = False
            for rule_number, (field, *arguments, rule) in enumerate(rules):
                #  not already calculated all the arguments are calculated
                if field not in props and all(arg in props for arg in arguments) and all(
                        props[arg] for arg in arguments):
                    new_props[field] = rule(*map(props.get, arguments))
                    if field == requested_prop:
                        conclusion = f"This restaurant is {'not ' if not new_props[field] else ''}recommended because of rule {arguments} > {'not' if not new_props[field] else ''} {field}"
                    change_happened = True
                    print(f"iteration{i}. Rule{rule_number}. {arguments} > {field} = {new_props[field]}")
                    # iteration: 1. Rule 1. [cheap, good food] > busy = True
            if not change_happened:
                break
            props.update(new_props)
        if requested_prop not in props:
            conclusion = f"I don't know whether this restaurant is {requested_prop}"
        print(conclusion + '\n')
        return conclusion, props
    
    print(restaurant['restaurantname'].title() + ':')
        
    return propagate_consequences(properties, test_rules, requested_prop)
    # extract_property_consequences({'name': 'milano', 'food': 'italian', 'pricerange': 'cheap'}, 'romantic')