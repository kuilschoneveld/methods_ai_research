import b.utils as utils
import b.globals as globals


def restart_state(act, user_words):
    """
    This state is called when the user is thought to request a restart of the conversation.
    If the user affirms, the preferences etc. are reset and the conversation restarted.
    If the user declines, the system continues with the conversation.
    If the user does neither, the system remains in the same state and asks again.
    Please note: The state will not be changed and no additional input will be processed unless the user affirms or negates the question.
    
    :param act: The current predicted act.
    :param user_words: The preprocessed user utterance.
    :return: The new state and the new system utterance.
    """
    if act == "affirm":
        globals.preferences['food_type'] = globals.preferences['area'] = globals.preferences['price_range'] = None
        globals.curr_cat = globals.curr_val = None
        globals.recommendations = []
        globals.current_recommendation = {}
        globals.current_recommendation_no = 0
        globals.is_alternative = 0
        return "hello", "The query was reset! How can I help you?"

    elif act == "negate":
        for specification, value in globals.preferences.items():
            if not value:
                return specification, 'The query was not reset! ' + globals.questions[specification]
        globals.recommendations = utils.lookup_restaurant(globals.preferences)
        if len(globals.recommendations) > 1:
            return "additional_preferences", "Do you have any additional preferences?"
        return utils.make_recommendation("Okay! ")
    else:
        return "restart", "Sorry, I don't understand. Would you like to start over?"
    
    
def hello_state(act, user_words):
    """
    This state is initially loaded when starting the application, when the preferences were reset and
    when changes to a complete preference set are made.
    If the act is hello or inform, the system checks which slots were filled with the user utterance.
    Depending on the settings, if clarification is needed (Levenshtein distance > 0 or slot is filled with "any"),
    the system will call the make_sure state to ensure that the system is understanding the user correctly.
    If the user input suffices to make a recommendation, the system will do so via the make_recommendation method or ask for additional preferences.
    If additional information is needed, the system will call the state corresponding to the first empty slot.
    Please note: If multiple slots are filled with "any" or have a Levenshtein distance > 0, the system will only ask for clarification of one of 
    the statements and not consider the rest.
    
    :param act: The current predicted act.
    :param user_words: The preprocessed user utterance.
    :return: The new state and the new system utterance.
    """
    if act == "hello" or act == "inform":
        # try to get info out of user utterance
        for category_list, specification in [(globals.food_types, 'food type'), (globals.areas, 'area'), (globals.price_ranges, 'price range')]:
            if utils.get_match(specification.split(" ")[0], user_words, category_list) is not None:
                cat, dist = utils.get_match(specification.split(" ")[0], user_words, category_list)
                if cat.split(" ")[0] == "any":
                    new_state = "make_sure"
                    system_utterance = f"You don't care about the {specification}, is that right?"
                    globals.curr_cat = specification.replace(' ', '_')
                    globals.curr_val = "any"
                elif dist > 0:
                    new_state = "make_sure"
                    if globals.switch_Lev_makesure:
                        system_utterance = f"Sorry, I can't find this {specification}. Did you mean {cat}?"
                    else:
                        system_utterance = f"I think you mean {cat}. Enter anything to continue."
                        globals.lev_defaultyes = True
                    globals.curr_cat = specification.replace(' ', '_')
                    globals.curr_val = cat
                else:
                    globals.preferences[specification.replace(' ', '_')] = cat
                    globals.current_recommendation_no = 0
                    globals.is_alternative = 0
        if utils.check_remaining_matches() != False:
            return utils.check_remaining_matches()

        # search for next state, depending on the already given information
        if not globals.curr_cat:
            for specification, value in globals.preferences.items():
                if not value:
                    return specification, globals.questions[specification]
            globals.recommendations = utils.lookup_restaurant(globals.preferences)
            if len(globals.recommendations) > 1:
                return "additional_preferences", "Do you have any additional preferences?"
            return utils.make_recommendation("")

        return new_state, system_utterance
    else:
        for specification, value in globals.preferences.items():
            if not value:
                return specification, "I'm afraid I don't understand that. " + globals.questions[specification]
        return "hello", "I'm afraid I don't understand. Could you please rephrase your query?"

    return new_state, system_utterance


def food_state(act, user_words):
    """
    This state is loaded if the system looked for missing information and the food slot was not filled yet.
    Depending on the settings, if clarification is needed (Levenshtein distance > 0 or slot is filled with "any"),
    the system will call the make_sure state to ensure that the system is understanding the user correctly.
    If the user input suffices to make a recommendation, the system will do so via the make_recommendation method or ask for additional preferences.
    If additional information is needed, the system will call the state corresponding to the next empty slot.
    
    :param act: The current predicted act.
    :param user_words: The preprocessed user utterance.
    :return: The new state and the new system utterance.
    """
    if utils.get_match("food", user_words, globals.food_types):
        cat, dist = utils.get_match("food", user_words, globals.food_types)
        if cat.split(" ")[0] == "any":
            new_state = "make_sure"
            system_utterance = f"You don't care about the food type, is that right?"
            globals.curr_cat = "food_type"
            globals.curr_val = "any"
            return new_state, system_utterance
        elif dist > 0:
            globals.curr_cat = "food_type"
            globals.curr_val = cat
            if globals.switch_Lev_makesure:
                return "make_sure", f"Sorry, I can't find this food type. Did you mean {cat}?"
            else:
                globals.lev_defaultyes = True
                return "make_sure", f"I think you mean {cat}. Enter anything to continue."

        else:
            globals.preferences['food_type'] = cat
            globals.current_recommendation_no = 0
            globals.is_alternative = 0
            if utils.check_remaining_matches() != False:
                return utils.check_remaining_matches()
            if not globals.preferences['area']:
                return "area", f"Okay, a restaurant with {globals.preferences['food_type']} food. In what region would you like to eat?"
            elif not globals.preferences['price_range']:
                return "price_range", f"Okay, a restaurant with {globals.preferences['food_type']} food. What price range are you comfortable with?"
            globals.recommendations = utils.lookup_restaurant(globals.preferences)
            if len(globals.recommendations) > 1:
                return "additional_preferences", "Do you have any additional preferences?"
            return utils.make_recommendation("")
    return "food_type", "Sorry, I don't know that type of food."


def area_state(act, user_words):
    """
    This state is loaded if the system looked for missing information and the area slot was not filled yet.
    Depending on the settings, if clarification is needed (Levenshtein distance > 0 or slot is filled with "any"),
    the system will call the make_sure state to ensure that the system is understanding the user correctly.
    If the user input suffices to make a recommendation, the system will do so via the make_recommendation method or ask for additional preferences.
    If additional information is needed, the system will call the state corresponding to the next empty slot.
    
    :param act: The current predicted act.
    :param user_words: The preprocessed user utterance.
    :return: The new state and the new system utterance.
    """
    if utils.get_match("area", user_words, globals.areas):
        cat, dist = utils.get_match("area", user_words, globals.areas)
        if cat.split(" ")[0] == "any":
            new_state = "make_sure"
            system_utterance = f"You don't care about the area is that right?"
            globals.curr_cat = "area"
            globals.curr_val = "any"
            return new_state, system_utterance
        elif dist > 0:
            globals.curr_cat = "area"
            globals.curr_val = cat
            if globals.switch_Lev_makesure:
                return "make_sure", f"Sorry, I can't find this area. Did you mean {cat}?"
            else:
                return "make_sure", f"I think you mean {cat}. Enter anything to continue."
        globals.preferences['area'] = cat
        globals.current_recommendation_no = 0
        globals.is_alternative = 0
        if utils.check_remaining_matches() != False:
            return utils.check_remaining_matches()
        if globals.preferences['price_range']:
            globals.recommendations = utils.lookup_restaurant(globals.preferences)
            if len(globals.recommendations) > 1:
                return "additional_preferences", "Do you have any additional preferences?"
            return utils.make_recommendation("")
        if globals.preferences['area'] == "any":
            return "price_range", f"Okay, a restaurant in {globals.preferences['area']} area. What price range are you comfortable with?"
        else:
            return "price_range", f"Okay, a restaurant in the {globals.preferences['area']}. What price range are you comfortable with?"
    return "area", "Sorry, I don't know that area. Could you choose between north, east, south, west or centre?"


def price_state(act, user_words):
    """
    This state is loaded if the system looked for missing information and the price slot was not filled yet.
    Depending on the settings, if clarification is needed (Levenshtein distance > 0 or slot is filled with "any"),
    the system will call the make_sure state to ensure that the system is understanding the user correctly.
    If the user input suffices to make a recommendation, the system will do so via the make_recommendation method or ask for additional preferences.
    If additional information is needed, the system will call the state corresponding to the next empty slot.
    
    :param act: The current predicted act.
    :param user_words: The preprocessed user utterance.
    :return: The new state and the new system utterance.
    """
    if utils.get_match("price", user_words, globals.price_ranges):
        cat, dist = utils.get_match("price", user_words, globals.price_ranges)
        if cat.split(" ")[0] == "any":
            new_state = "make_sure"
            system_utterance = f"You don't care about the price range is that right?"
            globals.curr_cat = "price_range"
            globals.curr_val = "any"
            return new_state, system_utterance
        elif dist > 0:
            globals.curr_cat = "price_range"
            globals.curr_val = cat
            if globals.switch_Lev_makesure:
                return "make_sure", f"Sorry, I can't find this price range. Did you mean {cat}?"
            else:
                return "make_sure", f"I think you mean {cat}. Enter anything to continue."
        else:
            globals.preferences['price_range'] = cat
            globals.current_recommendation_no = 0
            globals.is_alternative = 0
            globals.recommendations = utils.lookup_restaurant(globals.preferences)
            if len(globals.recommendations) > 1:
                return "additional_preferences", "Do you have any additional preferences?"
            return utils.make_recommendation("")
    else:
        return "price_range", "Sorry, I don't know that price. Could you choose between cheap, moderate priced and expensive?"



def additional_req_state(act, user_words):
    """
    This state is loaded if the system found more than one restaurant for the given user preference.
    If the user negates the question whether they have additional preferences, the first recommendation is loaded.
    If not, the system will search which additional preference the user is asking for and inform the user which restaurants fit their requirement.
    If no restaurant fits the users additional preference, the system will ask the user to pick another one.
    Please note: The system will remain in this state until a match including the additional preference was found or the user negates the question.    
    
    :param act: The current predicted act.
    :param user_words: The preprocessed user utterance.
    :return: The new state and the new system utterance.
    """
    globals.recommendations = utils.lookup_restaurant(globals.preferences)
    if act == "negate":
        return utils.make_recommendation()

    found_match = False
    new_recommendations = []

    add_props = ["busy", "long time", "romantic", "children", 'late night', 'noisy', 'cheap liquor']

    if utils.get_match(None, user_words, add_props):
        request, d = utils.get_match(None, user_words, add_props)

        for restaurant in globals.recommendations:
            _, properties = utils.extract_property_consequences(restaurant, request)
            if request in properties:
                if properties[request]:
                    new_recommendations.append(restaurant)
                    found_match = True

        if found_match:
            globals.recommendations = new_recommendations
            globals.current_recommendation = globals.recommendations[0]

            return "handle_requests", globals.current_recommendation['restaurantname'].capitalize() \
                   + f" is a {globals.current_recommendation['food']} restaurant with {globals.current_recommendation['pricerange']} prices in {globals.current_recommendation['area']} that satisfies your additional requirement. What additional information do you need?"
        else:
            return "additional_preferences", f"Sorry, there is no restaurant with the requirement '{request}', do you have another additional requirement?"
    else:
        return "additional_preferences", f"Sorry, I don't understand your requirement, please ask for another one."


def make_sure_state(act, user_words):
    """
    This state is loaded if the system needs clarification regarding one of the users statements.
    Depending on the settings, the system will automatically affirm and merely inform the user of this decision.
    If automatic affirmation is disabled, the user will be asked to affirm or negate the statement, after which the
    information will either be added to the preferences or discarded.
    Please note: The state will not be changed and no additional input will be processed unless the user affirms or negates the question.
    
    :param act: The current predicted act.
    :param user_words: The preprocessed user utterance.
    :return: The new state and the new system utterance.
    """
    if (not globals.switch_Lev_makesure) and globals.lev_defaultyes:
        act = "affirm"
        globals.lev_defaultyes = False

    if act == "affirm":
        globals.preferences[globals.curr_cat] = globals.curr_val
        globals.current_recommendation_no = 0
        globals.is_alternative = 0
        globals.curr_cat = None
        cur_val = None
        if utils.check_remaining_matches() != False:
            return utils.check_remaining_matches()
    elif act == "negate":
        globals.curr_cat = None
        cur_val = None
    else:
        return "make_sure", "I don't understand, can you please answer with yes or no?"

    for specification, value in globals.preferences.items():
        if not value:
            return specification, 'Okay! ' + globals.questions[specification]
    globals.recommendations = utils.lookup_restaurant(globals.preferences)
    if len(globals.recommendations) > 1:
        return "additional_preferences", "Do you have any additional preferences?"
    return utils.make_recommendation("")


def request_alternatives_state(act, user_words):
    """
    This state is loaded if there is no recommendation left that fits the user preferences exactly, but imperfect
    matches were found. The system analyses the user utterance whether they want to hear alternative matches or change their initial query.
    Please note: The state will not be changed and no additional input will be processed unless the user picks one of the two options.
    
    :param act: The current predicted act.
    :param user_words: The preprocessed user utterance.
    :return: The new state and the new system utterance.
    """
    alternative_list = ["alternative", "alternatives", "more", "other", "another", "anything else"]
    query_list = ["change", "changes", "different", "something else"]

    alternative_distance = None
    query_distance = None

    if utils.get_match(None, user_words, alternative_list) is not None:
        _, alternative_distance = utils.get_match(None, user_words, alternative_list)
    if utils.get_match(None, user_words, query_list) is not None:
        _, query_distance = utils.get_match(None, user_words, query_list)

    if alternative_distance is None and query_distance is not None:
        return "hello", "Okay! Tell me what you would like to change to your query."
    elif alternative_distance is not None and query_distance is None:
        globals.recommendations = globals.alternative_recommendations.copy()
        globals.is_alternative = 1
        globals.current_recommendation_no = 0
        return utils.make_recommendation("Okay! ")
    elif alternative_distance is not None and query_distance is not None:
        if alternative_distance > query_distance:
            return "hello", "Okay! Tell me what you would like to change to your query."
        elif alternative_distance < query_distance:
            globals.recommendations = globals.alternative_recommendations.copy()
            globals.is_alternative = 1
            globals.current_recommendation_no = 0
            return utils.make_recommendation("Okay! ")
        else:
            "request_alternatives", "Sorry, I'm unsure what your choice is, could you please be more clear?"
    else:
        return "request_alternatives", "I don't understand, can you please answer the question first?"


def handle_request_state(act, user_words):
    """
    This state is loaded whenever the system presented a new recommendation to the user.
    The user utterance is analysed whether the user is asking for address, phone number and/or postal code.
    If the act was predicted to be reqalts, the system will display the next recommendation (if available).
    Once there are no more recommendations, the system will look for similar matches and give the user further
    options depending on the result.
    Please note: The system will remain in this state until there are no more recommendations left in the queue or the user decides
    to reset the conversation.
    
    :param act: The current predicted act.
    :param user_words: The preprocessed user utterance.
    :return: The new state and the new system utterance.
    """
    system_utterance = ''

    if act == 'thankyou' or act == 'bye':
        return 'end', "Enjoy your meal. Goodbye!"

    if act == 'request':
        req_addr = utils.get_match(None, user_words, ["address"]) is not None
        req_phone = utils.get_match(None, user_words, ["phone number"]) is not None
        req_postcode = utils.get_match(None, user_words, ["postcode", "postal code", "post code", "telephone"]) is not None
        ongoing_sentence = False
        if req_phone:
            system_utterance = f"The phone number of {globals.current_recommendation['restaurantname'].capitalize()} is {globals.current_recommendation['phone']}"
            ongoing_sentence = True
        if req_addr:
            if ongoing_sentence:
                system_utterance += f", the address is {globals.current_recommendation['addr'].capitalize()}"
            else:
                system_utterance = f"The address of {globals.current_recommendation['restaurantname'].capitalize()} is {globals.current_recommendation['addr']}"
                ongoing_sentence = True
        if req_postcode:
            if ongoing_sentence:
                system_utterance += f" and the postal code is {globals.current_recommendation['postcode']}"
            else:
                system_utterance = f"The postal code of {globals.current_recommendation['restaurantname'].capitalize()} is {globals.current_recommendation['postcode']}"
                ongoing_sentence = True
        if ongoing_sentence:
            return "handle_requests", system_utterance + "."
        return "handle_requests", "I'm afraid I don't have that information. Would you like to know the phone number, address or postal code of the restaurant? "

    if act == 'reqalts':
        globals.current_recommendation_no += 1
        if not globals.current_recommendation_no >= len(globals.recommendations):
            globals.current_recommendation = globals.recommendations[globals.current_recommendation_no]
            return utils.make_recommendation("Yes! ")
        else:
            globals.alternative_recommendations = utils.find_alternatives(globals.preferences)
            if len(globals.alternative_recommendations) == 0 or globals.is_alternative == 1:
                return "hello", "I'm afraid there are no more alternatives. Tell me what you would like to change to your query."
            else:
                return "request_alternatives", "I'm afraid there are no more perfect matches. Would you like to hear similar alternatives or make changes to your query?"

    return "handle_requests", "I'm afraid I don't understand you."
