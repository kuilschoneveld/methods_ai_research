
def init():
    """
    A helper method used to access global variables from other modules.
    """
    global feature_matrix, price_ranges, areas, food_types, recommendations, current_recommendation, current_recommendation_no, alternative_recommendations
    global current_recommendation_no, alternative_recommendations, alternative_recommendations, is_alternative, curr_cat, curr_val, switch_distance
    global switch_Lev_makesure, switch_restart, switch_delay, switch_CAP, switch_text2speech, lev_defaultyes, switch_baseline1, switch_baseline2
    global switch_dectree, switch_nn, feature_matrix, preferences, questions, system_state, model

    feature_matrix = None

    price_ranges = []
    areas = []
    food_types = []
    
    recommendations = []
    current_recommendation = {}
    current_recommendation_no = 0
    alternative_recommendations = []
    is_alternative = 0
    
    curr_cat = None
    curr_val = None
    
    # Configurability
    switch_distance = True  # 1
    switch_Lev_makesure = True  # 2
    switch_restart = True  # 9
    switch_delay = False  # 10
    switch_CAP = False  # 14
    switch_text2speech = False  # 15
    lev_defaultyes = False  # initially need to be False, related to #2
    
    # Please only choose one of these models. Default is logistic regression.
    # Please note that the baseline models are barely functioning with this script.
    switch_baseline1 = False  # 5
    switch_baseline2 = False  # 5
    switch_dectree = False  # 5
    switch_nn = False  # 5
    
    # slots:
    preferences = {'food_type': None,
                   'area': None,
                   'price_range': None}
    
    questions = {'food_type': 'What kind of food are you looking for?',
                 'price_range': 'What price range are you comfortable with?',
                 'area': 'Where do you want to go to eat?'}
    
    system_state = ""
    
    model = None