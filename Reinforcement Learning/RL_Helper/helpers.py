import pandas as pd
import numpy as np

def round_to_nearest_grade(x):
    return round(x * 4) / 4

def add_non_choices(df, all_courses):
    students = df['StudentID'].unique()
    result = pd.DataFrame()
    for i in students:
        df_temp = df[df['StudentID'] == i]
        not_taken = (list(set(all_courses) - set(df_temp[df_temp['StudentID'] == i].CourseID.values)))
        df_temp2 = pd.DataFrame({
            'StudentID': [i] * len(not_taken),
            'CourseID': not_taken, 
            'Course_Taken': [0] * len(not_taken)
            })
        df_student = pd.concat([df_temp, df_temp2])
        result = pd.concat([result, df_student])
        
    return result

def aggregate_policy_reward(replay_df): # extracting mean reward per policy
    items = list(replay_df['item_id'].unique())
    df_result = pd.DataFrame()
    for i in items:
        grade = pd.DataFrame(replay_df[replay_df['item_id'] == i]).reward.mean()
        df_temp = pd.DataFrame({
            'CourseID': [i],
            'Rank': [grade]})
        df_result = pd.concat([df_result, df_temp])
    return df_result

# input student data (list of strings) of user j (index number as integer)

def filter_T1_courses(candidates): # df of candidates
    courses_not_taken = list(candidates.CourseID)
    prefix = 'T1'
    for course in courses_not_taken[:]:
        if course.startswith(prefix):
            courses_not_taken.remove(course)
    selection = candidates[candidates.CourseID.isin(courses_not_taken)]
    return selection

def bandit(candidates, x):# x is the number of recommendations to be obtained 
    random.seed(32)
    chosen_arms = pd.DataFrame()
    candidates = candidates.sort_values(by = 'Rank', ascending = False)
    df_temp = pd.DataFrame({ # adding the first 50 % of x recommendations needed to the list, as candidates are sorted in descending order
        'CourseID':candidates.CourseID[: int(x / 2)],
        'Rank':candidates.Rank[: int(x / 2)]
    })
    chosen_arms = pd.concat([chosen_arms, df_temp])
    count = int(x / 2)
    candidates = candidates[~candidates.CourseID.isin(chosen_arms.CourseID)] # filter candidates
    explore_arms = int(x / 6)
    count += explore_arms
    a = candidates.CourseID.to_list()
    ex = list(np.random.choice(a, explore_arms, replace = False))
    temp = pd.DataFrame()
    for i in ex:
        b = candidates.loc[candidates['CourseID'] == i]
        df = pd.DataFrame({
            'CourseID': b.CourseID.values,
            'Rank': b.Rank.values
        })
        temp = pd.concat([temp, df])
    chosen_arms = pd.concat([chosen_arms, temp])
    candidates = candidates[~candidates.CourseID.isin(chosen_arms.CourseID)] # filter candidates
    remaining = x - count
    df_temp = pd.DataFrame({ 
        'CourseID':candidates.CourseID[: remaining],
        'Rank':candidates.Rank[: remaining]
    })
    chosen_arms = pd.concat([chosen_arms, df_temp])
    return chosen_arms

# obtain recommendations: corresponds to Algorithm 6, p. 49
# input: test set, aggregated policy, x = number of recommendations to be obtained
def get_recommendations(test_df, corrected_policy, x):
    rec_df = pd.DataFrame()
    test_df_index = list(test_df.StudentID.unique())
    for i in test_df_index:
        student_data = test_df[test_df['StudentID'] == i].dropna(axis=0) 
        courses_taken = student_data.CourseID.to_list()
        candidates = corrected_policy[~corrected_policy.CourseID.isin(courses_taken)]
        recommendations = bandit(candidates, x)
        recommendations['StudentID'] = [i] * len(recommendations)
        rec_df = pd.concat([rec_df, recommendations])   
    return rec_df
        