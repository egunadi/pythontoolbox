from typing import List, Dict, Any

"""
1-1.  finding "key connectors" 
      with respect to "degree centrality"
      given user and friendship data
"""
users_list = [
  { "id": 0, "name": "Hero" },
  { "id": 1, "name": "Dunn" },
  { "id": 2, "name": "Sue" },
  { "id": 3, "name": "Chi" },
  { "id": 4, "name": "Thor" },
  { "id": 5, "name": "Clive" },
  { "id": 6, "name": "Hicks" },
  { "id": 7, "name": "Devin" },
  { "id": 8, "name": "Kate" },
  { "id": 9, "name": "Klein" }
]

friendship_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
                    (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

# refactor friendship_pairs as a dict with keys as user ids and values as list of friend ids
# first, initialize the dict with an empty list for each user id:
friendships_dict = {user["id"]: []
                    for user in users_list}

# next, loop over the friendship pairs to populate it
for i, j in friendship_pairs:
  friendships_dict[i].append(j)  # add j as a friend of user i
  friendships_dict[j].append(i)  # add i as a friend of user j

# print(friendships_dict)

# OUTPUT
# {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2, 4], 4: [3, 5], 
#  5: [4, 6, 7], 6: [5, 8], 7: [5, 8], 8: [6, 7, 9], 9: [8]}

#analysis
def get_number_of_friends(
  user: Dict[str, Any]  # keys are strings, values can be either int or string
                        # ex. { "id": 0, "name": "Hero" }
) ->    int:
  """How many friends does _user_ have?"""
  user_id = user["id"]
  friend_id_list = friendships_dict[user_id]
  return len(friend_id_list)

total_connections = sum(get_number_of_friends(user)
                        for user in users_list)

# print(total_connections) # 24

num_users = len(users_list)                     # length of the users list
avg_connections = total_connections / num_users # 24 / 10 == 2.4

# print(avg_connections) # 2.4

# create a list of (user_id, number_of_friends) tuples
num_friends_by_id_list = [(user["id"], get_number_of_friends(user))
                          for user in users_list]

num_friends_by_id_list.sort(         # sort the list "in place"
  key = lambda tuple: tuple[1], # by number_of_friends
  reverse = True                # largest to smallest
)

# print(num_friends_by_id_list)
# each pair is (user_id, number_of_friends)
# user ids listed first have highest "degree centrality"
# see ch. 22 for more complex notions of centrality

# OUTPUT
# [(1, 3), (2, 3), (3, 3), (5, 3), (8, 3), 
#  (0, 2), (4, 2), (6, 2), (7, 2), (9, 1)]

"""1-2. building a "people you may know" suggester"""
from collections import Counter

def get_friends_of_friends(
  user: Dict[str, Any]  # keys are strings, values can be either int or string
                        # ex. { "id": 0, "name": "Hero" }
) ->    Dict[int, int]:
  """foaf is short for "friend of a friend" """
  user_id = user["id"]
  counter = Counter(
    foaf_id
    for friend_id in friendships_dict[user_id]      # for each of my friends,
    for foaf_id in friendships_dict[friend_id]      # find their friends
    if foaf_id != user_id                           # who aren't me
      and foaf_id not in friendships_dict[user_id]  # and aren't my friends
  )
  return dict(counter)

# print(get_friends_of_friends(users_list[3])) # {0: 2, 5: 1}
# keys are user_ids, values are counts of foaf(s)

"""
1-3.  given a list of (user_id, interest) tuples,
      suggest friends based on similar interests
"""
from collections import defaultdict

interests_list = [
  (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
  (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
  (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
  (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
  (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
  (3, "statistics"), (3, "regression"), (3, "probability"),
  (4, "machine learning"), (4, "regression"), (4, "decision trees"),
  (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
  (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
  (6, "probability"), (6, "mathematics"), (6, "theory"),
  (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
  (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
  (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
  (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

# for efficiency, build two indexes 
# one for interests to users
user_ids_by_interest_dict = defaultdict(list)

# keys are interests, values are lists of user_ids with that interest
for user_id, interest in interests_list:
  user_ids_by_interest_dict[interest].append(user_id)

# and another for users to interests
interests_by_user_id_dict = defaultdict(list)

for user_id, interest in interests_list:
  interests_by_user_id_dict[user_id].append(interest)

def get_users_with_most_common_interests(
  user: Dict[str, Any]  # keys are strings, values can be either int or string
                        # ex. { "id": 0, "name": "Hero" }
) ->    Dict[int, int]:
  counter = Counter(
    interested_user_id
    for interest in interests_by_user_id_dict[user["id"]]         # for my interests,
    for interested_user_id in user_ids_by_interest_dict[interest] # find others also interested
    if interested_user_id != user["id"]                           # who aren't me
  )
  return dict(counter)

# print(get_users_with_most_common_interests(users_list[3])) # {5: 2, 2: 1, 6: 2, 4: 1}
# keys are user_ids, values are counts of common interest(s)

"""
1-4.  given a list of (salary, tenure) tuples,
      return average salary by tenure bucket
"""
salaries_and_tenures_list = [ (83000, 8.7), (88000, 8.1),
                              (48000, 0.7), (76000, 6),
                              (69000, 6.5), (76000, 7.5),
                              (60000, 2.5), (83000, 10),
                              (48000, 1.9), (63000, 4.2)  ]

def get_tenure_bucket(
  tenure: float
) ->      str:
  if tenure < 2:
    return "less than two"
  elif tenure < 5:
    return "between two and five"
  else:
    return "more than five"

# keys are tenure buckets, values are lists of salaries for that bucket
salary_by_tenure_bucket_dict = defaultdict(list)

for salary, tenure in salaries_and_tenures_list:
  bucket = get_tenure_bucket(tenure)
  salary_by_tenure_bucket_dict[bucket].append(salary)

# keys are tenure buckets, values are average salary for that bucket
average_salary_by_bucket_dict = {
  tenure_bucket: sum(salaries) / len(salaries)
  for tenure_bucket, salaries in salary_by_tenure_bucket_dict.items()
}

# print(average_salary_by_bucket_dict)

# OUTPUT
"""
{ 'more than five': 79166.66666666667, 
  'less than two': 48000.0, 
  'between two and five': 61500.0 }
"""

"""
1-5.  given a list of (user_id, interest) tuples,
      return a list of topics with the most interest
"""
words_counter = Counter(word
                        for user, interest in interests_list
                        for word in interest.lower().split())

# for word, count in words_counter.most_common():
#   if count > 1:
#     print(word, count)

# perhaps a cleaner/better approach:
common_words_list = [ (word, count)
                      for word, count in dict(words_counter.most_common()).items()
                      if count > 1  ]

# print(common_words_list)

# OUTPUT
# [('big', 3), ('data', 3), ('java', 3), ('python', 3), ('learning', 3), ('hadoop', 2), ('hbase', 2), ('cassandra', 2), ('scikit-learn', 2), ('r', 2), ('statistics', 2), ('regression', 2), ('probability', 2), ('machine', 2), ('neural', 2), ('networks', 2)]
