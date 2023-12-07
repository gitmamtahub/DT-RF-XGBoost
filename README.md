# DT-RF-XGBoost
3 mostly used and powerful supervised ML algorithms or models ----Decision Tree , Random Forest and XGBoost

Decision Tree - A decision tree is like a game of 20 Questions. 
You start with a big question at the top (the root), and as you answer yes or no, you follow the tree branches down to more specific questions (nodes). 
Each time you answer a question, you follow the path down to the next one. This continues until you reach the end of a path (a leaf node), where you find your final answer or decision.
Decision trees are a type of machine learning algorithm that make decisions by asking a series of questions. 
They’re simple to understand and can be drawn and explained visually, which is a big advantage.
Disadvantages - overfiting, sensitive to small changes in data, difficulty in capturing complex relationships of features

Random Forest   -   In a Random Forest, instead of just one decision tree making all the decisions, we create an entire “forest” of decision trees. 
Each tree gives its “opinion” or prediction based on the data it has seen. 
The final output is then determined by considering the output of all the trees in the forest.
Advantages - Robust to overfiting, Handles Large Datasets and feature Spaces, Parallelizable, Feature Importance
Disadvantages - Complexity,Less Interpretability, Longer prediction time.

XGBoost   -    XGBoost, which stands for “eXtreme Gradient Boosting,” is an advanced implementation of the gradient boosting algorithm.
Gradient boosting is a machine learning technique where the main idea is to combine many simple models, also known as “weak learners,” to create an ensemble model that is better at prediction.
The fundamental concept of XGBoost, like other boosting methods, is to add new models into the ensemble sequentially. 
However, unlike bagging methods like Random Forest where trees are grown in parallel, boosting methods train models one after another, each new tree helping to correct errors made by the previously trained tree.
Advantages - Performance, Speed, Versatility, Robustness
Disadvantages - Leads to overfiting if not tuned carefully, Less interpretability
