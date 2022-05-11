# Prediction of Community Approval of Reddit Posts

This repository contains the source code for reproducing the results in the report "Prediction of Community Approval of Reddit Posts".

## Abstract:

In this study, we predict of the approval of a new post in a specific community within the social media platform Reddit. We divide this task into three sub-tasks: predicting whether the net amount of upvotes on the post will surpass a certain threshold, predicting directly the upvote ratio the post will receive, and predicting which of three ranges the upvote ratio will fall into. We attempt to build models for these tasks using only textual data from the Reddit posts themselves, including the title and text, avoiding any features that may only be collected after the post has been released such as comment count. We also introduce a custom loss function for dealing with the imbalanced nature of Reddit upvote data. We train and test our models on several university subreddits and find that our models perform reasonably well with respect to a random baseline.