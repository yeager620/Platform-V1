# [DEPRECATED] Platform-V1
Data Analytics and Betting Engine

## Note for anyone looking through this repo:

This repo is *mostly* garbage and hasnt really been updated since 1/??/25. AFAIK any strategy that can be directly plucked from this repo is either not profitable (due to lookahead bias or the like) or not feasible.

### However;

Profitable strategies have been derived from the infrastructure available in this repo. The most difficult part (IMO) about a project like this one is the data pipelining and feature engineering infrastructure. Although quite slow and unreadable, this archive does have great data pipelining and feature engineering infrastructure with minimal 3rd party dependencies (with the exception of pandas of course, and a handful of libraries related to requests / web scraping and kalman filtering). At least when I begun this project, the publicly available APIs and SDKs for MLB data realy suck for meaningful data analysis and are mostly oriented towards people interested in fun facts / quick statistics, game summaries, and individual players, rather than serious predictive modeling.
