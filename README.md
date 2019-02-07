# SpellingCorrector

--author -- AR Dirkson 
--date -- 5-2-2019 

This script is a spelling correction module that uses unsupervised data to construct a list of candidates. The correction algorithm is a weighted Levenshtein distance algorithm. A decision process is used to determine if a word is a spelling mistake.

It makes use of the CELEX generic dictionary but this can be substituted by another generic dictionary. It is only used to determine if a word should not be corrected because it is a generic word. 

The matrices for edit weights are included (but change the paths!) and due to licensing, CELEX is not.

The grid used for the spelling mistake detection was [0.05 - 0.15] (steps of 0.01) for relative weighted edit distance max and [2-10] (steps of 1) for relative corpus frequency multiplier. F0.5 was used as a metric. This can be re-tuned (tuning not included in this script).

Note: the damlev module only works on Linux platforms and the input data needs to be tokenized
