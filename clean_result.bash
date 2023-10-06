{
    head -n 1 '/Users/a123/proj/bird/records/Experiment1-Bird Classification/eval_results.txt';
    tail -n +2 '/Users/a123/proj/bird/records/Experiment1-Bird Classification/eval_results.txt' | sort -n -t '.' -k1
} > '/Users/a123/proj/bird/records/Experiment1-Bird Classification/sorted_eval_results.txt'
