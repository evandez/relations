updates from me. I've been working on testing how methods generalize for some syntactical/simple reasoning relations. Here's a summary. 

I have tested for 4 cases

1. superlative/comerative of an adjective. good -> better -> best

Only corner isn't good enough. However `w = Jacobian` and `bias = corner` works great!

2. First letter of a token. 

Corner doesn't work without the jacobian. Jacobian + Bias is also working great! (sorry for the wrong report during the meeting. found a silly bug in my code)

3. `(n (spelled out), {} comes after, n-1 (spelled out))`

Nothing is working. Perhaps some sort of reasoning happens inside the model to “know” that sixteen comes after fifteen. Maybe, this knowledge isn’t factual.

4. Hypernym of a subject
```
 car is a vehicle.
 salmon is a fish.
 {} is a
```
Just corner isn’t enough. Needs jacobian. Jacobian + bias combination is bad.

For all of these tasks, 
* the full GPT-J can perform these tasks with reasonable accuracy. 
* I also filtered out the subjects and objects that were tokenized to multiple tokens.