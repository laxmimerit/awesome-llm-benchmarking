# Grader corrections

Two items in the trap set were wrong when the sweep ran, and the models were
right. Both are recorded here rather than quietly patched, because the
corrections change reported scores.

Correctness in this eval is a pure function of the extracted answer, and every
answer is stored in `results/quality_hard.jsonl`, so both fixes were applied by
re-running the graders offline (`research/rescore.py`) rather than by spending
another five GPU hours. The original file is left untouched;
`results/quality_hard_rescored.jsonl` carries `correct_original` and
`dept_original` on every row so the change is auditable.

## T8: not a trap at all

> A train leaves the station at 3:00 pm travelling at 60 miles per hour. A
> second train leaves at 4:00 pm travelling at 80 miles per hour. At what time
> does the second train catch the first?

I wrote this as unanswerable on the grounds that direction and separation are
never stated. That reasoning is wrong: leaving *the* station implies a shared
origin, and "catches the first" implies a shared direction. The chase is well
posed. At 4:00 pm the first train is 60 miles ahead, the closing speed is 20
mph, so it takes three hours and the answer is 7:00 pm.

All six models answered 7:00 pm. Reclassified as a solvable item with that
answer. Effect: every model gains one point, the solvable set grows from 20
items to 21, and the trap set shrinks from 8 to 7.

## T6: zero is a defensible answer

> A bag contains 3 red balls and 4 blue balls. Eight balls are drawn without
> replacement. What is the probability that every ball drawn is red?

The draw cannot happen, so IMPOSSIBLE is a good answer. But answering that the
probability is 0 is a reasonable reading of an impossible event rather than a
fabrication, and treating it as a hallucination would be unfair. The checker
now accepts both. Effect: Granite 4.2 8B gains one point.

## Items deliberately left alone

T5 (a "arithmetic" sequence that is geometric), T7 (a false injectivity
premise) and the other traps stand. In those the prompt asserts something
false, and a model that plays along and computes an answer is doing the thing
the item is designed to catch.
