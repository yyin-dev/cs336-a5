## Problem (math_baseline)

(a) `math_baseline.py`. I evaluated on GSM8k test set, using `r1_zero_reward_function`

With `fast=True`: 

* format=1, answer=1, reward=1: 0
* format=1, answer=0, reward=0: 258
* format=0, answer=0, reward=0: 1061.

With `fast=False`:

* format=1, answer=1, reward=1: 78
* format=1, answer=0, reward=0: 180
* format=0, answer=0, reward=0: 1061.

With `fast=True`, sometimes the reward function doesn't reward correct-looking responses, so it's important to use `fast=False`. From this point on, use `fast=False` unless otherwise specified. 

(b) Observed cases where the format reward is zero: 

* the grader requires both `</think> <answer>` and `</answer>`, but the model doesn't always do. Sometimes it forgot to close with `</answer>`, sometimes it output the answer doesn't without closing `</think>`. However, the model could be outputting correct answers.
* the model doesn't finish answer with in the answer token count. 

```
{"question": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?", "truth": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3", "response": "initial`` The problem can be broken down as follows:\n\n1. Determine how much white fiber is required based on the amount of blue fiber.\n2. Sum the amount of blue and white fibers to get the total amount required.\n\nHere's the step-by-step reasoning and the corresponding Python code to solve the problem.\n\n1. The problem states that the robe takes 2 bolts of blue fiber.\n2. It also states that it takes half the amount of white fiber compared to blue fiber. So, if the blue fiber is 2 bolts, the white fiber will be \\( \\frac{2}{2} = 1 \\) bolt.\n3. The total amount of fiber required is the sum of the blue and white fibers: \\( 2 + 1 = 3 \\) bolts.\n\nThe Python code to solve this problem is straightforward:\n\n```python\n# Define the amount of blue fiber\nblue_fiber = 2\n\n# Calculate the amount of white fiber\nwhite_fiber = blue_fiber / 2\n\n# Calculate the total amount of fiber required\ntotal_fiber = blue_fiber + white_fiber\n\n# Print the total fiber required\nprint(total_fiber)\n```\n```output\n3.0\n```\nThe total amount of fiber required to make the robe is \\( \\boxed{3} \\) bolts.", "reward": {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}}
```

I think the issue is with the model.

Observed cases where the format reward is 1 but the answer reward is 0: I observed 10 cases; 2 out of 10 the model output correct answer, but still the answer reward is 0. This is problem with the parser. 

```
{"question": "Peter plans to go to the movies this week. He always gets a ticket for $7 and popcorn for $7. If he has 42 dollars for the week, how many times can he go to the movies?", "truth": "It costs him $14 to go to the movies because 7 + 7 = <<7+7=14>>14\nHe can go to the movies 3 times because 42 / 14 = <<42/14=3>>3\n#### 3", "response": " Peter can buy 3 times the movies for 42 dollars. </think> <answer> 3 </answer>", "reward": {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}}

{"question": "A wooden bridge can carry no more than 5000 pounds. A delivery truck filled with identical boxes, each weighing 15 pounds, will pass over the bridge. The combined weight of the driver and the empty truck is 3755 pounds. What is the maximum number of boxes which can be loaded onto the truck while not exceeding the bridge's weight limit?", "truth": "The boxes can weigh up to 5000 pounds - 3755 pounds = <<5000-3755=1245>>1245 pounds in total.\nThere can be 1245 / 15 = <<1245/15=83>>83 boxes loaded onto the truck without exceeding the bridge's weight limit.\n#### 83", "response": " The truck's weight without boxes is 3755 pounds. The remaining capacity of the bridge is 5000 - 3755 = 1245 pounds. Each box weighs 15 pounds, so the maximum number of boxes that can be loaded is 1245 / 15 = 83 boxes. </think> <answer> 83 boxes (answer format) </answer>", "reward": {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}}
```

(c) On GSM8k test set: 

* Format correct ratio: 258/1319 = 19.5%. 
* Answer correct ratio: 78 / 1319 =5.9%



## problem (compute_entropy)

Starting with softmax:
$$
p(x) = \frac{e^{\text{logits}(x)}}{\sum_{x \in X} e^{\text{logits}(x)}}
$$
Taking the logarithm:
$$
\log p(x) = \log \frac{e^{\text{logits}(x)}}{\sum_{x \in X} e^{\text{logits}(x)}} = \log e^{\text{logits}(x)} - \log \sum_{x \in X} e^{\text{logits}(x)} = \text{logits}(x) - \text{logsumexp}(x)
$$
Therefore:
$$
\boxed{\log p(x) = x - \text{logsumexp}(x)}
$$
And equivalently:
$$
\boxed{p(x) = e^{(x - \text{logsumexp}(x))}}
$$
Thus, 
$$
p(x)log(p(x)) = e^{(x - \text{logsumexp}(x))} (x - \text{logsumexp}(x))
$$

## problem (sft_microbatch_train_step)

What's the loss function in SFT? Why do we use "negative log likelihood "?

* "likelihood" is just probability. LLM products probabilities at each generation step. The goal is to maximize the probability that the model generates the label response, which is the product of the probabilities that the model generates each token in the label.
* "negative": Because gradient descent minimizes the loss while we want to maximize the probability, we use the negative probability as loss. 
* "log": The process of multiplying a sequence of probabilities is numerically unstable and can cause underflow, using log probabilities turns multiplication into sum. 

How is cross-entropy loss related to negative log likelihood? Cross-entropy loss in this case is another name for negative log likelihood. Cross-entropy measures the difference between two distributions. When you apply cross-entropy loss with a "groud-truth" distribution that's a one-hot vector, the formula simplifies to negative log likelihood.

